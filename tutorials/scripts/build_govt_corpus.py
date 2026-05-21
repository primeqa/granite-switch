"""Build or load the ChromaDB government-services corpus (notebook section 3).

Adds per-batch latency and throughput reporting and CLI control over batch
size and embedding backend.

Usage:
    python build_govt_corpus.py [--batch-size N] [--backend pytorch|onnx|openvino]
                                [--dtype fp32|fp16|bf16] [--device cpu|cuda]
                                [--chroma-path PATH] [--jsonl-path PATH] [--full-corpus]
"""

import argparse
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

load_dotenv(Path(__file__).parent.parent / ".env", override=False)

import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from granite_switch.tutorials.govt_data_loader import (
    CHROMA_PATH as _DEFAULT_CHROMA_PATH,
    EMBEDDING_MODEL_ID,
    GOVT_JSONL_PATH as _DEFAULT_JSONL_PATH,
    GOVT_JSONL_URL,
    TUTORIAL_DOC_IDS,
    load_or_build_govt_chroma,
)


class TimedEmbeddingFunction(EmbeddingFunction):
    """Wraps model inference with per-batch latency and throughput tracking.

    Batching is delegated to ``SentenceTransformer.encode`` so passages can be
    length-sorted before grouping. Per-batch latency comes from a forward hook
    on the underlying transformer module (inference only — tokenization is
    not included).
    Timing covers tokenization + model inference for each batch.
    """

    def __init__(self, model_id: str, batch_size: int, backend: str, device: str,
                 dtype: torch.dtype = torch.float32, backend_file: str | None = None,
                 torch_compile: bool = False):
        super().__init__()
        self._model_id = model_id
        self._batch_size = batch_size
        self._backend = backend
        self._dtype = dtype
        self._device = device
        self._torch_compile = torch_compile
        self._batch_latencies: list[float] = []
        self._total_docs = 0
        self._wall_start: float | None = None
        self._batch_t0: float | None = None

        st_backend_map = {
            "pytorch":      "torch",
            "pytorch-int8": "torch",
            "torchao-int8": "torch",
            "onnx":         "onnx",
            "openvino":     "openvino",
        }
        st_backend = st_backend_map[backend]
        quant_backends = {"pytorch-int8", "torchao-int8"}

        if backend == "pytorch":
            model_kwargs: dict = {"torch_dtype": dtype}
        elif backend == "openvino":
            # ModernBERT's SDPA path trips a dtype-mismatch check during
            # torch.jit.trace (attn_mask=bf16 vs query=fp32). Force the eager
            # attention impl so the openvino export can complete.
            model_kwargs = {"attn_implementation": "eager"}
        elif backend in quant_backends:
            # Quant recipes mutate fp32 master weights, so load fp32 first.
            model_kwargs = {"torch_dtype": torch.float32}
        else:
            model_kwargs = {}
        if backend_file is not None:
            model_kwargs["file_name"] = backend_file
        if backend in {"onnx", "openvino"}:
            if backend_file is None:
                print(f"Exporting to {backend} (first run may take a few minutes)...")
            else:
                print(f"Loading {backend} from {backend_file}...")
        self._model: Any = SentenceTransformer(
            model_id, backend=st_backend, device=device, model_kwargs=model_kwargs
        )

        # Apply quantization in place on the inner HF model. The forward hook
        # below is registered on the SentenceTransformer Transformer wrapper, so
        # it still fires per batch regardless of how `auto_model` is mutated.
        if backend == "pytorch-int8":
            from torch.ao.quantization import quantize_dynamic
            transformer = self._model[0]
            transformer.auto_model = quantize_dynamic(
                transformer.auto_model.float(),
                {torch.nn.Linear},
                dtype=torch.qint8,
            ).eval()
            print("Applied torch.ao.quantization.quantize_dynamic({Linear}, qint8)")
        elif backend == "torchao-int8":
            from torchao.quantization import (  # type: ignore[import-not-found]
                Int8DynamicActivationInt8WeightConfig,
                quantize_,
            )
            transformer = self._model[0]
            quantize_(transformer.auto_model, Int8DynamicActivationInt8WeightConfig())
            transformer.auto_model = torch.compile(transformer.auto_model)  # type: ignore[assignment]
            # SentenceTransformer.encode unconditionally calls self.to(device).
            # nn.Module.to walks params via _apply, which dispatches an aten op
            # that torchao's LinearActivationQuantizedTensor subclass does not
            # implement, so the call raises. Since this benchmark stays on a
            # single device, override `to` with a no-op.
            self._model.to = lambda *_args, **_kwargs: self._model
            print("Applied torchao Int8DynamicActivationInt8Weight + torch.compile")

        # Hook the transformer submodule so we time forward() per batch
        # regardless of the outer encode() loop. Registered before torch.compile
        # so the wrapped module still owns the hook.
        first_module = next(iter(self._model.children()))

        def _pre(_mod, _inputs):
            self._batch_t0 = time.perf_counter()

        def _post(_mod, _inputs, _output):
            if self._batch_t0 is not None:
                self._batch_latencies.append(time.perf_counter() - self._batch_t0)
                self._batch_t0 = None

        first_module.register_forward_pre_hook(_pre)
        first_module.register_forward_hook(_post)

        if torch_compile:
            if backend != "pytorch":
                raise ValueError(f"--torch-compile requires --backend pytorch (got {backend})")
            self._model = torch.compile(self._model)

        dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype))
        if backend == "pytorch-int8":
            suffix = " (qint8 dynamic)"
        elif backend == "torchao-int8":
            suffix = " (w8a8 + compile)"
        elif torch_compile:
            suffix = " (compiled)"
        else:
            suffix = ""
        print(f"[{backend}/{dtype_name}] Embedding model ready on {device}{suffix}  ({model_id})")

    def __call__(self, documents: Documents) -> Embeddings:
        if self._wall_start is None:
            self._wall_start = time.perf_counter()

        docs = list(documents)
        embs = self._model.encode(
            docs,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._total_docs += len(docs)
        return embs.tolist()

    def warmup(self, documents: list[str]) -> None:
        """Run encode on `documents` and discard all timing state.

        Pays the JIT compile / CUDA kernel autotune / lazy-init costs so that
        the recorded run reflects steady-state inference performance.
        """
        self._model.encode(
            documents,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._batch_latencies.clear()
        self._total_docs = 0
        self._wall_start = None
        self._batch_t0 = None

    def report(self) -> None:
        if not self._batch_latencies:
            print("No batches embedded (collection already existed).")
            return

        lats = self._batch_latencies
        wall = time.perf_counter() - self._wall_start  # type: ignore[operator]
        throughput = self._total_docs / wall

        dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(self._dtype, str(self._dtype))
        print(f"\n── Embedding timing [{self._backend}/{dtype_name}, batch={self._batch_size}] ──")
        print(f"  Documents  : {self._total_docs:,}")
        print(f"  Batches    : {len(lats)}")
        print(f"  Wall time  : {wall:.2f}s")
        print(f"  Throughput : {throughput:.1f} docs/s")
        print(f"  Latency/batch (forward only)")
        print(f"    mean   : {statistics.mean(lats)*1000:.1f} ms")
        print(f"    median : {statistics.median(lats)*1000:.1f} ms")
        if len(lats) >= 20:
            qs = statistics.quantiles(lats, n=100)
            print(f"    p95    : {qs[94]*1000:.1f} ms")
            print(f"    p99    : {qs[98]*1000:.1f} ms")
        print(f"    min    : {min(lats)*1000:.1f} ms")
        print(f"    max    : {max(lats)*1000:.1f} ms")
        print("─" * 46)

    def save_json(self, path: str | os.PathLike) -> None:
        """Dump run metadata + per-batch latencies to a JSON file."""
        if not self._batch_latencies:
            return

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lats = self._batch_latencies
        wall = time.perf_counter() - self._wall_start  # type: ignore[operator]
        dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(self._dtype, str(self._dtype))

        data = {
            "model_id": self._model_id,
            "backend": self._backend,
            "dtype": dtype_name,
            "device": self._device,
            "torch_compile": self._torch_compile,
            "batch_size": self._batch_size,
            "documents": self._total_docs,
            "batches": len(lats),
            "wall_time_s": wall,
            "throughput_docs_per_s": self._total_docs / wall,
            "batch_latencies_ms": [t * 1000 for t in lats],
            "summary_ms": {
                "mean":   statistics.mean(lats)   * 1000,
                "median": statistics.median(lats) * 1000,
                "min":    min(lats)               * 1000,
                "max":    max(lats)               * 1000,
            },
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote timing JSON to {out_path}")


def _prepare_openvino_source(model_id: str) -> str:
    """Re-save the HF model locally with eager attention + fp32 so that
    optimum-intel's openvino exporter can JIT-trace it.

    optimum-intel drops the ``attn_implementation`` and ``torch_dtype``
    model_kwargs we pass through SentenceTransformer, so the model loads with
    SDPA + the checkpoint's native bf16 weights. ModernBERT's SDPA path then
    hits a mask-vs-query dtype mismatch during torch.jit.trace and the export
    aborts. Pre-exporting locally with the right config bakes both choices
    into the saved model so the kwargs are no longer needed at export time.
    """
    from transformers import AutoModel, AutoTokenizer

    cache_root = Path.home() / ".cache" / "granite-switch" / "ov-source"
    safe_id = model_id.replace("/", "__")
    out_dir = cache_root / safe_id
    sentinel = out_dir / "config.json"
    if sentinel.exists():
        return str(out_dir)

    print(f"Re-saving {model_id} with attn=eager, torch_dtype=fp32 -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer is None:
        raise RuntimeError(f"Could not load tokenizer for {model_id}")
    tokenizer.save_pretrained(out_dir)
    return str(out_dir)


def _read_warmup_samples(jsonl_path: str, n: int) -> list[str]:
    """Return up to `n` non-empty document texts from the corpus jsonl, or
    synthetic placeholders if the file isn't present yet (first-run case)."""
    if not os.path.exists(jsonl_path):
        return ["Warmup passage for embedding model kernel/cache initialization. " * 5] * n

    texts: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", "").strip()
            if text:
                texts.append(text)
            if len(texts) >= n:
                break
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or load the govt ChromaDB corpus with timing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=EMBEDDING_MODEL_ID, metavar="MODEL_ID",
        help="HuggingFace model ID or local path for the embedding model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Number of passages per embedding call",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "pytorch-int8", "torchao-int8", "onnx", "openvino"],
        default="pytorch",
        help="Embedding inference backend. 'pytorch-int8' applies "
             "torch.ao.quantization.quantize_dynamic (Linear weights -> qint8, "
             "activations dynamically quantized at runtime). 'torchao-int8' "
             "applies torchao Int8DynamicActivationInt8WeightConfig (W8A8) and "
             "wraps the inner module with torch.compile. Both quant paths are "
             "CPU-oriented.",
    )
    parser.add_argument(
        "--backend-file", default=None, metavar="FILENAME",
        help="Specific model file to load within the repo (e.g. model_qint8_avx512.onnx); "
             "forwarded as file_name to optimum's from_pretrained",
    )
    parser.add_argument(
        "--num-threads", type=int, default=None, metavar="N",
        help="Number of intra-op threads for PyTorch (default: PyTorch's own heuristic)",
    )
    parser.add_argument(
        "--num-interop-threads", type=int, default=None, metavar="N",
        help="Number of inter-op threads for PyTorch (default: PyTorch's own heuristic)",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
        help="Model weight dtype; fp16/bf16 reduce memory and can speed up GPU inference",
    )
    parser.add_argument(
        "--torch-compile", action="store_true",
        help="Wrap the embedding model with torch.compile() (pytorch backend only); "
             "the first batch pays a one-time compilation cost",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device for the pytorch backend (default: cuda if available, else cpu). "
             "Ignored for onnx and openvino.",
    )
    parser.add_argument(
        "--chroma-path", default=os.environ.get("CHROMA_PATH", _DEFAULT_CHROMA_PATH),
        metavar="PATH",
    )
    parser.add_argument(
        "--jsonl-path", default=os.environ.get("GOVT_JSONL_PATH", _DEFAULT_JSONL_PATH),
        metavar="PATH",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Remove --chroma-path if it exists before building",
    )
    parser.add_argument(
        "--num-samples", type=int, default=len(TUTORIAL_DOC_IDS), metavar="N",
        help=f"Number of passages to embed; pass -1 to embed the full corpus. "
             f"For N <= {len(TUTORIAL_DOC_IDS)}, draws from the curated tutorial subset "
             f"(the docs the demo queries actually retrieve). For N > {len(TUTORIAL_DOC_IDS)}, "
             f"draws the first N passages from the full corpus.",
    )
    args = parser.parse_args()

    if args.force and os.path.exists(args.chroma_path):
        shutil.rmtree(args.chroma_path)
        print(f"Removed {args.chroma_path}")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    if args.num_interop_threads is not None:
        torch.set_num_interop_threads(args.num_interop_threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    model_id_for_ef = args.model
    if args.backend == "openvino" and args.backend_file is None:
        model_id_for_ef = _prepare_openvino_source(args.model)

    ef = TimedEmbeddingFunction(
        model_id=model_id_for_ef,
        batch_size=args.batch_size,
        backend=args.backend,
        device=device,
        dtype=dtype,
        backend_file=args.backend_file,
        torch_compile=args.torch_compile,
    )

    warmup_n = 100
    warmup_docs = _read_warmup_samples(args.jsonl_path, warmup_n)
    print(f"Warming up on {len(warmup_docs)} samples (timings discarded)...")
    t_warm = time.perf_counter()
    ef.warmup(warmup_docs)
    print(f"Warmup done in {time.perf_counter() - t_warm:.2f}s")

    if args.num_samples == -1:
        load_only_tutorial = False
        max_docs = None
    elif args.num_samples <= len(TUTORIAL_DOC_IDS):
        load_only_tutorial = True
        max_docs = args.num_samples
    else:
        load_only_tutorial = False
        max_docs = args.num_samples

    t_total = time.perf_counter()
    chroma_collection = load_or_build_govt_chroma(
        chroma_path=args.chroma_path,
        jsonl_path=args.jsonl_path,
        jsonl_url=GOVT_JSONL_URL,
        embedding_model_id=args.model,
        device=device,
        load_only_tutorial_docs=load_only_tutorial,
        max_docs=max_docs,
        embedding_fn=ef,
    )
    total_time = time.perf_counter() - t_total

    ef.report()

    timing_name = (
        f"{time.strftime('%Y%m%d_%H%M%S')}"
        f"_{args.backend}_{args.dtype}_b{args.batch_size}"
        f"{'_compile' if args.torch_compile else ''}"
        f".json"
    )
    ef.save_json(Path("timings") / timing_name)

    print(f"\nCollection ready: {chroma_collection.count():,} passages in {args.chroma_path}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()