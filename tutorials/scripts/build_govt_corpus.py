"""Build or load the ChromaDB government-services corpus (notebook section 3).

Adds per-batch latency and throughput reporting and CLI control over batch
size and embedding backend.

Usage:
    python build_govt_corpus.py [--batch-size N] [--backend pytorch|onnx|openvino]
                                [--dtype fp32|fp16|bf16] [--device cpu|cuda]
                                [--chroma-path PATH] [--jsonl-path PATH] [--full-corpus]
"""

import argparse
import os
import shutil
import statistics
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from granite_switch.tutorials.govt_data_loader import (
    CHROMA_PATH as _DEFAULT_CHROMA_PATH,
    EMBEDDING_MODEL_ID,
    GOVT_JSONL_PATH as _DEFAULT_JSONL_PATH,
    GOVT_JSONL_URL,
    load_or_build_govt_chroma,
)


class TimedEmbeddingFunction(EmbeddingFunction):
    """Wraps model inference with per-batch latency and throughput tracking.

    Timing covers tokenization + model inference for each batch.
    """

    def __init__(self, model_id: str, batch_size: int, backend: str, device: str,
                 dtype: torch.dtype = torch.float32, backend_file: str | None = None):
        self._batch_size = batch_size
        self._backend = backend
        self._dtype = dtype
        self._device = device
        self._batch_latencies: list[float] = []
        self._total_docs = 0
        self._wall_start: float | None = None

        st_backend = {"pytorch": "torch", "onnx": "onnx", "openvino": "openvino"}[backend]
        model_kwargs: dict = {"torch_dtype": dtype} if backend == "pytorch" else {}
        if backend_file is not None:
            model_kwargs["file_name"] = backend_file
        if backend != "pytorch":
            print(f"Exporting to {backend} (first run may take a few minutes)...")
        self._model = SentenceTransformer(
            model_id, backend=st_backend, device=device, model_kwargs=model_kwargs
        )

        dtype_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype))
        print(f"[{backend}/{dtype_name}] Embedding model ready on {device}  ({model_id})")

    def __call__(self, input: Documents) -> Embeddings:
        if self._wall_start is None:
            self._wall_start = time.perf_counter()

        all_embs = []
        for i in range(0, len(input), self._batch_size):
            batch = list(input[i : i + self._batch_size])
            t0 = time.perf_counter()
            embs = self._model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            self._batch_latencies.append(time.perf_counter() - t0)
            self._total_docs += len(batch)
            all_embs.extend(embs.tolist())

        return all_embs

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
        print(f"  Latency/batch (tokenize + infer)")
        print(f"    mean   : {statistics.mean(lats)*1000:.1f} ms")
        print(f"    median : {statistics.median(lats)*1000:.1f} ms")
        if len(lats) >= 20:
            qs = statistics.quantiles(lats, n=100)
            print(f"    p95    : {qs[94]*1000:.1f} ms")
            print(f"    p99    : {qs[98]*1000:.1f} ms")
        print(f"    min    : {min(lats)*1000:.1f} ms")
        print(f"    max    : {max(lats)*1000:.1f} ms")
        print("─" * 46)


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
        "--backend", choices=["pytorch", "onnx", "openvino"], default="pytorch",
        help="Embedding inference backend",
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
        "--full-corpus", action="store_true",
        help="Embed the full corpus instead of the tutorial subset",
    )
    parser.add_argument(
        "--num-samples", type=int, default=130, metavar="N",
        help="Maximum number of passages to embed (default: 130)",
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

    ef = TimedEmbeddingFunction(
        model_id=args.model,
        batch_size=args.batch_size,
        backend=args.backend,
        device=device,
        dtype=dtype,
        backend_file=args.backend_file,
    )

    t_total = time.perf_counter()
    chroma_collection = load_or_build_govt_chroma(
        chroma_path=args.chroma_path,
        jsonl_path=args.jsonl_path,
        jsonl_url=GOVT_JSONL_URL,
        embedding_model_id=args.model,
        device=device,
        load_only_tutorial_docs=not args.full_corpus,
        max_docs=args.num_samples,
        embedding_fn=ef,
    )
    total_time = time.perf_counter() - t_total

    ef.report()
    print(f"\nCollection ready: {chroma_collection.count():,} passages in {args.chroma_path}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()