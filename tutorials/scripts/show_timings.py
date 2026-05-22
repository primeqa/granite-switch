"""Render a table of timing runs written by build_govt_corpus.py.

Reads every *.json file under the timings/ directory and prints one row per
run with the key headline metrics (throughput + per-batch latency stats).

Usage:
    python show_timings.py [--dir PATH] [--sort {file,throughput,mean,p95}]
"""

import argparse
import json
import statistics
from pathlib import Path


COLUMNS = [
    ("file",        18),
    ("model",       24),
    ("backend",     14),
    ("dtype",        5),
    ("device",       5),
    ("compile",      7),
    ("batch",        5),
    ("docs",         6),
    ("wall(s)",      8),
    ("docs/s",       8),
    ("mean(ms)",     9),
    ("p50(ms)",      8),
    ("p95(ms)",      8),
    ("p99(ms)",      8),
]


def _short_model(model_id: str) -> str:
    """Compress a HF model id to something that fits a narrow column.

    - Drops the "org/" prefix
    - Compresses common verbose tokens (embedding -> emb, english -> en)
    """
    short = model_id.rsplit("/", 1)[-1]
    for long, abbr in [
        ("embedding", "emb"),
        ("english",   "en"),
        ("multilingual", "ml"),
        ("instruct",  "inst"),
    ]:
        short = short.replace(long, abbr)
    return short


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (`pct` in [0, 100]). None if no data."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _row_from_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    lats = data.get("batch_latencies_ms") or []
    summary = data.get("summary_ms") or {}
    # Derive a short "file" label from the timestamp prefix in the filename.
    stem = path.stem
    parts = stem.split("_", 2)
    file_label = "_".join(parts[:2]) if len(parts) >= 2 else stem
    return {
        "file":     file_label,
        "model":    _short_model(data.get("model_id", "?")),
        "backend":  data.get("backend", "?"),
        "dtype":    data.get("dtype", "?"),
        "device":   data.get("device", "?"),
        "compile":  "yes" if data.get("torch_compile") else "no",
        "batch":    data.get("batch_size", 0),
        "docs":     data.get("documents", 0),
        "wall_s":   data.get("wall_time_s", 0.0),
        "tput":     data.get("throughput_docs_per_s", 0.0),
        "mean_ms":  summary.get("mean") or (statistics.mean(lats) if lats else None),
        "p50_ms":   summary.get("median") or (statistics.median(lats) if lats else None),
        "p95_ms":   _percentile(lats, 95),
        "p99_ms":   _percentile(lats, 99),
    }


def _fmt(value, width: int) -> str:
    if value is None:
        s = "-"
    elif isinstance(value, float):
        s = f"{value:.1f}"
    else:
        s = str(value)
    if len(s) > width:
        s = s[:width - 1] + "…"
    return s.ljust(width)


def _print_table(rows: list[dict]) -> None:
    header = "  ".join(name.ljust(w) for name, w in COLUMNS)
    print(header)
    print("  ".join("-" * w for _, w in COLUMNS))

    key_for_col = {
        "file": "file", "model": "model", "backend": "backend", "dtype": "dtype",
        "device": "device", "compile": "compile", "batch": "batch",
        "docs": "docs", "wall(s)": "wall_s", "docs/s": "tput",
        "mean(ms)": "mean_ms", "p50(ms)": "p50_ms",
        "p95(ms)": "p95_ms", "p99(ms)": "p99_ms",
    }
    for row in rows:
        print("  ".join(_fmt(row[key_for_col[name]], w) for name, w in COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a table of timing runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir", default="timings", help="Directory containing *.json timing files")
    parser.add_argument(
        "--sort", default="file",
        choices=["file", "throughput", "mean", "p95"],
        help="Sort key (throughput is descending, latency keys are ascending)",
    )
    args = parser.parse_args()

    timings_dir = Path(args.dir)
    if not timings_dir.is_dir():
        raise SystemExit(f"Directory not found: {timings_dir}")

    files = sorted(timings_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No *.json files in {timings_dir}")

    rows = [_row_from_json(p) for p in files]

    if args.sort == "throughput":
        rows.sort(key=lambda r: r["tput"] or 0, reverse=True)
    elif args.sort == "mean":
        rows.sort(key=lambda r: r["mean_ms"] if r["mean_ms"] is not None else float("inf"))
    elif args.sort == "p95":
        rows.sort(key=lambda r: r["p95_ms"] if r["p95_ms"] is not None else float("inf"))
    else:  # file
        rows.sort(key=lambda r: r["file"])

    _print_table(rows)


if __name__ == "__main__":
    main()
