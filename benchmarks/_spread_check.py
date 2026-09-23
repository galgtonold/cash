"""Is the ~15ms step in the mid-size cells of the cost-model matrix real?

The frozen matrix (`results/ser_deser_matrix.csv`) is non-monotonic in a way no
line can fit: list_flat deserialises 11.8 KB in 0.47 ms and 115 KB in 14.94 ms,
dict_shallow 10 KB in 0.47 ms and 95 KB in 16.88 ms. Those steps are what push
the fitted intercept to ~10 ms, which is the error the module documents.

`measure_one` reports a median of 3 repeats, so this re-measures the same cells
with every sample printed. A step that survives is a real cost; one that moves
around is noise the fit should not be shaped by.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmarks._object_generators import estimate_in_memory_size, make_object  # noqa: E402

CELLS = [
    ("list_flat", 10_000),
    ("list_flat", 100_000),
    ("list_flat", 1_000_000),
    ("dict_shallow", 10_000),
    ("dict_shallow", 100_000),
    ("dataframe_numeric", 10_000),
    ("dataframe_numeric", 100_000),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--cache-root", type=Path, default=Path("benchmarks/results/_spread_cache"))
    args = p.parse_args()

    from cash.backends.file_backend import FileBackend

    print(f"{'family':>18} {'size':>12} {'median':>9} {'min':>8} {'max':>8}  samples (ms)")
    for family, target in CELLS:
        obj = make_object(family, target)
        size = estimate_in_memory_size(obj)
        root = args.cache_root / f"{family}-{target}"
        root.mkdir(parents=True, exist_ok=True)
        backend = FileBackend(str(root))
        samples = []
        for i in range(args.repeats + 1):
            key = f"spread:{family}:{target}:{i}"
            backend.set(key, {"variables": {"v": obj}}, {"timestamp": 0.0})
            t0 = time.perf_counter()
            backend.get(key)
            samples.append((time.perf_counter() - t0) * 1000)
        samples = samples[1:]  # drop the warmup, as the matrix does
        print(
            f"{family:>18} {size:>12,} {statistics.median(samples):>7.2f}ms "
            f"{min(samples):>6.2f}ms {max(samples):>6.2f}ms  "
            f"{' '.join(f'{s:.1f}' for s in samples[:12])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
