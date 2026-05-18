"""Cost-model measurement matrix CLI.

Iterates over (family, target_bytes, backend) cells and writes one CSV
row per measurement to ``--out``. See the spec for the matrix shape.

Usage:
    python benchmarks/measure_ser_deser.py
        [--out PATH] [--cache-root DIR]
        [--families a,b,c] [--sizes 1000,10000,...] [--backends ram,disk]
        [--repeats N]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

# Make benchmarks/ importable as a top-level package when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._object_generators import FAMILIES
from benchmarks._ser_deser_measure import MeasureResult, measure_one


_DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
_DEFAULT_BACKENDS = ["ram", "disk"]


def _parse_list(s: str, conv=str) -> list:
    return [conv(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cost-model measurement matrix")
    p.add_argument("--out", type=Path,
                   default=Path("benchmarks/results/ser_deser_matrix.csv"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("benchmarks/results/_ser_deser_cache"))
    p.add_argument("--families", type=str, default=",".join(FAMILIES))
    p.add_argument("--sizes", type=str,
                   default=",".join(str(x) for x in _DEFAULT_SIZES))
    p.add_argument("--backends", type=str, default=",".join(_DEFAULT_BACKENDS))
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)

    families = _parse_list(args.families)
    sizes = _parse_list(args.sizes, int)
    backends = _parse_list(args.backends)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[MeasureResult] = []

    for family in families:
        for size in sizes:
            for backend_kind in backends:
                cell_cache = args.cache_root / f"{family}-{size}-{backend_kind}"
                r = measure_one(
                    family=family,
                    target_bytes=size,
                    backend_kind=backend_kind,
                    repeats=args.repeats,
                    cache_root=cell_cache,
                )
                rows.append(r)
                label = "OK" if r.error is None else f"ERR: {r.error[:40]}"
                print(f"[{family:18s} {size:>10} {backend_kind:4s}] "
                      f"ser={r.serialize_seconds*1000:.2f}ms "
                      f"deser={r.deserialize_seconds*1000:.2f}ms "
                      f"size={r.actual_size_bytes:>10} {label}")

    fields = [f.name for f in dataclasses.fields(MeasureResult)]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = dataclasses.asdict(r)
            d["error"] = d["error"] or ""
            w.writerow(d)

    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
