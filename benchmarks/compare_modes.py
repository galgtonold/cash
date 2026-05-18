"""Compare bench results across cash-off / cash-cold / cash-warm.

Reads per-mode result JSONs from a directory and emits a markdown table
showing wall-clock per cell, the cold-off overhead in ms, and the
relative overhead as a fraction of off-mode time.

Usage:
    python benchmarks/compare_modes.py <results-dir> <notebook-stem>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks._overhead_results import read_results


def _median_per_cell(results_dir: Path, stem: str, mode: str) -> dict[int, float]:
    """Median wall-seconds per cell across all repeats except the first
    (treated as warmup). Returns {cell_index: median_seconds}."""
    files = sorted(results_dir.glob(f"{stem}-{mode}-*.json"))
    if not files:
        return {}
    # Discard the repeat-0 warmup if more than one repeat is present.
    samples_by_cell: dict[int, list[float]] = defaultdict(list)
    for f in files:
        repeat = int(f.stem.rsplit("-", 1)[-1])
        if len(files) > 1 and repeat == 0:
            continue
        r = read_results(f)
        for cell in r.cells:
            samples_by_cell[cell.index].append(cell.wall_seconds)
    return {idx: statistics.median(samples) for idx, samples in samples_by_cell.items()}


def build_table(results_dir: Path, notebook_stem: str) -> str:
    off = _median_per_cell(results_dir, notebook_stem, "off")
    cold = _median_per_cell(results_dir, notebook_stem, "cold")
    warm = _median_per_cell(results_dir, notebook_stem, "warm")

    all_cells = sorted(set(off) | set(cold) | set(warm))
    lines = [
        f"## {notebook_stem}",
        "",
        "| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_off = total_cold = total_warm = 0.0
    for idx in all_cells:
        o = off.get(idx, 0.0)
        c = cold.get(idx, 0.0)
        w = warm.get(idx, 0.0)
        total_off += o
        total_cold += c
        total_warm += w
        diff = c - o
        ratio = (diff / o) if o > 0 else float("inf")
        lines.append(
            f"| cell {idx} | {o*1000:.2f} | {c*1000:.2f} | {w*1000:.2f} "
            f"| {diff*1000:+.2f} | {ratio:+.1%} |"
        )
    diff = total_cold - total_off
    ratio = (diff / total_off) if total_off > 0 else float("inf")
    lines.append(
        f"| **TOTAL** | **{total_off*1000:.2f}** | **{total_cold*1000:.2f}** "
        f"| **{total_warm*1000:.2f}** | **{diff*1000:+.2f}** | **{ratio:+.1%}** |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare bench results across modes")
    p.add_argument("results_dir", type=Path)
    p.add_argument("notebook_stem", type=str)
    args = p.parse_args(argv)
    print(build_table(args.results_dir, args.notebook_stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
