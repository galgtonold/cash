"""Do the refitted constants make BETTER promotion decisions than the shipped ones?

A fit with a lower residual is not the point. The constants exist to answer one
question per value:

    persist if  execution_time - estimated_restore > 0.20 * execution_time

so the only thing worth scoring is how often that answer is wrong -- judged
against the restore time the matrix actually MEASURED for that family and size.

For every measured (family, size) cell, this sweeps execution_time across five
decades, asks the question three ways (shipped constants, refitted constants,
measured truth), and counts the disagreements. A wrong answer is reported in the
direction it errs, because they are not equally bad:

  * "kept out" -- persisting would have paid and cash refused. The user loses a
    hit they should have had; the work is recomputed.
  * "let in" -- restoring costs more than recomputing and cash persisted anyway.
    The user gets a slow hit AND pays the disk for it, forever.

Usage:
    python benchmarks/_fit_decision_check.py benchmarks/results/ser_deser_matrix.frozen.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmarks.fit_cost_model import fit_all  # noqa: E402
from cash.notebook.cost_model import _COEFFS as SHIPPED  # noqa: E402

SAVINGS_PCT = 0.20  # config.min_cache_savings_pct
EXEC_TIMES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def persists(estimated_restore: float, execution_time: float) -> bool:
    return execution_time - estimated_restore > SAVINGS_PCT * execution_time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=Path)
    p.add_argument("--objective", default="relative")
    args = p.parse_args()

    refit = {(f.family, f.backend_kind, f.operation): (f.a, f.b) for f in fit_all(args.csv_path, args.objective)}

    measured: list[tuple[str, float, float]] = []
    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["error"] or row["backend_kind"] != "disk":
                continue
            measured.append((row["family"], float(row["actual_size_bytes"]), float(row["deserialize_seconds"])))

    tally = {"shipped": {"kept out": 0, "let in": 0}, "refit": {"kept out": 0, "let in": 0}}
    total = 0
    worst_examples: list[tuple[float, str]] = []

    for family, size, real in measured:
        for exec_time in EXEC_TIMES:
            total += 1
            truth = persists(real, exec_time)
            for label, coeffs in (("shipped", SHIPPED), ("refit", refit)):
                a, b = coeffs[(family, "disk", "deserialize")]
                got = persists(a + b * size, exec_time)
                if got != truth:
                    kind = "kept out" if truth else "let in"
                    tally[label][kind] += 1
                    if label == "shipped":
                        cost = abs(exec_time - real)
                        worst_examples.append(
                            (
                                cost,
                                f"{family} {int(size):,}B, body {exec_time * 1000:.0f}ms, "
                                f"real restore {real * 1000:.1f}ms: shipped says "
                                f"{'persist' if got else 'skip'}, truth says "
                                f"{'persist' if truth else 'skip'}",
                            )
                        )

    print(
        f"{total} decisions scored against measured restore times "
        f"({len(measured)} cells x {len(EXEC_TIMES)} body times)\n"
    )
    for label in ("shipped", "refit"):
        wrong = sum(tally[label].values())
        print(
            f"  {label:>8}: {wrong:>3} wrong ({wrong / total:5.1%})   "
            f"kept out {tally[label]['kept out']:>3}, let in {tally[label]['let in']:>3}"
        )

    print("\nWorst calls the shipped constants make:")
    for _cost, text in sorted(worst_examples, reverse=True)[:6]:
        print(f"  - {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
