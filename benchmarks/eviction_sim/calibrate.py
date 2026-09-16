"""Derive the workload's size/compute distributions from an overhead sweep.

``workload.Params`` is calibrated on real statement metrics: execution time
and ``cost_model_size_bytes`` per COMPUTED statement, as recorded by
``_rerun_sweep.py`` / ``bench_notebook_overhead.py``. Those result
directories (``benchmarks/results_*``) are gitignored, so re-run a sweep
first and point this at it:

    python benchmarks/_rerun_sweep.py ...          # writes benchmarks/results_<x>/
    python benchmarks/eviction_sim/calibrate.py benchmarks/results_<x>

It prints the quantiles the Params comment quotes. Then adjust
``compute_*``, ``size_*`` and ``p_large`` until a Project's statements
match (``--check``).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict

QS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def statement_metrics(obj):
    """Yield every dict that looks like a StatementMetric, however nested."""
    if isinstance(obj, dict):
        if "execution_time" in obj and "cost_model_size_bytes" in obj:
            yield obj
        for v in obj.values():
            yield from statement_metrics(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from statement_metrics(v)


def quantiles(xs, qs=QS):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs]


def describe(rows, label):
    if not rows:
        print(f"{label}: no rows")
        return
    et = [r[0] for r in rows]
    sz = [r[1] for r in rows]
    print(f"{label}: {len(rows)} statements")
    print("  exec s   " + "  ".join(f"p{int(q*100)}={v:.3g}" for q, v in zip(QS, quantiles(et))))
    print("  size B   " + "  ".join(f"p{int(q*100)}={v:.3g}" for q, v in zip(QS, quantiles(sz))))
    le = [math.log(max(1e-4, x)) for x in et]
    ls = [math.log(max(1, x)) for x in sz]
    if len(rows) > 2:
        print(f"  log-sd exec {statistics.pstdev(le):.2f}   log-sd size {statistics.pstdev(ls):.2f}   "
              f"log-corr {statistics.correlation(le, ls):.2f}")
    spm = [r[0] / max(r[1] / 1e6, 1e-9) for r in rows]
    print("  s per MB " + "  ".join(f"p{int(q*100)}={v:.3g}" for q, v in zip(QS, quantiles(spm))))


def load(results_dir, pattern):
    rows = []
    for f in glob.glob(os.path.join(results_dir, pattern)):
        with open(f) as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError:
                continue
        nb = os.path.basename(f).split("-cold")[0]
        for m in statement_metrics(d):
            if m.get("status") == "COMPUTED" and m.get("cost_model_size_bytes"):
                rows.append((m["execution_time"], m["cost_model_size_bytes"], nb))
    return rows


def check():
    """The same statistics over a synthetic Project, to compare against."""
    from workload import Params, Project
    rows = []
    for seed in range(5):
        proj = Project(Params(seed=seed))
        rows += [(s.compute, s.size, "synthetic") for s in proj.stmts if not s.loop_iters]
    describe(rows, "synthetic Project (5 seeds, loop statements excluded)")
    describe([r for r in rows if r[0] >= 0.1], "synthetic, disk-eligible (>= 0.1 s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("--pattern", default="*-cold-*.json",
                    help="cold runs only: their execution_time is real compute")
    ap.add_argument("--check", action="store_true", help="also describe the synthetic workload")
    a = ap.parse_args()
    if a.results_dir:
        rows = load(a.results_dir, a.pattern)
        describe(rows, "all computed statements")
        describe([r for r in rows if r[0] >= 0.1], "disk-eligible (>= 0.1 s)")
        per_nb = defaultdict(list)
        for r in rows:
            per_nb[r[2]].append(r)
        print("\nper notebook: statements, disk-eligible, bytes of disk-eligible results")
        for nb, rr in sorted(per_nb.items()):
            big = [r for r in rr if r[0] >= 0.1]
            print(f"  {nb:<34} {len(rr):4d} {len(big):4d} {sum(r[1] for r in big) / 1e6:10.1f} MB")
    if a.check or not a.results_dir:
        print()
        check()


if __name__ == "__main__":
    main()
