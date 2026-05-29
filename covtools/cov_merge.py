"""Merge multiple coverage data files into one (set-union of lines + arcs).

Used to build the NEW keep-set coverage without re-running the 895 tests
already measured:  keepers_new = keepers_all (895)  U  newly_protected (161).

Usage:
    python covtools/cov_merge.py OUT.cov IN1.cov IN2.cov [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

from coverage import CoverageData


def main() -> None:
    out_path = sys.argv[1]
    inputs = sys.argv[2:]
    if not inputs:
        raise SystemExit("need at least one input .cov")

    # Start clean so a stale OUT file can't leak old data.
    Path(out_path).unlink(missing_ok=True)
    out = CoverageData(basename=out_path)
    for p in inputs:
        d = CoverageData(basename=p)
        d.read()
        out.update(d)
        print(f"  merged {p}")
    out.write()

    cash = [f for f in out.measured_files() if "cash" in f.replace("\\", "/")]
    print(f"-> {out_path}: {len(cash)} cash files measured")


if __name__ == "__main__":
    main()
