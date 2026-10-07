"""Speed set: how much slower (or faster) cash is than plain Python, per scenario.

Each scenario (``_speed_scenarios.py``) is timed as plain Python and with
cash in this one process, and reported as the ratio cash / plain, which
stays comparable across machines. Medians of interleaved samples, CPU time.

Usage:
    python benchmarks/speed_set.py                 # full: ~4 min
    python benchmarks/speed_set.py --quick         # fewer samples: ~2 min
    python benchmarks/speed_set.py --only nb_ --json out.json

``--only`` keeps the scenarios whose name contains any of the given parts.
The JSON holds every sample; ``speed_compare.py`` runs this script on two
versions of cash and compares the ratios.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# One BLAS/OpenMP thread: idle worker threads spin and count as this
# process's CPU time, more so the busier the machine, which made the numpy
# and sklearn rows noisy. Both sides run with the same setting.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

FULL_REPEATS = 15
QUICK_REPEATS = 5


def select(scenarios, only: list[str] | None):
    if not only:
        return list(scenarios)
    return [s for s in scenarios if any(part in s.name for part in only)]


def run(only: list[str] | None, repeats: int, quick: bool, progress=None) -> dict:
    import cash
    from benchmarks._speed_harness import default_clock, measure, skipped_rows
    from benchmarks._speed_scenarios import SCENARIOS

    chosen = select(SCENARIOS, only)
    if not chosen:
        raise SystemExit(f"no scenario matches {only}")
    rows = []
    started = time.perf_counter()
    for sc in chosen:
        t0 = time.perf_counter()
        try:
            got = measure(sc, repeats=repeats, quick=quick)
        except Exception as exc:
            # One broken scenario (say, an older cash under A/B lacks what it
            # drives) is reported as such; the others still run.
            got = skipped_rows(sc, f"error: {type(exc).__name__}: {str(exc)[:200]}")
        rows.extend(got)
        if progress:
            progress(f"{sc.name}: {time.perf_counter() - t0:.1f}s")
    return {
        "meta": {
            "cash_version": getattr(cash, "__version__", "?"),
            "cash_path": str(Path(cash.__file__).resolve().parent),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.node(),
            "cpus": os.cpu_count(),
            "repeats": repeats,
            "clock": default_clock(),
            "quick": quick,
            "seconds": round(time.perf_counter() - started, 1),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
        "rows": [r.to_json() for r in rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick", action="store_true", help=f"{QUICK_REPEATS} samples per side instead of {FULL_REPEATS}"
    )
    parser.add_argument("--repeats", type=int, help="samples per side (overrides --quick)")
    parser.add_argument("--only", nargs="+", metavar="PART", help="run only scenarios whose name contains PART")
    parser.add_argument("--json", type=Path, help="write the full result here")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("-q", "--no-progress", action="store_true", help="no per-scenario progress lines")
    args = parser.parse_args(argv)

    if args.list:
        from benchmarks._speed_scenarios import SCENARIOS

        for sc in SCENARIOS:
            print(f"{sc.name:<24} {sc.group:<10} {sc.what}")
        return 0

    repeats = args.repeats or (QUICK_REPEATS if args.quick else FULL_REPEATS)
    progress = None if args.no_progress else (lambda line: print(line, file=sys.stderr, flush=True))
    result = run(args.only, repeats, args.quick, progress)

    from benchmarks._speed_harness import Row, format_table

    print(format_table([Row.from_json(r) for r in result["rows"]]))
    meta = result["meta"]
    print(
        f"\ncash {meta['cash_version']} from {meta['cash_path']}; {meta['repeats']} samples per side; "
        f"{meta['seconds']}s.\nTimes are per call (decorator), per cell (nb_long, nb_module_data) or per run of the "
        f"measured cells, median {meta['clock']} time.\nratio = cash / plain; noise = median deviation of the "
        f"per-round ratios from their median."
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
