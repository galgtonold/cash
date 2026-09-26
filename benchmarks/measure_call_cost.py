"""What one call through ``@cash.cache`` costs, next to lru_cache and joblib.

Times a hit and a miss of a function whose body does almost nothing, so the
figure is the cache's own cost per call:

* **small argument**: ``f(i)`` with an ``int``;
* **large array**: ``f(a)`` with an 8 MB ``float64`` array, which every cache
  that supports it must hash by content on each call.

The body returns a small value in both cases, so a miss stores a few bytes and
the array cost is the argument's hashing, not the result's writing.

``functools.lru_cache`` keeps results in a dict in RAM and cannot take an
array. ``joblib.Memory`` keeps them on disk only, so its hit reads a file.
A cash hit in the same process comes from its RAM tier; a hit in a later
process reads from disk, which the restore table on ``docs/benchmarks.md``
covers.

Usage:
    python benchmarks/measure_call_cost.py [--out PATH] [--rounds N] [--quick]

Each figure is the median of ``--rounds`` independent rounds (a fresh cache
each), so one busy moment on the machine does not set it. Writes one CSV row
per (tool, argument, outcome) with the seconds per call and the machine it
ran on, and prints the Markdown table
``docs/benchmarks.md`` shows. The committed result is
``benchmarks/call_cost.frozen.csv``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import os
import platform
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cash
from cash import Cash

ARRAY_ELEMENTS = 1_000_000  # 8 MB of float64
TOOLS = ("cash", "lru_cache", "joblib")
FIELDS = [
    "tool",
    "argument",
    "outcome",
    "seconds_per_call",
    "calls",
    "cpu",
    "cores",
    "python",
    "platform",
    "cash_version",
    "joblib_version",
    "numpy_version",
    "measured_on",
]


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _median_per_call(fn, args_list, batch: int) -> float:
    """Median over batches of the mean time per call.

    Each batch calls ``fn`` once per item of ``args_list[k:k + batch]``, so a
    miss batch never repeats an argument and a hit batch reuses the one given.
    """
    per_call = []
    for start in range(0, len(args_list), batch):
        chunk = args_list[start : start + batch]
        t0 = time.perf_counter()
        for a in chunk:
            fn(a)
        per_call.append((time.perf_counter() - t0) / len(chunk))
    return statistics.median(per_call)


def _build(tool: str, workdir: Path):
    def body(x):
        return x if isinstance(x, int) else float(x[0])

    if tool == "cash":
        c = Cash(cache_dir=str(workdir / "cash"), register_magic=False)
        return c.cache(body), c
    if tool == "lru_cache":
        return functools.lru_cache(maxsize=None)(body), None
    import joblib

    return joblib.Memory(location=str(workdir / "joblib"), verbose=0).cache(body), None


def measure(tool: str, argument: str, quick: bool) -> dict[str, tuple[float, int]]:
    """{outcome: (seconds per call, calls timed)} for one tool and argument."""
    if tool == "lru_cache" and argument == "array":
        return {}  # an ndarray is unhashable: lru_cache raises TypeError
    small = argument == "small"
    scale = 10 if quick else 1
    with tempfile.TemporaryDirectory() as tmp:
        fn, owner = _build(tool, Path(tmp))
        try:
            if small:
                warm = [10**9 + i for i in range(20)]
                misses = list(range(2000 // scale))
                hit_arg, n_hits = -1, 20_000 // scale
                batch_miss, batch_hit = 50, 1000
            else:
                rng = np.random.default_rng(0)
                warm = [rng.random(ARRAY_ELEMENTS) for _ in range(3)]
                misses = [rng.random(ARRAY_ELEMENTS) for _ in range(40 // scale or 4)]
                hit_arg, n_hits = rng.random(ARRAY_ELEMENTS), 60 // scale or 6
                batch_miss, batch_hit = 2, 3
            # The first calls pay one-off work (cash analyses the function,
            # joblib writes its function record); a steady-state cost is the
            # figure a hot loop sees.
            for a in warm:
                fn(a)
            fn(hit_arg)
            miss = _median_per_call(fn, misses, batch_miss)
            hit = _median_per_call(fn, [hit_arg] * n_hits, batch_hit)
        finally:
            if owner is not None:
                owner.shutdown()
    return {"hit": (hit, n_hits), "miss": (miss, len(misses))}


def machine() -> dict[str, str]:
    try:
        import joblib

        joblib_version = joblib.__version__
    except ImportError:
        joblib_version = ""
    return {
        "cpu": _cpu(),
        "cores": str(os.cpu_count()),
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "cash_version": cash.__version__,
        "joblib_version": joblib_version,
        "numpy_version": np.__version__,
        "measured_on": datetime.date.today().isoformat(),
    }


def format_seconds(s: float) -> str:
    """The unit the page uses: µs below a millisecond, ms from there."""
    if s < 1e-3:
        us = s * 1e6
        return f"{us:.2f} µs" if us < 1 else f"{us:.0f} µs" if us >= 10 else f"{us:.1f} µs"
    ms = s * 1e3
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"


def markdown(rows: list[dict[str, str]]) -> str:
    got = {(r["tool"], r["argument"], r["outcome"]): float(r["seconds_per_call"]) for r in rows}
    cols = [("small", "hit"), ("small", "miss"), ("array", "hit"), ("array", "miss")]
    names = {"cash": "`@cash.cache`", "lru_cache": "`lru_cache`", "joblib": "`joblib.Memory`"}
    out = [
        "| Cache | Hit, small argument | Miss, small argument | Hit, 8 MB array | Miss, 8 MB array |",
        "|---|---:|---:|---:|---:|",
    ]
    for tool in TOOLS:
        cells = [format_seconds(got[(tool, *c)]) if (tool, *c) in got else "not supported" for c in cols]
        out.append(f"| {names[tool]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/call_cost.csv"))
    p.add_argument("--rounds", type=int, default=3, help="independent rounds; the median is kept")
    p.add_argument("--quick", action="store_true", help="a tenth of the calls, for a smoke test")
    args = p.parse_args(argv)

    info = machine()
    rows = []
    with warnings.catch_warnings():
        # A body this cheap is exactly what cash's net-loss advisory is for.
        warnings.simplefilter("ignore")
        for tool in TOOLS:
            for argument in ("small", "array"):
                rounds = [measure(tool, argument, args.quick) for _ in range(max(1, args.rounds))]
                for outcome in rounds[0]:
                    seconds = statistics.median(r[outcome][0] for r in rounds)
                    calls = rounds[0][outcome][1]
                    rows.append(
                        {
                            "tool": tool,
                            "argument": argument,
                            "outcome": outcome,
                            "seconds_per_call": f"{seconds:.9f}",
                            "calls": str(calls),
                            **info,
                        }
                    )
                    print(f"{tool:10} {argument:6} {outcome:5} {format_seconds(seconds)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(markdown(rows))
    print(f"\n{info['cpu']}, {info['cores']} cores, Python {info['python']}, cash {info['cash_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
