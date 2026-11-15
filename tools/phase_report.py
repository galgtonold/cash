"""Adds up the phase times ``@cash.cache`` wrote with ``CASH_PHASE_TIMES_FILE``.

Each process that ran with ``CASH_PHASE_TIMES_FILE=<path>`` appended one JSON
line (`cash.phase_times.dump`). This reads any number of such files, or
directories of them, and prints one row per cached function (module and
qualified name, summed over every process)::

    python tools/phase_report.py out/                 # a directory of .jsonl files
    python tools/phase_report.py a.jsonl b.jsonl --json
    python tools/phase_report.py out/ --phases        # every phase, per function

Columns: calls; hits (fast + ordinary); misses (stored); refused (ran
without storing: no key, refused after the body, or raised); body seconds
(wall time of the bodies that ran); cash seconds on hits and on the rest;
overhead in seconds and as a percentage of the body time; median cash time of
a hit; and the three phases that took the most of cash's time.

Standard library only: needs neither cash nor its dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator

#: Histogram slots per doubling, and the exponent of slot 0, as
#: `cash.phase_times` writes them.
_STEPS = 8
_LOW_EXP = -30
#: Outcomes counted as refused: ran, nothing stored.
REFUSED = ("uncached", "not_stored", "raised")
#: The per-outcome rows that are not phases.
_TOTALS = ("total", "cash")


def _files(paths: Iterable[str]) -> Iterator[str]:
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith((".jsonl", ".json")):
                    yield os.path.join(path, name)
        else:
            yield path


def load(paths: Iterable[str]) -> tuple[dict[tuple[str, str], dict[str, dict]], int]:
    """``{(module, qualname): {"<outcome>.<phase>": {count, total, max,
    hist}}}`` summed over every process line in *paths*, and how many
    process lines there were."""
    merged: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    processes = 0
    for path in _files(paths):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    print(f"phase_report: skipped a damaged line in {path}", file=sys.stderr)
                    continue
                processes += 1
                for func in data.get("functions", []):
                    into = merged[(func.get("module", "?"), func.get("qualname", func.get("name", "?")))]
                    for key, stat in func.get("phases", {}).items():
                        have = into.get(key)
                        if have is None:
                            into[key] = {
                                "count": stat["count"],
                                "total": stat["total"],
                                "max": stat["max"],
                                "hist": {int(k): v for k, v in stat.get("hist", {}).items()},
                            }
                            continue
                        have["count"] += stat["count"]
                        have["total"] += stat["total"]
                        have["max"] = max(have["max"], stat["max"])
                        for k, v in stat.get("hist", {}).items():
                            have["hist"][int(k)] = have["hist"].get(int(k), 0) + v
    return dict(merged), processes


def median(hist: dict[int, int]) -> float:
    """The median a histogram stands for (within about 4%)."""
    total = sum(hist.values())
    seen = 0
    for slot in sorted(hist):
        seen += hist[slot]
        if seen * 2 >= total:
            e, step = divmod(slot, _STEPS)
            return math.ldexp(0.5 + (step + 0.5) / (2 * _STEPS), e + _LOW_EXP)
    return 0.0


def _get(stats: dict, key: str, field: str = "total") -> float:
    stat = stats.get(key)
    return stat[field] if stat is not None else 0


def row(stats: dict[str, dict]) -> dict:
    """One function's summary from its merged phase stats."""
    outcomes = {key.split(".", 1)[0] for key in stats}
    count = {o: _get(stats, f"{o}.total", "count") for o in outcomes}
    hits = count.get("hit", 0) + count.get("fast_hit", 0)
    misses = count.get("miss", 0)
    refused = sum(count.get(o, 0) for o in REFUSED)
    body = sum(_get(stats, f"{o}.body") for o in outcomes)
    cash_hit = _get(stats, "hit.cash") + _get(stats, "fast_hit.cash")
    cash_rest = sum(_get(stats, f"{o}.cash") for o in outcomes if o not in ("hit", "fast_hit"))
    overhead = cash_hit + cash_rest
    phases: dict[str, float] = defaultdict(float)
    for key, stat in stats.items():
        outcome, phase = key.split(".", 1)
        if phase in _TOTALS or phase == "body":
            continue
        phases[f"{outcome}.{phase}"] += stat["total"]
    if count.get("fast_hit"):
        phases["fast_hit"] = _get(stats, "fast_hit.cash")
    top = sorted(phases.items(), key=lambda kv: kv[1], reverse=True)[:3]
    hit_hist: dict[int, int] = {}
    for o in ("hit", "fast_hit"):
        for k, v in stats.get(f"{o}.cash", {}).get("hist", {}).items():
            hit_hist[k] = hit_hist.get(k, 0) + v
    return {
        "calls": sum(count.values()),
        "hits": hits,
        "fast_hits": count.get("fast_hit", 0),
        "misses": misses,
        "refused": refused,
        "body_s": body,
        "cash_hit_s": cash_hit,
        "cash_other_s": cash_rest,
        "overhead_s": overhead,
        "overhead_pct": 100.0 * overhead / body if body else None,
        "hit_median_s": median(hit_hist) if hit_hist else None,
        "top_phases": [(name, seconds, 100.0 * seconds / overhead if overhead else 0.0) for name, seconds in top],
    }


def _seconds(s: float | None) -> str:
    if s is None:
        return "-"
    if s >= 1:
        return f"{s:.2f}s"
    if s >= 1e-3:
        return f"{s * 1e3:.1f}ms"
    if s >= 1e-5:
        return f"{s * 1e6:.0f}us"
    return f"{s * 1e6:.1f}us"


def table(merged: dict[tuple[str, str], dict[str, dict]]) -> str:
    """The per-function table, costliest overhead first."""
    rows = sorted(((name, row(stats)) for name, stats in merged.items()), key=lambda r: -r[1]["overhead_s"])
    head = (
        f"{'function':<34} {'calls':>6} {'hits':>6} {'miss':>5} {'refus':>5} {'body':>8} "
        f"{'cash hit':>8} {'cash rest':>9} {'overhead':>8} {'%body':>7} {'hit med':>7}  top phases (share of cash time)"
    )
    lines = [head, "-" * len(head)]
    for (module, qualname), r in rows:
        name = f"{module}.{qualname}"
        if len(name) > 34:
            name = "..." + name[-31:]
        pct = f"{r['overhead_pct']:.1f}%" if r["overhead_pct"] is not None else "-"
        top = ", ".join(f"{n} {_seconds(s)} ({p:.0f}%)" for n, s, p in r["top_phases"])
        lines.append(
            f"{name:<34} {r['calls']:>6} {r['hits']:>6} {r['misses']:>5} {r['refused']:>5} "
            f"{_seconds(r['body_s']):>8} {_seconds(r['cash_hit_s']):>8} {_seconds(r['cash_other_s']):>9} "
            f"{_seconds(r['overhead_s']):>8} {pct:>7} {_seconds(r['hit_median_s']):>7}  {top}"
        )
    return "\n".join(lines)


def phases_table(merged: dict[tuple[str, str], dict[str, dict]]) -> str:
    """Every phase of every function: count, total, median, max."""
    lines = []
    for (module, qualname), stats in sorted(merged.items()):
        lines.append(f"{module}.{qualname}")
        for key in sorted(stats):
            s = stats[key]
            lines.append(
                f"  {key:<24} {s['count']:>7} total {_seconds(s['total']):>8}  "
                f"median {_seconds(median(s['hist'])):>7}  max {_seconds(s['max']):>8}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="files or directories written via CASH_PHASE_TIMES_FILE")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    parser.add_argument("--phases", action="store_true", help="also list every phase of every function")
    args = parser.parse_args(argv)
    merged, processes = load(args.paths)
    if not merged:
        print("phase_report: no phase times found", file=sys.stderr)
        return 1
    if args.json:
        out = {f"{m}.{q}": row(stats) for (m, q), stats in merged.items()}
        print(json.dumps({"processes": processes, "functions": out}, indent=2))
        return 0
    print(f"{processes} process(es), {len(merged)} cached function(s)")
    print(table(merged))
    if args.phases:
        print()
        print(phases_table(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
