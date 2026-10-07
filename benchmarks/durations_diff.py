"""Per-test durations: save them from a full run and flag the tests that got slower.

Run the suite with ``--junitxml`` (works with xdist), then:

    python benchmarks/durations_diff.py results.xml [more.xml ...] [--store DIR] [--label NAME]

It reads every test's time from the JUnit XML, compares it with the last run
saved in the store, prints the tests that got more than 2x AND more than 1 s
slower (both thresholds configurable), and saves this run as the new latest.
Exit code 1 when a test was flagged (``--warn-only``: 0), 2 on bad input.

The store defaults to ``~/.cash-test-durations`` (outside the repo, so
baselines stay per machine). Compare against a particular saved run with
``--against FILE``; compare without saving with ``--no-save``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STORE = Path.home() / ".cash-test-durations"


def case_id(case: ET.Element) -> str:
    """``tests/x/test_y.py::TestC::test_z`` style id from a <testcase>."""
    classname = case.get("classname", "")
    name = case.get("name", "")
    file = case.get("file")
    if file:
        # classname is "tests.x.test_y.TestC": keep the class part after the module.
        module = file.replace("\\", "/").removesuffix(".py").replace("/", ".")
        rest = classname[len(module) + 1 :] if classname.startswith(module + ".") else ""
        return "::".join(p for p in (file.replace("\\", "/"), rest, name) if p)
    return f"{classname}::{name}" if classname else name


def read_junit(paths: list[Path]) -> dict[str, float]:
    """Test id -> seconds. A test that appears twice (a rerun) keeps its longest time."""
    out: dict[str, float] = {}
    for path in paths:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        for case in root.iter("testcase"):
            if case.find("skipped") is not None:
                continue
            tid = case_id(case)
            out[tid] = max(out.get(tid, 0.0), float(case.get("time", "0") or 0))
    return out


def load_run(path: Path) -> dict[str, float]:
    if path.suffix == ".xml":
        return read_junit([path])
    return json.loads(path.read_text(encoding="utf-8"))["durations"]


@dataclass
class Slower:
    test: str
    before: float
    after: float


def diff(before: dict[str, float], after: dict[str, float], factor: float, min_seconds: float) -> list[Slower]:
    """Tests in both runs that took more than ``factor`` times as long AND
    ``min_seconds`` longer, slowest increase first."""
    flagged = [
        Slower(t, before[t], after[t])
        for t in after
        if t in before and after[t] > factor * before[t] and after[t] - before[t] > min_seconds
    ]
    return sorted(flagged, key=lambda s: s.before - s.after)


def latest(store: Path) -> Path | None:
    runs = sorted(store.glob("durations-*.json"))
    return runs[-1] if runs else None


def save(store: Path, durations: dict[str, float], label: str | None) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = store / f"durations-{stamp}{'-' + label if label else ''}.json"
    path.write_text(json.dumps({"label": label, "saved": stamp, "durations": durations}, indent=0), encoding="utf-8")
    return path


def report(before: dict[str, float], after: dict[str, float], flagged: list[Slower], baseline: str) -> str:
    common = [t for t in after if t in before]
    lines = [
        f"against {baseline}: {len(after)} tests now, {len(before)} before, {len(common)} in both; "
        f"total {sum(before[t] for t in common):.0f}s -> {sum(after[t] for t in common):.0f}s for those in both"
    ]
    if flagged:
        lines.append(f"{len(flagged)} slower:")
        lines += [
            f"  {s.before:7.2f}s -> {s.after:7.2f}s  ({s.after / max(s.before, 1e-9):5.1f}x)  {s.test}" for s in flagged
        ]
    else:
        lines.append("no test got slower past the thresholds")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("junit", nargs="+", type=Path, help="JUnit XML from pytest --junitxml (several: xdist shards)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help=f"saved runs (default {DEFAULT_STORE})")
    parser.add_argument(
        "--against", type=Path, help="compare with this saved run (.json) or JUnit file instead of the latest"
    )
    parser.add_argument("--label", help="a name for this run in the store (e.g. the commit)")
    parser.add_argument(
        "--factor", type=float, default=2.0, help="flag a test more than this many times slower (default 2)"
    )
    parser.add_argument(
        "--min-seconds", type=float, default=1.0, help="... and more than this many seconds slower (default 1)"
    )
    parser.add_argument("--no-save", action="store_true", help="do not save this run")
    parser.add_argument("--warn-only", action="store_true", help="exit 0 even when tests are flagged")
    args = parser.parse_args(argv)

    try:
        after = read_junit(args.junit)
    except (OSError, ET.ParseError) as exc:
        print(f"cannot read {args.junit}: {exc}", file=sys.stderr)
        return 2
    if not after:
        print("no test times in the JUnit files", file=sys.stderr)
        return 2
    baseline = args.against or latest(args.store)
    flagged: list[Slower] = []
    if baseline is None:
        print(f"no saved run in {args.store} yet; this run becomes the baseline")
    else:
        before = load_run(baseline)
        flagged = diff(before, after, args.factor, args.min_seconds)
        print(report(before, after, flagged, str(baseline)))
    if not args.no_save:
        print(f"saved {save(args.store, after, args.label)}")
    return 1 if flagged and not args.warn_only else 0


if __name__ == "__main__":
    sys.exit(main())
