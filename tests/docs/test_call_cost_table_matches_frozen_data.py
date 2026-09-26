"""The per-call cost table on benchmarks.md is the committed measurement.

The table compares `@cash.cache` with `functools.lru_cache` and
`joblib.Memory` per hit and miss. A figure typed by hand drifts, or flatters,
so the page must show exactly what ``benchmarks/measure_call_cost.py`` prints
for ``benchmarks/call_cost.frozen.csv``, and the prose must name the machine
that file records.
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "benchmarks.md"
FROZEN = REPO / "benchmarks" / "call_cost.frozen.csv"
SCRIPT = REPO / "benchmarks" / "measure_call_cost.py"


def _script():
    spec = importlib.util.spec_from_file_location("measure_call_cost", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows() -> list[dict[str, str]]:
    with FROZEN.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"{FROZEN} is empty"
    return rows


def test_the_page_shows_the_frozen_table():
    expected = _script().markdown(_rows())
    assert expected in PAGE.read_text(encoding="utf-8"), (
        "benchmarks.md's per-call table differs from the frozen measurement. "
        "Re-run `python benchmarks/measure_call_cost.py --out "
        "benchmarks/call_cost.frozen.csv` and paste the table it prints:\n" + expected
    )


def test_every_tool_and_case_was_measured():
    got = {(r["tool"], r["argument"], r["outcome"]) for r in _rows()}
    for tool in ("cash", "joblib"):
        for argument in ("small", "array"):
            for outcome in ("hit", "miss"):
                assert (tool, argument, outcome) in got, (tool, argument, outcome)
    assert ("lru_cache", "small", "hit") in got


def test_the_page_names_the_machine_the_file_records():
    row = _rows()[0]
    text = " ".join(PAGE.read_text(encoding="utf-8").split())
    python = ".".join(row["python"].split(".")[:2])
    for fact in (f"{row['cores']}-core", f"Python {python}", f"cash {row['cash_version']}"):
        assert fact in text, f"the per-call table's machine is not described: {fact!r} missing"
