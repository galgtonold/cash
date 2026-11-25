"""The per-call cost table on benchmarks.md is the committed measurement.

The table compares `@cash.cache` with `functools.lru_cache` and
`joblib.Memory` per hit and miss. A figure typed by hand drifts, or flatters,
so the page must show exactly what ``benchmarks/measure_call_cost.py`` prints
for ``benchmarks/call_cost.frozen.csv``, and the prose must name the machine
that file records.

The figures sit between ``docnum`` markers, which ``scripts/doc_numbers.py
--update`` fills from the same file, here and on the pages that build advice
on them; this test reads the page as a reader sees it, markers removed.
"""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "benchmarks.md"
FROZEN = REPO / "benchmarks" / "call_cost.frozen.csv"
SCRIPT = REPO / "benchmarks" / "measure_call_cost.py"


_DOCNUM = re.compile(r"<!--\s*/?docnum(?::[a-z0-9_]+)?\s*-->")


def _page() -> str:
    """The page with the ``docnum`` markers taken out."""
    return _DOCNUM.sub("", PAGE.read_text(encoding="utf-8"))


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
    assert expected in _page(), (
        "benchmarks.md's per-call table differs from the frozen measurement. "
        "Re-run `python benchmarks/measure_call_cost.py --out "
        "benchmarks/call_cost.frozen.csv`, then `python scripts/doc_numbers.py --update`; "
        "the table it prints:\n" + expected
    )


def test_every_tool_and_case_was_measured():
    got = {(r["tool"], r["argument"], r["outcome"]) for r in _rows()}
    for tool in ("cash", "joblib"):
        for argument in ("small", "array"):
            for outcome in ("hit", "first_hit", "miss"):
                assert (tool, argument, outcome) in got, (tool, argument, outcome)
    assert ("lru_cache", "small", "hit") in got


def test_the_page_names_the_machine_the_file_records():
    row = _rows()[0]
    text = " ".join(_page().split())
    python = ".".join(row["python"].split(".")[:2])
    for fact in (f"{row['cores']}-core", f"Python {python}", f"cash {row['cash_version']}"):
        assert fact in text, f"the per-call table's machine is not described: {fact!r} missing"


def test_the_pages_that_advise_on_the_figures_quote_the_file():
    """*Why cash*, the FAQ, the decorator guide and the threads guide say
    when a call is too cheap to cache. Their figures are markers filled from
    the frozen file, so a new measurement moves the advice with the table."""
    pages = (
        "why-cash.md",
        "faq.md",
        "decorator.md",
        "how-it-works/decorator-path.md",
        "tutorials/feature-guides/thread-safety.md",
    )
    for page in pages:
        text = (REPO / "docs" / page).read_text(encoding="utf-8")
        assert "<!-- docnum:cc_cash_small_" in text, page
