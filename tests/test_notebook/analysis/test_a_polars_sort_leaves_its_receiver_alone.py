"""polars' ``sort`` returns a new frame; ``list.sort`` sorts in place.

``raw.group_by("host").len().sort("host")`` was badged "In-place mutation on:
raw": the statement never cached, and ``raw``'s lineage moved, so what was
built from it ran again. A ``sort`` that acts on a new object -- one a method
earlier in the chain built, or whose value is chained on or bound with a
column to sort by -- leaves its receiver alone. The in-place forms still
change their receiver.
"""

from __future__ import annotations

import pytest

from cash.analysis.cacheability import analyze_statement
from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    "code",
    [
        'raw.group_by("host").len().sort("host")',
        'top = raw.sort("ts").head(5)',
        'out = raw.sort("ts")',
        'out = raw.sort(["host", "ts"], descending=True)',
        "df.groupby('a').sum().sort('a')",
    ],
)
def test_a_sort_that_returns_a_new_frame_changes_nothing(code):
    assert analyze_statement(code, None).all_mutated_vars == frozenset()


@pytest.mark.parametrize(
    "code, changed",
    [
        ("rows.sort()", "rows"),
        ("rows.sort(key=len)", "rows"),
        ("x = rows.sort()", "rows"),
        ("arr.sort(0)", "arr"),
        ("d.setdefault(k, []).sort()", "d"),
        ("a.transpose().sort()", "a"),
        ('s = raw.sort("ts", in_place=True)', "raw"),
        ("cfg.update(extra)", "cfg"),
    ],
)
def test_a_sort_in_place_still_changes_its_receiver(code, changed):
    assert changed in analyze_statement(code, None).all_mutated_vars


def test_a_polars_summary_line_keeps_the_frames_lineage(cash_magics):
    pytest.importorskip("polars")
    run_cash_cell(cash_magics, 'import polars as pl\nraw = pl.DataFrame({"host": ["b", "a", "b"], "n": [1, 2, 3]})')
    before = cash_magics.cash_status("dict")["lineage"]["raw"]
    for cell in ('summary = raw.group_by("host").len().sort("host")', 'raw.group_by("host").len().sort("host")'):
        run_cash_cell(cash_magics, cell)
        status = cash_magics.cash_status("dict")
        assert status["lineage"]["raw"] == before, cell
        statements = status["last_cell"]["statements"]
        assert not any("In-place mutation" in str(m) for m in statements), statements
