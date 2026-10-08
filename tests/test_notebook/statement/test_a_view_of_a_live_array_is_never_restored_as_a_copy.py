"""A statement whose output is, or holds, a numpy view of an array something
else holds is never restored as a copy.

A hit binds a deserialised array that owns its memory, so writes through the
restored view no longer reach the base. The live-alias check used to look
only at an output that IS a view of an array bound to a NAME: views inside a
list (``parts = np.split(a, 2)``) and views of an array held in a dict
(``w = window(data['x'], 2)``) were restored as detached copies on the next
Run All (bug-hunt-5 AL-04). Each case runs "Run All" twice and must match
plain Python both times.
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement.derivation_edges import is_uncacheable_alias
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

SETUP = f"import time\nimport numpy as np\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x"

NOTEBOOKS = {
    "a list of views": (
        [SETUP, "a = np.zeros(10)", "parts = slow(np.split(a, 2))", "parts[0][:] = 1", "r = a.sum()"],
        5.0,
    ),
    "a view of an array in a dict": (
        [SETUP, "data = {'x': np.zeros(10)}", "w = slow(data['x'][2:5])", "w[:] = 5", "r = data['x'].sum()"],
        15.0,
    ),
    "a dict of views": (
        [SETUP, "a = np.zeros(10)", "halves = slow({'lo': a[:5], 'hi': a[5:]})", "halves['hi'][:] = 2", "r = a.sum()"],
        10.0,
    ),
}


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_a_second_run_all_still_writes_into_the_base(cash_magics, name):
    cells, expected = NOTEBOOKS[name]
    for run in ("first", "second"):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] == expected, f"{run} Run All"


def _stored(metrics) -> bool:
    return metrics["status"] == CacheStatus.COMPUTED and not metrics.get("uncacheable_reasons")


@pytest.mark.parametrize(
    "statement",
    ["parts = slow(np.split(np.zeros(10), 2))", "grid = slow(np.arange(10.0).reshape(2, 5))"],
    ids=["views-of-their-own-array", "reshape-of-a-fresh-array"],
)
def test_views_of_an_array_only_they_hold_are_stored(cash_magics, statement_processor, statement):
    """Control: restoring a copy of a base nothing else holds is correct."""
    run_cash_cell(cash_magics, SETUP)
    metrics = statement_processor.process_statement(statement)
    assert _stored(metrics), metrics.get("uncacheable_reasons")


def test_where_the_base_is_held_decides():
    a = np.zeros(10)
    data = {"x": np.zeros(10)}
    user_ns = {"a": a, "data": data}
    assert is_uncacheable_alias([a[:2], 1], user_ns), "a view of a named array, in a list"
    assert is_uncacheable_alias(data["x"][2:5], user_ns), "a view of an array a dict holds"
    buf = np.zeros(5)
    pair = (buf, buf[:3])
    del buf
    assert is_uncacheable_alias(pair, user_ns), "a view of an array the output holds too"
    # Built outside the assert, whose rewriting would keep the base alive.
    own = np.split(np.zeros(10), 2)
    assert not is_uncacheable_alias(own, user_ns)
    assert not is_uncacheable_alias([a[:2].copy()], user_ns)
    held = (np.arange(4.0), "key")
    assert is_uncacheable_alias(held[0][1:], user_ns)
    assert not is_uncacheable_alias(held[0][1:], user_ns, cash_held=[held]), "cash's own reference"
