"""``v = holder[len(holder) - 1]()`` calls a function a list holds beside
more objects of the user's classes than cash looks through
(`_INSTANCE_LIMIT`): the index is computed, so its key cannot name which
function runs or what it reads; the statement is refused, says why, and runs
every time. A constant index (``holder[-1]``) is read exactly instead
(`handed_callables`), and its function's reads are keyed.

A function whose code is swapped in place (``inner.__code__ = ...``) is
seen by its trace: the statement runs again, and the check of the cells
above its reader must not bring the old value back.
"""

from __future__ import annotations

import pytest

import cash.analysis.handed_callables as hc
import cash.notebook.shadow as shadow
from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

SETUP = (
    "K = 1\nclass Box:\n    def __init__(self, i):\n        self.i = i\n"
    "def g():\n    return K\nholder = [Box(i) for i in range(200)] + [g]"
)


def _reasons(magics) -> list[str]:
    statements = magics._last_cell_metrics.get("statements", [])
    return [reason for s in statements for reason in s.get("uncacheable_reasons") or ()]


def test_a_holder_past_the_limit_refuses_the_statement(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    monkeypatch.setattr(shadow, "SHADOW", None)
    monkeypatch.setattr(hc, "_INSTANCE_LIMIT", 50)
    cells = [SETUP, "v = holder[len(holder) - 1]()", "w = [v, 'read']"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["v"] == 1
    exec("K = 2", ns)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert ns["v"] == 2, "served what K = 1 gave"


def test_a_reader_below_code_swapped_in_place_gets_the_new_value(cash_magics, monkeypatch):
    if shadow.SHADOW is None:
        pytest.skip("traces are recorded on Python 3.12+ only")
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    cells = ["def inner(x):\n    return x * 2", "v = inner(3)", "w = [v, 'read']"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["v"] == 6
    exec("inner.__code__ = (lambda x: x * 10).__code__", ns)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["v"] == 30, "the statement or the check above its reader served the old value"
