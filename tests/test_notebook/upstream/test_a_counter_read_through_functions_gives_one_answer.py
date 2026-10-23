"""A counter two cells draw from, through functions of the user's or
directly, run out of order: re-running a reader rebuilds the counter first
and runs every draw before it again, as the documented rule says, so the
same cells in the same order give the same values every time.

``w = [v, u]`` below a re-run of ``v = a()`` re-ran ``v = a()`` and
``u = b()`` on the counter where earlier runs had left it: ``[1, 20]`` in
one run and ``[0, 10]`` in another, depending on what the call cache held.
"""

from __future__ import annotations

import pytest

import cash.notebook.call_unit as call_unit
from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

FORMS = {
    "through functions": [
        "import itertools\nc = itertools.count()\ndef a():\n    return next(c)\ndef b():\n    return next(c) * 10",
        "v = a()",
        "u = b()",
        "w = [v, u]",
    ],
    "directly": ["import itertools\nc = itertools.count()", "v = next(c)", "u = next(c) * 10", "w = [v, u]"],
}
ORDER = [0, 1, 2, 3, 2, 3, 1, 3, 3]


def _run(cash_magics, cells):
    for step in ORDER:
        run_cash_cell(cash_magics, cells[step], cells=cells)
    ns = cash_magics.shell.user_ns
    return ns["v"], ns["u"], ns["w"]


@pytest.fixture
def no_floors(monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    monkeypatch.setattr(call_unit, "_COST_FLOOR_S", 0)
    monkeypatch.setenv("CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS", "0")
    monkeypatch.setenv("CASH_CALL_COST_FLOOR_SECONDS", "0")


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("attempt", range(3))
def test_the_cells_give_what_a_run_from_the_top_gives(cash_magics, no_floors, form, attempt):
    """Each attempt runs in a fresh kernel (and, under xdist, often a fresh
    process): one answer, the top-down one."""
    assert _run(cash_magics, FORMS[form]) == (0, 10, [0, 10])


@pytest.mark.parametrize("form", list(FORMS))
def test_the_same_order_again_in_one_kernel_gives_the_same_values(cash_magics, no_floors, form):
    first = _run(cash_magics, FORMS[form])
    assert [_run(cash_magics, FORMS[form]) for _ in range(2)] == [first, first]
    assert first == (0, 10, [0, 10])
