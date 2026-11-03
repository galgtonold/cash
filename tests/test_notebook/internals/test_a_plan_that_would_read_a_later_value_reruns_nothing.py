"""A plan that would run a statement on a value from below re-runs nothing.

Whatever pass leaves it out, a statement the plan re-runs must read each
input at the version a run from the top gives it there. When the version
it would read is one a later statement makes, cash re-runs nothing: the
cell runs as written, on what it held, uncached, with
``NOTEBOOK-REPAIR-GAP`` -- not on values a run from the top never has.
The gap is made here by switching off the completion that brings in the
``k = 5`` the append reads.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook.ipython._types import RunInstead
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner
from tests._cell_driver import run_cash_cell

CELLS = ["base = 1", "X, Y = [base], [10, 20]", "k = 5\nY.append(k)", "k = 100", "print(X)"]
EDITED = ["base = 2", *CELLS[1:]]


def _run_reader(magics, cells):
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        patch.object(ReexecutionPlanner, "_complete_shadowed_var_producers", lambda self, indices, *a, **k: indices),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = magics._cell_executor.execute_cell(cells[-1], ttl=None, cell_id=None)
    return result, [w.message for w in caught if isinstance(w.message, CashCacheIneffectiveWarning)]


def test_nothing_reruns_on_the_later_k(cash_magics, mock_shell):
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    result, told = _run_reader(cash_magics, EDITED)

    assert mock_shell.user_ns["Y"] == [10, 20, 5], mock_shell.user_ns["Y"]
    assert mock_shell.user_ns["X"] == [1], "the repair was not put back"
    assert isinstance(result, RunInstead) and result.source == EDITED[-1], result
    assert [m.code for m in told] == ["NOTEBOOK-REPAIR-GAP"], told
    assert "Y.append(k)" in str(told[0]) and "k = 5" in str(told[0]), told[0]


def test_with_the_completion_the_append_reads_k_as_it_was(cash_magics, mock_shell):
    """The control: unpatched, the plan re-runs ``k = 5`` before the append."""
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    run_cash_cell(cash_magics, EDITED[-1], cells=EDITED)

    assert mock_shell.user_ns["X"] == [2]
    assert mock_shell.user_ns["Y"] == [10, 20, 5], mock_shell.user_ns["Y"]
    assert mock_shell.user_ns["k"] == 100
