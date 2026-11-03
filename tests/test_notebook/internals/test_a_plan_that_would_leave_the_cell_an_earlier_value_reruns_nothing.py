"""A plan that would leave the cell a name at an earlier version re-runs nothing.

A re-run statement writes all of its outputs again. When a statement below
it writes one of them anew, and the plan leaves that one out, the cell
would read the re-run's version, which a run from the top never hands it:
in E8 (homl ch07) a repair re-ran the ``train_test_split`` of the moons
data, not the MNIST split below it, and ``estimator.score(X_test, ...)``
failed on 2 features for 784. Cash re-runs nothing instead: the cell runs
as written, on what it held, uncached, with ``NOTEBOOK-REPAIR-GAP``. The
gap is made here by taking ``Y = [7]`` out of what the backward scan
schedules, and switching off the completion that would bring it back.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from cash.exceptions import CashCacheIneffectiveWarning
from cash.notebook.ipython._types import RunInstead
from cash.notebook.upstream.mismatch_classifier import MismatchClassifier
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner
from tests._cell_driver import run_cash_cell

CELLS = ["base = 1", "X, Y = [base], [10, 20]", "Y = [7]", "Z = (X, Y)"]
EDITED = ["base = 2", *CELLS[1:]]


_scan = MismatchClassifier.backward_scan_pass


def _scan_without_y(self, sim, result, **kwargs):
    run, restored, cost = _scan(self, sim, result, **kwargs)
    return [i for i in run if sim.trace[i].stmt_code != "Y = [7]"], restored, cost


def _run_reader(magics, cells):
    with (
        patch.object(MismatchClassifier, "backward_scan_pass", _scan_without_y),
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        patch.object(ReexecutionPlanner, "_complete_shadowed_var_producers", lambda self, indices, *a, **k: indices),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = magics._cell_executor.execute_cell(cells[-1], ttl=None, cell_id=None)
    return result, [w.message for w in caught if isinstance(w.message, CashCacheIneffectiveWarning)]


def test_nothing_reruns_over_the_later_y(cash_magics, mock_shell):
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    result, told = _run_reader(cash_magics, EDITED)

    assert [m.code for m in told] == ["NOTEBOOK-REPAIR-GAP"], (told, result, mock_shell.user_ns.get("Z"))
    assert isinstance(result, RunInstead) and result.source == EDITED[-1], result
    assert mock_shell.user_ns["Y"] == [7], mock_shell.user_ns["Y"]
    assert mock_shell.user_ns["X"] == [1], "the repair was not put back"
    assert "X, Y = ([base]" in str(told[0]) and "Y = [7]" in str(told[0]) and "after it" in str(told[0]), told[0]


def test_with_the_completion_the_cell_reads_y_as_it_is_below(cash_magics, mock_shell):
    """The control: unpatched, the plan re-runs ``Y = [7]`` after the split."""
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    run_cash_cell(cash_magics, EDITED[-1], cells=EDITED)

    assert mock_shell.user_ns["X"] == [2]
    assert mock_shell.user_ns["Y"] == [7], mock_shell.user_ns["Y"]
    assert mock_shell.user_ns["Z"] == ([2], [7]), mock_shell.user_ns["Z"]
