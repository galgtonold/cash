"""A repair that fails for a statement cash left out does not fail the cell.

To bring a cell up to date cash re-runs statements above it. When one of them
fails because the plan left out a statement above it that sets what it reads,
the statement never got its inputs as the notebook has them: the failure is
cash's, and a run from the top does not have it. The cell then runs as
written, on what it held before the repair, uncached, with
``NOTEBOOK-REPAIR-GAP``; it used to fail with ``UpstreamStateError``.

The gap is made here by taking the producer out of the plan, so the test pins
how the check reacts to one, whatever leaves it.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from cash.exceptions import CashCacheIneffectiveWarning, UpstreamStateError
from cash.notebook.ipython._types import RunInstead
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner
from tests._cell_driver import run_cash_cell

CELLS = ["a = 1", "c = 5\nb = a + c", "print(b)"]
EDITED = ["a = 1", "c = 6\nb = a + c", "print(b)"]


def _without(statement):
    plan = ReexecutionPlanner.plan

    def planned(self, *args, **kwargs):
        made = plan(self, *args, **kwargs)
        return made._replace(statements=[s for s in made.statements if s != statement])

    return patch.object(ReexecutionPlanner, "plan", planned)


def _run_reader(magics, cells):
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        result = magics._cell_executor.execute_cell(cells[2], ttl=None, cell_id=None)
    return result, [w.message for w in caught if isinstance(w.message, CashCacheIneffectiveWarning)]


def _set_up(magics, shell):
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)
    del shell.user_ns["a"]


def test_the_cell_runs_as_written_with_a_warning(cash_magics, mock_shell):
    _set_up(cash_magics, mock_shell)
    with _without("a = 1"):
        result, told = _run_reader(cash_magics, EDITED)

    assert isinstance(result, RunInstead), result
    assert result.source == EDITED[2], f"the cell did not run as written: {result.source!r}"
    assert [m.code for m in told] == ["NOTEBOOK-REPAIR-GAP"], told
    assert "a = 1" in str(told[0]) and "uncached" in str(told[0]), told[0]


def test_what_the_repair_rebound_is_put_back(cash_magics, mock_shell, statement_processor):
    _set_up(cash_magics, mock_shell)
    lineage_before = statement_processor.tracking_state.variable_lineage["c"]
    with _without("a = 1"):
        _run_reader(cash_magics, EDITED)

    assert mock_shell.user_ns["c"] == 5, "c = 6 ran in the repair and was left bound"
    assert mock_shell.user_ns["b"] == 6
    assert "a" not in mock_shell.user_ns
    assert statement_processor.tracking_state.variable_lineage["c"] == lineage_before


def test_a_repair_that_really_fails_still_fails_the_cell(cash_magics, mock_shell):
    """Positive control: with nothing left out, the statement's own failure
    stops the cell, as a run from the top stops at it."""
    _set_up(cash_magics, mock_shell)
    broken = ["a = 1", "c = 6\nb = a + c + undefined_name", "print(b)"]
    mock_shell.user_ns["a"] = 1
    with pytest.raises(UpstreamStateError) as raised:
        run_cash_cell(cash_magics, broken[2], cells=broken)
    assert "undefined_name" in str(raised.value)
    assert type(raised.value) is UpstreamStateError
