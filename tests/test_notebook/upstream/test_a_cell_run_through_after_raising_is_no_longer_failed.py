"""A cell that raised once and then ran through is no longer taken to raise.

cash records where a cell's last run raised (``TrackingState.failed_cells``)
and the upstream simulation stops the cell there. A cell reading a name only
a cell below binds raises ``NameError`` before that cell has run; once it
has, the cell runs through on the name in memory, but as written, uncached
(NOTEBOOK-FORWARD-READ), and only a run cash makes of the cell cleared the
record. The simulation went on stopping the cell at its loop, and a repair
re-ran the ``nps = {}`` above it without the loop that fills it.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from cash.source_norm import exact_source_digest
from tests._cell_driver import run_cash_cell

CELLS = ["nps = {}\nfor v in sv:\n    nps[v] = v * 2", "sv = [1, 2]", "out = dict(nps)"]


def _run_as_the_kernel_does(cash_magics, code):
    """Through cash's ``run_cell`` hook, so a cell the check steps aside for
    runs through IPython's own ``run_cell`` as written."""
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(CELLS)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        outcome = cash_magics._run_cell_cached(code, (), {})
    return outcome, [str(w.message) for w in caught]


def test_a_forward_read_run_that_runs_through_clears_where_it_raised(cash_magics):
    state = cash_magics.tracking_state
    digest = exact_source_digest(CELLS[0])
    with pytest.raises(NameError):
        run_cash_cell(cash_magics, CELLS[0], cells=CELLS)
    assert state.failed_cells.get(digest) == 1, "the loop raised"

    run_cash_cell(cash_magics, CELLS[1], cells=CELLS)
    outcome, said = _run_as_the_kernel_does(cash_magics, CELLS[0])

    assert outcome.success
    assert any("NOTEBOOK-FORWARD-READ" in m for m in said), said
    assert cash_magics.shell.user_ns["nps"] == {1: 2, 2: 4}
    assert digest not in state.failed_cells, "the cell ran whole, yet is still taken to stop at its loop"
    assert digest not in state.failed_cell_names


def test_a_forward_read_run_that_raises_again_keeps_where_it_raised(cash_magics):
    """Control: run as written and raising again, the record stays."""
    state = cash_magics.tracking_state
    digest = exact_source_digest(CELLS[0])
    with pytest.raises(NameError):
        run_cash_cell(cash_magics, CELLS[0], cells=CELLS)
    run_cash_cell(cash_magics, CELLS[1], cells=CELLS)
    cash_magics.shell.user_ns["sv"] = None  # not iterable: the loop raises again

    outcome, _ = _run_as_the_kernel_does(cash_magics, CELLS[0])

    assert not outcome.success
    assert state.failed_cells.get(digest) == 1
