"""A check simulates the cells whose inputs changed, not every cell below.

Before each cell the upstream check simulates the cells above it. Its cache
kept a snapshot per cell only up to the cell last checked, and only up to
the first one that changed: after an edit near the top, or a run of a cell
near the top, the next check further down simulated every cell in between
again, a few milliseconds a statement, growing with the notebook. Each cell
now keeps what its last simulation read and did (``cell_memo``); one whose
text and inputs are as they were is not simulated again.

Counted in simulated statements, not timed.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.virtual_lineage import VirtualLineage
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls


@pytest.fixture(autouse=True)
def _memos_unchecked(monkeypatch):
    """The verify mode simulates each cell a memo stands in for as well,
    which is the work counted here."""
    monkeypatch.delenv("CASH_VERIFY_INCREMENTAL_SIM", raising=False)


def _notebook(n: int, tag: str) -> list[str]:
    """A base cell, a cell reading it, then *n* cells each reading the one
    before it in pairs: an edit of the base reaches the second cell only."""
    cells = ["base = 1", f"{tag}_reader = base + 1"]
    for i in range(n):
        cells.append(f"{tag}{i} = {i}" if i % 2 == 0 else f"{tag}{i} = {tag}{i - 1} + 1")
    cells.append(f"{tag}_last = {tag}_reader")
    return cells


def _simulated_for_the_last_cell(magics, cells: list[str], before_last) -> int:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    cells = before_last(cells)
    with method_calls(VirtualLineage, "simulate_one_node") as simulated:
        run_cash_cell(magics, cells[-1], cells=cells)
    return simulated.calls


def _edit_the_base(magics):
    def edit(cells):
        cells = list(cells)
        cells[0] = "base = 2"
        run_cash_cell(magics, cells[0], cells=cells)
        return cells

    return edit


def _rerun_the_reader(magics):
    def rerun(cells):
        run_cash_cell(magics, cells[1], cells=cells)
        return cells

    return rerun


def test_an_edit_near_the_top_simulates_the_cells_it_reaches(cash_magics):
    short = _simulated_for_the_last_cell(cash_magics, _notebook(4, "s"), _edit_the_base(cash_magics))
    long = _simulated_for_the_last_cell(cash_magics, _notebook(40, "l"), _edit_the_base(cash_magics))

    # The edited cell and the one reading it; nothing else changed.
    assert 0 < short <= 4, short
    assert long <= short, f"{long} statements simulated at 40 cells below the edit against {short} at 4"


def test_a_cell_rerun_near_the_top_leaves_the_cells_below_unsimulated(cash_magics):
    short = _simulated_for_the_last_cell(cash_magics, _notebook(4, "s"), _rerun_the_reader(cash_magics))
    long = _simulated_for_the_last_cell(cash_magics, _notebook(40, "l"), _rerun_the_reader(cash_magics))

    # The cell that ran, whose records its run moved.
    assert 0 < short <= 2, short
    assert long <= short, f"{long} statements simulated at 40 cells below the rerun against {short} at 4"


def test_the_last_cell_still_sees_the_edit(cash_magics, mock_shell):
    cells = _notebook(10, "v")
    _simulated_for_the_last_cell(cash_magics, cells, _edit_the_base(cash_magics))

    assert mock_shell.user_ns["v_reader"] == 3
    assert mock_shell.user_ns["v_last"] == 3
