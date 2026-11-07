"""A statement whose text comes again lower down does not make the check
simulate the cells between the two again.

The simulation keeps a snapshot per cell and starts the next check after the
last snapshot that still holds. One that holds no more is a cell one of whose
statements linked other shared-object edges since it was simulated
(``TrackingState.share_edge_records``). Each run of a statement was noted as
such a change when it ran with inputs no run of that text had read before,
though what the simulation reads for the statement was the same. A teaching
notebook that builds a second ``DataLoader`` with the lines of the first
then simulated every cell between the two again, before the next cell, on a
straight run from the top (7 cells, 68 statements and 0.29 s before a
one-line cell of E11).

Counted in simulated statements over a run of one-line cells, not timed.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.virtual_lineage import VirtualLineage
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls

_BETWEEN = 20


@pytest.fixture(autouse=True)
def _memos_unchecked(monkeypatch):
    """The verify mode simulates each cell a memo stands in for as well,
    which is the work counted here."""
    monkeypatch.delenv("CASH_VERIFY_INCREMENTAL_SIM", raising=False)


def _notebook() -> list[str]:
    """``rows = list(data)`` twice, over two values of ``data``, with
    one-line cells between them and below."""
    cells = ["data = [1, 2, 3]", "rows = list(data)\nfirst = len(rows)"]
    cells += [f"step{i} = len(rows) + {i}" for i in range(_BETWEEN)]
    cells += ["data = [4, 5, 6, 7]", "rows = list(data)\nsecond = len(rows)"]
    cells += [f"after{i} = {i} * 2" for i in range(6)]
    return cells


def test_one_line_cells_below_simulate_only_the_cell_before_them(cash_magics):
    cells = _notebook()
    below = cells.index("rows = list(data)\nsecond = len(rows)") + 1
    for cell in cells[:below]:
        run_cash_cell(cash_magics, cell, cells=cells)

    with method_calls(VirtualLineage, "simulate_one_node") as simulated:
        for cell in cells[below:]:
            run_cash_cell(cash_magics, cell, cells=cells)

    # Each check simulates the cell run before it: two statements for the
    # second ``rows`` cell, one for each cell after it. Simulating the cells
    # between the two ``rows`` cells again is _BETWEEN statements more.
    assert simulated.calls <= len(cells) - below + 2, simulated.calls
    assert cash_magics.shell.user_ns["second"] == 4


def test_a_cell_above_edited_is_still_simulated_again(cash_magics):
    """The snapshots kept are the ones nothing changed: an edit above the
    repeated statement still re-runs what reads it."""
    cells = _notebook()
    below = cells.index("rows = list(data)\nsecond = len(rows)") + 1
    for cell in cells[: below + 1]:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells = list(cells)
    cells[0] = "data = [1, 2, 3, 4, 5]"
    cells.append("print(first, second)")
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert cash_magics.shell.user_ns["first"] == 5
    assert cash_magics.shell.user_ns["second"] == 4
