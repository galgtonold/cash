"""A saved edit counts as a change above the cell wherever it sits among the
cells the upstream check simulates again.

The check looks for the first cell above whose snapshot it cannot reuse and
simulates from there. It called the notebook changed only when that first
cell's text had changed. With a cell simulated again for another reason
above an edited one (after a restart and a rebuild, the first cell: its
statements ran since), the edit was not counted, and the value the old code
left in memory was taken for an edit the user ran and did not save, or for a
loop's trusted output, and kept.

Pins the scan; the notebook behaviour is in
``tests/test_notebook_integration/restart/test_a_saved_edit_after_a_rebuild_reaches_the_cells_below.py``.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

CELLS = ["n = 1", "x = 1", "x = x + 100", "w = x * 2 + n"]
EDITED = [*CELLS[:2], "x = x + 200", CELLS[3]]


def _start(magics, cells: list[str], *, first_cell_holds: bool):
    lineage = magics._upstream_checker.simulator.virtual_lineage
    if not first_cell_holds:
        # What a changed file leaves on a snapshot: simulated again, with
        # no change to its text.
        entries = lineage.cache.entries
        entries[0] = entries[0]._replace(vars_with_stale_files={"n"})
    return lineage.find_incremental_start(3, cells)


def _run_all(magics) -> None:
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)


def test_an_edit_below_the_first_cell_simulated_again_is_a_modification(cash_magics):
    _run_all(cash_magics)
    start = _start(cash_magics, EDITED, first_cell_holds=False)
    assert start.first_changed_cell == 0
    assert start.cache_had_hash_mismatch


def test_a_cell_simulated_again_with_no_edit_is_not_one(cash_magics):
    """Negative control: simulating again is not an edit."""
    _run_all(cash_magics)
    start = _start(cash_magics, CELLS, first_cell_holds=False)
    assert start.first_changed_cell == 0
    assert not start.cache_had_hash_mismatch


def test_an_edit_found_by_the_scan_itself_still_is(cash_magics):
    _run_all(cash_magics)
    start = _start(cash_magics, EDITED, first_cell_holds=True)
    assert start.first_changed_cell == 2
    assert start.cache_had_hash_mismatch
