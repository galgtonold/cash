"""A cell whose memo is reused snapshots the modules as they are now.

A check reuses a cell's memo when what the cell reads is as it was. The
modules it compares are only those the cell names (`state_of`): an
import added above it changes nothing it can see. But the snapshot the
reuse stored for the cell kept the modules of the memo's own time, all
of them: after ``import collections`` was added to the cell above, the
cell's snapshot still lacked it, and a later check starting after the
cell began without it. expert_8's replay logged it 89 times as "cell 1's
memo differs from simulating it: snapshot 0 differs in fields [2]".
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream import virtual_lineage
from tests._cell_driver import run_cash_cell


@pytest.fixture
def verifying(monkeypatch):
    """Each reused memo checked against simulating its cell
    (``CASH_VERIFY_INCREMENTAL_SIM``); the differences found."""
    monkeypatch.setenv("CASH_VERIFY_INCREMENTAL_SIM", "1")
    found = virtual_lineage.VERIFY_FAILURES
    del found[:]
    yield found
    del found[:]


def test_the_snapshot_of_a_reused_cell_holds_an_import_added_above(cash_magics, verifying):
    cells = ["import zipfile", "x = 1", "y = x + 1", "z = y + 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[0] = "import zipfile\nimport collections"
    run_cash_cell(cash_magics, cells[0], cells=cells)
    # Simulates the edited cell and reuses the memo of the one below it.
    run_cash_cell(cash_magics, cells[3], cells=cells)

    entries = cash_magics._upstream_checker.simulator.cache.entries
    assert "collections" in entries[1].virtual_modules, entries[1].virtual_modules
    assert not verifying, verifying
