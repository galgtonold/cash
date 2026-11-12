"""The upstream check restores a statement that wrote into an array into
the array itself.

Run below a statement it skipped (``arr[3] = 7``), the upstream check
restores that statement's entry. Bound to the stored copy, ``arr`` was a new
object, and ``reg = {id(arr): 'a'}`` taken since no longer found it, where
plain Python, which writes into the array, does. The state now goes into
the live array, as at a hit in the statement's own place.
"""

from __future__ import annotations

import weakref

from tests._cell_driver import run_cash_cell

MAKE = "import numpy as np\narr = np.zeros(10)"
BODY = "arr[3] = 7"


def test_an_id_key_taken_since_the_store_finds_the_restored_array(cash_magics, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    ns = cash_magics.shell.user_ns
    cells = [MAKE, BODY, "r = 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells = [MAKE, "reg = {id(arr): 'a'}", BODY, "r = reg.get(id(arr)) == 'a'"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    # Held weakly: a reference of the test's own would be another holder.
    held = weakref.ref(ns["arr"])
    # The cell below the skipped one: the upstream check restores it first.
    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert ns["arr"] is held()
    assert ns["arr"][3] == 7
    assert ns["r"] is True
