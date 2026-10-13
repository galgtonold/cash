"""What holds a notebook function, or a stored call of it, sees a global
the function reads edited above, run or not.

A key reads a notebook function by its code, and the globals it reads as
they are at the call. The function's lineage still folds those globals as
they were when its ``def`` ran: that is what rebuilds a list holding the
function, and what the simulation keys a stored call with while an edited
cell above has not run yet.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

COLS = "cols = ['a', 'b']"
EDITED = "cols = ['a', 'b', 'c']"
DEF = "def make():\n    return [c * 2 for c in cols]"
CALL = "r = make()"


def test_a_function_held_in_a_list_still_sees_an_edited_global(cash_magics):
    """Only the key reads the function by its code: the lineage still moves
    when the ``def`` runs again over the edited global, so the list holding
    the function is rebuilt and the call through it re-keyed."""
    cash_magics.cash_persist("on")
    cells = [COLS, DEF, "fns = [make]", "r = fns[0]()"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["r"] == ["aa", "bb"]

    cells[0] = EDITED
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"]


def test_an_edited_global_not_run_yet_reaches_a_stored_call(cash_magics):
    cash_magics.cash_persist("on")
    cells = [COLS, DEF, CALL]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells[0] = EDITED
    run_cash_cell(cash_magics, CALL, cells=cells)

    assert cash_magics.shell.user_ns["r"] == ["aa", "bb", "cc"]
