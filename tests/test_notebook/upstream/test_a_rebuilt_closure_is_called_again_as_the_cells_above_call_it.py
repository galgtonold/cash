"""A closure the cells above rebuild is called again as they call it.

``draw, rebind = make(BIG)`` and then ``rebind([0])`` leave ``draw`` reading
``[0]``. Re-run alone below an edit of ``make`` (or a change to ``BIG``
below, whose share edge with ``draw`` came and went with the process's string
hashes), the reader got ``draw, rebind = make(BIG)`` again, but not the call
that points the shared closure at ``[0]``: it binds nothing, so no pass took
it for part of what built ``draw``, and ``y`` was ``len(BIG)`` where a
restart and Run All give 1.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

MAKE = (
    "import time\nBIG = [{'id': i} for i in range(2000)]\n"
    "def make(rows):\n    def draw(k):\n        time.sleep(0.02)\n        return len(rows)EXTRA\n"
    "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, rebind\n"
)
READER = "y = draw(1)"


def test_an_edit_above_rebuilds_the_closure_with_its_later_call(cash_magics):
    cells = [MAKE.replace("EXTRA", ""), "draw, rebind = make(BIG)\nrebind([0])", READER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[0] = MAKE.replace("EXTRA", " + 10")
    run_cash_cell(cash_magics, READER, cells=cells)
    assert cash_magics.shell.user_ns["y"] == 11


def test_a_change_below_leaves_the_reader_as_run_all_gives_it(cash_magics):
    cells = [MAKE.replace("EXTRA", "") + "draw, rebind = make(BIG)\nrebind([0])", READER, "BIG.append({'id': -1})"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    for _ in range(3):
        run_cash_cell(cash_magics, READER, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    run_cash_cell(cash_magics, READER, cells=cells)
    assert cash_magics.shell.user_ns["y"] == 1


def test_a_call_of_a_function_that_changes_nothing_is_not_run_again(cash_magics):
    """Control: ``draw`` only reads what its closure holds, so ``n =
    draw(0)`` below the rebuilt statement is no part of what built it."""
    cells = [MAKE.replace("EXTRA", ""), "draw, rebind = make(BIG)\nrebind([0])\nn = draw(0)", READER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[0] = MAKE.replace("EXTRA", " + 10")
    run_cash_cell(cash_magics, READER, cells=cells)
    assert cash_magics.shell.user_ns["y"] == 11
    ran = [s.get("code", "") for s in cash_magics._last_cell_metrics["statements"]]
    assert "rebind([0])" in ran
    assert "n = draw(0)" not in ran


def test_a_rebind_reached_through_a_dict_is_called_on_the_rebuilt_closure(cash_magics):
    """``HOOKS = {'r': rebind}`` and ``setup()`` calling ``HOOKS['r']([0])``:
    the rebuild re-ran ``make`` and ``setup()``, but ``HOOKS`` still held the
    old ``rebind``, so the call pointed the old closure at ``[0]`` and the
    rebuilt ``draw`` read all of ``BIG`` (2001 where Run All gives 1)."""
    make = MAKE.replace("EXTRA", "")
    hooks = "draw, rebind = make(BIG)\nHOOKS = {'r': rebind}\ndef setup():\n    HOOKS['r']([0])"
    cells = [make, hooks, "setup()", READER, "BIG.append({'id': -1})"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["y"] == 1
    run_cash_cell(cash_magics, READER, cells=cells)
    assert ns["y"] == 1
