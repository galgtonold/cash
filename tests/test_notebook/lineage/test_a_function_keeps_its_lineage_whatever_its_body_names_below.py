"""A ``def`` whose body names a global bound in a cell below gives its
function one lineage, on a first run and on the next run of the same kernel.

The body's globals are looked up at the call, and a key reading the
function folds them there. Folded into the function's lineage as they stood
when the ``def`` ran, the helper was missing the first time and present the
next, so what held the function took another lineage and missed.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

DEF = "def solve(n):\n    return seg(n) + 1"
HELPER = "def seg(n):\n    return n * {k}"


def test_a_function_naming_a_helper_below_keeps_its_lineage_on_the_next_run(cash_magics):
    cells = [DEF, HELPER.format(k=2), "pair = (len, solve)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    again = cash_magics.tracking_state.variable_lineage

    assert again["solve"] == first["solve"]
    assert again["pair"] == first["pair"]


def test_a_function_whose_default_names_a_global_takes_its_lineage(cash_magics):
    """Control: what the ``def`` evaluates itself -- a default -- is read
    when it runs, so a new value moves the function's lineage."""
    cells = ["k = 2", "def scale(n, k=k):\n    return n * k"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = cash_magics.tracking_state.variable_lineage["scale"]

    cells[0] = "k = 3"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert cash_magics.tracking_state.variable_lineage["scale"] != first
