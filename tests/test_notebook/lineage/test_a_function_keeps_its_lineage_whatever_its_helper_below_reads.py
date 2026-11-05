"""A ``def`` whose helper, defined below it in the same cell, reads an
attribute or draws with pyplot gives its function one lineage, on a first
run and on the next run of the same kernel.

The helper is left out of the function's lineage, but the ``def``
statement still took two things from its body once it was bound: the
globals its code names (``tree.get``), through the key's walk of the
called functions, and the verdict that the ``def`` draws through it, which
put the function's value into its lineage. Missing the first time and
present the next, what held the function took another lineage and missed.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

TREE = (
    "def walk(tree):\n"
    "    return list(preorder(tree, 0))\n"
    "\n"
    "def preorder(tree, root):\n"
    "    yield root\n"
    "    for child in tree.get(root, ()):\n"
    "        yield from preorder(tree, child)"
)
PLOT = (
    "def walk(tree, plot=False):\n"
    "    if plot:\n"
    "        plt.figure()\n"
    "        plot_segment(list(tree))\n"
    "    return sorted(tree)\n"
    "\n"
    "def plot_segment(segment):\n"
    "    plt.plot(segment)"
)
SETUP = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"


@pytest.mark.parametrize("helper", [TREE, PLOT], ids=["reads_an_attribute", "draws_with_pyplot"])
def test_a_function_keeps_its_lineage_on_the_next_run(cash_magics, helper):
    cells = [SETUP, helper, "pair = (len, walk)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    again = cash_magics.tracking_state.variable_lineage

    assert again["walk"] == first["walk"]
    assert again["pair"] == first["pair"]


def test_a_helper_above_still_moves_the_function_when_it_changes(cash_magics):
    """Control: a helper bound above the ``def`` is read as the ``def``
    finds it, so new code in it moves the function's lineage."""
    cells = ["def preorder(tree, root):\n    return [root]", "def walk(tree):\n    return preorder(tree, 0)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = cash_magics.tracking_state.variable_lineage["walk"]

    cells[0] = "def preorder(tree, root):\n    return [root, root]"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert cash_magics.tracking_state.variable_lineage["walk"] != first
