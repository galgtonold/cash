"""A statement restored for one of its outputs leaves the others a
statement below it bound again.

``a, b = split()`` is restored for ``b``, which the cell being run rebinds
from itself. ``a = list(a)`` below the unpack bound ``a`` again; restored
too, ``a`` went back to the array ``split()`` made, and the cell read what
a run from the top never shows it.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytest.importorskip("numpy")

SPLIT = (
    "import time, numpy as np\n"
    "def split():\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return np.arange(4), np.arange(2)\n"
    "def slow_list(x):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return list(x)"
)


def _run_all_then_last_again(cash_magics, cells):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for _ in range(2):
        run_cash_cell(cash_magics, cells[-1], cells=cells)
    return cash_magics.shell.user_ns


@pytest.mark.parametrize("rebind", ["a = list(a)", "a.append(9)"], ids=["rebound", "changed_in_place"])
def test_the_other_output_keeps_what_the_statement_below_gave_it(cash_magics, rebind):
    split = SPLIT.replace("return np.arange(4)", "return list(range(4))") if "append" in rebind else SPLIT
    ns = _run_all_then_last_again(cash_magics, [split, "a, b = split()", rebind, "b = b * 10"])
    assert ns["b"].tolist() == [0, 10]
    if "append" in rebind:
        assert ns["a"] == [0, 1, 2, 3, 9]
    else:
        assert type(ns["a"]) is list


def test_an_output_still_needed_below_is_restored(cash_magics):
    """The cell rebinds ``a`` too, so ``a = slow_list(a)`` is restored as
    well: the unpack restored after it leaves the ``a`` it gave back."""
    cells = [SPLIT, "a, b = split()", "a = slow_list(a)", "a = a + ['x']\nb = b * 10"]
    ns = _run_all_then_last_again(cash_magics, cells)
    assert ns["a"] == [0, 1, 2, 3, "x"]
    assert ns["b"].tolist() == [0, 10]


def test_a_restored_loop_leaves_the_name_rebound_below_it(cash_magics):
    """A loop is repaired one iteration at a time, and the rebinding below
    it runs again after it: ``a`` ends as that rebinding leaves it."""
    loop = "for k in range(2):\n    a, b = split()"
    ns = _run_all_then_last_again(cash_magics, [SPLIT, loop, "a = list(a)", "b = b * 10"])
    assert ns["b"].tolist() == [0, 10]
    assert type(ns["a"]) is list
