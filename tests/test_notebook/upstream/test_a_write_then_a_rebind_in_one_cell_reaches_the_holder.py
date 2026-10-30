"""A cell that changes an object and then rebinds its name keeps the change
where another variable holds the object.

``g = [t]``, then one cell ``t.tag = 2`` and ``t = O()``: the cell reads and
rebinds ``t``, which was taken for a swap (``a, b = b, a``) and put back
from its entry before the cell ran. The write landed on that copy, and
``g[0].tag`` stayed unset from the first Run All, where plain Python sets it.
"""

import pytest

from tests._cell_driver import run_cash_cell

SHAPES = {
    "attribute": ("class O:\n    pass\nt = O()", "t.tag = 2\nt = O()", "r = getattr(g[0], 'tag', None)"),
    "attribute_one_line": ("class O:\n    pass\nt = O()", "t.tag = 2; t = O()", "r = getattr(g[0], 'tag', None)"),
    "item": ("t = {}", "t['tag'] = 2\nt = {}", "r = g[0].get('tag')"),
}


@pytest.mark.parametrize("shape", SHAPES)
def test_the_holder_sees_the_write(monkeypatch, request, shape):
    # Every statement stored, however quick: the class, ``t`` and ``g``.
    monkeypatch.setenv("CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS", "0")
    cash_magics = request.getfixturevalue("cash_magics")
    make, change, read = SHAPES[shape]
    cells = [make, "g = [t]", change, read]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["r"] == 2


def test_a_swap_is_still_put_back_before_it_runs_again(cash_magics):
    """Control: ``a, b = b, a`` run alone again swaps the values the cells
    above made, not the ones it swapped last time."""
    cells = ["a = [1]\nb = [2]", "a, b = b, a"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert (cash_magics.shell.user_ns["a"], cash_magics.shell.user_ns["b"]) == ([2], [1])
