"""A function that calls one defined in a later cell.

``def a(n): return b(n) * 2`` in cell 1 and ``def b(n): return n + 1`` in
cell 2: after cell 2 was edited and only the cell computing ``r = a(3)`` was
run, cash checked what ``a`` came from (cell 1, unchanged) and never cell 2,
so the old ``b`` answered: 8 instead of 26, without a word.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    "caller",
    [
        "def a(n):\n    return b(n) * 2",
        "def a(n):\n    return [b(i) for i in range(n, n + 1)][0] * 2",
        "def a(n):\n    return c(n)\ndef c(n):\n    return b(n) * 2",
    ],
    ids=["direct", "in-a-comprehension", "through-another-helper"],
)
def test_an_edit_to_a_later_helper_reruns_it(cash_magics, mock_shell, caller):
    cells = [caller, "def b(n):\n    return n + 1", "r = a(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["r"] == 8

    cells[1] = "def b(n):\n    return n + 10"
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert mock_shell.user_ns["r"] == 26


def test_a_later_helper_left_as_it_is_is_not_rerun(cash_magics, mock_shell):
    cells = ["def a(n):\n    return b(n) * 2", "calls = []\ndef b(n):\n    calls.append(n)\n    return n + 1", "r = a(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert mock_shell.user_ns["r"] == 8
    assert mock_shell.user_ns["calls"] == [3]

