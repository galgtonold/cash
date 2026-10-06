"""The statements after the one a cell raised at are not run by a cell below.

A cell that fails part-way leaves what came after the error unrun, as in a
plain kernel. The upstream check credited the whole cell, saw the names
bound after the error lag behind, and ran those lines -- past a failed
``assert`` -- when the next cell ran, silently.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


def test_a_failed_assert_does_not_let_the_next_line_run(cash_magics, mock_shell):
    cells = [
        "prices = [10, -1, 30]",
        "assert all(p > 0 for p in prices), 'negative price'\nprices = [p * 1.2 for p in prices]",
        "total = sum(prices)",
    ]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with pytest.raises(AssertionError):
        run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["total"] == 39


def test_a_value_bound_before_the_error_is_kept(cash_magics, mock_shell):
    cells = ["x = 2\n1 / 0\nx = 3", "y = x * 10"]
    with pytest.raises(ZeroDivisionError):
        run_cash_cell(cash_magics, cells[0], cells=cells)
    run_cash_cell(cash_magics, cells[1], cells=cells)

    assert mock_shell.user_ns["y"] == 20


def test_the_cell_counts_whole_once_it_runs_through(cash_magics, mock_shell):
    cells = ["ok = False", "assert ok\nz = 1", "w = z + 1"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with pytest.raises(AssertionError):
        run_cash_cell(cash_magics, cells[1], cells=cells)
    mock_shell.user_ns["ok"] = True
    run_cash_cell(cash_magics, cells[1], cells=cells)
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["w"] == 2
