"""A cell that starts with a space or a tab runs like any other cell.

IPython takes the first line's indentation off every line, so such a cell
runs. Handed to a parser as it is, the text is refused, and the cell ran
outside the pipeline: ` df.loc["total"] = ...` changed the frame with no
lineage bump, and the next `df` was served from the entry stored before it.
Pasting a line out of a function is how a cell gets a leading space.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize("indent", [" ", "    ", "\t"])
def test_an_indented_cell_runs_through_the_pipeline(cash_magics, mock_shell, indent):
    run_cash_cell(cash_magics, "n = 1")
    run_cash_cell(cash_magics, f"{indent}n = 2")
    assert mock_shell.user_ns["n"] == 2


@pytest.mark.parametrize("indent", [" ", "\t"])
def test_a_change_made_by_an_indented_cell_reaches_the_cells_that_read_it(cash_magics, mock_shell, indent):
    run_cash_cell(cash_magics, "d = {'a': 1}")
    run_cash_cell(cash_magics, "size = len(d)")
    assert mock_shell.user_ns["size"] == 1
    run_cash_cell(cash_magics, f"{indent}d['b'] = 2")
    run_cash_cell(cash_magics, "size = len(d)")
    assert mock_shell.user_ns["size"] == 2, "served from the entry stored before the change"


def test_every_line_of_an_indented_block_loses_the_first_lines_indentation(cash_magics, mock_shell):
    run_cash_cell(cash_magics, "total = 0")
    run_cash_cell(cash_magics, "  for i in range(3):\n      total += i")
    assert mock_shell.user_ns["total"] == 3
