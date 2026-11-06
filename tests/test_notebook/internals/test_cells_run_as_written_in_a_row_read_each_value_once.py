"""Cells run as written one after another fingerprint a value once each.

Around a cell cash steps aside from (here ``NOTEBOOK-FORWARD-READ``: it reads
a name only a cell below binds), every variable with a lineage is
fingerprinted before and after, to forget the ones the cell changed in place
(`StatementProcessor.running_as_written`). A participant's notebook ran forty
such cells in a row next to a 2-million-row frame and the lists it was built
from: two fingerprints of every variable per cell, 13-16 s a cell where plain
Python took 0.03 s. The fingerprint a value had after one such cell is its
fingerprint before the next while it is the same object under the same
lineage.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from cash.notebook.statement import processor as processor_module

CELLS = [
    "data = [[i] for i in range(300)]",
    "n = len(later)",
    "later = [1, 2]",
]


@pytest.fixture
def shell(cash_magics, mock_shell):
    """The notebook shell under ``%cash_on``; a cell goes through the
    magics' ``run_cell`` hook, which runs a cell cash steps aside from
    through the shell's own ``run_cell``."""
    cash_magics.cash_on("")
    return mock_shell


def _run(shell, cell: str, cells: list[str] = CELLS) -> None:
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        result = shell.run_cell(cell)
    error = getattr(result, "error_in_exec", None)
    if isinstance(error, BaseException):
        raise error


@pytest.fixture
def reads_of_data(shell):
    """How many times a fingerprint read ``data``'s list."""
    count = [0]
    real = processor_module._fingerprint

    def counting(value):
        if value is shell.user_ns.get("data"):
            count[0] += 1
        return real(value)

    with patch.object(processor_module, "_fingerprint", counting):
        yield count


def test_a_forward_read_runs_as_written(shell, statement_processor):
    """Control: the cell is one cash steps aside from, and the value it only
    read keeps its lineage."""
    _run(shell, CELLS[0])
    _run(shell, CELLS[2])
    lineage = statement_processor.tracking_state.variable_lineage["data"]
    with patch.object(statement_processor, "running_as_written", wraps=statement_processor.running_as_written) as ran:
        _run(shell, CELLS[1])
    assert ran.called
    assert shell.user_ns["n"] == 2
    assert statement_processor.tracking_state.variable_lineage["data"] == lineage


def test_cells_in_a_row_fingerprint_a_value_once_each(shell, reads_of_data):
    _run(shell, CELLS[0])
    _run(shell, CELLS[2])
    for _ in range(4):
        _run(shell, CELLS[1])
    # Two for the first cell, then only the one after each cell.
    assert reads_of_data[0] == 5


def test_a_change_the_next_cell_makes_is_still_seen(shell, statement_processor):
    _run(shell, CELLS[0])
    _run(shell, CELLS[2])
    _run(shell, CELLS[1])
    changer = "n = len(later)\ndata[0].append(-1)"
    _run(shell, changer, [CELLS[0], changer, CELLS[2]])
    assert shell.user_ns["data"][0] == [0, -1]
    assert "data" not in statement_processor.tracking_state.variable_lineage


def test_a_value_bound_again_between_the_cells_is_read_again(shell, reads_of_data, statement_processor):
    _run(shell, CELLS[0])
    _run(shell, CELLS[2])
    _run(shell, CELLS[1])
    _run(shell, "data = [[i] for i in range(301)]")
    _run(shell, CELLS[1])
    # The new list: fingerprinted before and after the second cell.
    assert reads_of_data[0] == 4
    assert "data" in statement_processor.tracking_state.variable_lineage
