"""A cell that raises before it writes into a large value rebuilds nothing
above it on its next run.

A cell that changes a value in place (``rows[0] = ...``) and has run before
starts again from the value as the cells above leave it: the upstream check
rebuilds a large value (one whose content it does not compare) when the cell
started from that very object last time. A run that raised before its write
(``rows[0] = rows['typo']``) left the object as it found it, so each run of
the cell while its error is fixed rebuilt the value for nothing, seconds for
an error plain Python shows at once.

The producer counts its runs with an ``os.open`` append (a write cash does
not restore); the value keeps its identity when nothing rebuilt it.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

# More items than cash hashes by content, so the identity check applies.
_MAKE = """\
import os
def make(n):
    fd = os.open({log!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b'x')
    os.close(fd)
    return [[i] for i in range(n)]"""
_PRODUCE = "rows = make(300)"


def _runs(log) -> int:
    return len(log.read_bytes()) if log.exists() else 0


def _attempt(magics, cells, source):
    cells = [*cells, source]
    with pytest.raises(TypeError, match="list indices must be integers"):
        run_cash_cell(magics, source, cells=cells)


@pytest.mark.parametrize(
    "attempts",
    [
        pytest.param(["rows[0] = rows['a']", "rows[0] = rows['b']", "rows[0] = rows['b']"], id="first-statement"),
        pytest.param(
            ["n = len(rows)\nrows[0] = rows['a']", "n = len(rows)\nrows[0] = rows['b']"], id="later-statement"
        ),
        pytest.param(["rows[0] += rows['a']", "rows[0] += rows['b']"], id="augmented"),
    ],
)
def test_no_rebuild_while_the_error_is_fixed(cash_magics, mock_shell, tmp_path, attempts):
    log = tmp_path / "runs.log"
    cells = [_MAKE.format(log=str(log)), _PRODUCE]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    rows = mock_shell.user_ns["rows"]
    assert _runs(log) == 1

    for source in attempts:
        _attempt(cash_magics, cells, source)

    assert _runs(log) == 1, "the producer above re-ran for a cell that changed nothing"
    assert mock_shell.user_ns["rows"] is rows
    assert rows == [[i] for i in range(300)]


def test_a_write_before_the_error_still_rebuilds(cash_magics, mock_shell, tmp_path):
    """The control: a statement that wrote before the error (``rows[5]``,
    then the item store that raises) leaves the cell's own output, and the
    next run starts from the value as the cells above leave it."""
    log = tmp_path / "runs.log"
    cells = [_MAKE.format(log=str(log)), _PRODUCE]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    _attempt(cash_magics, cells, "rows[5], rows['k'] = ['w'], 1")
    assert mock_shell.user_ns["rows"][5] == ["w"]
    _attempt(cash_magics, cells, "rows[0], rows['k'] = [7], 1")

    rows = mock_shell.user_ns["rows"]
    assert rows[5] == [5], "the write of the run before was kept"
    assert rows[0] == [7]


def test_a_statement_before_the_error_still_rebuilds(cash_magics, mock_shell, tmp_path):
    """The control: a statement that ran whole before the one that raised
    (``rows.append``) changed the value, and the next run starts from the
    value as the cells above leave it, as Restart & Run All does."""
    log = tmp_path / "runs.log"
    cells = [_MAKE.format(log=str(log)), _PRODUCE]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    _attempt(cash_magics, cells, "rows.append([1])\nm = rows['a']")
    _attempt(cash_magics, cells, "rows.append([1])\nm = rows['b']")

    assert len(mock_shell.user_ns["rows"]) == 301


def test_a_write_of_an_earlier_run_still_rebuilds(cash_magics, mock_shell, tmp_path):
    """The control: the cell wrote into the list on an earlier run, then an
    edit that raised read only ``other``. The next run that writes the list
    again starts from it as the cells above leave it."""
    log = tmp_path / "runs.log"
    cells = [_MAKE.format(log=str(log)), _PRODUCE + "\nother = make(300)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    run_cash_cell(cash_magics, "rows[5] = ['w']", cells=[*cells, "rows[5] = ['w']"])
    _attempt(cash_magics, cells, "other[0] = other['a']")
    _attempt(cash_magics, cells, "rows[0] = [7]\nm = rows['a']")

    rows = mock_shell.user_ns["rows"]
    assert rows[5] == [5], "the write of an earlier run was kept"
    assert rows[0] == [7]
