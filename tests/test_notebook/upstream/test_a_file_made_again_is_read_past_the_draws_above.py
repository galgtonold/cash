"""A file open for reading that the upstream repair opens again is read past
what the statements above drew from it, as a run from the top reads it.

``f = open(path)``, then ``header = f.readline()``, then a cell reading
``rows = f.readlines()``. When the repair has to make ``f`` again (the kernel
no longer holds it, after a restart), re-running ``f = open(path)`` alone
handed the cell a file standing at its top: the header came back as a row.
A run from the top reads the header first. The same when the cell is re-run
on its own after ``next(f)`` drew the header: a text file refuses ``tell()``
while it is iterated, so where it stood could not be compared, and the cell
read the rest of a file it had already read to its end.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

DATA = "h1,h2\n1,2\n3,4\n5,6\n"
READ = "rows = f.readlines()"


@pytest.fixture
def data_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "d.csv"
    path.write_text(DATA, encoding="utf-8")
    return path


@pytest.mark.parametrize("skip", ["header = f.readline()", "header = next(f)"])
def test_a_file_the_kernel_lost_is_read_past_the_header(cash_magics, data_file, skip):
    cells = ["f = open('d.csv')", skip, READ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["rows"] == ["1,2\n", "3,4\n", "5,6\n"]

    ns.pop("f").close()
    run_cash_cell(cash_magics, READ, cells=cells)

    assert ns["rows"] == ["1,2\n", "3,4\n", "5,6\n"]


def test_a_bare_draw_the_trace_counts_as_a_change_is_run_again_too(cash_magics, data_file):
    """``f.readline()`` alone is a call on ``f`` the simulation already
    counts as changing it: re-run for the same reason, before and now."""
    cells = ["f = open('d.csv')", "f.readline()", READ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    ns.pop("f").close()
    run_cash_cell(cash_magics, READ, cells=cells)

    assert ns["rows"] == ["1,2\n", "3,4\n", "5,6\n"]


@pytest.mark.parametrize("skip", ["header = next(f)", "next(f)"])
def test_a_reader_re_run_after_next_reads_what_a_run_from_the_top_reads(cash_magics, data_file, skip):
    cells = ["f = open('d.csv')", skip, READ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    run_cash_cell(cash_magics, READ, cells=cells)
    assert ns["rows"] == ["1,2\n", "3,4\n", "5,6\n"]
    run_cash_cell(cash_magics, READ, cells=cells)
    assert ns["rows"] == ["1,2\n", "3,4\n", "5,6\n"]


def test_a_list_made_again_does_not_re_run_what_reads_it(cash_magics, data_file):
    """The draws are re-run only for a value that moves as it is read: a
    list made again stands as it was made, and what summed it is left."""
    cells = ["items = [1, 2, 3]", "total = sum(items)", "n = len(items)"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["total"] == 6

    ns.pop("items")
    ns["total"] = "kept"
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert ns["n"] == 3
    assert ns["total"] == "kept"
