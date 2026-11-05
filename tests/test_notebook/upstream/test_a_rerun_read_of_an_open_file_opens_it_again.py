"""A statement re-run above the cell that reads an open file, a generator
or an iterator gets it made again first, as a run from the top has it.

Found by replaying a real session (JuNE student_2): ``f = open(path)`` and
``lines = f.readlines()`` in one cell, then a cell copying ``lines`` into
``lines2`` and ending ``lines = lines2``. The rule that rebuilds a name a
cell reads and rebinds re-ran ``lines = f.readlines()`` alone, on the handle
the first run had already read to its end: ``lines`` came back empty, in a
plain Run All too, and every cell below ran on no lines without a word. A run
from the top opens the file again before reading it.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

OPEN = 'data_path = "log.txt"\n\nf = open(data_path, "r")\nlines = f.readlines()\nlen(lines)'
COPY = "lines2 = []\nfor line in lines:\n    lines2.append(line)\n\nlines = lines2\nlen(lines)"
COPY_EDITED = "lines2 = []\nfor line in lines:\n    assert line.count('@') <= 2\n    lines2.append(line)\n\nlines = lines2\nlen(lines)"
COUNT = "n = len(lines)"


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "log.txt").write_text("".join(f"@User{i}: Action_1 -> Action_2\n" for i in range(50)), encoding="utf-8")


def test_a_run_all_reads_every_line(cash_magics, log_file):
    cells = [OPEN, COPY, COUNT]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["n"] == 50


def test_an_edited_cell_reading_the_lines_reads_every_line(cash_magics, log_file):
    ns = cash_magics.shell.user_ns
    cells = [OPEN, COPY, COUNT]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells = [OPEN, COPY_EDITED, COUNT]
    run_cash_cell(cash_magics, COPY_EDITED, cells=cells)
    assert len(ns["lines"]) == 50
    run_cash_cell(cash_magics, COUNT, cells=cells)
    assert ns["n"] == 50


def test_a_header_read_between_is_read_again_too(cash_magics, log_file):
    """A statement between the open and the re-run read that draws from the
    file too runs again with them, or the re-run read starts at the top."""
    cells = [
        'f = open("log.txt")\nheader = f.readline()',
        "lines = f.readlines()",
        "lines2 = list(lines)\nlines = lines2",
        "n = len(lines)",
    ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["n"], ns["header"]) == (49, "@User0: Action_1 -> Action_2\n")


def test_a_file_no_cell_opens_is_not_read_again(cash_magics, log_file):
    """No statement of the notebook opens ``f``: cash cannot hand the read
    the file as a run from the top does, so it re-runs nothing and the cell
    runs as written, on the lines in memory (the driver raises what the
    kernel would run the cell past)."""
    from cash.notebook.upstream.replay import UpstreamRepairGap

    ns = cash_magics.shell.user_ns
    ns["f"] = open("log.txt", encoding="utf-8")  # read by the cells below
    cells = ["lines = f.readlines()", "lines2 = list(lines)\nlines = lines2", "n = len(lines)"]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    with pytest.raises(UpstreamRepairGap, match="already read"):
        run_cash_cell(cash_magics, cells[1], cells=cells)
    assert len(ns["lines"]) == 50
    ns["f"].close()
