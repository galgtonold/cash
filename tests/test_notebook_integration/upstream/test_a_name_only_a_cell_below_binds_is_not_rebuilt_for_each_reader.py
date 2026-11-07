"""A cell that splits what a cell saved below it read is not run again for every reader.

Found by replaying a real session (JuNE student_2) with the notebook known to
cash. After the notebook was saved, the cell reading the log
(``lines = f.readlines()``) sat below the cell that splits it
(``lines2 = []``, a loop appending the parts, ``lines = lines2``); the user
ran the reading cell first, then the cells in order. Nothing above the split
binds ``lines``, so the check took the kernel's ``lines`` for the version the
loop read; but that is the one the cell itself made from it. The loop looked
stale, ran again before every cell below, and the run moved ``lines`` on
again: ``len(deltas)`` re-ran the split, the parse and a 224 s loop each
time.

The check now takes the version the statement read when it last ran. The
cells below keep what memory holds, as plain Jupyter does.
"""

import re

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
WRITE_LOG = (
    "with open('log.txt', 'w') as fh:\n"
    "    for i in range(400):\n"
    "        fh.write('u%d:%d@u%d:%d\\n' % (i % 7, i, i % 5, 3 * i))"
)
SPLIT = (
    "def split_lines(line):\n"
    "    ind = line[1:].find('@')\n"
    "    if ind == -1:\n"
    "        return [line]\n"
    "    ind += 1\n"
    "    return [line[:ind]] + split_lines(line[ind:])\n\n"
    "lines2 = []\n"
    "for line in lines:\n"
    "    for l in split_lines(line):\n"
    "        lines2.append(l)\n"
    "lines = lines2\n"
    "len(lines)"
)
PARSE = "parsed = [(l.split(':')[0], int(l.split(':')[1])) for l in lines]\nparsed = sorted(parsed, key=lambda x: x[1])"
DELTAS = (
    "import time\n"
    "deltas = []\n"
    "for (name, t), (_, u) in zip(parsed, parsed[1:]):\n"
    "    deltas.append(u - t)\n"
    "time.sleep(0.05)"
)
LEN = "print('LEN', len(deltas), sum(deltas))"
READ = "f = open('log.txt')\nlines = f.readlines()\nlen(lines)"

CELLS = [SETUP, WRITE_LOG, SPLIT, PARSE, DELTAS, LEN, READ]
READ_CELL, LEN_CELL = 7, 6


def _plain(order, tmp_path):
    """What plain Python prints for LEN, running the cells in *order*."""
    import contextlib
    import io
    import os

    here = os.getcwd()
    out = io.StringIO()
    ns: dict = {}
    try:
        os.chdir(tmp_path)
        with contextlib.redirect_stdout(out):
            for cell in order:
                exec(compile(CELLS[cell - 1], "<cell>", "exec"), ns)
        ns["f"].close()
    finally:
        os.chdir(here)
    return re.findall(r"^LEN .*$", out.getvalue(), re.M)[-1]


def _upstream_runs(output):
    return [line for line in output.splitlines() if re.match(r"\s+\^(EXECUTED|NOT CACHED|CACHED|RESTORED)", line)]


def test_the_cells_below_a_split_of_a_name_read_below_run_nothing_above(nb_runner, tmp_path):
    order = [1, 2, READ_CELL, 3, 4, 5, LEN_CELL]
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_cells(order)
    expected = _plain(order[1:], tmp_path)

    for _ in range(2):
        nb_runner.run_cell(LEN_CELL)
        output = nb_runner.get_output(LEN_CELL)
        assert re.search(r"^LEN .*$", output, re.M).group(0) == expected, output
        assert not _upstream_runs(output), output
    assert nb_runner.peek("len(lines)") == "800"
