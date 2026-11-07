"""An upstream re-run interrupted part way leaves the names as they were.

Found by replaying a real session (JuNE student_2): the kernel held the log's
lines unsplit (``len(lines)`` the raw ``readlines`` count) from some point
until the restart, under a cell that splits them (``lines2 = []``, a loop
appending the parts, ``lines = lines2``). The cell had been edited and saved;
a cell below re-ran it upstream, the user interrupted the slow loop, and:

- ``lines = f.readlines()`` had re-run, the loop had not: ``lines`` unsplit;
- the re-run ``lines2 = []`` matched the edited cell, and a name a loop fills
  is trusted once the statement binding it matches: the half-filled
  ``lines2`` passed for the loop's;
- the check had already taken in the edit, so the next one saw no change
  above and trusted the loop's names as they were.

The next cell re-ran nothing and worked on the unsplit lines. An interrupted
repair now puts back what it rebound, with its lineage, and what the check
saw of the cells: the kernel holds what plain Jupyter holds after an
interrupted cell, and the next cell repairs it again, as a run from the top
gives it.
"""

import asyncio
import re
import threading

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(180), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
WRITE_LOG = (
    "with open('log.txt', 'w') as fh:\n"
    "    for i in range(400):\n"
    "        fh.write('u%d:%d@u%d:%d\\n' % (i % 7, i, i % 5, 3 * i))"
)
READ = "f = open('log.txt')\nlines = f.readlines()\nlen(lines)"
SPLIT = (
    "import time\n"
    "def split_lines(line):\n"
    "    time.sleep(0.004)\n"
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
PARSE = "parsed = [(l.split(':')[0], int(l.split(':')[1])) for l in lines]\nprint('PARSED', len(parsed))"
DELTAS = "deltas = []\nfor (name, t), (_, u) in zip(sorted(parsed, key=lambda x: x[1]), parsed[1:]):\n    deltas.append(u - t)"
LEN = "print('LEN', len(deltas))"
CELLS = [SETUP, WRITE_LOG, READ, SPLIT, PARSE, DELTAS, LEN]
SPLIT_CELL, PARSE_CELL, LEN_CELL = 4, 5, 7
EDITED = SPLIT.replace("lines2 = []", "lines2 = list()")


def _upstream_runs(output):
    return [line for line in output.splitlines() if re.match(r"\s+\^(EXECUTED|NOT CACHED|CACHED|RESTORED)", line)]


def _interrupted(nb_runner, cell, after):
    """Run *cell*, interrupting the kernel *after* seconds; its output."""

    def interrupt():  # Kernel > Interrupt
        res = nb_runner.client.km.interrupt_kernel()
        if asyncio.iscoroutine(res):
            asyncio.run_coroutine_threadsafe(res, nb_runner._loop)

    timer = threading.Timer(after, interrupt)
    timer.start()
    try:
        nb_runner.run_cell(cell)
        return nb_runner.get_output(cell)
    except CellExecutionError as e:
        return str(e)
    finally:
        timer.cancel()


@pytest.mark.parametrize("reader", [LEN_CELL, PARSE_CELL], ids=["a_reader_of_deltas", "the_parse_cell"])
def test_an_interrupted_rerun_of_an_edited_split_leaves_the_lines_split(nb_runner, reader):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("len(lines)") == "800"

    nb_runner.set_cell_source(SPLIT_CELL, EDITED)
    output = _interrupted(nb_runner, reader, after=1.5)
    assert "KeyboardInterrupt" in output, output
    # What plain Jupyter holds after an interrupted cell: what it held before.
    assert nb_runner.peek("(len(lines), len(lines2))") == "(800, 800)"

    nb_runner.run_cell(reader)
    output = nb_runner.get_output(reader)
    assert _upstream_runs(output), output  # the edited cell, re-run this time
    assert nb_runner.peek("(len(lines), len(parsed), len(deltas))") == "(800, 800, 799)"
    nb_runner.run_cell(PARSE_CELL)
    assert "PARSED 800" in nb_runner.get_output(PARSE_CELL)

    nb_runner.run_cell(reader)
    assert not _upstream_runs(nb_runner.get_output(reader))
