"""A cell below the frame does not rebuild it because a later cell is unrun.

Found by replaying a real session (JuNE student_2) with the notebook known to
cash. Its cells read a log (``f = open(...)``, ``lines = f.readlines()``),
split the lines and rebind ``lines`` to the split ones, parse them and build
a frame in a loop; a cell further down reads the log again as it was. After a
restart the user ran every cell but that second read, then a loop filling
``deltas`` from the frame, then ``len(deltas)``, a histogram and
``np.quantile(deltas, 0.95)``. Before each of those cells cash re-ran the
``deltas = []`` init and the loop that fills it (150 s a cell there): the
kernel's ``lines`` is the split one and the cells above end with the read
one, so everything ever built from ``lines`` was taken as stale, though it was
built from the very version a run from the top gives it, and re-running it
never made the two agree.

A value built on the version of a name a run from the top gives it now stays.
Edits above still re-run what a run from the top would.
"""

import os
import re

import pytest

pytestmark = [pytest.mark.timeout(240), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
WRITE_LOG = (
    "with open('log.txt', 'w', encoding='utf-8') as fh:\n"
    "    for i in range(300):\n"
    "        acts = '|'.join('a%d=%d' % (j % 3, 1000 * i + 37 * j) for j in range(1 + i % 5))\n"
    "        fh.write('u%d:%s@u%d:a1=%d\\n' % (i % 7, acts, i % 5, 7 * i))"
)
READ = "f = open('log.txt', encoding='utf-8')\nlines = f.readlines()\nlen(lines)"
SPLIT = (
    "lines2 = []\n"
    "for line in lines:\n"
    "    for part in line.strip().split('@'):\n"
    "        lines2.append(part)\n"
    "lines = lines2\n"
    "len(lines)"
)
PARSE = (
    "parsed = []\n"
    "for line in lines:\n"
    "    name, rest = line.split(':')\n"
    "    parsed.append((name, [(a.split('=')[0], int(a.split('=')[1])) for a in rest.split('|')]))"
)
FRAME = (
    "import pandas as pd\n\n"
    "session_num = dict()\n"
    "user_id_col = []\n"
    "session_num_col = []\n"
    "action_time_col = []\n"
    "action_name_col = []\n"
    "for (name, actions) in parsed:\n"
    "    curr_session = session_num[name] + 1 if name in session_num else 0\n"
    "    session_num[name] = curr_session\n"
    "    for (action_name, time) in actions:\n"
    "        user_id_col.append(name)\n"
    "        session_num_col.append(curr_session)\n"
    "        action_time_col.append(time)\n"
    "        action_name_col.append(action_name)\n"
    "df = pd.DataFrame({'user_id': user_id_col, 'session_num': session_num_col,\n"
    "                   'action_time': action_time_col, 'action_name': action_name_col})"
)
READ_AGAIN = "# the log as it was\n" + READ
DELTAS = (
    "deltas = []\n\n"
    "df2 = df.sort_values('action_time')\n\n"
    "for _, group in df2.groupby(['user_id', 'session_num', 'action_name']):\n"
    "    for delta in group['action_time'].diff().to_numpy()[1:]:\n"
    "        deltas.append(delta)"
)
LEN = "print('LEN', len(deltas))"
QUANTILE = "import numpy as np\n\nprint('Q', np.quantile(deltas, 0.95))"

CELLS = [SETUP, WRITE_LOG, READ, SPLIT, PARSE, FRAME, READ_AGAIN, DELTAS, LEN, QUANTILE]
DELTAS_CELL, LEN_CELL, QUANTILE_CELL = 8, 9, 10


def _run_all_plain(cells, tmp_path):
    """What a restart and Run All prints for LEN and Q, in plain Python."""
    here = os.getcwd()
    out = {}
    ns = {"print": lambda *a: out.__setitem__(a[0], a[1])}
    try:
        os.chdir(tmp_path)
        for src in cells[1:]:
            exec(compile(src, "<cell>", "exec"), ns)
        ns["f"].close()
    finally:
        os.chdir(here)
    return f"LEN {out['LEN']}", f"Q {out['Q']}"


def _value(output, label):
    found = re.search(rf"^{label} \S+", output, re.M)
    return found.group(0) if found else output


def _upstream(output):
    """The upstream statements the badge lists for a cell."""
    block = output.split("Upstream:", 1)
    return block[1] if len(block) == 2 else ""


def _after_a_restart_without_the_read_again(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_cells(list(range(2, len(CELLS) + 1)))
    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3, 4, 5, 6, DELTAS_CELL])


def test_the_cells_below_the_loop_do_not_run_it_again(nb_runner, tmp_path):
    _after_a_restart_without_the_read_again(nb_runner)
    expected = _run_all_plain(CELLS, tmp_path)

    for _ in range(2):
        nb_runner.run_cells([LEN_CELL, QUANTILE_CELL])
        for cell, label, value in ((LEN_CELL, "LEN", expected[0]), (QUANTILE_CELL, "Q", expected[1])):
            output = nb_runner.get_output(cell)
            assert _value(output, label) == value, output
            assert "deltas = []" not in _upstream(output), output


def test_an_edited_init_still_runs_again_with_its_loop(nb_runner, tmp_path):
    _after_a_restart_without_the_read_again(nb_runner)
    edited = DELTAS.replace("deltas = []", "deltas = [0.0]")
    nb_runner.set_cell_source(DELTAS_CELL, edited)
    expected = _run_all_plain(CELLS[: DELTAS_CELL - 1] + [edited] + CELLS[DELTAS_CELL:], tmp_path)

    nb_runner.run_cells([LEN_CELL, QUANTILE_CELL])

    assert _value(nb_runner.get_output(LEN_CELL), "LEN") == expected[0], nb_runner.get_output(LEN_CELL)
    assert _value(nb_runner.get_output(QUANTILE_CELL), "Q") == expected[1], nb_runner.get_output(QUANTILE_CELL)


def test_an_edited_loop_still_runs_again(nb_runner, tmp_path):
    _after_a_restart_without_the_read_again(nb_runner)
    edited = DELTAS.replace("deltas.append(delta)", "deltas.append(delta * 2)")
    nb_runner.set_cell_source(DELTAS_CELL, edited)
    expected = _run_all_plain(CELLS[: DELTAS_CELL - 1] + [edited] + CELLS[DELTAS_CELL:], tmp_path)

    nb_runner.run_cells([QUANTILE_CELL, LEN_CELL])

    assert _value(nb_runner.get_output(QUANTILE_CELL), "Q") == expected[1], nb_runner.get_output(QUANTILE_CELL)
    assert _value(nb_runner.get_output(LEN_CELL), "LEN") == expected[0], nb_runner.get_output(LEN_CELL)


def test_an_edited_split_above_the_frame_still_rebuilds_it(nb_runner, tmp_path):
    _after_a_restart_without_the_read_again(nb_runner)
    edited = SPLIT.replace("for part in line.strip().split('@'):", "for part in line.strip().split('@')[:1]:")
    nb_runner.set_cell_source(4, edited)
    expected = _run_all_plain(CELLS[:3] + [edited] + CELLS[4:], tmp_path)

    nb_runner.run_cells([LEN_CELL, QUANTILE_CELL])

    assert _value(nb_runner.get_output(LEN_CELL), "LEN") == expected[0], nb_runner.get_output(LEN_CELL)
    assert _value(nb_runner.get_output(QUANTILE_CELL), "Q") == expected[1], nb_runner.get_output(QUANTILE_CELL)
