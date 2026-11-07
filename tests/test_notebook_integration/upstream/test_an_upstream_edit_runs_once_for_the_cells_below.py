"""An upstream edit saved but not run is run once; the cells below then reuse it.

Shaped on a real session (a log parsed in a loop into ``parsed``, sorted,
turned into a frame cell by cell, and cells below grouping that frame). The
user edits one of the parsing cells in the notebook file without running it,
then runs a cell below that reads the frame, edits that cell and runs it
again. The first run brings the frame up to date with the saved notebook, as
a restart and Run All would; the runs after it find the frame current and run
nothing above. Re-running the parse every time cost 20-45 s a cell there.
"""

import re

import pytest

pytestmark = [pytest.mark.timeout(240), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
WRITE_LOG = (
    "import random\n"
    "random.seed(3)\n"
    "with open('log.txt', 'w') as fh:\n"
    "    for i in range(2000):\n"
    "        acts = ' -> '.join('a%d (%02d:%02d)' % (random.randrange(5), j, random.randrange(60))"
    " for j in range(1 + random.randrange(6)))\n"
    "        fh.write('@u%d: %s\\n' % (random.randrange(40), acts))"
)
READ = "f = open('log.txt')\nlines = f.readlines()\nlen(lines)"
PARSE = (
    "parsed = []\n"
    "for line in lines:\n"
    "    name, rest = line.split(':', 1)\n"
    "    actions = []\n"
    "    for a in rest.split('->'):\n"
    "        act, t = a.strip().split(' ')\n"
    "        actions.append((act, int(t[1:3]) * 60 + int(t[4:6])))\n"
    "    parsed.append((name, actions))\n"
    "parsed = list(filter(lambda x: len(x[1]) > 0, parsed))"
)
SORT = "parsed = sorted(parsed, key=lambda x: x[1][0][1])"
FRAME = (
    "import pandas as pd\n\n"
    "session_num = dict()\n"
    "user_id_col = []\n"
    "session_num_col = []\n"
    "action_time_col = []\n"
    "action_name_col = []\n"
    "for (name, actions) in parsed:\n"
    "    curr_session = 0\n"
    "    if name in session_num:\n"
    "        curr_session = session_num[name] + 1\n"
    "    session_num[name] = curr_session\n"
    "    for (action_name, time) in actions:\n"
    "        user_id_col.append(name)\n"
    "        session_num_col.append(curr_session)\n"
    "        action_time_col.append(time)\n"
    "        action_name_col.append(action_name)\n"
    "df = pd.DataFrame({'user_id': user_id_col, 'session_num': session_num_col,\n"
    "                   'action_time': action_time_col, 'action_name': action_name_col})"
)
READER = (
    "df2 = df.groupby(['user_id', 'session_num'])['action_time'].apply(lambda s: s.max() - s.min())\n"
    "count = 0\n"
    "for _, group in df.groupby(['user_id', 'session_num']):\n"
    "    if group['action_name'].nunique() < len(group):\n"
    "        count += 1\n"
    "print('OUT', len(df), int(df2.sum()), count)"
)
READER_EDITED = READER.replace("count += 1", "count += 2")
CELLS = [SETUP, WRITE_LOG, READ, PARSE, SORT, FRAME, READER]
READER_CELL = 7

EDITS = {
    "parse": (4, "len(x[1]) > 0", "len(x[1]) > 1"),
    "sort": (5, "x[1][0][1]", "x[1][-1][1]"),
    "frame": (6, "session_num[name] + 1", "session_num[name] + 2"),
}


def _plain(cells, tmp_path):
    """What a restart and Run All prints for the reader, in plain Python."""
    import contextlib
    import io
    import os

    here = os.getcwd()
    out = io.StringIO()
    ns: dict = {}
    try:
        os.chdir(tmp_path)
        with contextlib.redirect_stdout(out):
            for src in cells[1:]:
                exec(compile(src, "<cell>", "exec"), ns)
        ns["f"].close()
    finally:
        os.chdir(here)
    return re.search(r"^OUT .*$", out.getvalue(), re.M).group(0)


def _out(output):
    found = re.search(r"^OUT .*$", output, re.M)
    return found.group(0) if found else output


def _upstream_runs(output):
    """The statements above that the badge says ran again for this cell."""
    return [line for line in output.splitlines() if re.match(r"\s+\^(EXECUTED|NOT CACHED)", line)]


@pytest.mark.parametrize("edit", list(EDITS))
def test_a_saved_upstream_edit_runs_once_then_the_cell_below_reuses_it(nb_runner, tmp_path, edit):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    cell, old, new = EDITS[edit]
    saved = list(CELLS)
    saved[cell - 1] = CELLS[cell - 1].replace(old, new)
    nb_runner.set_cell_source(cell, saved[cell - 1])

    nb_runner.run_cell(READER_CELL)
    first = nb_runner.get_output(READER_CELL)
    assert _out(first) == _plain(saved, tmp_path), first
    assert _upstream_runs(first), first

    saved[READER_CELL - 1] = READER_EDITED
    nb_runner.set_cell_source(READER_CELL, READER_EDITED)
    expected = _plain(saved, tmp_path)
    for _ in range(2):
        nb_runner.run_cell(READER_CELL)
        output = nb_runner.get_output(READER_CELL)
        assert _out(output) == expected, output
        assert not _upstream_runs(output), output
