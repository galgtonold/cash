"""A cell below a cell that raised part way runs on what that cell left.

``tot = {}`` then a loop filling ``tot`` raises half way (a helper it calls
was changed). Running a cell further down, cash re-ran ``tot = {}`` and the
cell between (``final = {... tot ...}``) on that state, silently: tot came
back empty and final with it. A plain kernel keeps the partial tot and the
previous final; after a restart final is not there at all.
"""

import os

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(180)]

PROC = """\
def weight(s):
    if s > {limit}:
        raise ValueError(s)
    return s ** 0.5
"""

CELLS = [
    "import cash\n%cash_on",
    "import proc\nby_user = {'a': [1, 4], 'b': [9, 16], 'c': [25]}",
    "tot = {}\nfor u in by_user:\n    tot[u] = sum(proc.weight(s) for s in by_user[u])\nprint(tot)",
    "final = {k: v for k, v in tot.items() if v > 2}\nprint(len(final))",
    "print('final' in dir(), len(final) if 'final' in dir() else None, tot)",
]


def _write(nb_runner, limit, stamp):
    path = nb_runner.work_dir / "proc.py"
    path.write_text(PROC.format(limit=limit), encoding="utf-8")
    os.utime(path, (stamp, stamp))


def _run_until_error(nb_runner):
    for n in range(1, len(CELLS) + 1):
        try:
            nb_runner.run_cell(n)
        except CellExecutionError:
            return n
    return None


def test_a_cell_below_a_failed_loop_sees_what_the_loop_left(nb_runner):
    _write(nb_runner, 99, 1_700_000_000)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    assert _run_until_error(nb_runner) is None

    _write(nb_runner, 15, 1_700_000_100)
    assert _run_until_error(nb_runner) == 3
    nb_runner.run_cell(5)
    out = nb_runner.get_output(5)
    assert out.splitlines()[0] == "True 3 {'a': 3.0}", out
    assert "NOTEBOOK-FAILED-CELL" in nb_runner.get_raw_output(5)

    nb_runner.restart()
    assert _run_until_error(nb_runner) == 3
    assert nb_runner.peek("__import__('sys').modules.get('cash.notebook') is not None") == "True"
    nb_runner.run_cell(5)
    assert nb_runner.get_output(5).splitlines()[0] == "False None {'a': 3.0}", nb_runner.get_output(5)
