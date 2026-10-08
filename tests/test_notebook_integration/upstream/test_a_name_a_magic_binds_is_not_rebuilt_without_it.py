"""A name a magic binds or changes is never rebuilt from the Python above it
without the magic.

The upstream check read a cell's Python with its magic lines deleted. A name
that ``%time df = clean(df)`` or ``p = !cmd`` rebound, and what the cell
computed from it, looked built by the code above the magic: on a plain first
Run All the cell below re-ran ``df = load()`` and ``k = len(df)`` and saw the
uncleaned ``df``. After an edit above ``model = M(k)\n%time model.fit()``,
the cell below got ``model = M(k)`` re-run without the fit: an untrained
model, a state neither run gives. Now the magic's names keep a lineage of
their own. A ``%time``, ``%timeit`` or ``%prun`` statement is Python, and a
rebuild runs it again with the Python above it, as a top-to-bottom run does.
Any other magic or shell command is never run for the user: when what it read
has changed, cash keeps the value and warns (NOTEBOOK-MAGIC-STALE).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

HELPERS = (
    "CALLS = []\n"
    "def load():\n"
    "    CALLS.append(1)\n"
    "    return [3, 1, None, 2]\n"
    "def clean(v):\n"
    "    return sorted(x for x in v if x is not None)"
)

MODEL = (
    "class M:\n"
    "    def __init__(self, k):\n"
    "        self.k = k; self.trained = False\n"
    "    def fit(self):\n"
    "        self.trained = True; return self\n"
    "    def predict(self):\n"
    "        return ('trained' if self.trained else 'UNTRAINED', self.k)"
)


def _lines(nb_runner, cell):
    return [line for line in nb_runner.get_output(cell).splitlines() if not line.startswith(("CPU times", "Wall"))]


def test_a_run_all_keeps_what_the_magic_bound(nb_runner):
    nb_runner.create_notebook(
        [
            HELPERS,
            "df = load()",
            "%time df = clean(df)\nk = len(df)",
            "print('A', df, k)",
            "d = 10\n%time d = d + 5\ne = d * 2",
            "print('B', d, e)",
            "p = load()\np = !echo a b\nq = len(p)",
            "print('C', p, q, len(CALLS))",
        ]
    )
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert _lines(nb_runner, 4) == ["A [1, 2, 3] 3"]
        assert _lines(nb_runner, 6) == ["B 15 30"]
        assert _lines(nb_runner, 8) == ["C ['a b'] 1 2"]
        assert "NOTEBOOK-MAGIC-STALE" not in nb_runner.get_raw_output(8)


def test_an_edit_above_runs_the_time_magic_again(nb_runner):
    nb_runner.create_notebook([MODEL, "k = 3", "model = M(k)\n%time model.fit()", "print('pred', model.predict())"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(2, "k = 4")
    nb_runner.run_cells([2, 4])
    out = nb_runner.get_raw_output(4)
    assert "pred ('trained', 4)" in out  # what Restart & Run All prints
    assert "NOTEBOOK-MAGIC-STALE" not in out

    nb_runner.run_all()
    assert "pred ('trained', 4)" in nb_runner.get_raw_output(4)
    assert "NOTEBOOK-MAGIC-STALE" not in nb_runner.get_raw_output(4)


def test_an_edit_above_runs_a_time_assignment_again(nb_runner):
    nb_runner.create_notebook(
        [
            "raw = [3, 1, None]",
            "df = list(raw)",
            "%time df = sorted(x for x in df if x is not None)\nk = len(df)",
            "print('A', df, k)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _lines(nb_runner, 4) == ["A [1, 3] 2"]

    nb_runner.set_cell_source(1, "raw = [5, 4, None, 1]")
    nb_runner.run_cells([1, 4])
    assert _lines(nb_runner, 4) == ["A [1, 4, 5] 3"]
    assert "NOTEBOOK-MAGIC-STALE" not in nb_runner.get_raw_output(4)


def test_an_edit_above_a_shell_command_keeps_the_value_and_warns(nb_runner):
    nb_runner.create_notebook(["k = 3", "res = {'k': k}", "res['out'] = !echo hi", "print('res', res)"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(1, "k = 4")
    nb_runner.run_cells([1, 4])
    out = nb_runner.get_raw_output(4)
    assert "res {'k': 3, 'out': ['hi']}" in out  # as the plain kernel keeps it
    assert "NOTEBOOK-MAGIC-STALE" in out
    assert "re-run cell 3" in out

    nb_runner.run_cells([2, 3, 4])
    out = nb_runner.get_raw_output(4)
    assert "res {'k': 4, 'out': ['hi']}" in out
    assert "NOTEBOOK-MAGIC-STALE" not in out


def test_after_a_restart_the_time_magic_runs_again(nb_runner):
    nb_runner.create_notebook([MODEL, "k = 3", "model = M(k)\n%time model.fit()", "print('pred', model.predict())"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()

    nb_runner.run_cell(4)
    out = nb_runner.get_raw_output(4)
    assert "pred ('trained', 3)" in out
    assert "NOTEBOOK-MAGIC-STALE" not in out


def test_after_a_restart_a_shell_command_is_not_rebuilt_without_it(nb_runner):
    nb_runner.create_notebook(["p = [0]", "p = !echo a b", "print('p', p)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()

    with pytest.raises(Exception):
        nb_runner.run_cell(3)
    out = nb_runner.get_raw_output(3)
    assert "p [0]" not in out
    assert "NOTEBOOK-MAGIC-STALE" in out
