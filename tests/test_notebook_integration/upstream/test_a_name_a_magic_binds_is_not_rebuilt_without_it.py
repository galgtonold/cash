"""A name a magic binds or changes is never rebuilt from the Python above it.

The upstream check read a cell's Python with its magic lines deleted. A name
that ``%time df = clean(df)`` or ``p = !cmd`` rebound, and what the cell
computed from it, looked built by the code above the magic: on a plain first
Run All the cell below re-ran ``df = load()`` and ``k = len(df)`` and saw the
uncleaned ``df``. After an edit above ``model = M(k)\\n%time model.fit()``,
the cell below got ``model = M(k)`` re-run without the fit: an untrained
model, a state neither run gives. Now the magic's names keep a lineage of
their own; when what the magic read has changed, cash keeps the value and
warns (NOTEBOOK-MAGIC-STALE), as it does not re-run a magic.
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


def test_an_edit_above_keeps_the_value_and_warns(nb_runner):
    nb_runner.create_notebook([MODEL, "k = 3", "model = M(k)\n%time model.fit()", "print('pred', model.predict())"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(2, "k = 4")
    nb_runner.run_cells([2, 4])
    out = nb_runner.get_raw_output(4)
    assert "pred ('trained', 3)" in out  # not an untrained model built from k = 4
    assert "NOTEBOOK-MAGIC-STALE" in out
    assert "re-run cell 3" in out

    nb_runner.run_cells([3, 4])
    out = nb_runner.get_raw_output(4)
    assert "pred ('trained', 4)" in out
    assert "NOTEBOOK-MAGIC-STALE" not in out


def test_after_a_restart_it_is_not_rebuilt_without_the_magic(nb_runner):
    nb_runner.create_notebook([MODEL, "k = 3", "model = M(k)\n%time model.fit()", "print('pred', model.predict())"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()

    with pytest.raises(Exception):
        nb_runner.run_cell(4)
    out = nb_runner.get_raw_output(4)
    assert "UNTRAINED" not in out
    assert "NOTEBOOK-MAGIC-STALE" in out
