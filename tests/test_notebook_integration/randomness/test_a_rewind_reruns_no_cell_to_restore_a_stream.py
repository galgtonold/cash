"""Putting the random stream back before a draw re-runs no user code.

Run after a restart, a cell below seeds and draws gets the values plain
restart + Run All gives at that cell. Restoring the stream must not re-run a
whole cell above (it would rebind what a later cell set), must keep each seed
at its own place in the run order (three ``np.random.seed(1)`` cells are three
seeds, not one), and must never run a statement again for its draw when the
statement also changes the user's variables (``df = df.reindex(...)``).
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

pytestmark = [pytest.mark.libraries, pytest.mark.fresh_kernel]

ON = "import cash\n%cash_on"


def _printed(text: str) -> list[str]:
    return re.findall(r"^R .*$", text, re.M)


def _plain(cells: list[str]) -> list[str]:
    """What plain Python prints for *cells* run top to bottom."""
    out: list[str] = []
    ns: dict = {"print": lambda *a, **k: out.append(" ".join(str(x) for x in a))}
    for src in cells:
        exec(compile(src, "<plain>", "exec"), ns)
    return [line for line in out if line.startswith("R ")]


def _run(nb_runner, cells: list[str], last: int) -> list[str]:
    """Run All, restart, then run only the ON cell and cell *last* (1-based)."""
    nb_runner.create_notebook([ON, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner.run_cell(1)
    nb_runner.run_cell(last)
    return _printed(nb_runner.get_output(last))


# Three cells that open with the same seed statement, each then drawing.
SAME_SEEDS = [
    "import time\nimport numpy as np",
    f"np.random.seed(1)\na = (time.sleep({F}), np.random.rand(3))[1]",
    "np.random.seed(1)\nb = np.random.rand(2)",
    "np.random.seed(1)\nc = np.random.rand(4) + np.random.rand(4)",
    "print('R', repr(float(a.sum())), repr(float(b.sum())), repr(float(c.sum())))",
]


@pytest.mark.timeout(180)
def test_each_seed_counts_at_its_own_place(nb_runner):
    want = _plain(SAME_SEEDS)
    assert _run(nb_runner, SAME_SEEDS, len(SAME_SEEDS) + 1) == want


# A seed-and-draw cell whose `model` a later, slow cell replaces.
REBOUND = [
    "import time\nimport numpy as np",
    "np.random.seed(1)\nw = np.random.rand(3)\nmodel = 'first'",
    f"model = (time.sleep({F}), 'second')[1]",
    "np.random.seed(2)\nw = np.random.rand(5)",
    "print('R', model, repr(float(w.sum())), repr(float(np.random.rand())))",
]


@pytest.mark.timeout(180)
def test_no_cell_above_is_run_again_over_a_later_binding(nb_runner):
    want = _plain(REBOUND)
    assert _run(nb_runner, REBOUND, len(REBOUND) + 1) == want


# The shape of mlbook ch08: a seed, a branch that shuffles a frame (the branch
# taken draws from its own generator, the other from the global stream), and
# a slow grid-search fit that draws from the global stream inside a method;
# below them a draw whose input is edited.
SHUFFLE = [
    "import time\nimport numpy as np\nimport pandas as pd\nruns = []\n"
    "class GS:\n"
    "    def fit(self):\n"
    "        runs.append(1)\n"
    f"        time.sleep({F})\n"
    "        self.best_ = np.random.rand(2)\n"
    "        return self\n"
    "gs = GS()",
    "np.random.seed(0)\ndf = pd.DataFrame({'v': range(8)})\n"
    "if len(df) > 0:\n"
    "    df = df.sample(frac=1, random_state=0)\n"
    "else:\n"
    "    df = df.reindex(np.random.permutation(df.index))",
    "gs.fit()",
    "k = 3",
    "z = np.random.rand(k)",
    "print('R', list(df['v']), repr(float(gs.best_.sum())), repr(float(z.sum())), len(runs))",
]


@pytest.mark.timeout(180)
def test_a_statement_that_changes_user_state_is_not_run_again(nb_runner):
    """The draw re-run for the edit starts where the fit left the stream;
    neither the shuffle nor the fit runs again (plain ran each once)."""
    nb_runner.create_notebook([ON, *SHUFFLE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, "k = 4")
    nb_runner.run_cell(5)
    nb_runner.run_cell(7)
    want = _plain([*SHUFFLE[:3], "k = 4", *SHUFFLE[4:]])
    assert _printed(nb_runner.get_output(7)) == want


# Cheap draws one after another below a seed, nothing stored.
CHAIN = [
    "import numpy as np",
    "np.random.seed(0)\nn = 2",
    "a = np.random.rand(n)",
    "b = np.random.rand(3)",
    "print('R', repr(float(b.sum())), repr(float(np.random.rand())))",
]


@pytest.mark.timeout(180)
def test_a_draw_above_that_only_binds_its_name_runs_again_in_place(nb_runner):
    want = _plain(CHAIN)
    assert _run(nb_runner, CHAIN, len(CHAIN) + 1) == want


# The shape of E11: loops above that draw and bind their loop names, rebound
# by a slow loop below; an edited draw under them.
LOOPS = [
    "import time\nimport numpy as np",
    "np.random.seed(1)\nfor i in range(3):\n    example = np.random.rand()",
    f"for i in range(5):\n    example = (time.sleep({F / 5}), i * 10)[1]",
    "k = 3",
    "w = np.random.rand(k)",
    "print('R', i, example, repr(float(w.sum())), repr(float(np.random.rand())))",
]


@pytest.mark.timeout(180)
def test_loop_names_a_later_loop_bound_keep_their_values(nb_runner):
    nb_runner.create_notebook([ON, *LOOPS])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, "k = 4")
    nb_runner.run_cell(len(LOOPS) + 1)
    want = _plain([*LOOPS[:3], "k = 4", *LOOPS[4:]])
    assert _printed(nb_runner.get_output(len(LOOPS) + 1)) == want
