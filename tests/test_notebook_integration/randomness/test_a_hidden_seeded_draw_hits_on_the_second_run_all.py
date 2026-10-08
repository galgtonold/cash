"""A seeded draw hidden in a helper is restored on the second Run All.

``X = make_data(1000)`` under ``np.random.seed(42)`` spells no draw, so its
first key does not read numpy's RNG variable. The run sees it draw; the value
is stored under the key the next run builds, which reads that variable, so a
second Run All in the same kernel restores it instead of drawing again (it
used to run once more, and only the third Run All hit).

After a restart with a new seed, the value stored under the old seed is not
served: the key it is stored under reads the seed.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(300)]

np = pytest.importorskip("numpy")

CELLS = [
    "import time\nimport numpy as np\nnp.random.seed(42)",
    "def make_data(n):\n    time.sleep(0.4)  # slow enough to store\n    return np.random.normal(size=n)",
    "X = make_data(1000)",
    "print('SUM %.12f' % X.sum())",
]

_STATUSES = (
    "[str(s.get('status')) for s in "
    "get_ipython().magics_manager.magics['line']['cash_status'].__self__"
    "._last_cell_metrics['statements']]"
)


def _sum(seed: int) -> str:
    np.random.seed(seed)
    return "SUM %.12f" % np.random.normal(size=1000).sum()


def test_the_second_run_all_restores_it_and_a_new_seed_is_not_served_it(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _sum(42) in nb_runner.get_output(4)

    for cell in (1, 2, 3):
        nb_runner.run_cell(cell)
    assert ast.literal_eval(nb_runner.peek(_STATUSES)) == ["RESTORED"], "the draw ran again on the second Run All"
    nb_runner.run_cell(4)
    assert _sum(42) in nb_runner.get_output(4)

    nb_runner.set_cell_source(1, CELLS[0].replace("seed(42)", "seed(7)"))
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _sum(7) in nb_runner.get_output(4)
