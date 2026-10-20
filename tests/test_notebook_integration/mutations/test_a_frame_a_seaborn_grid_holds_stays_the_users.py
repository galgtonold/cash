"""The DataFrame a seaborn grid holds stays the user's, also when the grid is served.

``PairGrid.data`` and ``FacetGrid.data`` are the user's frame itself. The
grid's figure is held back by its axes, so what the figure holds is left
out of the sharing check as a figure of its own is; what the grid holds
directly besides it must still be found. Served without it, ``g.data``
would be a copy of the frame: a change made in place to ``df`` later would
not reach the grid, and one made through ``g.data`` would not reach ``df``.

A grid built by a helper is the case only that check finds: the statement
names no frame, so ``df`` is found up the referrers of what the grid holds.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(300)]

pytest.importorskip("seaborn")

SETUP = (
    "import cash\n%cash_on\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns"
)
FRAME = (
    "df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': ['x', 'y', 'x']})\n"
    "def pair():\n    g = sns.PairGrid(df, vars=['a', 'b'])\n    g.map(plt.scatter)\n    return g\n"
    "def facet():\n    g = sns.FacetGrid(df, col='c')\n    g.map(plt.scatter, 'a', 'b')\n    return g\n"
    "t0 = time.perf_counter()"
)
GRIDS = {
    "pair": "g = sns.PairGrid(df, vars=['a', 'b'])\ng.map(plt.scatter)",
    "facet": "g = sns.FacetGrid(df, col='c')\ng.map(plt.scatter, 'a', 'b')",
    "pair_by_a_helper": "g = pair()",
    "facet_by_a_helper": "g = facet()",
}
TOOK = "took = time.perf_counter() - t0"
CHANGE = "df.loc[0, 'a'] = 100.0\ng.data.loc[1, 'b'] = 50.0"
READ = "print('SAME', g.data is df, g.data.loc[0, 'a'], df.loc[1, 'b'])"


def _check(nb_runner, served):
    assert "SAME True 100.0 50.0" in nb_runner.get_output(6), nb_runner.get_raw_output(6)
    took = float(nb_runner.peek("took"))
    assert (took < 0.9) if served else (took >= 1.0), took


@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_the_grid_and_the_frame_see_each_others_changes(nb_runner, grid):
    nb_runner.create_notebook([SETUP, FRAME, "time.sleep(1.0)\n" + GRIDS[grid], TOOK, CHANGE, READ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    _check(nb_runner, served=False)

    # The frame cell makes a new frame; the grid cell is served onto it.
    nb_runner.run_all()
    _check(nb_runner, served=True)

    nb_runner.restart()
    nb_runner.run_all()
    _check(nb_runner, served=True)
