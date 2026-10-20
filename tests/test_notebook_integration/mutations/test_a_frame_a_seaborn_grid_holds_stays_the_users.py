"""The DataFrame a seaborn grid holds stays the user's on every run.

``PairGrid.data`` and ``FacetGrid.data`` are the user's frame itself. Run
again in the same kernel and after a restart, as in plain Python, a change
made in place to ``df`` reaches the grid and one made through ``g.data``
reaches ``df``, whether the cell names ``df`` or a helper builds the grid.
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
)
GRIDS = {
    "pair": "g = sns.PairGrid(df, vars=['a', 'b'])\ng.map(plt.scatter)",
    "facet": "g = sns.FacetGrid(df, col='c')\ng.map(plt.scatter, 'a', 'b')",
    "pair_by_a_helper": "g = pair()",
    "facet_by_a_helper": "g = facet()",
}
CHANGE = "df.loc[0, 'a'] = 100.0\ng.data.loc[1, 'b'] = 50.0"
READ = "print('SAME', g.data is df, g.data.loc[0, 'a'], df.loc[1, 'b'])"


def _check(nb_runner):
    assert "SAME True 100.0 50.0" in nb_runner.get_output(5), nb_runner.get_raw_output(5)


@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_the_grid_and_the_frame_see_each_others_changes(nb_runner, grid):
    nb_runner.create_notebook([SETUP, FRAME, "time.sleep(1.0)\n" + GRIDS[grid], CHANGE, READ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    _check(nb_runner)

    nb_runner.run_all()
    _check(nb_runner)

    nb_runner.restart()
    nb_runner.run_all()
    _check(nb_runner)
