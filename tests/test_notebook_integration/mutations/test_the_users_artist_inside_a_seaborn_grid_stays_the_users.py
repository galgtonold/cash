"""The user's own artist inside a seaborn grid's figure stays the user's.

A helper adds the user's ``rect`` to the grid's axes and puts ``meta`` on its
figure. On every run, as in plain Python, the grid's patch is ``rect`` and its
``meta`` is ``meta``, so changes to them show through the grid. Making the
grid opens a pyplot figure, so the statement runs every time.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(300)]

pytest.importorskip("seaborn")

SETUP = (
    "import cash\n%cash_on\nimport time\nimport matplotlib\nmatplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\n"
    "from matplotlib.patches import Rectangle"
)
DATA = (
    "df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': ['x', 'y', 'x']})\n"
    "rect = Rectangle((0, 0), 1, 1)\n"
    "meta = {'run': 1}\ncfg = {'patch': rect, 'meta': meta}\n"
    "def facet():\n    g = sns.FacetGrid(df, col='c')\n    g.map(plt.scatter, 'a', 'b')\n"
    "    g.axes[0, 0].add_patch(cfg['patch'])\n    g.figure.meta = cfg['meta']\n    return g\n"
    "t0 = time.perf_counter()"
)
GRID = "time.sleep(1.0)\ng = facet()"
TOOK = "took = time.perf_counter() - t0"
CHANGE = "rect.set_width(5.0)\nmeta['run'] = 2"
READ = (
    "p = g.axes[0, 0].patches[0]\nprint('SAME', p is rect, p.get_width(), g.figure.meta is meta, g.figure.meta['run'])"
)
PLAIN = "SAME True 5.0 True 2"


def test_users_patch_in_a_grid_stays_the_users(nb_runner):
    nb_runner.create_notebook([SETUP, DATA, GRID, TOOK, CHANGE, READ])
    nb_runner.start_kernel()
    outs = []
    nb_runner.run_all()
    outs.append((nb_runner.get_raw_output(6), nb_runner.peek("took")))
    nb_runner.run_all()
    outs.append((nb_runner.get_raw_output(6), nb_runner.peek("took")))
    nb_runner.restart()
    nb_runner.run_all()
    outs.append((nb_runner.get_raw_output(6), nb_runner.peek("took")))
    for out, _ in outs:
        assert PLAIN in out, outs
