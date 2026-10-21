"""A grid built by a helper that holds the user's axes keeps them.

The helper reaches the user's figure through pyplot (``plt.figure(1)``),
naming no variable, and keeps its axes on the grid. On every run, as in plain
Python, ``g.inset`` is ``ax0`` and holds ``rect``.
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
    "plt.close('all')\n"
    "df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': ['x', 'y', 'x']})\n"
    "fig0 = plt.figure(1)\nax0 = fig0.add_subplot()\nrect = ax0.add_patch(Rectangle((0, 0), 1, 1))\n"
    "def mk():\n    g = sns.FacetGrid(df, col='c')\n    g.inset = plt.figure(1).axes[0]\n    return g\n"
    "t0 = time.perf_counter()"
)
GRID = "time.sleep(1.0)\ng = mk()"
TOOK = "took = time.perf_counter() - t0"
CHANGE = "rect.set_width(5.0)\nax0.set_title('t')"
READ = "print('SAME', g.inset is ax0, g.inset.patches[0] is rect, g.inset.patches[0].get_width(), g.inset.get_title())"
PLAIN = "SAME True True 5.0 t"


def test_users_axes_in_a_grid_stay_the_users(nb_runner):
    nb_runner.create_notebook([SETUP, DATA, GRID, TOOK, CHANGE, READ])
    nb_runner.start_kernel()
    outs = []
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        outs.append((nb_runner.get_raw_output(6), nb_runner.peek("took")))
    for out, _ in outs:
        assert PLAIN in out, outs
