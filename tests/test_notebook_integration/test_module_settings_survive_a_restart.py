"""A setting made on a module above is in force when a cell below runs after a restart.

Round 24's r24s2 styled every chart with ``plt.rcParams.update({...})`` in a
setup cell. After a kernel restart, running the chart cell first rebuilt what
it needed -- the data, the imports -- and drew all eight pages in matplotlib's
default style: the statement setting the style produces no variable, so
rebuilding variables never reached it. No warning; the data was right and the
pictures were not.

The same holds for any setting a module keeps: ``pd.set_option``,
``plt.style.use``, ``np.seterr``, ``warnings.filterwarnings``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(180)]

DATA = "data = np.sort(np.random.default_rng(0).normal(size=3_000_000))\nprint('N', len(data))"


@pytest.mark.parametrize(
    "setting, probe, expected",
    [
        (
            "plt.rcParams.update({'axes.grid': True, 'axes.titleweight': 'bold'})",
            "(plt.rcParams['axes.grid'], plt.rcParams['axes.titleweight'])",
            "(True, 'bold')",
        ),
        ("plt.rcParams['axes.grid'] = True", "plt.rcParams['axes.grid']", "True"),
        ("plt.style.use('ggplot')", "plt.rcParams['axes.facecolor']", "'#E5E5E5'"),
        ("pd.set_option('display.max_columns', 3)", "pd.get_option('display.max_columns')", "3"),
    ],
    ids=["rcparams_update", "rcparams_store", "style_use", "pandas_option"],
)
def test_a_module_setting_above_holds_after_a_restart(nb_runner, setting, probe, expected):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
            "import numpy as np\nimport pandas as pd\n" + setting,
            DATA,
            f"print('SETTING', repr({probe}), len(data))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert f"SETTING {expected} 3000000" in nb_runner.get_output(4), "before the restart: " + nb_runner.get_output(4)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)

    assert f"SETTING {expected} 3000000" in nb_runner.get_output(4), nb_runner.get_output(4)
