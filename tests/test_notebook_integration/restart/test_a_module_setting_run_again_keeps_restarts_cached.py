"""A module setting re-run in a session keeps what reads the module cached after a restart.

``pd.set_option('display.width', 120)`` and ``plt.style.use('ggplot')`` are
real settings of their module, so each run moves the module's lineage. Each
run built on the lineage the last one left, so a session that ran the
setting cell three times keyed every statement reading ``pd`` or ``plt``
apart from a Run All after a restart, which ran it once. Headless (the
notebook not found), nothing derives the lineages from the cells, and every
such statement recomputed after the restart.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(240)]

IMPORTS = "%cash_badge print\nimport time\nimport pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"
SETTINGS = "pd.set_option('display.width', 120)\nplt.style.use('ggplot')"
FRAME = "df = pd.DataFrame({'a': [time.sleep(0.4) or i for i in range(3)]})"
STYLE = "face = plt.rcParams['axes.facecolor'] + ('' if time.sleep(1.2) else '')"


def _session(nb_runner, settings: str = SETTINGS, found: bool = False) -> None:
    nb_runner.create_notebook([IMPORTS, settings, FRAME, STYLE])
    # Headless by default, as the run that found it: the notebook is not found.
    nb_runner.start_kernel(inject_notebook_path=found)


@pytest.mark.parametrize("found", [False, True], ids=["headless", "notebook_found"])
def test_readers_of_a_setting_run_three_times_are_served_after_a_restart(nb_runner, found):
    _session(nb_runner, found=found)
    nb_runner.run_cells([1, 2, 2, 2, 3, 4])
    nb_runner.run_cells([3])
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_raw_output(3)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_raw_output(3)
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_raw_output(4)
    assert nb_runner.peek("face") == "'#E5E5E5'"
    assert nb_runner.peek("pd.get_option('display.width')") == "120"


def test_a_different_style_is_not_served_the_entry_of_the_other(nb_runner):
    _session(nb_runner)
    nb_runner.run_cells([1, 2, 2, 3, 4])
    assert nb_runner.peek("face") == "'#E5E5E5'"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(2, "pd.set_option('display.width', 120)\nplt.style.use('dark_background')")
    nb_runner.run_all()

    assert not shows_cached(nb_runner.get_output(4)), nb_runner.get_raw_output(4)
    assert nb_runner.peek("face") == "'black'"


def test_settings_run_in_another_order_are_not_served_each_others_entry(nb_runner):
    # The later style wins: ggplot then dark_background leaves a black face,
    # the other order a grey one.
    two = "plt.style.use('ggplot')\nplt.style.use('dark_background')"
    _session(nb_runner, two)
    nb_runner.run_cells([1, 2, 2, 4])
    assert nb_runner.peek("face") == "'black'"

    nb_runner.set_cell_source(2, "plt.style.use('dark_background')\nplt.style.use('ggplot')")
    nb_runner.run_cells([1, 2, 4])

    assert not shows_cached(nb_runner.get_output(4)), nb_runner.get_raw_output(4)
    assert nb_runner.peek("face") == "'#E5E5E5'"
