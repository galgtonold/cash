"""A statement calling a helper that can draw is cached when it did not draw.

E16's ``benchmarks(...)`` calls ``run(tsp, cities, plot=False)``, whose
``plot_tour`` is behind ``if plot:``. Its code can reach pyplot, so it ran
every time: ~200 s on each restart. Where what runs is recorded (Python
3.12+), whether pyplot's code ran decides; a run that draws still runs
every time.
"""

from __future__ import annotations

import sys

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="what runs is recorded on Python 3.12+")

HELPERS = (
    "import time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "def plot_tour(tour):\n    plt.figure()\n    plt.plot(tour)\n"
    "def run(n, plot=True):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    tour = list(range(n))\n    if plot:\n        plot_tour(tour)\n    return sum(tour)"
)
QUIET = "quiet = run(5, plot=False)"
DRAWN = "drawn = run(6)"
CELLS = [HELPERS, QUIET, DRAWN]


@pytest.fixture
def shadow_on():
    from cash.notebook import shadow

    was = shadow.SHADOW
    if shadow.enable() is None:
        pytest.skip("what runs cannot be recorded here")
    yield
    if was is None:
        shadow.disable()


def _status(magics):
    return magics.cash_status("dict")["last_cell"]["statements"][-1].get("status")


def test_the_run_that_did_not_draw_is_served_and_the_one_that_drew_runs_again(cash_magics, shadow_on):
    plt.close("all")
    try:
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        assert cash_magics.shell.user_ns["drawn"] == 15
        figures = plt.get_fignums()

        run_cash_cell(cash_magics, QUIET, cells=CELLS)
        assert _status(cash_magics) == CacheStatus.RESTORED
        assert cash_magics.shell.user_ns["quiet"] == 10
        # Served, it drew nothing, as its run drew nothing.
        assert plt.get_fignums() == figures

        run_cash_cell(cash_magics, DRAWN, cells=CELLS)
        assert _status(cash_magics) != CacheStatus.RESTORED
        assert len(plt.get_fignums()) == len(figures) + 1
    finally:
        plt.close("all")


def test_without_the_record_it_runs_every_time(cash_magics):
    from cash.notebook import shadow

    was = shadow.SHADOW
    shadow.disable()
    plt.close("all")
    try:
        for cell in CELLS[:2]:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        run_cash_cell(cash_magics, QUIET, cells=CELLS)
        assert _status(cash_magics) != CacheStatus.RESTORED
    finally:
        plt.close("all")
        if was is not None:
            shadow.enable()


READER = (
    "import time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    f"def nlines():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return len(plt.gca().lines)"
)


def test_a_helper_that_only_reads_the_figure_still_runs_every_time(cash_magics, shadow_on):
    cells = [READER, "plt.figure()\nplt.plot([1, 2])", "n = nlines()"]
    try:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["n"] == 1
        cells[1] = "plt.figure()\nplt.plot([1, 2])\nplt.plot([2, 1])"
        for cell in cells[1:]:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["n"] == 2
    finally:
        plt.close("all")
