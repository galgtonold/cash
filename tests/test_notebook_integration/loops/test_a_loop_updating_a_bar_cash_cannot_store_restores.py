"""A loop that only moves a progress bar cash cannot store is restored.

``pbar = pyprind.ProgBar(...)`` is refused (its own parts hold it), and each
``for file in ...: ...; pbar.update()`` run in a pass of an outer loop
changed it in place, so the loop ran on every Run All and after every
restart. When nothing after the loop reads the bar, a hit leaves the bar as
it was; a cell that comes to read it builds it again.
"""

import pytest

from tests._nbharness.runner import CASH_TEST_PIN_THRESHOLDS

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

STATUS = (
    "[str(r['status']) for r in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
)

SETUP = (
    "import cash\n%cash_on\n" + CASH_TEST_PIN_THRESHOLDS + "import sys, time\n"
    "class Bar:\n"
    "    def __init__(self, stream):\n"
    "        self.n = 0\n"
    "        self.stream = stream\n"
    "        self.me = self\n"
    "    def update(self):\n"
    "        self.n += 1\n"
)
FILL = (
    "bar = Bar(sys.stdout)\n"
    "rows = 0\n"
    "for s in (1, 2):\n"
    "    for i in range(1500):\n"
    "        rows = rows + i * s\n"
    "        time.sleep(0.0002)\n"
    "        bar.update()"
)


def _run_all_after_a_restart(nb_runner, cells: int) -> list[str]:
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    statuses = []
    for cell in range(1, cells + 1):
        nb_runner.run_cell(cell)
        if cell == 2:
            statuses = nb_runner.peek(STATUS)
    return statuses


def test_the_passes_restore_when_nothing_reads_the_bar(nb_runner):
    nb_runner.create_notebook([SETUP, FILL, "total = rows * 2"])
    nb_runner.start_kernel()
    statuses = _run_all_after_a_restart(nb_runner, 3)
    assert "RESTORED" in statuses, statuses
    assert nb_runner.peek("total") == str(2 * (1499 * 1500 // 2) * 3)


def test_a_cell_added_later_that_reads_the_bar_builds_it_again(nb_runner):
    nb_runner.create_notebook([SETUP, FILL, "total = rows * 2"])
    nb_runner.start_kernel()
    statuses = _run_all_after_a_restart(nb_runner, 3)
    assert "RESTORED" in statuses, statuses
    nb_runner.set_cell_source(3, "total = rows * 2\nseen = bar.n")
    nb_runner.run_cell(3)
    assert nb_runner.peek("seen") == "3000", nb_runner.get_raw_output(3)


def test_a_bar_read_below_keeps_the_passes_running(nb_runner):
    """Control: a cell below reads the bar, so a pass may not leave it as
    it was."""
    nb_runner.create_notebook([SETUP, FILL, "seen = bar.n"])
    nb_runner.start_kernel()
    statuses = _run_all_after_a_restart(nb_runner, 3)
    assert "RESTORED" not in statuses, statuses
    assert nb_runner.peek("seen") == "3000"
