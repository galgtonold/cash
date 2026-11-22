"""A statement reading an item of matplotlib's ``rcParams`` sees it set anew.

``seen = plt.rcParams["font.size"]`` reads the setting by a subscript, which
no call of matplotlib's makes: set in another cell by a subscript too
(``plt.rcParams["font.size"] = 7``), which no setting call makes either,
the reader was served 10.0 where plain Python gives 7.
"""

from __future__ import annotations

import sys

import pytest

from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sys.monitoring needs Python 3.12: below it what a statement finds by a lookup is not watched "
    "(NOTEBOOK-LOOKUPS-UNSEEN)",
)

SETUP = "import matplotlib\nimport matplotlib.pyplot as plt\ndef size_now():\n    return plt.rcParams['font.size']"


@pytest.fixture(autouse=True)
def matplotlib_defaults():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("matplotlib.pyplot")
    yield
    matplotlib.rcdefaults()


@pytest.mark.parametrize(
    "change",
    ["plt.rcParams['font.size'] = 7", "matplotlib.rcParams['font.size'] = 7"],
    ids=["through-pyplot", "through-matplotlib"],
)
@pytest.mark.parametrize(
    "reader",
    [
        "seen = plt.rcParams['font.size']",
        "seen = matplotlib.rcParams['font.size']",
        "seen = plt.rcParams.get('font.size')",
        "seen = size_now()",
    ],
    ids=["an-item", "through-matplotlib", "by-get", "in-a-function"],
)
def test_a_reader_of_a_setting_runs_again_after_it_is_set(cash_magics, cash_instance, reader, change):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [SETUP, reader, f"# elsewhere\n{change}", f"# again\n{reader}"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 7


def test_a_reader_of_a_setting_is_served_while_it_holds(cash_magics, cash_instance):
    """Nothing set in between: the next run all serves it."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [SETUP, "seen = plt.rcParams['font.size']"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["RESTORED"], statuses


@pytest.mark.parametrize(
    "setter",
    ["plt.rcParams['font.size'] = 7", "plt.rcParams.update({'font.size': 7})", "x = plt.rcParams['font.size'] = 7"],
    ids=["an-item", "update", "bound-too"],
)
def test_a_cell_setting_it_runs_again_once_it_was_set_back(cash_magics, cash_instance, setter):
    """The setting set back to its default elsewhere, the cell setting it
    runs again: a hit would leave the default."""
    import matplotlib

    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [SETUP, setter, "seen = plt.rcParams['font.size']"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, "# elsewhere\nmatplotlib.rcdefaults()", cells=cells)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert matplotlib.rcParams["font.size"] == 7
    assert cash_magics.shell.user_ns["seen"] == 7
