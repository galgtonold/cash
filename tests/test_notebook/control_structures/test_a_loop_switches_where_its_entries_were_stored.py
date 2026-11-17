"""A loop whose rest runs as one unit switches where its entries were stored.

A loop of known length runs its first passes one by one and, once they
measured cheap, its rest as one unit (`single_unit_policy.rest_worth_one_unit`,
about 10 ms of bookkeeping a pass). That pass is timed, so it moved between
runs: a run that switched after pass 6 where the stored one had after pass 5
found no entry for its unit, and one that switched where the stored run had
gone on one by one found none for it either; the passes past it ran again.
Where the loop switched is now recorded with the entries it stored
(`LoopSplitStore.rest_after`), and a run whose passes are all served
switches there, whatever its own timing says.
"""

from __future__ import annotations

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.control_structures import single_unit_policy
from tests._cell_driver import run_cash_cell

SETUP = "import numpy as np\narr = np.zeros(10)"
LOOP = "for i in range(10):\n    arr[i] = i + 1"
NEVER = 99


def _switching_at(monkeypatch, points):
    """Make each run of the loop switch after the next pass of *points*
    (`NEVER`: it runs every pass one by one), as timing would."""
    runs = iter(points)
    at = {"k": NEVER}

    def cheap(shares):
        if len(shares) == 1:
            at["k"] = next(runs)
        return len(shares) >= at["k"]

    monkeypatch.setattr(single_unit_policy, "passes_were_cheap", cheap)
    monkeypatch.setattr(single_unit_policy, "rest_worth_one_unit", lambda overheads, remaining: remaining > 0)


def _loop_statuses(cash_magics):
    return {m.get("status") for m in cash_magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")}


@pytest.mark.parametrize(
    "points",
    [(5, 6), (NEVER, 6), (6, NEVER), (5, 5)],
    ids=["a-later-switch", "a-switch-where-it-ran-one-by-one", "no-switch-where-it-switched", "the-same-switch"],
)
def test_a_second_run_is_served_whole_whatever_its_timing(cash_magics, cash_instance, monkeypatch, points):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    _switching_at(monkeypatch, points)
    ns = cash_magics.shell.user_ns
    cells = [SETUP, LOOP]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert list(ns["arr"]) == [float(i + 1) for i in range(10)]
    assert _loop_statuses(cash_magics) == {CacheStatus.RESTORED}


def test_new_inputs_switch_by_their_own_timing(cash_magics, cash_instance, monkeypatch):
    """The control: a run whose passes are not served (the array is new)
    switches where its timing says, as before, and stores that."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    _switching_at(monkeypatch, (5, 7, 7))
    ns = cash_magics.shell.user_ns
    for setup in (SETUP, "import numpy as np\narr = np.ones(10)", "import numpy as np\narr = np.ones(10)"):
        cells = [setup, LOOP]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert list(ns["arr"]) == [float(i + 1) for i in range(10)]
    assert _loop_statuses(cash_magics) == {CacheStatus.RESTORED}
