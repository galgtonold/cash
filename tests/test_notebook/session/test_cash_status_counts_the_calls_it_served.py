"""``%cash_status`` credits the calls a statement had served from the cache.

A loop whose helper calls all came from the cache was reported COMPUTED with
0 s saved, while the badge and ``%cash_stats`` showed the hits: a tool reading
``%cash_status`` saw no reuse at all. Such a statement is MIXED, and what its
calls saved counts in its ``saved_time`` and in ``total_restored_time``.
"""

from __future__ import annotations

import json

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\ndef slow(i):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return i * 2"
LOOP = "tot = 0\nfor i in range(3):\n    tot += slow(i)"
# Other glue around the same calls: the loop runs, its calls hit.
SAME_CALLS = "tot2 = 1\nfor i in range(3):\n    tot2 += slow(i)"


def _last_cell(cash_magics):
    return cash_magics.cash_status("dict")["last_cell"]


def test_a_loop_of_served_calls_is_mixed_and_counts_what_they_saved(cash_magics):
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, LOOP)
    first = _last_cell(cash_magics)
    # Control: the calls ran the first time, so nothing was served.
    assert first["status"] == "COMPUTED"
    assert first["total_restored_time"] == 0.0
    assert all(row["status"] == CacheStatus.COMPUTED for row in first["statements"])

    run_cash_cell(cash_magics, SAME_CALLS)
    last = _last_cell(cash_magics)
    body = [row for row in last["statements"] if "slow(i)" in row["code"]]
    assert len(body) == 3, last
    for row in body:
        assert row["status"] == CacheStatus.MIXED
        assert row["calls"]["restored"] == 1
        assert row["saved_time"] >= ABOVE_PERSISTENCE_FLOOR_S * 0.5
    assert last["status"] == "MIXED"
    assert last["total_restored_time"] >= 3 * ABOVE_PERSISTENCE_FLOOR_S * 0.5
    assert json.loads(cash_magics.cash_status("json"))["last_cell"]["status"] == "MIXED"
