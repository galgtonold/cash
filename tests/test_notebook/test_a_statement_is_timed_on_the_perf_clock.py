"""A statement is timed on the clock its calls are timed on.

The statement pipeline timed a statement with ``time.time()`` while a cached
call inside it was timed with the performance counter. On Windows the wall
clock moves in steps -- a few tenths of a millisecond, up to 15.6 ms on an
idle machine, longer than the persistence floor -- so a statement wrapping a
0.2 s call was recorded as costing less than the call. The wall clock is for
timestamps; every duration is read off the performance counter.
"""

from __future__ import annotations

import time

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell

_STILL = 1_700_000_000.0


def test_a_statement_is_timed_while_the_wall_clock_stands_still(cash_magics, monkeypatch):
    """A wall clock that does not move between two of its ticks, as Windows'
    does, leaves a statement's measured time as it is."""
    magics = cash_magics
    run_cash_cell(magics, "import time")
    monkeypatch.setattr(time, "time", lambda: _STILL)

    metrics = magics._statement_processor.process_statement("x = time.sleep(0.03)")

    assert metrics["status"] == CacheStatus.COMPUTED, metrics.get("status")
    assert metrics["execution_time"] >= 0.025, metrics["execution_time"]
    assert metrics["compute_cost"] >= 0.025, metrics["compute_cost"]
    assert metrics["total_time"] >= metrics["execution_time"]
