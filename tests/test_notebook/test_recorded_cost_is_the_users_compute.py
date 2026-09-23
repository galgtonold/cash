"""A statement's recorded cost is what its own code costs, not cash's overhead.

``%cash_stats`` "saved 16.50s" for a folder read that takes 1.8 s
without cash -- the recorded cost was the statement's wall time under
cash, file tracking and call caching included, and a later hit credited all of
it. And "saved 6.55s" for a 50-100 s dict of fits: the statement that
built it was served its calls from the cache, so it recorded only what was left
over, and a later hit credited that. Cash's own time comes off; what the calls
it served would have cost goes on.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell


def _cost_of(backend, needle):
    for meta in backend.list_entries() or ():
        if needle in str(meta.get("code", "")):
            return meta.get("execution_time")
    raise AssertionError(f"no entry for {needle!r}")


DEFS = "import time\ndef slow(i):\n    time.sleep(0.2)\n    return i"


def test_calls_served_from_the_cache_count_toward_the_statements_cost(cash_magics, mock_shell, clean_backend):
    run_cash_cell(cash_magics, DEFS)
    run_cash_cell(cash_magics, "r = [slow(i) for i in range(3)]")
    run_cash_cell(cash_magics, "r2 = [slow(i) for i in range(3)] + []")  # a new statement; its calls hit
    assert mock_shell.user_ns["r2"] == [0, 1, 2]
    assert _cost_of(clean_backend, "+ []") >= 0.5, "the served calls' compute was left out"


def test_cash_tracking_time_is_not_counted_as_the_statements(cash_magics, mock_shell, clean_backend, monkeypatch):
    clock = [0.0]

    # Every statement is charged 0.45 s of tracking against a 0.5 s sleep, so
    # the right record is ~0.05 s plus whatever the machine adds, and one that
    # kept the tracking is over 0.5 s. The bound sits between with 350 ms of
    # room: macOS runners add up to ~170 ms to a statement this size, which
    # used up the 25 ms the first version of this test left.
    def tracked():
        clock[0] += 0.45
        return clock[0]

    monkeypatch.setattr("cash.notebook.statement.call_routing.tracking_seconds", tracked)
    run_cash_cell(cash_magics, "import time\nx = (time.sleep(0.5), 7)[1]")
    assert mock_shell.user_ns["x"] == 7
    assert _cost_of(clean_backend, "time.sleep(0.5)") < 0.4
