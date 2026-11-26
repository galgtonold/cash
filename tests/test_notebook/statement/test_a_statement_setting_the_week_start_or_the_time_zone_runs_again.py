"""A statement that sets the week's first day or the time zone runs again.

``calendar.setfirstweekday(6)`` and ``time.tzset()`` (after ``TZ`` was
set) change a setting the process keeps outside any variable. Restored
from the cache, the statement did not make it again: after a restart
(here, the setting put back as a new process has it) ``calendar.month``
started its weeks on Monday and ``time.localtime`` answered in the old
zone. Such a statement is not stored, and it is kept on record so that a
cell below run alone after a restart gets it run again first.
"""

from __future__ import annotations

import calendar
import os
import textwrap
import time

import pytest

from tests._cell_driver import run_cash_cell

SETUP = "import calendar, os, time\n"

#: name -> (statement, whether its effect is there, undo it as a new
#: process would not have it).
CASES = {
    "week-start": ("calendar.setfirstweekday(6)", "calendar.firstweekday() == 6", "calendar.setfirstweekday(0)"),
    # ``TZ`` is in the environment already (`zone`): the statement only
    # makes the process read it.
    "time-zone": ("time.tzset()", "time.timezone == -32400", "_as_started()"),
}


def _as_started() -> None:
    """The zone a process started without ``TZ=JST-9`` has, with the
    variable left as the test set it."""
    os.environ["TZ"] = "UTC0"
    time.tzset()
    os.environ["TZ"] = "JST-9"


@pytest.fixture
def zone(monkeypatch):
    """``TZ=JST-9`` in the environment, not yet read; the process's own
    zone and week start back at the end."""
    if not hasattr(time, "tzset"):
        yield False
        return
    first = calendar.firstweekday()
    monkeypatch.setenv("TZ", "JST-9")
    try:
        yield True
    finally:
        calendar.setfirstweekday(first)
        monkeypatch.undo()
        time.tzset()


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _params():
    for kind in CASES:
        yield pytest.param(kind, False, id=f"{kind}-in-the-cell")
        yield pytest.param(kind, True, id=f"{kind}-in-a-function")


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize(("kind", "in_a_function"), list(_params()))
def test_the_setting_is_made_again(kind, in_a_function, cash_magics, mock_shell, zone):
    if kind == "time-zone" and not zone:
        pytest.skip("no time.tzset on this platform")
    statement, there, undo = CASES[kind]
    setup = SETUP
    if in_a_function:
        setup += "def setting():\n" + textwrap.indent(statement, "    ") + "\n"
        statement = "setting()"
    cells = [setup, statement]
    ns = mock_shell.user_ns
    run_cash_cell(cash_magics, setup, cells=cells)
    ns["_as_started"] = _as_started
    exec(undo, ns)

    run_cash_cell(cash_magics, statement, cells=cells)
    assert eval(there, ns)
    exec(undo, ns)
    assert not eval(there, ns)
    run_cash_cell(cash_magics, statement, cells=cells)

    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert eval(there, ns), f"restored without making the setting again: {statuses}"


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("kind", list(CASES))
def test_the_setting_is_kept_for_a_cell_run_alone_after_a_restart(kind, cash_magics, mock_shell, zone):
    from cash.notebook.cache_key import statement_source_hash

    if kind == "time-zone" and not zone:
        pytest.skip("no time.tzset on this platform")
    statement, _there, undo = CASES[kind]
    cells = [SETUP, statement]
    ns = mock_shell.user_ns
    run_cash_cell(cash_magics, SETUP, cells=cells)
    ns["_as_started"] = _as_started
    exec(undo, ns)

    run_cash_cell(cash_magics, statement, cells=cells)

    writers = cash_magics.tracking_state.process_state_writers
    assert writers.get(statement_source_hash(statement)) == frozenset({"registry"})


@pytest.mark.usefixtures("store_everything", "zone")
def test_a_statement_that_only_reads_them_is_still_stored(cash_magics):
    cells = [SETUP, "start = calendar.firstweekday()\nzone = time.tzname\ntotal = sum(range(10))"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses[-1] == "RESTORED", statuses
