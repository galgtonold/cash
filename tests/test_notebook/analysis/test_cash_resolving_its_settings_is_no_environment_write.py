"""Cash resolving its settings while a statement runs is no write of the
statement's to the environment.

Each resolution of where "here" is re-exports it to the environment for
pool workers (``_CASH_PARENT_LAUNCH``, through ``os.putenv``). When cash's
own work in the middle of a statement resolved it -- a function's text
looked up in the scan store once the process's parse budget is spent, under
a ``Cash`` of the user's own that is not the default -- the statement was
taken to change the environment, which no variable holds, and was never
stored: ``G = build()`` ran every time, and only after enough other code had
been read in the process.
"""

from __future__ import annotations

import os
import uuid

import pytest

from cash import _active, _location
from cash.backends.persistence_policy import PersistencePolicy
from cash.decorator import hit_memo
from cash.notebook.cache_status import CacheStatus
from cash.tracking import process_registries
from tests._cell_driver import run_cash_cell

_ENVIRONMENT = "the environment (os.environ, os.putenv, os.unsetenv)"


def test_cash_resolving_where_here_is_changes_no_registry(monkeypatch):
    # Another value first: the comparison of os.environ, not only the
    # count of os.putenv calls, leaves cash's own key out.
    monkeypatch.setenv(_location.LAUNCH_ENV, "{}")
    before = process_registries.snapshot()
    _location.project_anchor()
    assert os.environ[_location.LAUNCH_ENV] != "{}"
    assert process_registries.changed(before) == []


@pytest.mark.parametrize("write", ["environ", "putenv"])
def test_a_write_of_the_users_is_still_seen(monkeypatch, write):
    monkeypatch.setenv("CASH_TEST_OWN_LAUNCH_CONTROL", "a")
    before = process_registries.snapshot()
    _location.project_anchor()
    if write == "environ":
        os.environ["CASH_TEST_OWN_LAUNCH_CONTROL"] = "b"
    else:
        os.putenv("CASH_TEST_OWN_LAUNCH_CONTROL", "b")
    assert _ENVIRONMENT in process_registries.changed(before)


SETUP = f"""
import time

class Graph:
    def __init__(self):
        self.rows = list(range(1000))

    def size(self) -> int:
        return len(self.rows)

def build():
    # {uuid.uuid4().hex}: a text this process has not looked up before
    time.sleep(0.2)
    return Graph()
"""


def test_a_statement_during_which_cash_resolves_its_settings_is_served(cash_magics, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    # Spent, as after enough code was read in the process: a function's text
    # is looked up in the scan store, whose folder comes from the settings,
    # resolved afresh with no default instance.
    monkeypatch.setattr(hit_memo, "_PARSED_BUDGET", [0])
    # A Cash of the user's own runs the magics: no default instance holds
    # settings already resolved.
    monkeypatch.setattr(_active, "_default", None)
    run_cash_cell(cash_magics, SETUP)
    monkeypatch.setenv(_location.LAUNCH_ENV, "{}")
    run_cash_cell(cash_magics, "G = build()")
    assert os.environ[_location.LAUNCH_ENV] != "{}", "cash resolved no settings for the statement: nothing is tested"
    run_cash_cell(cash_magics, "G = build()")
    assert cash_magics._last_cell_metrics["statements"][-1]["status"] == CacheStatus.RESTORED
