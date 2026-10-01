"""A statement's compute time is its own run, not cash's setup around it.

The first statement in a process installs the reader patches before it runs
(tens of milliseconds). Charged to the statement, ``x = 1 + 1`` would look
expensive enough to store, and a re-run would restore it instead of
recomputing it.
"""

from __future__ import annotations

import time

from cash.notebook.cache_status import CacheStatus
from cash.tracking import io_watch

SETUP_SECONDS = 0.05


def test_slow_observation_setup_is_not_the_statements_cost(statement_processor, monkeypatch):
    def slow_install() -> None:
        time.sleep(SETUP_SECONDS)

    monkeypatch.setattr(io_watch, "_patchers", [*io_watch._patchers, (slow_install, lambda: None)])
    p = statement_processor

    first = p.process_statement("x = 1 + 1")
    assert first["execution_time"] < SETUP_SECONDS
    # Too cheap to store -> recomputed, not restored.
    assert p.process_statement("x = 1 + 1")["status"] == CacheStatus.COMPUTED
