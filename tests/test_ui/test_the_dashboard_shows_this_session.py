"""``show_stats()`` shows the session that recorded the events.

The dashboard used to make its own `AnalyticsManager`, with a new random
session id, so "Current Session" was always empty; and its flush drained its
own empty buffer, so "All Time" left out the events this session still had
buffered (up to a thousand).
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def analytics_home(tmp_path, monkeypatch):
    """Analytics on, into a database of this test's own. Listed before
    ``cash_magics`` so the processor's manager is created under it."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("CASH_ANALYTICS", raising=False)


def test_the_dashboard_reads_the_events_this_session_recorded(analytics_home, cash_magics, cash_instance, monkeypatch):
    import cash.ui.dashboard as dashboard

    for i in range(3):
        run_cash_cell(cash_magics, f"shown_{i} = {i} + 1")
    recorded = len(cash_magics._statement_processor.analytics_manager._event_buffer)
    assert recorded >= 3, "the cells should have recorded events"

    shown = []
    monkeypatch.setattr(dashboard, "HAS_WIDGETS", True)
    monkeypatch.setattr(dashboard, "show_analytics_dashboard", lambda mgr=None: shown.append(mgr))
    cash_instance.show_stats()

    (mgr,) = shown
    assert mgr is not None, "the dashboard was left to make a manager of its own"
    assert mgr.get_session_stats()["total_events"] == recorded
    assert mgr.get_global_stats()["total_events"] == recorded, "buffered events were left out of All Time"
