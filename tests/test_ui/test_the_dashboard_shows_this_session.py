"""``show_stats()`` shows the session that recorded the events.

The dashboard used to make its own `AnalyticsManager`, with a new random
session id, so "Current Session" was always empty; and its flush drained its
own empty buffer, so "All Time" left out the events this session still had
buffered (up to a thousand).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from tests._cell_driver import run_cash_cell


def _point_the_per_user_cache_at(tmp_path, monkeypatch):
    """Every platform's per-user cache root into *tmp_path*: XDG_CACHE_HOME on
    Linux, LOCALAPPDATA on Windows, and HOME for macOS, whose
    ``~/Library/Caches`` reads neither of the other two."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def analytics_home(tmp_path, monkeypatch):
    """Analytics on, into a database of this test's own. Listed before
    ``cash_magics`` so the processor's manager is created under it."""
    from cash.analytics import default_db_path

    _point_the_per_user_cache_at(tmp_path, monkeypatch)
    monkeypatch.delenv("CASH_ANALYTICS", raising=False)
    assert default_db_path().is_relative_to(tmp_path), "the test would read the machine's own analytics"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_the_database_is_the_tests_own_on_every_platform(platform, tmp_path, monkeypatch):
    """The macOS root is under HOME, not XDG_CACHE_HOME: left out, the test
    read the shared ``~/Library/Caches`` database the whole suite writes to.
    (On Windows ``os.name`` decides first, and LOCALAPPDATA is set too.)"""
    from cash import _location
    from cash.analytics import default_db_path

    monkeypatch.setattr(_location, "sys", types.SimpleNamespace(platform=platform))
    _point_the_per_user_cache_at(tmp_path, monkeypatch)
    assert default_db_path().is_relative_to(tmp_path)


def test_the_dashboard_reads_the_events_this_session_recorded(
    analytics_home, cash_magics, cash_instance, monkeypatch, tmp_path
):
    import cash.core
    import cash.ui.dashboard as dashboard

    for i in range(3):
        run_cash_cell(cash_magics, f"shown_{i} = {i} + 1")
    recorded = len(cash_magics._statement_processor.analytics_manager._event_buffer)
    assert recorded >= 3, "the cells should have recorded events"

    shown = []
    monkeypatch.setattr(dashboard, "HAS_WIDGETS", True)
    # show_stats() only draws the widgets inside a kernel; this test has none.
    monkeypatch.setattr(cash.core, "_in_kernel", lambda: True)
    monkeypatch.setattr(dashboard, "show_analytics_dashboard", lambda mgr=None: shown.append(mgr))
    cash_instance.show_stats()

    (mgr,) = shown
    assert mgr is not None, "the dashboard was left to make a manager of its own"
    assert Path(mgr.db_path).is_relative_to(tmp_path), "the manager is not writing this test's database"
    assert mgr.get_session_stats()["total_events"] == recorded
    assert mgr.get_global_stats()["total_events"] == recorded, "buffered events were left out of All Time"
