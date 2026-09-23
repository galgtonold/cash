"""Smoke tests for UI modules to catch import and basic wiring regressions."""

from __future__ import annotations

import types

from cash.ui import dashboard, explorer


def test_dashboard_no_widgets_path(monkeypatch):
    """Dashboard should exit cleanly when widgets are unavailable."""
    monkeypatch.setattr(dashboard, "HAS_WIDGETS", False)
    assert dashboard.show_analytics_dashboard() is None


def test_explorer_list_entries_smoke():
    """Explorer should return cache entries and enrich source metadata."""

    class DummyBackend:
        def list_entries(self):
            return [{"key": "k1", "func_name": "f1", "timestamp": 1}]

    app = types.SimpleNamespace(backend=DummyBackend(), functions={})
    cache_explorer = explorer.CacheExplorer(app)

    entries = cache_explorer.list_entries()
    assert len(entries) == 1
    assert entries[0]["key"] == "k1"
    assert "source_code" in entries[0]
