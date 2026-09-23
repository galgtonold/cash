"""Smoke tests for the cash.ui modules: import and basic wiring.

The dashboard depends on ipywidgets, which may not be installed, so it must
import either way and exit cleanly without widgets.
"""

from __future__ import annotations

import types

from cash.ui import dashboard, explorer
from cash.ui.dashboard import HAS_WIDGETS, show_analytics_dashboard


class TestDashboardSmoke:
    """Basic smoke tests for the analytics dashboard."""

    def test_module_imports(self):
        """The dashboard module should always be importable."""
        assert hasattr(dashboard, "show_analytics_dashboard")

    def test_function_is_callable(self):
        assert callable(show_analytics_dashboard)

    def test_has_widgets_is_boolean(self):
        assert isinstance(HAS_WIDGETS, bool)

    def test_dashboard_no_widgets_path(self, monkeypatch):
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
