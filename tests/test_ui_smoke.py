"""Smoke tests for the cash.ui dashboard.

These modules depend on IPython/ipywidgets which may not be available
in all test environments, so tests gracefully handle ImportError.
"""

from cash.ui.dashboard import HAS_WIDGETS, show_analytics_dashboard


class TestDashboardSmoke:
    """Basic smoke tests for the analytics dashboard."""

    def test_module_imports(self):
        """The dashboard module should always be importable."""
        from cash.ui import dashboard

        assert hasattr(dashboard, "show_analytics_dashboard")

    def test_function_is_callable(self):
        assert callable(show_analytics_dashboard)

    def test_has_widgets_is_boolean(self):
        assert isinstance(HAS_WIDGETS, bool)
