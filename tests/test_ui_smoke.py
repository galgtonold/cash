"""Smoke tests for cash.ui modules (dashboard and visualizer).

These modules depend on IPython/ipywidgets which may not be available
in all test environments, so tests gracefully handle ImportError.
"""

from cash.ui.dashboard import show_analytics_dashboard, HAS_WIDGETS


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


class TestVisualizerSmoke:
    """Basic smoke tests for the cache visualizer."""

    def test_module_imports(self):
        """The visualizer module should always be importable."""
        from cash.ui import visualizer
        assert hasattr(visualizer, "format_time")

    def test_format_time_none(self):
        from cash.ui.visualizer import format_time
        assert format_time(None) == "N/A"

    def test_format_time_zero(self):
        from cash.ui.visualizer import format_time
        result = format_time(0.0)
        assert isinstance(result, str)
        assert "ms" in result or "< 1ms" in result

    def test_format_time_seconds(self):
        from cash.ui.visualizer import format_time
        result = format_time(5.0)
        assert isinstance(result, str)
        assert "5" in result

    def test_format_time_subsecond(self):
        from cash.ui.visualizer import format_time
        result = format_time(0.5)
        assert isinstance(result, str)

    def test_status_constants(self):
        from cash.ui.visualizer import STATUS_HIT, STATUS_MISS
        assert STATUS_HIT == "HIT"
        assert STATUS_MISS == "MISS"

    def test_format_memory_none(self):
        from cash.ui.visualizer import format_memory
        assert format_memory(None) == "N/A"

    def test_format_memory_zero(self):
        from cash.ui.visualizer import format_memory
        assert format_memory(0) == "0 B"

    def test_format_memory_bytes(self):
        from cash.ui.visualizer import format_memory
        assert format_memory(512) == "512 B"

    def test_format_memory_kilobytes(self):
        from cash.ui.visualizer import format_memory
        result = format_memory(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_format_memory_megabytes(self):
        from cash.ui.visualizer import format_memory
        result = format_memory(5 * 1024 * 1024)
        assert "MB" in result

    def test_visualize_notebook_is_callable(self):
        from cash.ui.visualizer import visualize_notebook
        assert callable(visualize_notebook)

    def test_visualize_cache_is_callable(self):
        from cash.ui.visualizer import visualize_cache
        assert callable(visualize_cache)

    def test_build_results_dataframe_is_callable(self):
        from cash.ui.visualizer import _build_results_dataframe
        assert callable(_build_results_dataframe)
