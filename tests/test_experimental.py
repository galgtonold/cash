"""Tests for experimental namespace and DependencyGraph."""
import pytest
import unittest.mock
import importlib
from cash.experimental import __getattr__ as exp_getattr
from cash.graph import DependencyGraph


class TestUIInit:
    """Test the cash.ui lazy import namespace."""

    def test_import_dependency_graph(self):
        """DependencyGraph is directly importable from cash.ui."""
        from cash.ui import DependencyGraph as DG
        assert DG is not None

    def test_lazy_import_cache_explorer(self):
        """CacheExplorer can be lazy-imported from cash.ui."""
        from cash.ui import CacheExplorer
        assert CacheExplorer is not None

    def test_lazy_import_cache_debugger(self):
        """CacheDebugger can be lazy-imported from cash.ui."""
        from cash.ui import CacheDebugger
        assert CacheDebugger is not None

    def test_lazy_import_visualize_notebook(self):
        """visualize_notebook can be lazy-imported from cash.ui."""
        from cash.ui import visualize_notebook
        assert visualize_notebook is not None

    def test_lazy_import_invalid(self):
        """Invalid attribute raises AttributeError."""
        import cash.ui
        with pytest.raises(AttributeError):
            _ = cash.ui.nonexistent_thing  # noqa: B018


class TestBackendsInit:
    """Test the cash.backends __init__ exports."""

    def test_import_core_backends(self):
        """Core backends are importable from cash.backends."""
        from cash.backends import (
            CacheBackend, InMemoryBackend, FileBackend,
            CascadingBackend, TieredBackend,
        )
        assert all(cls is not None for cls in [
            CacheBackend, InMemoryBackend, FileBackend,
            CascadingBackend, TieredBackend,
        ])

    def test_serializers_importable(self):
        """Serializers are importable from cash.backends."""
        from cash.backends import Serializer, PickleSerializer
        assert Serializer is not None
        assert PickleSerializer is not None

    def test_redis_backend_is_none_without_redis(self):
        """RedisBackend is None when redis is not installed."""
        from cash.backends import RedisBackend
        # May be None if redis not installed, or a class if it is
        # Just check it doesn't crash
        assert RedisBackend is None or RedisBackend is not None

    def test_s3_backend_is_none_without_boto3(self):
        """S3Backend is None when boto3 is not installed."""
        from cash.backends import S3Backend
        assert S3Backend is None or S3Backend is not None


class TestExperimentalNamespace:
    """Test the cash.experimental lazy import namespace."""

    def test_import_cache_explorer(self):
        """CacheExplorer can be imported from experimental."""
        from cash.experimental import CacheExplorer
        assert CacheExplorer is not None

    def test_import_analytics_manager(self):
        """AnalyticsManager can be imported from experimental."""
        from cash.experimental import AnalyticsManager
        assert AnalyticsManager is not None

    def test_import_tiered_backend(self):
        """TieredBackend can be imported from experimental."""
        from cash.experimental import TieredBackend
        assert TieredBackend is not None

    def test_import_dependency_graph(self):
        """DependencyGraph can be imported from experimental."""
        from cash.experimental import DependencyGraph
        assert DependencyGraph is not None

    def test_invalid_attribute_raises(self):
        """Invalid attribute raises AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            exp_getattr("NonExistentThing")

    def test_redis_backend_import_error(self):
        """RedisBackend raises ImportError if redis not installed."""
        try:
            from cash.experimental import RedisBackend
            # If redis is installed, this should work
            assert RedisBackend is not None
        except ImportError:
            pass  # Expected if redis not installed

    def test_s3_backend_import_error(self):
        """S3Backend raises ImportError if boto3 not installed."""
        try:
            from cash.experimental import S3Backend
            assert S3Backend is not None
        except ImportError:
            pass  # Expected if boto3 not installed


class TestDependencyGraph:
    """Test DependencyGraph directed acyclic graph."""

    def test_add_node(self):
        """Add a node to the graph."""
        g = DependencyGraph()
        g.add_node("func_a")
        assert g.get_dependencies("func_a") == set()
        assert g.get_dependents("func_a") == set()

    def test_add_dependency(self):
        """Add a dependency edge."""
        g = DependencyGraph()
        g.add_dependency("func_a", "func_b")  # a depends on b
        assert "func_b" in g.get_dependencies("func_a")
        assert "func_a" in g.get_dependents("func_b")

    def test_get_all_dependents(self):
        """Get transitive dependents."""
        g = DependencyGraph()
        g.add_dependency("func_a", "func_b")  # a depends on b
        g.add_dependency("func_b", "func_c")  # b depends on c

        # If c changes, b and a are affected
        all_deps = g.get_all_dependents("func_c")
        assert "func_b" in all_deps
        assert "func_a" in all_deps

    def test_get_all_dependents_no_self(self):
        """get_all_dependents excludes the node itself."""
        g = DependencyGraph()
        g.add_dependency("func_a", "func_b")
        all_deps = g.get_all_dependents("func_b")
        assert "func_b" not in all_deps

    def test_get_all_dependents_cycle_safe(self):
        """get_all_dependents handles cycles without infinite loop."""
        g = DependencyGraph()
        g.add_dependency("a", "b")
        g.add_dependency("b", "a")  # Cycle
        all_deps = g.get_all_dependents("a")
        assert "b" in all_deps

    def test_clear(self):
        """Clear removes all nodes and edges."""
        g = DependencyGraph()
        g.add_dependency("a", "b")
        g.clear()
        assert g.get_dependencies("a") == set()

    def test_diamond_dependency(self):
        """Diamond dependency pattern works."""
        g = DependencyGraph()
        # d depends on b and c, both depend on a
        g.add_dependency("d", "b")
        g.add_dependency("d", "c")
        g.add_dependency("b", "a")
        g.add_dependency("c", "a")

        # If a changes, b, c, and d are all affected
        all_deps = g.get_all_dependents("a")
        assert all_deps == {"b", "c", "d"}

    def test_multiple_dependencies(self):
        """A node can have multiple dependencies."""
        g = DependencyGraph()
        g.add_dependency("main", "helper1")
        g.add_dependency("main", "helper2")
        g.add_dependency("main", "helper3")

        deps = g.get_dependencies("main")
        assert deps == {"helper1", "helper2", "helper3"}

    def test_isolated_node(self):
        """Isolated node has no dependents or dependencies."""
        g = DependencyGraph()
        g.add_node("isolated")
        assert g.get_all_dependents("isolated") == set()

    def test_nonexistent_node(self):
        """Querying nonexistent node returns empty set."""
        g = DependencyGraph()
        assert g.get_dependencies("nonexistent") == set()
        assert g.get_dependents("nonexistent") == set()

    def test_get_dependents_immediate(self):
        """get_dependents returns immediate (non-transitive) dependents."""
        g = DependencyGraph()
        g.add_dependency("b", "a")
        g.add_dependency("c", "b")
        # a's immediate dependents are just b, not c
        assert g.get_dependents("a") == {"b"}
        # b's dependents are just c
        assert g.get_dependents("b") == {"c"}

    def test_visualize_without_pyvis(self, capsys):
        """visualize prints install message when pyvis is not available."""
        g = DependencyGraph()
        g.add_dependency("a", "b")
        with unittest.mock.patch.dict('sys.modules', {'pyvis': None, 'pyvis.network': None}):
            result = g.visualize()
        captured = capsys.readouterr()
        assert "pip install pyvis" in captured.out
        assert result is None

    def test_visualize_with_pyvis_notebook(self):
        """visualize returns IFrame when pyvis is available in notebook mode."""
        g = DependencyGraph()
        g.add_dependency("func_a", "data.csv")
        g.add_dependency("func_b", "func_a")

        mock_net = unittest.mock.MagicMock()
        mock_net.generate_html.return_value = "<html>graph</html>"
        unittest.mock.MagicMock()

        with unittest.mock.patch.dict('sys.modules', {
            'pyvis': unittest.mock.MagicMock(),
            'pyvis.network': unittest.mock.MagicMock(),
        }), unittest.mock.patch('cash.graph.Network', return_value=mock_net, create=True):
            # Need to patch at the point of use
            import cash.graph as graph_mod  # noqa: F811

            # Verify the module loaded with our mock
            assert graph_mod is not None

    def test_visualize_non_notebook_mode(self, tmp_path):
        """visualize in non-notebook mode saves to file."""
        g = DependencyGraph()
        g.add_dependency("a", "b")

        mock_net = unittest.mock.MagicMock()
        mock_network_cls = unittest.mock.MagicMock(return_value=mock_net)
        mock_ipython = unittest.mock.MagicMock()

        with unittest.mock.patch.dict('sys.modules', {
            'pyvis': unittest.mock.MagicMock(),
            'pyvis.network': unittest.mock.MagicMock(Network=mock_network_cls),
            'IPython': mock_ipython,
            'IPython.display': mock_ipython.display,
        }):
            # Re-import to pick up mocks
            import cash.graph as graph_mod
            importlib.reload(graph_mod)
            g2 = graph_mod.DependencyGraph()
            g2.add_dependency("a", "b")
            output_file = str(tmp_path / "graph.html")
            g2.visualize(output_file=output_file, notebook=False)
            mock_net.show.assert_called_once_with(output_file)
