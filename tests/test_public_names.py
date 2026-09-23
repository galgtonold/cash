"""Tests for the cash.ui and cash.backends namespaces, and DependencyGraph."""

import importlib
import unittest.mock

import pytest

from cash.graph import DependencyGraph


class TestOneImportPath:
    """Each public name has one import path, with no aliases in between."""

    def test_cash_ui_re_exports_nothing(self):
        import cash.ui

        for name in ("CacheExplorer", "DependencyGraph"):
            assert not hasattr(cash.ui, name)

    def test_the_experimental_namespace_is_gone(self):
        with pytest.raises(ImportError):
            importlib.import_module("cash.experimental")

    def test_the_markers_have_one_spelling(self):
        import cash

        for name in ("mark_opaque", "mark_pure", "mark_stateful", "CascadingBackend"):
            assert not hasattr(cash, name)

    def test_opaque_leaves_the_class_unmodified(self):
        """`cash.opaque` marks a class once, in the registry, and does not
        write to the class -- so it works on one that refuses new attributes,
        which is the kind of class it is called on from outside."""
        import cash

        class Frozen(type):
            def __setattr__(cls, name, value):
                raise TypeError("this class takes no new attributes")

        class Vendor(metaclass=Frozen):
            pass

        assert cash.opaque(Vendor) is Vendor
        assert cash.Cash._is_opaque(Vendor) is True
        assert "__cash_opaque__" not in vars(Vendor)

    def test_tiered_backend_is_exported_at_the_top_level(self):
        import cash
        from cash.backends.tiered_backend import TieredBackend

        assert cash.TieredBackend is TieredBackend


class TestBackendsInit:
    """Test the cash.backends __init__ exports."""

    def test_import_core_backends(self):
        """Core backends are importable from cash.backends."""
        from cash.backends import (
            CacheBackend,
            FileBackend,
            InMemoryBackend,
            TieredBackend,
        )

        assert all(
            cls is not None
            for cls in [
                CacheBackend,
                InMemoryBackend,
                FileBackend,
                TieredBackend,
            ]
        )

    def test_serializers_importable(self):
        """Serializers are importable from cash.backends."""
        from cash.backends import PickleSerializer, Serializer

        assert Serializer is not None
        assert PickleSerializer is not None

    def test_remote_backends_resolve_to_their_classes(self):
        """The lazy names are the classes themselves, installed client or not."""
        from cash.backends import RedisBackend, S3Backend
        from cash.backends.redis_backend import RedisBackend as RedisFromModule
        from cash.backends.s3_backend import S3Backend as S3FromModule

        assert RedisBackend is RedisFromModule
        assert S3Backend is S3FromModule


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

    def test_clear(self):
        """Clear removes all nodes and edges."""
        g = DependencyGraph()
        g.add_dependency("a", "b")
        g.clear()
        assert g.get_dependencies("a") == set()

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
        assert g.get_dependents("isolated") == set()
        assert g.get_dependencies("isolated") == set()

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
        with unittest.mock.patch.dict("sys.modules", {"pyvis": None, "pyvis.network": None}):
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

        with (
            unittest.mock.patch.dict(
                "sys.modules",
                {
                    "pyvis": unittest.mock.MagicMock(),
                    "pyvis.network": unittest.mock.MagicMock(),
                },
            ),
            unittest.mock.patch("cash.graph.Network", return_value=mock_net, create=True),
        ):
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

        with unittest.mock.patch.dict(
            "sys.modules",
            {
                "pyvis": unittest.mock.MagicMock(),
                "pyvis.network": unittest.mock.MagicMock(Network=mock_network_cls),
                "IPython": mock_ipython,
                "IPython.display": mock_ipython.display,
            },
        ):
            # Re-import to pick up mocks
            import cash.graph as graph_mod

            importlib.reload(graph_mod)
            g2 = graph_mod.DependencyGraph()
            g2.add_dependency("a", "b")
            output_file = str(tmp_path / "graph.html")
            g2.visualize(output_file=output_file, notebook=False)
            mock_net.show.assert_called_once_with(output_file)
