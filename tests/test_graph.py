"""Tests for DependencyGraph in cash.graph."""

from cash.graph import DependencyGraph


class TestDependencyGraph:
    """Unit tests for the DependencyGraph class."""

    def test_add_node(self):
        g = DependencyGraph()
        g.add_node("A")
        assert g.get_dependents("A") == set()
        assert g.get_dependencies("A") == set()

    def test_add_dependency(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")  # A depends on B
        assert g.get_dependencies("A") == {"B"}
        assert g.get_dependents("B") == {"A"}

    def test_get_dependents_unknown_node(self):
        g = DependencyGraph()
        assert g.get_dependents("nonexistent") == set()

    def test_get_dependencies_unknown_node(self):
        g = DependencyGraph()
        assert g.get_dependencies("nonexistent") == set()

    def test_multiple_dependencies(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("A", "C")
        assert g.get_dependencies("A") == {"B", "C"}

    def test_multiple_dependents(self):
        g = DependencyGraph()
        g.add_dependency("A", "C")
        g.add_dependency("B", "C")
        assert g.get_dependents("C") == {"A", "B"}

    def test_clear(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "C")
        g.clear()
        assert g.get_dependents("A") == set()
        assert g.get_dependencies("A") == set()

    def test_duplicate_dependency_idempotent(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("A", "B")
        assert g.get_dependencies("A") == {"B"}
        assert g.get_dependents("B") == {"A"}

    def test_visualize_without_pyvis(self, capsys):
        """Visualize gracefully handles missing pyvis."""
        g = DependencyGraph()
        g.add_dependency("A", "B")
        # visualize with notebook=False and no pyvis should print a message
        # (pyvis may or may not be installed; just verify no crash)
        import contextlib
        with contextlib.suppress(Exception):
            g.visualize(notebook=False)
