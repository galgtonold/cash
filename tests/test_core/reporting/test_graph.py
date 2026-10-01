"""Tests for DependencyGraph in cash.graph."""

from cash.graph import DependencyGraph


class TestDependencyGraph:
    """Unit tests for the DependencyGraph class."""

    def test_add_node(self):
        g = DependencyGraph()
        g.add_node("A")
        assert g.get_dependencies("A") == set()

    def test_add_dependency(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")  # A depends on B
        assert g.get_dependencies("A") == {"B"}
        assert g.get_dependencies("B") == set()

    def test_get_dependencies_unknown_node(self):
        g = DependencyGraph()
        assert g.get_dependencies("nonexistent") == set()

    def test_multiple_dependencies(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("A", "C")
        assert g.get_dependencies("A") == {"B", "C"}

    def test_dependencies_are_immediate(self):
        g = DependencyGraph()
        g.add_dependency("C", "B")
        g.add_dependency("B", "A")
        assert g.get_dependencies("C") == {"B"}

    def test_add_node_keeps_existing_edges(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_node("A")
        assert g.get_dependencies("A") == {"B"}

    def test_clear(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("B", "C")
        g.clear()
        assert g.get_dependencies("A") == set()
        assert g.get_dependencies("B") == set()

    def test_duplicate_dependency_idempotent(self):
        g = DependencyGraph()
        g.add_dependency("A", "B")
        g.add_dependency("A", "B")
        assert g.get_dependencies("A") == {"B"}
