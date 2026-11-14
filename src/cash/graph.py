"""Function dependency graph behind ``Cash.graph``."""

from __future__ import annotations

__all__ = ["DependencyGraph"]


class DependencyGraph:
    """Which cached functions and data sources each cached function depends on.

    Nodes are string identifiers (function keys or data-source ids). An edge
    ``A -> B`` records that A depends on B.
    """

    def __init__(self) -> None:
        self._dependencies: dict[str, set[str]] = {}

    def add_node(self, node: str) -> None:
        self._dependencies.setdefault(node, set())

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Record that *dependent* depends on *dependency*."""
        self.add_node(dependency)
        self._dependencies.setdefault(dependent, set()).add(dependency)

    def get_dependencies(self, node: str) -> set[str]:
        """The immediate dependencies of *node*."""
        return self._dependencies.get(node, set())

    def has_dependents(self, node: str) -> bool:
        """Does any other node depend on *node*?"""
        return any(node in deps for dependent, deps in self._dependencies.items() if dependent != node)

    def clear(self) -> None:
        self._dependencies.clear()
