"""Variable dependency graph for tracking lineage relationships.

This module is the primary home for :class:`DependencyGraph`.
``cash.ui.graph`` re-exports it for backwards compatibility.
"""

from __future__ import annotations

__all__ = ["DependencyGraph"]



class DependencyGraph:
    """
    Directed Acyclic Graph to track dependencies between functions and data sources.
    Nodes are string identifiers (e.g., function names or file paths).
    Edges represent dependencies: A -> B means A depends on B.
    """

    def __init__(self) -> None:
        # Adjacency list: node -> list of dependencies (children)
        # If A depends on B, we store A -> [B] ?
        # Wait, for invalidation: if B changes, we need to invalidate A.
        # So we need to traverse from B to A.
        # Let's store: node -> list of dependents (parents)
        # If A depends on B, we add an edge B -> A.
        # So when B changes, we can find A.
        self._dependents: dict[str, set[str]] = {}
        self._dependencies: dict[str, set[str]] = {}  # Inverse for convenience

    def add_node(self, node: str) -> None:
        if node not in self._dependents:
            self._dependents[node] = set()
        if node not in self._dependencies:
            self._dependencies[node] = set()

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """
        Record that 'dependent' depends on 'dependency'.
        When 'dependency' changes, 'dependent' is invalid.
        """
        self.add_node(dependent)
        self.add_node(dependency)

        self._dependents[dependency].add(dependent)
        self._dependencies[dependent].add(dependency)

    def get_dependents(self, node: str) -> set[str]:
        """Get immediate dependents of a node (parents)."""
        return self._dependents.get(node, set())

    def get_dependencies(self, node: str) -> set[str]:
        """Get immediate dependencies of a node (children)."""
        return self._dependencies.get(node, set())

    def get_all_dependents(self, node: str) -> set[str]:
        """Get all transitive dependents of a node (cascading)."""
        visited = set()
        stack = [node]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            for dependent in self._dependents.get(current, []):
                stack.append(dependent)

        visited.remove(node)  # Exclude the node itself
        return visited

    def clear(self) -> None:
        self._dependents.clear()
        self._dependencies.clear()

    def visualize(self, output_file: str = "dependency_graph.html", notebook: bool = True) -> None:
        """
        Visualize the dependency graph using pyvis.

        Args:
            output_file: Path to save the HTML file.
            notebook: If True, return the HTML for display in Jupyter.
        """
        try:
            import IPython.display
            from pyvis.network import Network
        except ImportError:
            print("Please install pyvis to use visualization: pip install pyvis")
            return None

        # cdn_resources='in_line' ensures JS is embedded, avoiding 404s for local files
        net = Network(height="600px", width="100%", notebook=True, directed=True, cdn_resources='in_line')

        # Add nodes
        for node in self._dependencies:
            color = "#97c2fc"  # Default blue
            if "/" in node or "\\" in node or "." in node and not node.isidentifier():
                color = "#ffb3b3"  # Red for files?

            net.add_node(node, label=node, color=color)

        # Add edges
        # Data Flow: Dependency -> Dependent.
        for dependency, dependents in self._dependents.items():
            for dependent in dependents:
                net.add_edge(dependency, dependent)

        if notebook:
            # Generate HTML string
            # pyvis show() writes to file. generate_html() returns string.
            html_content = net.generate_html()

            # Use a data URI to embed the HTML content in an IFrame
            # This avoids issues with srcdoc length limits and escaping,
            # and ensures isolation from the notebook's CSS.
            import base64
            html_base64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            data_uri = f"data:text/html;base64,{html_base64}"

            return IPython.display.IFrame(src=data_uri, width="100%", height="650px")
        net.show(output_file)
        print(f"Graph saved to {output_file}")
        return None
