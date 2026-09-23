"""Graph & tree data structures — cash caching with graph algorithms."""

import textwrap

import pytest


@pytest.mark.stress
class TestTreePatterns:
    """Test tree data structure caching."""

    def test_tree_change_propagates(self, nb_runner):
        """Changing tree structure propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Node:
                    def __init__(self, v, children=None):
                        self.v = v
                        self.children = children or []

                def tree_sum(node):
                    return node.v + sum(tree_sum(c) for c in node.children)

                tree = Node(1, [Node(2), Node(3, [Node(4), Node(5)])])
            """),
                textwrap.dedent("""\
                total = tree_sum(tree)
                print(f"total={total}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15" in nb_runner.get_output(2)

        # Rebuild tree with different values
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Node:
                def __init__(self, v, children=None):
                    self.v = v
                    self.children = children or []

            def tree_sum(node):
                return node.v + sum(tree_sum(c) for c in node.children)

            tree = Node(10, [Node(20), Node(30, [Node(40), Node(50)])])
        """),
        )
        nb_runner.run_all()
        assert "total=150" in nb_runner.get_output(2)


@pytest.mark.stress
class TestGraphPatterns:
    """Test graph data structure patterns."""

    def test_adjacency_list_bfs(self, nb_runner):
        """Graph with adjacency list and BFS."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from collections import deque

                class Graph:
                    def __init__(self):
                        self.adj = {}

                    def add_edge(self, u, v):
                        self.adj.setdefault(u, []).append(v)
                        self.adj.setdefault(v, []).append(u)

                    def bfs(self, start):
                        visited = set()
                        queue = deque([start])
                        order = []
                        while queue:
                            node = queue.popleft()
                            if node in visited:
                                continue
                            visited.add(node)
                            order.append(node)
                            for neighbor in sorted(self.adj.get(node, [])):
                                if neighbor not in visited:
                                    queue.append(neighbor)
                        return order
            """),
                textwrap.dedent("""\
                g = Graph()
                for u, v in [(1,2), (1,3), (2,4), (3,4), (4,5)]:
                    g.add_edge(u, v)
                bfs_order = g.bfs(1)
                print(f"bfs={bfs_order}")
            """),
                textwrap.dedent("""\
                # Use same graph for different start
                bfs_from_5 = g.bfs(5)
                print(f"from_5={bfs_from_5}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bfs=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
        assert "from_5=[5, 4, 2, 3, 1]" in nb_runner.get_output(3)
