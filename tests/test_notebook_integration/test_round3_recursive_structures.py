"""
Recursive data structures, tree/graph traversal, and
nested container patterns across cells.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestTreePatterns:
    """Test tree data structures across cells."""

    def test_tree_modification_propagates(self, nb_runner):
        """Modify tree structure → traversal changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class TreeNode:
                    def __init__(self, v, children=None):
                        self.v = v
                        self.children = children or []
                    def leaf_values(self):
                        if not self.children:
                            return [self.v]
                        result = []
                        for c in self.children:
                            result.extend(c.leaf_values())
                        return result
            """),
                textwrap.dedent("""\
                tree = TreeNode('root', [
                    TreeNode('a', [TreeNode('x'), TreeNode('y')]),
                    TreeNode('b', [TreeNode('z')])
                ])
            """),
                textwrap.dedent("""\
                leaves = tree.leaf_values()
                print(leaves)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['x', 'y', 'z']" in nb_runner.get_output(3)

        # Add a new branch
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            tree = TreeNode('root', [
                TreeNode('a', [TreeNode('x'), TreeNode('y')]),
                TreeNode('b', [TreeNode('z')]),
                TreeNode('c', [TreeNode('w1'), TreeNode('w2')])
            ])
        """),
        )
        nb_runner.run_all()
        assert "['x', 'y', 'z', 'w1', 'w2']" in nb_runner.get_output(3)


class TestGraphPatterns:
    """Test graph algorithms across cells."""

    def test_adjacency_list_bfs(self, nb_runner):
        """BFS on adjacency list across cells."""
        nb_runner.create_notebook(
            [
                "from collections import deque",
                textwrap.dedent("""\
                graph = {
                    'A': ['B', 'C'],
                    'B': ['D'],
                    'C': ['D', 'E'],
                    'D': [],
                    'E': []
                }
            """),
                textwrap.dedent("""\
                def bfs(graph, start):
                    visited = []
                    queue = deque([start])
                    seen = {start}
                    while queue:
                        node = queue.popleft()
                        visited.append(node)
                        for neighbor in graph.get(node, []):
                            if neighbor not in seen:
                                seen.add(neighbor)
                                queue.append(neighbor)
                    return visited
            """),
                textwrap.dedent("""\
                order = bfs(graph, 'A')
                print(order)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "'A'" in output
        # BFS visits A first, then B, C, then D, E
        assert output.index("'A'") < output.index("'B'")

    def test_graph_change_propagation(self, nb_runner):
        """Change graph edges → BFS order changes."""
        nb_runner.create_notebook(
            [
                "from collections import deque",
                textwrap.dedent("""\
                graph = {'A': ['B'], 'B': ['C'], 'C': []}
            """),
                textwrap.dedent("""\
                def bfs(g, start):
                    visited = []
                    q = deque([start])
                    seen = {start}
                    while q:
                        n = q.popleft()
                        visited.append(n)
                        for nb in g.get(n, []):
                            if nb not in seen:
                                seen.add(nb)
                                q.append(nb)
                    return visited
            """),
                textwrap.dedent("""\
                print(bfs(graph, 'A'))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['A', 'B', 'C']" in nb_runner.get_output(4)

        # Add edge A→C directly
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            graph = {'A': ['B', 'C'], 'B': [], 'C': ['D'], 'D': []}
        """),
        )
        nb_runner.run_all()
        assert "['A', 'B', 'C', 'D']" in nb_runner.get_output(4)


class TestNestedContainers:
    """Test deeply nested containers across cells."""

    def test_nested_dict_access(self, nb_runner):
        """Deeply nested dict accessed across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                config = {
                    'database': {
                        'host': 'localhost',
                        'port': 5432,
                        'credentials': {
                            'user': 'admin',
                            'password': 'secret'
                        }
                    },
                    'cache': {
                        'ttl': 300
                    }
                }
            """),
                textwrap.dedent("""\
                host = config['database']['host']
                user = config['database']['credentials']['user']
                ttl = config['cache']['ttl']
                print(f"host={host} user={user} ttl={ttl}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost user=admin ttl=300" in nb_runner.get_output(2)


class TestRecursiveAlgorithms:
    """Test recursive algorithms defined and used across cells."""

    def test_fibonacci_memo(self, nb_runner):
        """Memoized fibonacci across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                memo = {}
                def fib(n):
                    if n in memo:
                        return memo[n]
                    if n <= 1:
                        return n
                    memo[n] = fib(n-1) + fib(n-2)
                    return memo[n]
            """),
                textwrap.dedent("""\
                result = fib(30)
                print(f"fib(30)={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib(30)=832040" in nb_runner.get_output(2)
