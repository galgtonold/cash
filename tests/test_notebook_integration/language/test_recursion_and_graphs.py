"""Recursive functions, nested data and graph algorithms across cells."""

import textwrap

import pytest


# Recursive data structures, tree/graph traversal, and
# nested container patterns across cells.
class TestTreePatterns:
    """Test tree data structures across cells."""

    @pytest.mark.integration
    @pytest.mark.stress
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

    # Graph & tree data structures — cash caching with graph algorithms.
    @pytest.mark.stress
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


class TestGraphPatterns:
    """Test graph algorithms across cells."""

    @pytest.mark.integration
    @pytest.mark.stress
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

    @pytest.mark.integration
    @pytest.mark.stress
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

    @pytest.mark.stress
    def test_bfs_on_a_graph_class(self, nb_runner):
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


@pytest.mark.integration
@pytest.mark.stress
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


@pytest.mark.integration
@pytest.mark.stress
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


# Deep nesting and complex structure edit tests.
#
# Tests editing cells with deeply nested data structures, mixed types,
# and complex data patterns to verify proper cache invalidation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDeepNestingEdits:
    """Editing deeply nested and complex data structures."""

    def test_edit_nested_dict_value(self, nb_runner):
        """Edit a nested dict value at depth 2 and verify propagation."""
        nb_runner.create_notebook(
            [
                "config = {'db': {'host': 'localhost', 'port': 5432}, 'debug': True}",
                "host = config['db']['host']\nport = config['db']['port']\nprint(f'host={host} port={port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "host=localhost port=5432" in nb_runner.get_output(2)

        # Change host
        nb_runner.set_cell_source(1, "config = {'db': {'host': '10.0.0.1', 'port': 5432}, 'debug': True}")
        nb_runner.run_all()
        assert "host=10.0.0.1 port=5432" in nb_runner.get_output(2)

    def test_edit_list_of_records(self, nb_runner):
        """Edit a list of dicts (records pattern)."""
        nb_runner.create_notebook(
            [
                "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}]",
                "names = [r['name'] for r in records]\nprint(f'names = {names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(2)

        # Add a record
        nb_runner.set_cell_source(
            1,
            "records = [{'name': 'Alice', 'score': 90}, {'name': 'Bob', 'score': 85}, {'name': 'Charlie', 'score': 95}]",
        )
        nb_runner.run_all()
        assert "Charlie" in nb_runner.get_output(2)

    def test_edit_dict_with_tuple_keys(self, nb_runner):
        """Edit a dict with tuple keys."""
        nb_runner.create_notebook(
            [
                "grid = {(0, 0): 'X', (0, 1): 'O', (1, 0): '.'}",
                "val = grid.get((0, 0), '.')\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = X" in nb_runner.get_output(2)

        # Change the value
        nb_runner.set_cell_source(1, "grid = {(0, 0): 'O', (0, 1): 'O', (1, 0): '.'}")
        nb_runner.run_all()
        assert "val = O" in nb_runner.get_output(2)

    def test_edit_3_level_deep_nested(self, nb_runner):
        """Edit a deeply nested structure (3+ levels)."""
        nb_runner.create_notebook(
            [
                "tree = {'a': {'b': {'c': 42}}}",
                "leaf = tree['a']['b']['c']\nprint(f'leaf = {leaf}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "leaf = 42" in nb_runner.get_output(2)

        # Change deep value
        nb_runner.set_cell_source(1, "tree = {'a': {'b': {'c': 99}}}")
        nb_runner.run_all()
        assert "leaf = 99" in nb_runner.get_output(2)


# Nested data structure interaction tests.
#
# Tests where nested dicts, lists, and objects are modified
# across cells and dependencies must propagate correctly.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNestedDictEdits:
    """Edit cells producing/consuming nested dicts."""

    def test_edit_nested_key(self, nb_runner):
        """Change a nested dict key."""
        nb_runner.create_notebook(
            [
                "config = {'db': {'host': 'localhost', 'port': 5432}}",
                "addr = f\"{config['db']['host']}:{config['db']['port']}\"\nprint(f'addr = {addr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr = localhost:5432" in nb_runner.get_output(2)

        # Change the host
        nb_runner.set_cell_source(1, "config = {'db': {'host': '10.0.0.1', 'port': 5432}}")
        nb_runner.run_all()
        assert "addr = 10.0.0.1:5432" in nb_runner.get_output(2)

    def test_add_nested_level(self, nb_runner):
        """Add a deeper nesting level."""
        nb_runner.create_notebook(
            [
                "data = {'a': 1}",
                "total = sum(v if isinstance(v, int) else sum(v.values()) for v in data.values())\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 1" in nb_runner.get_output(2)

        # Add nested dict
        nb_runner.set_cell_source(1, "data = {'a': 1, 'b': {'x': 10, 'y': 20}}")
        nb_runner.run_all()
        assert "total = 31" in nb_runner.get_output(2)

    def test_edit_dict_comprehension_source(self, nb_runner):
        """Edit dict comprehension inputs."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']\nvals = [1, 2, 3]",
                "mapping = dict(zip(keys, vals))\nprint(f'mapping = {mapping}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "keys = ['x', 'y', 'z']\nvals = [10, 20, 30]")
        nb_runner.run_all()
        assert "'x': 10" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNestedListEdits:
    """Edit cells producing/consuming nested lists."""

    def test_nested_list_processing_chain(self, nb_runner):
        """Chain of nested list operations."""
        nb_runner.create_notebook(
            [
                "raw = [[1, 2, 3], [4, 5, 6]]",
                "sums = [sum(row) for row in raw]",
                "total = sum(sums)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "raw = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "total = 100" in nb_runner.get_output(3)


# Recursive data structure interaction tests.
#
# Tests editing cells with recursive data structures
# (trees, linked lists) and verifying propagation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestRecursiveDataEdits:
    """Editing recursive data structure patterns."""

    def test_edit_tree_data(self, nb_runner):
        """Edit a nested dict tree and check traversal."""
        nb_runner.create_notebook(
            [
                "tree = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}",
                "def collect(node):\n    if node is None:\n        return []\n    return [node['val']] + collect(node['left']) + collect(node['right'])\nvals = collect(tree)\nprint(f'vals = {vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 2, 3]" in nb_runner.get_output(2)

        # Edit tree
        nb_runner.set_cell_source(
            1, "tree = {'val': 10, 'left': {'val': 20, 'left': None, 'right': None}, 'right': None}"
        )
        nb_runner.run_all()
        assert "vals = [10, 20]" in nb_runner.get_output(2)

    def test_edit_nested_dict_depth(self, nb_runner):
        """Edit depth of nested dictionary."""
        nb_runner.create_notebook(
            [
                "data = {'a': {'b': {'c': 42}}}",
                "def depth(d, level=0):\n    if not isinstance(d, dict):\n        return level\n    return max(depth(v, level + 1) for v in d.values())\nresult = depth(data)\nprint(f'depth = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "depth = 3" in nb_runner.get_output(2)

        # Deepen
        nb_runner.set_cell_source(1, "data = {'a': {'b': {'c': {'d': {'e': 99}}}}}")
        nb_runner.run_all()
        assert "depth = 5" in nb_runner.get_output(2)

    def test_edit_flat_list_to_tree(self, nb_runner):
        """Edit flat list that gets converted to nested pairs."""
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3, 4]",
                "def nest(lst):\n    if len(lst) <= 1:\n        return lst[0] if lst else None\n    mid = len(lst) // 2\n    return (nest(lst[:mid]), nest(lst[mid:]))\nresult = nest(items)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ((1, 2), (3, 4))" in nb_runner.get_output(2)

        # Change items
        nb_runner.set_cell_source(1, "items = [10, 20]")
        nb_runner.run_all()
        assert "result = (10, 20)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRecursiveDataStruct:
    """recursive data structures (trees, nested dicts)."""

    def test_tree_traversal(self, nb_runner):
        nb_runner.create_notebook(
            [
                "tree = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}",
                "def inorder(node):\n    if node is None:\n        return []\n    return inorder(node['left']) + [node['val']] + inorder(node['right'])\nresult = inorder(tree)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[2, 1, 3]" in nb_runner.get_output(2)

    def test_nested_dict_flatten(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}",
                "def flatten_dict(d, prefix=''):\n    items = {}\n    for k, v in d.items():\n        new_key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            items.update(flatten_dict(v, new_key))\n        else:\n            items[new_key] = v\n    return items\nresult = flatten_dict(nested)\nprint(f'result={dict(sorted(result.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        assert "'b.c': 2" in nb_runner.get_output(2)
        assert "'b.d.e': 3" in nb_runner.get_output(2)

    def test_recursive_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def depth(obj):\n    if not isinstance(obj, (list, dict)):\n        return 0\n    if isinstance(obj, list):\n        return 1 + max((depth(x) for x in obj), default=0)\n    return 1 + max((depth(v) for v in obj.values()), default=0)",
                "data = {'a': [1, [2, [3]]]}",
                "d = depth(data)\nprint(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=4" in nb_runner.get_output(3)
        # Edit data
        nb_runner.set_cell_source(2, "data = {'a': 1}")
        nb_runner.run_all()
        assert "d=1" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRecursiveFibTree:
    """recursive functions fibonacci and tree."""

    def test_fibonacci_memo(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import lru_cache",
                "@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\nresults = [fib(i) for i in range(10)]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]" in nb_runner.get_output(2)

    def test_recursive_flatten(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "def depth(lst):\n    if not isinstance(lst, list): return 0\n    if not lst: return 1\n    return 1 + max(depth(item) for item in lst)\ntree = [1, [2, [3, [4]]]]\nprint(f'depth={depth(tree)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "depth=4" in nb_runner.get_output(2)

    def test_recursive_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pass  # setup",
                "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nresult = factorial(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=120" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nresult = factorial(7)\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=5040" in nb_runner.get_output(2)


# Recursive data structure traversal interaction tests.
# Tests tree, linked list, and nested dict traversal with cache invalidation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRecursiveTraversalInteraction:
    """Test recursive traversal patterns with cache invalidation."""

    def test_tree_sum_edit(self, nb_runner):
        """Editing a tree structure should propagate to traversal results."""
        nb_runner.create_notebook(
            [
                (
                    "class TNode:\n"
                    "    def __init__(self, val, children=None):\n"
                    "        self.val = val\n"
                    "        self.children = children or []"
                ),
                "tree = TNode(1, [TNode(2, [TNode(4)]), TNode(3, [TNode(5)])])",
                (
                    "def tree_sum(node):\n"
                    "    return node.val + sum(tree_sum(c) for c in node.children)\n"
                    "total = tree_sum(tree)"
                ),
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=15" in out

        nb_runner.set_cell_source(2, "tree = TNode(10, [TNode(20, [TNode(40)]), TNode(30, [TNode(50)])])")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=150" in out

    def test_linked_list_sum_edit(self, nb_runner):
        """Editing a linked list should propagate to sum computation."""
        nb_runner.create_notebook(
            [
                (
                    "class LNode:\n"
                    "    def __init__(self, val, nxt=None):\n"
                    "        self.val = val\n"
                    "        self.nxt = nxt"
                ),
                "head = LNode(1, LNode(2, LNode(3)))",
                (
                    "def lsum(node):\n"
                    "    t = 0\n"
                    "    while node:\n"
                    "        t += node.val\n"
                    "        node = node.nxt\n"
                    "    return t\n"
                    "s = lsum(head)"
                ),
                "print(f'sum={s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sum=6" in out

        nb_runner.set_cell_source(2, "head = LNode(10, LNode(20, LNode(30, LNode(40))))")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sum=100" in out

    def test_nested_dict_flatten_edit(self, nb_runner):
        """Editing nested dict should propagate through recursive flattening."""
        nb_runner.create_notebook(
            [
                "nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}",
                (
                    "def flatten(d, prefix=''):\n"
                    "    result = {}\n"
                    "    for k, v in d.items():\n"
                    "        key = f'{prefix}.{k}' if prefix else k\n"
                    "        if isinstance(v, dict):\n"
                    "            result.update(flatten(v, key))\n"
                    "        else:\n"
                    "            result[key] = v\n"
                    "    return result\n"
                    "flat = flatten(nested)"
                ),
                "keys = sorted(flat.keys())\nvals = [flat[k] for k in keys]",
                "print(f'keys={keys},vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "keys=['a', 'b.c', 'b.d.e']" in out
        assert "vals=[1, 2, 3]" in out

        nb_runner.set_cell_source(1, "nested = {'x': 10, 'y': {'z': 20}}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "keys=['x', 'y.z']" in out
        assert "vals=[10, 20]" in out


# graph algorithms: BFS, DFS, shortest path.
@pytest.mark.stress
@pytest.mark.integration
class TestGraphAlgorithms:
    """Graph algorithm patterns."""

    def test_bfs(self, nb_runner):
        """Breadth-first search traversal."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from collections import deque

                def bfs(graph, start):
                    visited = []
                    queue = deque([start])
                    seen = {start}
                    while queue:
                        node = queue.popleft()
                        visited.append(node)
                        for neighbor in sorted(graph.get(node, [])):
                            if neighbor not in seen:
                                seen.add(neighbor)
                                queue.append(neighbor)
                    return visited

                graph = {
                    'A': ['B', 'C'],
                    'B': ['A', 'D', 'E'],
                    'C': ['A', 'F'],
                    'D': ['B'],
                    'E': ['B', 'F'],
                    'F': ['C', 'E'],
                }
                order = bfs(graph, 'A')
            """),
                "print(f'bfs={order}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "bfs=" in out
        assert out.index("A") < out.index("D")  # A before D in BFS

    def test_topological_sort(self, nb_runner):
        """Topological sort of a DAG."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def topo_sort(deps):
                    # deps maps task -> list of prerequisites
                    # Build adjacency: prerequisite -> tasks that depend on it
                    all_nodes = set(deps.keys())
                    for prereqs in deps.values():
                        all_nodes.update(prereqs)
                    adj = {n: [] for n in all_nodes}
                    in_degree = {n: 0 for n in all_nodes}
                    for task, prereqs in deps.items():
                        for p in prereqs:
                            adj[p].append(task)
                            in_degree[task] += 1
                    queue = sorted([n for n in in_degree if in_degree[n] == 0])
                    result = []
                    while queue:
                        u = queue.pop(0)
                        result.append(u)
                        for v in sorted(adj[u]):
                            in_degree[v] -= 1
                            if in_degree[v] == 0:
                                queue.append(v)
                                queue.sort()
                    return result

                deps = {
                    'install': [],
                    'build': ['install'],
                    'test': ['build'],
                    'lint': ['install'],
                    'deploy': ['test', 'lint'],
                }
                order = topo_sort(deps)
            """),
                "print(f'order={order}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "order=" in out
        # Parse the list from output: order=['install', 'lint', 'build', 'test', 'deploy']
        import ast

        order_str = out.strip().split("order=")[1]
        order_list = ast.literal_eval(order_str)
        # install must come before build, build before test, test before deploy
        assert order_list.index("install") < order_list.index("build")
        assert order_list.index("build") < order_list.index("test")
        assert order_list.index("test") < order_list.index("deploy")

    def test_graph_propagation(self, nb_runner):
        """Graph with upstream edge change propagation."""
        nb_runner.create_notebook(
            [
                "edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]",
                textwrap.dedent("""\
                from collections import defaultdict
                graph = defaultdict(list)
                for u, v in edges:
                    graph[u].append(v)
                reachable = set()
                stack = ['A']
                while stack:
                    node = stack.pop()
                    if node in reachable:
                        continue
                    reachable.add(node)
                    stack.extend(graph[node])
                reachable = sorted(reachable)
            """),
                "print(f'reachable={reachable}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['A', 'B', 'C', 'D']" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "edges = [('A', 'B'), ('A', 'C')]")
        nb_runner.run_cells([1, 2, 3])
        out = nb_runner.get_output(3)
        assert "A" in out
        assert "B" in out
        assert "C" in out
        assert "D" not in out  # D no longer reachable


# Recursive function edit propagation.
#
# Tests editing recursive functions and verifying downstream re-evaluation.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestRecursiveFunctionEdit:
    """Edit recursive functions, verify downstream propagation."""

    def test_recursive_sum_edit(self, nb_runner):
        """Edit recursive list sum approach."""
        nb_runner.create_notebook(
            [
                "def rsum(lst):\n    if not lst:\n        return 0\n    return lst[0] + rsum(lst[1:])",
                "data = [10, 20, 30, 40]\nval = rsum(data)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 100" in nb_runner.get_output(2)

        # Change to product
        nb_runner.set_cell_source(
            1,
            "def rsum(lst):\n    if not lst:\n        return 1\n    return lst[0] * rsum(lst[1:])",
        )
        nb_runner.run_all()
        # 10*20*30*40 = 240000
        assert "val = 240000" in nb_runner.get_output(2)
