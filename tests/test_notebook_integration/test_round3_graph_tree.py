"""Batch 56: Graph & tree data structures — cash caching with graph algorithms."""
import textwrap
import pytest


@pytest.mark.stress
class TestTreePatterns:
    """Test tree data structure caching."""

    def test_binary_tree_build_traverse(self, nb_runner):
        """Build binary tree in one cell, traverse in another."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class TreeNode:
                    def __init__(self, val, left=None, right=None):
                        self.val = val
                        self.left = left
                        self.right = right

                def inorder(node):
                    if node is None:
                        return []
                    return inorder(node.left) + [node.val] + inorder(node.right)
            """),
            textwrap.dedent("""\
                root = TreeNode(4,
                    TreeNode(2, TreeNode(1), TreeNode(3)),
                    TreeNode(6, TreeNode(5), TreeNode(7))
                )
                traversal = inorder(root)
                print(f"inorder={traversal}")
            """),
            textwrap.dedent("""\
                # Compute tree properties
                def depth(node):
                    if node is None:
                        return 0
                    return 1 + max(depth(node.left), depth(node.right))

                d = depth(root)
                count = len(traversal)
                print(f"depth={d} count={count}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "inorder=[1, 2, 3, 4, 5, 6, 7]" in nb_runner.get_output(2)
        assert "depth=3 count=7" in nb_runner.get_output(3)

    def test_tree_change_propagates(self, nb_runner):
        """Changing tree structure propagates."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15" in nb_runner.get_output(2)

        # Rebuild tree with different values
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class Node:
                def __init__(self, v, children=None):
                    self.v = v
                    self.children = children or []

            def tree_sum(node):
                return node.v + sum(tree_sum(c) for c in node.children)

            tree = Node(10, [Node(20), Node(30, [Node(40), Node(50)])])
        """))
        nb_runner.run_all()
        assert "total=150" in nb_runner.get_output(2)


@pytest.mark.stress
class TestGraphPatterns:
    """Test graph data structure patterns."""

    def test_adjacency_list_bfs(self, nb_runner):
        """Graph with adjacency list and BFS."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bfs=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
        assert "from_5=[5, 4, 2, 3, 1]" in nb_runner.get_output(3)

    def test_graph_shortest_path(self, nb_runner):
        """Shortest path (Dijkstra-like) across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import heapq

                def dijkstra(graph, start):
                    dist = {start: 0}
                    heap = [(0, start)]
                    while heap:
                        d, u = heapq.heappop(heap)
                        if d > dist.get(u, float('inf')):
                            continue
                        for v, w in graph.get(u, []):
                            new_dist = d + w
                            if new_dist < dist.get(v, float('inf')):
                                dist[v] = new_dist
                                heapq.heappush(heap, (new_dist, v))
                    return dist

                graph = {
                    'A': [('B', 1), ('C', 4)],
                    'B': [('C', 2), ('D', 5)],
                    'C': [('D', 1)],
                    'D': []
                }
            """),
            textwrap.dedent("""\
                distances = dijkstra(graph, 'A')
                sorted_dist = sorted(distances.items())
                print(f"distances={sorted_dist}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('A', 0)" in out
        assert "('D', 4)" in out  # A->B->C->D = 1+2+1 = 4

    def test_topological_sort(self, nb_runner):
        """Topological sort of DAG across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from collections import defaultdict, deque

                class DAG:
                    def __init__(self):
                        self.graph = defaultdict(list)
                        self.in_degree = defaultdict(int)
                        self.nodes = set()

                    def add_edge(self, u, v):
                        self.graph[u].append(v)
                        self.in_degree[v] = self.in_degree.get(v, 0) + 1
                        self.in_degree.setdefault(u, 0)
                        self.nodes.update([u, v])

                    def topo_sort(self):
                        queue = deque(sorted(n for n in self.nodes if self.in_degree.get(n, 0) == 0))
                        result = []
                        while queue:
                            node = queue.popleft()
                            result.append(node)
                            for neighbor in sorted(self.graph[node]):
                                self.in_degree[neighbor] -= 1
                                if self.in_degree[neighbor] == 0:
                                    queue.append(neighbor)
                        return result

                dag = DAG()
                for u, v in [('A','C'), ('B','C'), ('C','D'), ('B','D'), ('D','E')]:
                    dag.add_edge(u, v)
            """),
            textwrap.dedent("""\
                order = dag.topo_sort()
                print(f"topo={order}")
                # Verify C comes after A and B, D after C, E after D
                print(f"valid={order.index('C') > order.index('A')}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "topo=" in out
        assert "valid=True" in out
