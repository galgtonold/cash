"""Batch 97 – graph algorithms: BFS, DFS, shortest path."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestGraphAlgorithms:
    """Graph algorithm patterns."""

    def test_bfs(self, nb_runner):
        """Breadth-first search traversal."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "bfs=" in out
        assert out.index("A") < out.index("D")  # A before D in BFS

    def test_dfs(self, nb_runner):
        """Depth-first search traversal."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def dfs(graph, start):
                    visited = []
                    stack = [start]
                    seen = set()
                    while stack:
                        node = stack.pop()
                        if node in seen:
                            continue
                        seen.add(node)
                        visited.append(node)
                        for neighbor in sorted(graph.get(node, []), reverse=True):
                            if neighbor not in seen:
                                stack.append(neighbor)
                    return visited

                graph = {
                    'A': ['B', 'C'],
                    'B': ['D', 'E'],
                    'C': ['F'],
                    'D': [],
                    'E': ['F'],
                    'F': [],
                }
                order = dfs(graph, 'A')
            """),
            "print(f'dfs={order}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "dfs=" in out
        assert "A" in out
        assert "F" in out

    def test_dijkstra(self, nb_runner):
        """Dijkstra's shortest path algorithm."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import heapq

                def dijkstra(graph, start):
                    dist = {node: float('inf') for node in graph}
                    dist[start] = 0
                    prev = {node: None for node in graph}
                    heap = [(0, start)]
                    while heap:
                        d, u = heapq.heappop(heap)
                        if d > dist[u]:
                            continue
                        for v, w in graph[u]:
                            alt = dist[u] + w
                            if alt < dist[v]:
                                dist[v] = alt
                                prev[v] = u
                                heapq.heappush(heap, (alt, v))
                    return dist, prev

                graph = {
                    'A': [('B', 1), ('C', 4)],
                    'B': [('C', 2), ('D', 5)],
                    'C': [('D', 1)],
                    'D': [],
                }
                distances, _ = dijkstra(graph, 'A')
            """),
            "print(f'dist={distances}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': 0" in out
        assert "'B': 1" in out
        assert "'C': 3" in out  # A→B→C = 1+2
        assert "'D': 4" in out  # A→B→C→D = 1+2+1

    def test_topological_sort(self, nb_runner):
        """Topological sort of a DAG."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "order=" in out
        # Parse the list from output: order=['install', 'lint', 'build', 'test', 'deploy']
        import ast
        order_str = out.strip().split("order=")[1]
        order_list = ast.literal_eval(order_str)
        # install must come before build, build before test, test before deploy
        assert order_list.index('install') < order_list.index('build')
        assert order_list.index('build') < order_list.index('test')
        assert order_list.index('test') < order_list.index('deploy')

    def test_graph_propagation(self, nb_runner):
        """Graph with upstream edge change propagation."""
        nb_runner.create_notebook([
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
        ])
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
