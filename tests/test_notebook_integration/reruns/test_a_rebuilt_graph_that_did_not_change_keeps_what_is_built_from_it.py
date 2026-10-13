"""A networkx graph rebuilt to the same content keeps what was cached from it.

In a social-graph notebook a new day of interactions rebuilt ``G``, and
``H = G.subgraph(giant).copy()`` came out the same giant component as
before. Its lineage named the new ``G``, so betweenness, Louvain and label
propagation on ``H`` ran again: about 118 s after the second restart. A
graph a statement executed to bind now takes the lineage of an earlier one
with the same content, as a frame does.

Counted with ``os.write`` from inside the function built on ``H``.
"""

from pathlib import Path

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("networkx")

pytestmark = [pytest.mark.integration]

ON = "import cash\n%cash_on"
SETUP = (
    "import os, time\n"
    "import networkx as nx\n"
    "def slow_summary(g):\n"
    "    fd = os.open('runs.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "    time.sleep(0.2)\n"
    "    w = sum(d['w'] for _, _, d in g.edges(data=True))\n"
    "    r = sum(d['r'] for _, d in g.nodes(data=True))\n"
    "    return (type(g).__name__, g.number_of_edges(), w, r)"
)
EDGES = "edges = [(i, i + 1, {'w': i % 3}) for i in range(2000)]"
GRAPH = "G = nx.Graph(edges)\nnx.set_node_attributes(G, {n: n % 5 for n in G}, 'r')"
LCC = "H = G.subgraph(max(nx.connected_components(G), key=len)).copy()"
SUMMARY = "summary = slow_summary(H)\nprint('SUMMARY', summary)"
#: ``edges`` with a pair outside the giant component: ``G`` changes, ``H`` does not.
MORE_EDGES = "edges = [(i, i + 1, {'w': i % 3}) for i in range(2000)] + [(-1, -2, {'w': 0})]"
FIRST = "SUMMARY ('Graph', 2000, 1999, 4000)"


def _runs(runner) -> int:
    log = Path(runner.work_dir) / "runs.log"
    return log.read_text(encoding="utf-8").count("x") if log.exists() else 0


def _notebook(lcc: str = LCC) -> list[str]:
    return [ON, SETUP, EDGES, GRAPH, lcc, SUMMARY]


def test_a_graph_rebuilt_to_the_same_content_keeps_what_is_built_from_it(nb_runner):
    nb_runner.create_notebook(_notebook())
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert FIRST in nb_runner.get_output(6)
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, MORE_EDGES)
    nb_runner.run_all()
    assert FIRST in nb_runner.get_output(6)
    assert _runs(nb_runner) == 1, "a graph rebuilt to the same content ran what was built on it again"

    # After a restart the alias kept in the cache folder names the rebuilt
    # graph by the first one's lineage: still served.
    nb_runner.restart()
    nb_runner.run_all()
    assert FIRST in nb_runner.get_output(6)
    assert _runs(nb_runner) == 1


@pytest.mark.parametrize(
    ("edited", "expected"),
    [
        # One more edge inside the giant component.
        (MORE_EDGES.replace("(-1, -2, {'w': 0})", "(0, 2, {'w': 0})"), "SUMMARY ('Graph', 2001, 1999, 4000)"),
        # One edge attribute.
        (MORE_EDGES.replace("[(-1, -2, {'w': 0})]", "[(5, 6, {'w': 7})]"), "SUMMARY ('Graph', 2000, 2004, 4000)"),
    ],
    ids=["edge", "edge_attribute"],
)
def test_a_graph_that_changed_runs_what_is_built_from_it(nb_runner, edited, expected):
    nb_runner.create_notebook(_notebook())
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, edited)
    nb_runner.run_all()

    assert expected in nb_runner.get_output(6)
    assert _runs(nb_runner) == 2


def test_a_node_attribute_or_graph_type_change_runs_what_is_built_from_it(nb_runner):
    nb_runner.create_notebook(_notebook())
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(4, GRAPH.replace("n % 5", "n % 5 + (n == 7)"))
    nb_runner.run_all()
    assert "SUMMARY ('Graph', 2000, 1999, 4001)" in nb_runner.get_output(6)
    assert _runs(nb_runner) == 2

    nb_runner.set_cell_source(4, GRAPH.replace("nx.Graph(edges)", "nx.MultiGraph(edges)"))
    nb_runner.run_all()
    assert "SUMMARY ('MultiGraph', 2000, 1999, 4000)" in nb_runner.get_output(6)
    assert _runs(nb_runner) == 3


def test_a_graph_rebuilt_equal_to_a_restored_one_keeps_what_is_built_from_it(nb_runner):
    """After a restart ``H`` comes back from the cache, not from running.
    Rebuilt to the same content from a ``G`` that did change, it takes the
    restored one's lineage: the entry kept the digest ``H`` had when it ran."""
    cells = _notebook("# @cash:persist\n" + LCC)
    cells[0] = ON + "\n%cash_badge print"
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _runs(nb_runner) == 1

    nb_runner.restart()
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(5)), nb_runner.get_output(5)
    assert _runs(nb_runner) == 1

    nb_runner.set_cell_source(3, MORE_EDGES)
    nb_runner.run_all()
    assert FIRST in nb_runner.get_output(6)
    assert _runs(nb_runner) == 1, "a graph rebuilt equal to a restored one ran what was built on it again"
