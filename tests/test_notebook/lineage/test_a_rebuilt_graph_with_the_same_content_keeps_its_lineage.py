"""A networkx graph rebuilt to the same content takes the earlier one's lineage.

The same-value cutoff gave a frame, series or array that came out equal to
an earlier one its lineage back; a graph kept a new one. In a social-graph
notebook ``H = G.subgraph(giant).copy()`` came out the same after new data
rebuilt ``G``, and everything built on ``H`` ran again (118 s). A
``Graph``, ``DiGraph``, ``MultiGraph`` or ``MultiDiGraph`` now joins the
cutoff. Its digest covers all code can see of it, so any difference keeps
the new lineage: an edge, an attribute of a node, an edge or the graph, the
order of nodes or of edges, the graph's class.
"""

import uuid

import pytest

from cash.notebook.ipython.magics import CashMagics
from cash.notebook.value_cutoff import ValueCutoff
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

nx = pytest.importorskip("networkx")

EDGES = [(1, 2, {"w": 1.0}), (2, 3, {"w": 2.0}), (3, 4, {"w": 1.0})]


def _lineage():
    """A lineage of this test's own: a cutoff records its aliases for the
    whole process (`lineage_formula.record_lineage_alias`), so a lineage
    another test aliased would resolve to that test's."""
    return uuid.uuid4().hex * 2


def _graph(kind=None, edges=EDGES, nodes=(1, 2, 3, 4)):
    g = (kind or nx.Graph)()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def _with(change):
    g = _graph()
    change(g)
    return g


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_an_equal_graph_built_again_takes_the_first_ones_lineage(kind):
    a, b = _lineage(), _lineage()
    cutoff = ValueCutoff()
    assert cutoff.lineage_for(_graph(kind), a, compute_hash) == a
    assert cutoff.lineage_for(_graph(kind), b, compute_hash) == a


def test_a_graph_printed_or_looked_at_is_still_the_same():
    """networkx fills ``G.nodes``, ``G.adj`` and the like on first use:
    views, which hold nothing the graph does not."""
    a, b = _lineage(), _lineage()
    cutoff = ValueCutoff()
    cutoff.lineage_for(_graph(), a, compute_hash)
    looked_at = _graph()
    str(looked_at)
    list(looked_at.edges(data=True))
    dict(looked_at.degree())
    assert cutoff.lineage_for(looked_at, b, compute_hash) == a


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(lambda: _graph(edges=[*EDGES, (1, 4, {"w": 1.0})]), id="one_more_edge"),
        pytest.param(lambda: _graph(edges=EDGES[:2]), id="one_edge_less"),
        pytest.param(lambda: _graph(nodes=(1, 2, 3, 4, 5)), id="one_more_node"),
        pytest.param(lambda: _with(lambda g: g.nodes[2].update(region="north")), id="node_attribute"),
        pytest.param(lambda: _with(lambda g: g.edges[1, 2].update(w=1.5)), id="edge_attribute_value"),
        pytest.param(lambda: _with(lambda g: g.edges[1, 2].update(label="x")), id="edge_attribute_added"),
        pytest.param(lambda: _with(lambda g: g.graph.update(name="giant")), id="graph_attribute"),
        pytest.param(lambda: _with(lambda g: g.edges[1, 2].update(w=1)), id="int_for_equal_float"),
        pytest.param(lambda: _graph(nodes=(4, 3, 2, 1)), id="node_order"),
        pytest.param(lambda: _graph(edges=[EDGES[1], EDGES[0], EDGES[2]]), id="edge_order"),
        pytest.param(lambda: _graph(nx.DiGraph), id="digraph"),
        pytest.param(lambda: _graph(nx.MultiGraph), id="multigraph"),
        pytest.param(lambda: _graph(nx.MultiDiGraph), id="multidigraph"),
    ],
)
def test_a_graph_that_differs_keeps_its_own_lineage(other):
    """Edge order: node 2's neighbours are 3 then 1 instead of 1 then 3, the
    order ``G.adj[2]`` and every algorithm walking it sees."""
    a, b = _lineage(), _lineage()
    cutoff = ValueCutoff()
    cutoff.lineage_for(_graph(), a, compute_hash)
    assert cutoff.lineage_for(other(), b, compute_hash) == b


def test_a_graph_holding_what_cannot_be_hashed_keeps_its_own_lineage():
    """A digest that is only an identity says nothing of the content."""

    class Unpicklable:
        def __reduce__(self):
            raise TypeError("no")

    a, b = _lineage(), _lineage()
    cutoff = ValueCutoff()
    first = _with(lambda g: g.graph.update(handle=Unpicklable()))
    second = _with(lambda g: g.graph.update(handle=Unpicklable()))
    assert cutoff.lineage_for(first, a, compute_hash) == a
    assert cutoff.lineage_for(second, b, compute_hash) == b


SETUP = (
    "import time\n"
    "import networkx as nx\n"
    "def summary(g):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return g.number_of_edges()\n"
)
EDGES_CELL = "edges = [(i, i + 1) for i in range(50)]"
GRAPH = "G = nx.Graph(edges)"
LCC = "H = G.subgraph(max(nx.connected_components(G), key=len)).copy()  # @cash:persist"
SUMMARY = "s = summary(H)"
NAMES = ("edges", "G", "H", "s", "summary", "nx", "time")


def test_a_graph_rebuilt_equal_in_the_session_keeps_what_is_built_on_it(cash_magics, mock_shell):
    cells = [SETUP, EDGES_CELL, GRAPH, LCC, SUMMARY]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    # A pair outside the giant component: G changes, H does not.
    cells[1] = "edges = [(i, i + 1) for i in range(50)] + [(-1, -2)]"
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    state = cash_magics.tracking_state
    assert state.variable_lineage["G"] != first["G"]
    assert state.variable_lineage["H"] == first["H"]
    assert state.variable_lineage["s"] == first["s"]
    assert mock_shell.user_ns["s"] == 50


def test_a_graph_that_changed_in_the_session_takes_a_new_lineage(cash_magics, mock_shell):
    cells = [SETUP, EDGES_CELL, GRAPH, LCC, SUMMARY]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    cells[1] = "edges = [(i, i + 1) for i in range(50)] + [(0, 2)]"
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert cash_magics.tracking_state.variable_lineage["H"] != first["H"]
    assert mock_shell.user_ns["s"] == 51


def test_a_restored_graph_lends_its_lineage_to_an_equal_one(cash_magics, mock_shell, cash_instance):
    cells = [SETUP, EDGES_CELL, GRAPH, LCC, SUMMARY]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = dict(cash_magics.tracking_state.variable_lineage)

    # A restarted kernel over the same cache: a second CashMagics (the
    # fixtures' shell and Cash, an empty namespace), which has seen nothing.
    for name in NAMES:
        mock_shell.user_ns.pop(name, None)
    restarted = CashMagics(mock_shell, cash_instance)
    for cell in cells:
        run_cash_cell(restarted, cell, cells=cells)
    state = restarted.tracking_state
    assert state.variable_lineage["H"] == first["H"]  # restored, not run

    cells[1] = "edges = [(i, i + 1) for i in range(50)] + [(-1, -2)]"
    for cell in cells[1:]:
        run_cash_cell(restarted, cell, cells=cells)

    assert mock_shell.user_ns["s"] == 50
    assert state.variable_lineage["H"] == first["H"]
    assert state.variable_lineage["s"] == first["s"]
