"""``print(G)`` is not an in-place change of a networkx graph.

networkx fills ``G.degree`` (a ``functools.cached_property``) when
``print(G)`` asks for the edge count, so G's pickle grew a view and the
badge said ``In-place mutation on: G (observed; receiver lineage bumped)``
for a statement that changed nothing. A view holds nothing the graph does
not; adding an edge is still a change.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

nx = pytest.importorskip("networkx")

BUILD = "import networkx as nx\nG = nx.gnm_random_graph(60, 120, seed=1)"


def _reasons(metrics) -> str:
    return " ".join(metrics.get("uncacheable_reasons") or ())


def test_printing_a_graph_does_not_change_it(cash_magics, statement_processor):
    run_cash_cell(cash_magics, BUILD, cells=[BUILD])
    lineage = cash_magics.tracking_state.variable_lineage.get("G")

    metrics = statement_processor.process_statement("print(G)")

    assert "G" not in _reasons(metrics), _reasons(metrics)
    assert cash_magics.tracking_state.variable_lineage.get("G") == lineage


def test_adding_an_edge_still_does(cash_magics, statement_processor):
    run_cash_cell(cash_magics, BUILD, cells=[BUILD])

    metrics = statement_processor.process_statement("G.add_edge(1000, 1001)")

    assert "In-place mutation on: G" in _reasons(metrics), _reasons(metrics)
