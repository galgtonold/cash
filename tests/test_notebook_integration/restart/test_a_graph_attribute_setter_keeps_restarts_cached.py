"""Seeded graph algorithms are served after a restart once their setter cell ran twice.

A social-graph notebook built ``G``, set node attributes with
``nx.set_node_attributes(G, region, "region")`` and ran sampled betweenness,
Louvain and label propagation (``seed=42``). The attributes cell counted as
changing the module ``nx``, so each re-run of it moved the lineage of every
statement reading ``nx``. After a restart and Run All, which ran it once,
none of the seeded algorithms matched the entries the longer session had
stored: about 150 s recomputed on an unchanged graph. ``print(G)`` was
reported as an in-place change of ``G`` on the way.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytest.importorskip("networkx")

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(240)]

CELLS = [
    "%cash_badge print\nimport networkx as nx",
    "G = nx.gnm_random_graph(2000, 8000, seed=1)\nprint(G)",
    "nx.set_node_attributes(G, {n: n % 5 for n in G}, 'region')",
    "bc = nx.betweenness_centrality(G, k=150, seed=42)\nprint('BC', round(sum(bc.values()), 6))",
]


def test_the_seeded_call_is_served_after_a_restart(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel(inject_notebook_path=False)  # headless, as the run that found it
    nb_runner.run_cells([1, 2, 3])
    nb_runner.run_cells([2, 3, 4])  # the setter ran twice before the result was stored
    expected = nb_runner.peek("round(sum(bc.values()), 6)")
    assert "In-place mutation on: G" not in nb_runner.get_output(2), nb_runner.get_output(2)
    assert "nx" not in nb_runner.get_output(3).split("In-place mutation on:", 1)[-1], nb_runner.get_output(3)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_raw_output(4)
    assert nb_runner.peek("round(sum(bc.values()), 6)") == expected
