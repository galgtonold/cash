"""A networkx graph a statement binds is read once for its lineage and its session hash.

The same-value cutoff takes the content digest of a graph output to compare
it with earlier ones, and the session hash of that output is the same
digest. A graph has no cheaper digest than pickling all of it (0.3 s for
350,000 edges) and no sound "unchanged" signal, so the cutoff's digest is
handed to the session hash rather than taken twice.
"""

from __future__ import annotations

import pytest

from cash import kept_state
from tests._cell_driver import run_cash_cell

nx = pytest.importorskip("networkx")


def _counting_pickles(monkeypatch, mock_shell, name):
    """Count the pickles a digest takes of *name*: whole
    (`kept_state.dumps_also_keyed` writing to a hashing sink) or keyed apart
    (`kept_state.dumps`). A store's pickle of its own is not a read."""
    seen = []
    real, real_also = kept_state.dumps, kept_state.dumps_also_keyed

    def counted(value):
        if value is not None and value is mock_shell.user_ns.get(name):
            seen.append(1)

    def dumps(value, *a, **k):
        counted(value)
        return real(value, *a, **k)

    def dumps_also_keyed(value, *a, **k):
        if k.get("sink") is not None:
            counted(value)
        return real_also(value, *a, **k)

    monkeypatch.setattr(kept_state, "dumps", dumps)
    monkeypatch.setattr(kept_state, "dumps_also_keyed", dumps_also_keyed)
    return seen


def test_a_graph_output_is_pickled_once(cash_magics, mock_shell, monkeypatch):
    run_cash_cell(cash_magics, "import networkx as nx")
    run_cash_cell(cash_magics, "G = nx.gnm_random_graph(200, 800, seed=1)")
    reads = _counting_pickles(monkeypatch, mock_shell, "H")
    mock_shell.user_ns["H"] = None

    run_cash_cell(cash_magics, "H = G.subgraph(range(100)).copy()")

    assert len(reads) == 1
    state = cash_magics.tracking_state
    digest = state.value_cutoff.digest_of(state.variable_lineage["H"])
    assert digest is not None and digest.endswith(state.current_session_hashes["H"])
