"""A call result the RAM tier keeps as pickle bytes is pickled once.

A call's store digested its result (`call_refs.digest_and_size`, for the
statements that refer to it) and the RAM tier then pickled it again to keep
it: two full pickles of a graph or a model, back to back, with nothing in
between that could change it. The digest is now taken over the very bytes
the tier keeps, when those bytes are also the value's keyed bytes; else
over the keyed bytes, as before.

What must hold: the bytes kept are what the value was at the store, so an
edit afterwards never reaches a hit; an unchanged value digests alike
later (a statement refers to it); a changed one does not -- including a
change only a key sees (an attribute the class's own reduce leaves out).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from cash import kept_state
from cash.backends import memory_backend
from cash.notebook.call_refs import DIGEST_FIELD, digest_and_size, digest_of

ROWS = 1 << 18  # 2 MiB of floats: kept as pickle bytes, its array out of band

PICKLED: list[int] = []


class Model:
    """A fitted model: arrays and settings, counted each time it is pickled."""

    def __init__(self) -> None:
        self.coef = np.arange(ROWS, dtype=float)
        self.names = ["a", "b"]

    def __reduce_ex__(self, protocol):
        PICKLED.append(1)
        return object.__reduce_ex__(self, protocol)


class Thresholded:
    """A model whose own ``__getstate__`` leaves a setting out of its state."""

    def __init__(self) -> None:
        self.coef = np.arange(ROWS, dtype=float)
        self.threshold = 0.5

    def __getstate__(self):
        return {"coef": self.coef}


@pytest.fixture(autouse=True)
def _counted():
    PICKLED.clear()
    yield
    memory_backend.release_premade()


def test_the_digest_and_the_kept_copy_are_one_pickle(call_unit_harness):
    unit = call_unit_harness(lineage={}, user_ns={})
    model = Model()
    unit._entries.store("call:fit", model, 5.0)
    assert PICKLED == [1], "pickled once to digest it and again to keep it"
    assert type(unit._cash.backend._store["call:fit"][1]) is memory_backend._Pickled
    meta = unit._cash.backend.get_metadata("call:fit")
    assert meta[DIGEST_FIELD] == digest_of(model)


def test_an_edit_after_the_store_never_reaches_a_hit(call_unit_harness):
    unit = call_unit_harness(lineage={}, user_ns={})
    model = Model()
    digest = unit._entries.store("call:fit", model, 5.0)[0]
    model.coef[0] = -1.0
    model.names.append("z")
    _meta, served = unit._cash.backend.get("call:fit")
    assert served.coef[0] == 0.0 and served.names == ["a", "b"]
    assert digest_of(model) != digest, "an edited result still digests as the stored one"


def test_a_change_only_a_key_sees_changes_the_digest():
    model = Thresholded()
    before = digest_of(model)
    model.threshold = 0.9
    assert digest_of(model) != before


def test_a_pickle_made_for_a_store_is_taken_only_by_that_value():
    model, other = Model(), Model()
    made = memory_backend.premade_snapshot(model)
    assert made is not None
    kept = memory_backend._snapshot(other)
    assert kept.stream is not made.stream
    assert memory_backend._snapshot(model).stream is made.stream
    assert memory_backend._snapshot(model).stream is not made.stream, "taken twice"


def test_a_digest_for_a_store_matches_a_later_digest():
    model = Model()
    found = digest_and_size(model, for_store=True)
    memory_backend.release_premade()
    assert found == digest_and_size(model)


def test_a_record_array_with_padding_is_not_digested_from_its_stored_bytes():
    dtype = np.dtype({"names": ["a", "b"], "formats": ["f8", "u1"], "offsets": [0, 16], "itemsize": 24})
    array = np.zeros(4, dtype=dtype)
    _stream, also_keyed = kept_state.dumps_also_keyed(array, 5)
    assert not also_keyed
    plain = np.zeros(4)
    assert kept_state.dumps_also_keyed(plain, 5)[1]


def test_an_object_leaving_out_a_setting_is_not_digested_from_its_stored_bytes():
    assert not kept_state.dumps_also_keyed(Thresholded(), 5)[1]
    assert kept_state.dumps_also_keyed(Model(), 5)[1]


def test_a_graph_is_stored_without_its_views_and_digests_alike_after_them():
    nx = pytest.importorskip("networkx")
    graph = nx.path_graph(50)
    graph.add_edge(0, 7, weight=2.0)
    _ = graph.adj  # filled by building a graph from an edge list
    stream, also_keyed = kept_state.dumps_also_keyed(graph, 5)
    assert also_keyed
    loaded = pickle.loads(stream)
    assert "adj" not in vars(loaded)
    assert nx.utils.graphs_equal(loaded, graph) and loaded.adj[0][7] == {"weight": 2.0}
    before = digest_of(graph)
    _ = graph.degree, graph.nodes, graph.edges  # print(G) fills these
    assert digest_of(graph) == before
    graph.add_edge(1, 9)
    assert digest_of(graph) != before
