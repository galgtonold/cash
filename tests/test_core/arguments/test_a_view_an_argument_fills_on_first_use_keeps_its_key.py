"""A view an object fills on first use does not move the key it is passed under.

networkx fills ``G.nodes``, ``G.edges``, ``G.adj`` and ``G.degree``
(``functools.cached_property``) the first time they are read, and a
``print(G)`` reads one. The pickle of ``G`` then held the view too, so a
cached call handed ``G`` missed once its view was filled -- inside the call
itself (``G.number_of_edges()`` fills ``G.degree``), so even the second call
with the same graph ran again. A view holds nothing the object does not; a
cached value that is a snapshot of the object (a number) may be stale, and
still keys apart.
"""

from __future__ import annotations

import functools

import pytest

from cash.value_hash import compute_hash


class Node:
    def __init__(self, items):
        self._items = dict(items)

    @functools.cached_property
    def view(self):
        return ItemsView(self._items)

    @functools.cached_property
    def total(self):
        return sum(self._items.values())


class ItemsView:
    __slots__ = ("_items", "_weight")

    def __init__(self, items, weight=None):
        self._items = items
        self._weight = weight


def test_a_filled_view_leaves_the_hash_alone():
    node = Node({"a": 1, "b": 2})
    before = compute_hash(node)
    node.view  # filled on first use
    assert compute_hash(node) == before


def test_a_cached_snapshot_still_keys_apart():
    stale, fresh = Node({"a": 1}), Node({"a": 1})
    stale.total  # cached at 1
    stale._items["b"] = 2
    fresh._items["b"] = 2
    assert compute_hash(stale) != compute_hash(fresh)


def test_a_networkx_graph_keys_alike_before_and_after_print(disk_cash):
    nx = pytest.importorskip("networkx")
    runs = []

    @disk_cash.cache
    def edges(graph):
        runs.append(1)
        return graph.number_of_edges()  # fills graph.degree

    graph = nx.gnm_random_graph(40, 80, seed=1)
    before = compute_hash(graph)
    assert edges(graph) == 80
    str(graph)
    assert compute_hash(graph) == before
    assert edges(graph) == 80
    assert edges(nx.gnm_random_graph(40, 80, seed=1)) == 80
    assert len(runs) == 1
    graph.add_edge(100, 101)
    assert compute_hash(graph) != before
