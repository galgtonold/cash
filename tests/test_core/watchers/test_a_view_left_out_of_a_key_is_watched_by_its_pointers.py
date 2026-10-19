"""An object whose ``functools.cached_property`` views cash leaves out of
its key (a networkx graph's ``G.nodes``, ``G.adj``) is watched like any
other value: its own data in full, and each view by what makes it a view
(its class and the pointers it holds), so a view that stops being one is
seen, and filling a view is a change the next hash reads again.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property

import pytest

from cash import value_watch
from cash.value_hash import _compute_hash


class NodeView(Mapping):
    """A slot-only view, as networkx's: a mapping over its owner's dict."""

    __slots__ = ("_nodes",)

    def __init__(self, nodes):
        self._nodes = nodes

    def __getitem__(self, k):
        return self._nodes[k]

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)


class DegreeView:
    """A view with a ``__dict__``: its owner, and a setting."""

    def __init__(self, net, weight=None):
        self._net = net
        self._weight = weight


class Net:
    def __init__(self, n):
        self._node = {i: {"w": i} for i in range(n)}
        self._adj = {i: {(i + 1) % n: {"weight": 1.0}} for i in range(n)}
        self.name = "net"

    @cached_property
    def nodes(self):
        return NodeView(self._node)

    @cached_property
    def degree(self):
        return DegreeView(self)

    @cached_property
    def sizes(self):
        return [len(self._node)]  # a snapshot, not a view: keyed


def _net():
    net = Net(30)
    _ = net.nodes, net.degree, net.sizes
    return net


def watched(value, hash_fn):
    first = value_watch.digest(value, hash_fn)
    assert value_watch.digest(value, hash_fn) == first
    return first


def reread(value, hash_fn) -> bool:
    before = hash_fn.calls
    assert value_watch.digest(value, hash_fn) == _compute_hash(value)
    return hash_fn.calls > before


def test_an_object_with_views_is_watched(watching, counted):
    net = _net()
    watched(net, counted)
    assert value_watch._cw.info(net) is not None
    assert not reread(net, counted)


class _Elsewhere:
    pass


CHANGES = {
    "its data": lambda n: n._node[3].__setitem__("w", -1),
    "an edge added": lambda n: n._adj[1].__setitem__(5, {"weight": 2.0}),
    "a plain attribute": lambda n: setattr(n, "name", "other"),
    "a snapshot filled in place": lambda n: n.sizes.append(1),
    "a view replaced": lambda n: n.__dict__.__setitem__("nodes", NodeView(n._node)),
    "a view dropped": lambda n: n.__dict__.pop("nodes"),
    "a view's slot pointed elsewhere": lambda n: setattr(n.nodes, "_nodes", {"not": "held"}),
    "a view's slot pointed at other data": lambda n: setattr(n.nodes, "_nodes", n._adj),
    "a view's setting": lambda n: setattr(n.degree, "_weight", "weight"),
    "a view's dict grown": lambda n: setattr(n.degree, "extra", [1]),
    "a view's dict pointed elsewhere": lambda n: setattr(n.degree, "_net", _Elsewhere()),
    "a view's class": lambda n: setattr(NodeView, "tag", 1),
}


@pytest.mark.parametrize("change", list(CHANGES))
def test_a_change_to_the_data_or_a_view_is_read_again(watching, counted, change):
    net = _net()
    watched(net, counted)
    try:
        CHANGES[change](net)
        assert reread(net, counted)
        value_watch.digest(net, counted)
        assert value_watch.digest(net, counted) == _compute_hash(net)
    finally:
        if "tag" in NodeView.__dict__:
            del NodeView.tag


def test_filling_a_view_keeps_the_digest_without_reading_again(watching, counted):
    """networkx fills ``G.edges`` or ``G.degree`` on first use, and each
    first use had the whole graph hashed again for the same digest."""
    net = Net(30)
    before = watched(net, counted)
    _ = net.nodes
    _ = net.degree
    assert not reread(net, counted)
    assert value_watch.digest(net, counted) == before
    assert value_watch._cw.stats()["views_kept"] >= 1


FILLS = {
    "a snapshot, not a view": lambda n: n.sizes,
    "a plain attribute": lambda n: setattr(n, "extra", [1]),
    "a view and a change to the data": lambda n: (n.nodes, n._node[3].__setitem__("w", -1)),
    "a change to the data and a view": lambda n: (n._node[3].__setitem__("w", -1), n.nodes),
    "a view that stops being one": lambda n: setattr(n.nodes, "_nodes", {"not": "held"}),
    "a view, then a plain attribute": lambda n: (n.nodes, setattr(n, "extra", [1])),
    "a view, then dropped": lambda n: (n.nodes, n.__dict__.pop("nodes"), setattr(n, "extra", 1)),
}


@pytest.mark.parametrize("fill", list(FILLS))
def test_a_key_added_that_is_not_only_a_view_is_read_again(watching, counted, fill):
    net = Net(30)
    watched(net, counted)
    FILLS[fill](net)
    assert reread(net, counted)
    value_watch.digest(net, counted)
    assert value_watch.digest(net, counted) == _compute_hash(net)


def test_a_view_that_stops_being_one_changes_the_digest(watching, counted):
    net = _net()
    before = watched(net, counted)
    net.nodes._nodes = {"not": "held"}
    after = value_watch.digest(net, counted)
    assert after != before
    assert after == _compute_hash(net)


class LooksUp(Mapping):
    __slots__ = ("_nodes",)

    def __init__(self, nodes):
        self._nodes = nodes

    def __getattr__(self, name):
        raise AttributeError(name)

    def __getitem__(self, k):
        return self._nodes[k]

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)


def test_a_view_whose_class_looks_attributes_up_is_hashed_every_time(watching, counted):
    net = _net()
    net.__dict__["nodes"] = LooksUp(net._node)
    value_watch.digest(net, counted)
    value_watch.digest(net, counted)
    assert value_watch._cw.info(net) is None
    assert reread(net, counted)


def test_a_networkx_graph_changed_through_its_api_is_read_again(watching, counted):
    nx = pytest.importorskip("networkx")
    g = nx.gnm_random_graph(200, 600, seed=3)
    for u, v, d in g.edges(data=True):
        d["weight"] = 1.0
    print(g.nodes, g.edges, g.degree, g.adj)  # fills the views
    watched(g, counted)
    assert value_watch._cw.info(g) is not None, "a graph is not watched"
    assert not reread(g, counted)
    steps = [
        lambda: g.add_edge(1, 2, weight=3.0),
        lambda: g.remove_node(5),
        lambda: nx.set_node_attributes(g, {7: "x"}, "region"),
        lambda: g.graph.__setitem__("name", "g"),
        lambda: g.edges[1, 2].__setitem__("weight", 4.0),
        lambda: nx.relabel_nodes(g, {0: "zero"}, copy=False),
        lambda: g.clear_edges(),
    ]
    for step in steps:
        step()
        assert reread(g, counted)
        value_watch.digest(g, counted)
        assert not reread(g, counted)


def test_a_networkx_graph_is_not_read_again_when_a_call_fills_its_views(watching, counted):
    nx = pytest.importorskip("networkx")
    g = nx.gnm_random_graph(200, 600, seed=3)
    before = watched(g, counted)
    for call in (lambda: nx.density(g), lambda: dict(g.degree(weight="weight")), lambda: list(g.edges)):
        call()
        assert not reread(g, counted)
    assert value_watch.digest(g, counted) == before
