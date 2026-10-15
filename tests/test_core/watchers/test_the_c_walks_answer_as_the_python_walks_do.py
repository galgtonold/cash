"""The C walk kernels (``cwalk.c``, `cash._native.kernels`) give exactly the
answers of the Python walks they stand in for.

Each walk keeps its Python version as the reference and the fallback; here
both run on random structures and on real objects (a networkx graph, a
fitted model, a frame), and every answer must be the same: the same
functions and classes in the same order, the same sizes, the same groups.
"""

from __future__ import annotations

import collections
import functools
import itertools
import operator
import random
import types

import pytest

from cash import _native
from cash.notebook import held_functions as hf

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")

K = _native.kernels


# ----------------------------------------------------------------- objects


class Plain:
    pass


class WithSlots:
    __slots__ = ("a", "b", "__private")

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class SlotsNoDict:
    __slots__ = ()


class MyDict(dict):
    pass


class MyList(list):
    def __iter__(self):  # a subclass iterating its own way: never read in C
        return iter(list.__iter__(self))


class Meta(type):
    def __hash__(cls):
        return 1

    def __eq__(cls, other):
        return cls is other


class Odd(metaclass=Meta):
    pass


class NotebookClass:
    def method(self):
        return 1

    @property
    def prop(self):
        return 2

    @staticmethod
    def stat():
        return 3


def _notebook_namespace():
    ns: dict = {"__name__": "__main__"}
    exec(
        "def f(x):\n    return x + 1\n"
        "def g(x, k=1):\n    return f(x) * k\n"
        "h = lambda y: y\n"
        "class C:\n    def m(self):\n        return f(1)\n",
        ns,
    )
    return ns


def _random_value(rng: random.Random, ns: dict, depth: int, pool: list, odd: bool = True):
    """A random structure of the kinds the held walk takes apart; with
    *odd*, objects of a class whose metaclass hashes its own way too."""
    if pool and rng.random() < 0.08 and odd:
        return rng.choice(pool)  # shared: met twice
    leaves = [
        lambda: rng.randint(-5, 5),
        lambda: rng.random(),
        lambda: f"s{rng.randint(0, 9)}",
        lambda: None,
        lambda: True,
        lambda: b"x",
        lambda: bytearray(b"y"),
        lambda: 1j,
    ]
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(leaves)()
    n = rng.randint(0, 6)

    def child():
        return _random_value(rng, ns, depth - 1, pool, odd)

    makers = [
        lambda: [child() for _ in range(n)],
        lambda: tuple(child() for _ in range(n)),
        lambda: {rng.choice(leaves[:6])() if rng.random() < 0.8 else (1, "a"): child() for _ in range(n)},
        lambda: {(rng.randint(0, 3), rng.randint(0, 3)): child() for _ in range(n)},
        lambda: {(1, Plain()): child()},
        lambda: {rng.randint(0, 50) for _ in range(n)} | ({frozenset({1})} if rng.random() < 0.3 else set()),
        lambda: frozenset(f"k{rng.randint(0, 20)}" for _ in range(n)),
        lambda: MyDict({i: child() for i in range(n)}),
        lambda: MyList(child() for _ in range(n)),
        lambda: collections.OrderedDict((i, child()) for i in range(n)),
        lambda: ns["f"],
        lambda: ns["g"],
        lambda: ns["h"],
        lambda: ns["C"],
        lambda: ns["C"](),
        lambda: functools.partial(ns["g"], child(), k=child()),
        lambda: ns["C"]().m,
        lambda: NotebookClass,
        lambda: _attrs(Plain(), child, n),
        lambda: WithSlots(child(), child()),
        lambda: SlotsNoDict(),
        *((lambda: Odd(), lambda: Odd) if odd else ()),
        lambda: object(),
        lambda: len,
        lambda: types,
        lambda: staticmethod(ns["f"]),
        lambda: property(ns["f"]),
        lambda: [{"a": rng.randint(0, 9), "b": "x"} for _ in range(n * 10)],
        lambda: [(i, f"r{i}") for i in range(n * 10)],
    ]
    made = rng.choice(makers)()
    if rng.random() < 0.2:
        pool.append(made)
    return made


def _attrs(obj, child, n):
    for i in range(n):
        setattr(obj, f"a{i}", child())
    return obj


def _cycle():
    a: list = [1]
    b = {"a": a}
    a.append(b)
    return a


# ----------------------------------------------------------------- helpers


def _walk_both(value, ns, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(hf, "_K", None)
        python = hf._walk(value, ns)
    c = hf._walk(value, ns)
    return python, c


def _same(a, b):
    """Two `_walk` answers are the same objects in the same order."""
    fa, ca, done_a = a
    fb, cb, done_b = b
    assert done_a == done_b
    assert [id(x) for x in fa] == [id(x) for x in fb]
    assert [id(x) for x in ca] == [id(x) for x in cb]


# ----------------------------------------------------------------- the held walk


@pytest.mark.parametrize("seed", range(150))
def test_the_held_walk_finds_the_same_code_in_the_same_order(seed, monkeypatch):
    rng = random.Random(seed)
    ns = _notebook_namespace()
    pool: list = []
    value = [_random_value(rng, ns, rng.randint(1, 5), pool) for _ in range(rng.randint(1, 8))]
    if seed % 7 == 0:
        value.append(_cycle())
    python, c = _walk_both(value, ns, monkeypatch)
    _same(python, c)


def test_a_graph_a_model_and_a_frame_are_walked_alike(monkeypatch):
    nx = pytest.importorskip("networkx")
    pd = pytest.importorskip("pandas")
    ns = _notebook_namespace()
    graph = nx.gnm_random_graph(300, 1200, seed=1)
    for u, v in graph.edges:
        graph.edges[u, v]["w"] = u * v
    graph.graph["builder"] = ns["f"]
    frame = pd.DataFrame({"a": range(50), "b": [f"x{i}" for i in range(50)], "o": [[i] for i in range(50)]})
    values = [graph, frame, {"g": graph, "fn": ns["g"], "frame": frame}]
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import FunctionTransformer

        model = LinearRegression().fit([[0], [1], [2]], [0, 1, 2])
        values += [model, FunctionTransformer(ns["f"])]
    except ImportError:
        pass
    for value in values:
        python, c = _walk_both(value, ns, monkeypatch)
        _same(python, c)
    assert hf.held_notebook_code(graph, ns).codes == (ns["f"].__code__,)


def test_a_walk_past_the_limit_stops_at_the_same_place(monkeypatch):
    ns = _notebook_namespace()
    monkeypatch.setattr(hf, "WALK_LIMIT", 500)
    value = [[Plain() for _ in range(40)] for _ in range(40)] + [ns["f"]]
    python, c = _walk_both(value, ns, monkeypatch)
    _same(python, c)
    assert python[2] is False


# ----------------------------------------------------------------- kernels one by one


def _py_leaf_items(group, mode, leaves):
    if mode == 0:
        items = itertools.chain.from_iterable(group)
    elif mode == 1:
        items = itertools.chain.from_iterable(map(dict.values, group))
    elif mode == 2:
        items = itertools.chain(
            itertools.chain.from_iterable(group), itertools.chain.from_iterable(map(dict.values, group))
        )
    else:
        items = itertools.chain.from_iterable(group)
    return set(map(type, items)) <= leaves


def _py_dict_children(group, leaves):
    children: list = []
    keys = list(itertools.chain.from_iterable(group))
    key_kinds = set(map(type, keys))
    if not key_kinds <= leaves and not (
        key_kinds <= leaves | {tuple}
        and set(map(type, itertools.chain.from_iterable(hf._of_type(keys, tuple)))) <= leaves
    ):
        children.extend(keys)
    if not set(map(type, itertools.chain.from_iterable(map(dict.values, group)))) <= leaves:
        children.extend(itertools.chain.from_iterable(map(dict.values, group)))
    return children


def _py_seq_children(group, leaves):
    children: list = []
    if not set(map(type, itertools.chain.from_iterable(group))) <= leaves:
        children.extend(itertools.chain.from_iterable(group))
    return children


def _ids(objects):
    return [id(x) for x in objects]


@pytest.mark.parametrize("seed", range(60))
def test_each_kernel_matches_its_python_step(seed):
    rng = random.Random(1000 + seed)
    ns = _notebook_namespace()
    pool: list = []
    leaves = hf._LEAVES
    if seed % 3 == 0:
        leaves = leaves | {tuple}
    dicts = [
        rng.choice(
            [
                {i: rng.randint(0, 3) for i in range(rng.randint(0, 4))},
                {(i, "k"): None for i in range(rng.randint(0, 3))},
                {(i, Plain()): None for i in range(rng.randint(1, 2))},
                {"x": _random_value(rng, ns, 2, pool, odd=False)},
                {Plain(): 1},
            ]
        )
        for _ in range(rng.randint(0, 12))
    ]
    # dicts: values, keys and values, keys alone, and the children step
    for mode in (1, 2, 3):
        assert K.leaf_items(dicts, mode, leaves) == _py_leaf_items(dicts, mode, leaves)
    children: list = ["before"]
    assert K.dict_children(dicts, leaves, children) is None
    assert _ids(children[1:]) == _ids(_py_dict_children(dicts, leaves))

    seqs = [
        rng.choice(
            [
                [rng.randint(0, 9) for _ in range(rng.randint(0, 5))],
                tuple(f"t{i}" for i in range(rng.randint(0, 3))),
                {rng.randint(0, 99) for _ in range(rng.randint(0, 5))},
                frozenset({(1, 2), "a"}),
                [_random_value(rng, ns, 2, pool, odd=False) for _ in range(rng.randint(0, 3))],
            ]
        )
        for _ in range(rng.randint(0, 12))
    ]
    assert K.leaf_items(seqs, 0, leaves) == _py_leaf_items(seqs, 0, leaves)
    children = []
    assert K.seq_children(seqs, leaves, children) is None
    assert _ids(children) == _ids(_py_seq_children(seqs, leaves))

    # a level split by type, and its dedup against what the walk met
    objects = [_random_value(rng, ns, 1, pool, odd=False) for _ in range(rng.randint(0, 80))]
    objects += rng.sample(objects, min(len(objects), 5))
    kinds = set(map(type, objects))
    inert = {kind for kind in kinds if hf._inert(kind)}
    skipped = leaves | inert if inert else leaves
    others = list(itertools.compress(objects, map(operator.not_, map(skipped.__contains__, map(type, objects)))))
    others.extend(inert)
    expected = hf._by_type(others)
    got = K.split_level(objects, skipped, inert)
    assert [(kind, _ids(group)) for kind, group in got] == [(kind, _ids(group)) for kind, group in expected]
    assert K.type_set(objects) == kinds
    seen_py: set = set(rng.sample(_ids(objects), min(3, len(objects))))
    seen_c = set(seen_py)
    for kind, group in expected:
        by_identity = hf._identity_hashed(kind)
        found = dict(zip(group if by_identity else map(id, group), group))
        for key in seen_py & found.keys():
            del found[key]
        seen_py.update(found)
        assert _ids(K.new_in(group, by_identity, seen_c)) == _ids(found.values())
    assert seen_c == seen_py


def test_a_kernel_hands_back_what_it_cannot_answer():
    leaves = hf._LEAVES
    assert K.leaf_items([MyDict(a=1)], 2, leaves) is NotImplemented
    assert K.leaf_items([MyList([1])], 0, leaves) is NotImplemented
    assert K.dict_children([MyDict(a=1)], leaves, []) is NotImplemented
    assert K.seq_children([MyList([1])], leaves, []) is NotImplemented
    # a class whose metaclass hashes in its own way: its membership is asked in Python
    children: list = []
    assert K.seq_children([[Odd()]], leaves, children) is NotImplemented
    assert children == []
    assert K.split_level([Odd()], leaves, set()) is NotImplemented
