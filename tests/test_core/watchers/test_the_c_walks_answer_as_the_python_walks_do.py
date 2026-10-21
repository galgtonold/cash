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


# ----------------------------------------------------------------- the store's walks

import datetime
import decimal

from cash import _plain_data as pd
from cash import sizing
from cash.backends import memory_backend as mb


def _json_like(rng: random.Random, depth: int, pool: list, odd: bool):
    """Random JSON-like data, with now and then a part held twice, a value
    no walk accepts (with *odd*), an immortal empty tuple, a deep chain."""
    if pool and rng.random() < 0.05:
        return rng.choice(pool)
    leaf = rng.choice(
        [
            lambda: rng.randint(-(10**20), 10**20),
            lambda: rng.randint(0, 9),
            lambda: rng.random(),
            lambda: "s" * rng.randint(0, 40),
            lambda: "é" * rng.randint(1, 5),
            lambda: None,
            lambda: rng.random() < 0.5,
            lambda: b"b" * rng.randint(0, 9),
            lambda: bytearray(b"y"),
            lambda: 2j,
            lambda: datetime.date(2020, 1, rng.randint(1, 28)),
            lambda: decimal.Decimal("1.5"),
        ]
    )
    if depth <= 0 or rng.random() < 0.25:
        if odd and rng.random() < 0.02:
            return rng.choice([{1, 2}, Plain(), frozenset(), MyDict(), MyList()])
        return leaf()
    n = rng.randint(0, 7)
    kind = rng.random()
    if kind < 0.35:
        made = {
            rng.choice([f"k{i}", i, 1.5, None, (1, 2) if odd and rng.random() < 0.05 else "t"]): _json_like(
                rng, depth - 1, pool, odd
            )
            for i in range(n)
        }
    elif kind < 0.7:
        made = [_json_like(rng, depth - 1, pool, odd) for _ in range(n)]
    elif kind < 0.95:
        made = tuple(_json_like(rng, depth - 1, pool, odd) for _ in range(n))
    else:
        made = ()
    if rng.random() < 0.15:
        pool.append(made)
    return made


def _deep(n):
    value: list = [1]
    for _ in range(n):
        value = [value]
    return value


def _walk_shape(walk):
    """A walk's levels by identity: the containers and items of each level,
    its types, and which level lists are the list above's items."""
    if walk is None:
        return None
    out = []
    for k, (level, flat, kinds, key_kinds) in enumerate(walk):
        out.append(
            (
                [id(x) for x in level],
                [id(x) for x in flat],
                kinds,
                key_kinds,
                type(level) is list and type(flat) is list,
                k > 0 and level is walk[k - 1][1],
            )
        )
    return out


def _both(fn, monkeypatch, *modules):
    """*fn*() without the kernels and with them, each answer reduced to
    plain data before the other runs (a walk's lists hold references the
    other run would count)."""
    answers = []
    for kernels in (None, K):
        with monkeypatch.context() as m:
            for module in modules:
                m.setattr(module, "_K", kernels)
            answers.append(fn())
    return answers


def _facts_of(value):
    def run():
        facts, walk = pd.tree_facts_and_walk(value)
        shape = _walk_shape(walk)
        del walk
        return (
            pd.tree_facts(value),
            facts,
            shape,
            _walk_shape(pd.tree_walk(value)),
            pd.tree_size(value),
            pd.size_of(value),
            pd.held_items_facts(value, 1) if type(value) is list else None,
        )

    return run


@pytest.mark.parametrize("seed", range(250))
def test_the_store_walks_find_the_same_facts_sizes_and_levels(seed, monkeypatch):
    rng = random.Random(5000 + seed)
    pool: list = []
    value = _json_like(rng, rng.randint(1, 6), pool, odd=seed % 4 == 0)
    if seed % 25 == 0:
        value = [value, _deep(14 + seed % 5)]
    if seed % 31 == 0 and type(value) is list:
        value.append(value)  # a cycle
    python, c = _both(_facts_of(value), monkeypatch, pd)
    assert python == c
    del pool


@pytest.mark.parametrize("shape", ["rows", "records", "wide", "mixed"])
def test_levels_past_the_sampling_threshold_are_sampled_alike(shape, monkeypatch):
    n = pd.SIZE_EXACT_UP_TO + 4321
    if shape == "rows":
        value = [(f"a{i}", i, i * 0.5) for i in range(n)]
    elif shape == "records":
        value = [{"id": i, "name": f"n{i}" * (i % 3), "tags": [i, "x"]} for i in range(n)]
    elif shape == "wide":
        value = {f"key{i}" * (1 + i % 4): i for i in range(n)}
    else:
        value = {"rows": [(i, str(i)) for i in range(n)], "d": [{"a": i} for i in range(n)], "rng": (3, (1,) * 625)}
    python, c = _both(_facts_of(value), monkeypatch, pd)
    assert python == c
    assert python[0] is not None and python[0].size > 0


def test_a_tree_held_twice_below_the_top_is_not_held_once(monkeypatch):
    part = {"a": [1, 2]}
    for value, once in (([part, part], False), ([{"a": [1]}, {"b": ()}], False), ([{"a": [1]}, tuple([1])], True)):
        python, c = _both(lambda value=value: pd.tree_facts(value), monkeypatch, pd)
        assert python == c
        assert python.held_once is once


def test_a_copy_over_either_walk_is_the_same(monkeypatch):
    rng = random.Random(7)
    for _ in range(40):
        value = _json_like(rng, 4, [], odd=False)

        def copied(value=value):
            walk = pd.tree_walk(value)
            got = pd.spine_copy(value, walk=walk) if walk is not None else None
            return repr(got), got == value if got is not None else None

        python, c = _both(copied, monkeypatch, pd)
        assert python == c


def _parts_value(rng: random.Random):
    rows = [(f"r{i}", i) for i in range(mb._SHARED_ROWS_MIN + rng.randint(0, 3))]
    inner = {"rows": rows, "short": [(1,)], "nested": {"more": list(rows)}, "x": 1}
    value = {
        "variables": inner,
        "again": rows,
        "listed": [list(rows)],
        "mutable_rows": [[1, 2]] * mb._SHARED_ROWS_MIN,
        "big_dict": {f"k{i}": rows for i in range(mb._PARTS_KEYS + rng.randint(-2, 2))},
        "deep": {"a": {"b": {"c": tuple(rows)}}},
        "records": {f"k{i}": {"a": i, "e": (i, i)} for i in range(200)},
    }
    return value


@pytest.mark.parametrize("seed", range(10))
def test_the_shared_parts_are_the_same_parts(seed, monkeypatch):
    value = _parts_value(random.Random(seed))
    python, c = _both(lambda: mb._shared_parts(value), monkeypatch, mb)
    assert list(python) == list(c)
    assert [id(v) for v in python.values()] == [id(v) for v in c.values()]
    assert python


@pytest.mark.parametrize("seed", range(80))
def test_the_pickled_size_estimate_and_the_footprint_are_the_same(seed, monkeypatch):
    np = pytest.importorskip("numpy")
    rng = random.Random(9000 + seed)
    pool: list = []
    value = _json_like(rng, rng.randint(1, 5), pool, odd=True)
    arrays = [np.zeros(rng.randint(0, 50)), np.arange(3)]
    extra = [arrays[0], (arrays[1], {"a": arrays[0]}), [[[arrays[1]]]], WithSlots(arrays[0]), Plain()]
    value = [value, *rng.sample(extra, rng.randint(0, len(extra)))]
    if seed % 3:
        value = {"v": value, "t": tuple(value)}
    python, c = _both(
        lambda: (sizing.pickled_size_estimate(value), sizing.memory_footprint(value)), monkeypatch, sizing, pd
    )
    assert python == c


def test_the_estimate_sums_what_it_is_handed_as_python_does(monkeypatch):
    calls = []
    real = sizing.pickled_size_estimate

    def counting(value, _depth=0, _seen=None):
        calls.append((type(value).__name__, _depth))
        if type(value) is Plain:
            return 2**70  # past a C integer
        return real(value, _depth, _seen)

    monkeypatch.setattr(sizing, "pickled_size_estimate", counting)
    value = [Plain(), (Plain(), [Plain()]), {"a": Plain()}, 1, "x"]
    python, c = _both(lambda: (calls.clear(), counting(value), list(calls))[1:], monkeypatch, sizing)
    # The same objects estimated in the same order at the same depths; the
    # C loop goes into a tuple, a list or a dict without a call.
    containers = {"list", "tuple", "dict"}
    assert python[0] == c[0]
    assert [call for call in python[1] if call[0] not in containers] == c[1][1:]
    assert python[0] == 3 * 2**70


# ------------------------------------------------------------ heap read


def _referrers_in_python(heap, ids):
    """`shared_objects._referrers_of`'s Python read of the heap."""
    import gc

    return [obj for obj in heap if not ids.isdisjoint(map(id, gc.get_referents(obj)))]


@pytest.mark.parametrize("seed", range(5))
def test_the_heap_read_finds_the_same_referrers_in_the_same_order(seed):
    import gc

    rng = random.Random(seed)
    targets = [[i] for i in range(20)] + [Plain(), WithSlots(), {"k": 1}, (1, 2), "text", 7]
    holders = []
    for _ in range(200):
        picked = rng.sample(targets, rng.randint(1, 3))
        kind = rng.choice((list, tuple, dict, Plain, WithSlots, set))
        if kind is dict:
            holders.append({str(i): t for i, t in enumerate(picked)})
        elif kind is Plain:
            holder = Plain()
            holder.parts = picked
            holders.append(holder)
        elif kind is WithSlots:
            holders.append(WithSlots(*picked[:2]))
        elif kind is set:
            holders.append({t for t in picked if isinstance(t, (tuple, str, int))})
        else:
            holders.append(kind(picked))
    for size in (0, 1, 3, len(targets)):
        ids = {id(t) for t in rng.sample(targets, size)}
        heap = gc.get_objects()
        assert list(map(id, K.referrers_in(heap, ids))) == list(map(id, _referrers_in_python(heap, ids)))


def test_the_heap_read_hands_back_an_id_it_cannot_read():
    assert K.referrers_in([], {"not an id"}) is NotImplemented
    with pytest.raises(TypeError):
        K.referrers_in((), set())
