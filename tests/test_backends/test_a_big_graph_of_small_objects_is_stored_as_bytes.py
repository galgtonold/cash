"""A big graph of small objects is stored at about the cost of its pickle.

A trie of 250,000 slotted nodes builds in 0.3 s and pickles in 0.45 s. Its
store took 4.7 s: the RAM tier walked every node for its size, copied it
by a pickle round trip with a Python call per object, and the disk tier
pickled it again, with `kept_state`'s Python reducer (`_names_apart`) on
every node. A hit in the same process was that round trip again (4.6 s).

The RAM tier now keeps such a value as its pickle bytes (`_Pickled`): it
looks at a few objects first (`_at_least`, which reads ``__slots__`` and
calls a value big when it outgrows the objects it looks at), sizes the
entry by the bytes, and the disk tier writes those very bytes. A hit
unpickles them. The reducer's renaming of attribute names runs in C
(``reduce_apart``) where the extension is built.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import subprocess
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash, _native
from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend

TRIE = textwrap.dedent("""
    import random

    class Node:
        __slots__ = ("end", "children")

        def __init__(self):
            self.end = False
            self.children = {}

    class Trie:
        __slots__ = ("root",)

        def __init__(self, words):
            self.root = Node()
            for word in words:
                node = self.root
                for ch in word:
                    child = node.children.get(ch)
                    if child is None:
                        child = node.children[ch] = Node()
                    node = child
                node.end = True

    def words(n, seed=1):
        r = random.Random(seed)
        alphabet = "abcdefghijklmnopqrstuvwxyz\\u0e01\\u0e02\\u0e04\\u0e07\\u0e08\\u0e09\\u0e0a\\u0e0b"
        return ["".join(r.choice(alphabet) for _ in range(r.randint(3, 9))) for _ in range(n)]

    def shape(trie):
        # Every node's path and flag, in a fixed order.
        out, stack = [], [("", trie.root)]
        while stack:
            path, node = stack.pop()
            out.append((path, node.end))
            stack.extend((path + ch, child) for ch, child in sorted(node.children.items(), reverse=True))
        return out
""")

PROGRAM = textwrap.dedent("""
    import hashlib, sys, time, warnings
    warnings.simplefilter("ignore")
    from cash import Cash, _native
    from trie_nodes import Trie, shape, words

    cash = Cash(cache_dir=sys.argv[1], register_magic=False)
    WORDS = words(72_000)

    @cash.cache(assume_safe=True)
    def build(n):
        return Trie(WORDS[:n])

    started = time.perf_counter()
    trie = build(len(WORDS))
    took = time.perf_counter() - started
    print(len(shape(trie)), hashlib.sha256(repr(shape(trie)).encode()).hexdigest(), round(took, 3))
""")


@pytest.fixture
def trie_module(tmp_path, monkeypatch):
    folder = tmp_path / "mod"
    folder.mkdir()
    (folder / "trie_nodes.py").write_text(TRIE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(folder))
    sys.modules.pop("trie_nodes", None)
    import trie_nodes

    yield trie_nodes
    sys.modules.pop("trie_nodes", None)


def _run(folder, cache_dir):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(folder), *sys.path]))
    done = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(cache_dir)], capture_output=True, text=True, env=env, timeout=300
    )
    assert done.returncode == 0, done.stderr
    nodes, digest, _took = done.stdout.split()
    return int(nodes), digest


@pytest.mark.timeout(600)
def test_a_fresh_process_hit_returns_an_equal_trie(tmp_path, trie_module):
    words = trie_module.words(72_000)
    expected = trie_module.shape(trie_module.Trie(words))
    assert len(expected) > 230_000
    digest = hashlib.sha256(repr(expected).encode()).hexdigest()
    cache_dir = tmp_path / "cache"
    assert _run(tmp_path / "mod", cache_dir) == (len(expected), digest)  # the miss
    assert _run(tmp_path / "mod", cache_dir) == (len(expected), digest)  # the hit, from disk


@pytest.mark.timeout(600)
def test_a_miss_pickles_it_once_and_neither_walks_nor_copies_it(tmp_path, trie_module, monkeypatch):
    """The work a miss does with the value, counted: what made the store
    4.7 s was a size walk of every node, a pickle round trip to copy it and
    a second pickle by the disk tier."""
    from cash import kept_state

    cash = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    words = trie_module.words(72_000)

    @cash.cache(assume_safe=True)
    def build(n):
        return trie_module.Trie(words[:n])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(10)  # the function's first call, keyed and stored while small
    pickles, sized, loaded = [], [], []
    real_dumps, real_size, real_load = kept_state._dumps, memory_backend.memory_footprint, memory_backend._Pickled.load

    def dumps(*args, **kwargs):
        out = real_dumps(*args, **kwargs)
        pickles.append(len(out))
        return out

    def size(value, *args):
        sized.append(type(value))
        return real_size(value, *args)

    def load(self):
        loaded.append(self.nbytes)
        return real_load(self)

    monkeypatch.setattr(kept_state, "_dumps", dumps)
    monkeypatch.setattr(memory_backend, "memory_footprint", size)
    monkeypatch.setattr(memory_backend._Pickled, "load", load)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trie = build(len(words))
    cash.shutdown()  # a write left to a background thread is counted too
    whole = len(pickle.dumps(trie, protocol=5))
    assert len([n for n in pickles if n > whole // 2]) == 1, (pickles, whole)
    assert trie_module.Trie not in sized and trie_module.Node not in sized, sized
    assert not [n for n in loaded if n > whole // 2], loaded


@pytest.mark.timeout(600)
def test_a_miss_stores_it_in_a_few_times_its_pickle(tmp_path, trie_module):
    cash = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    words = trie_module.words(72_000)

    @cash.cache(assume_safe=True)
    def build(n):
        return trie_module.Trie(words[:n])

    # It was over 10 (4.7 s for a 0.45 s pickle). With the C extension
    # 2.3; without it the renaming of attribute names is a Python call per
    # node, 3.4 to 4 (on Python 3.10, whose pickle is slower, 2 to 2.7).
    bound = 3.0 if getattr(_native.kernels, "reduce_apart", None) is not None else 5.0
    # The best of a few misses: a loaded runner (5.2 once on Windows 3.10)
    # slows one, not all; a store that does more work is slower in every
    # one. The work itself is counted above.
    ratios = []
    for n in range(len(words), len(words) - 6, -1):
        started = time.perf_counter()
        plain = trie_module.Trie(words[:n])
        built = time.perf_counter() - started
        started = time.perf_counter()
        pickle.dumps(plain, protocol=5)
        pickled = time.perf_counter() - started
        del plain
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            started = time.perf_counter()
            build(n)
            missed = time.perf_counter() - started
        ratios.append((missed - built) / pickled)
        if len(ratios) >= 3 and min(ratios) < bound:
            break
    assert min(ratios) < bound, ratios


def test_its_size_is_the_bytes_kept_and_a_hit_is_its_own(trie_module):
    walked = []
    real = memory_backend.memory_footprint
    ram = InMemoryBackend()
    trie = trie_module.Trie(trie_module.words(20_000))
    try:
        memory_backend.memory_footprint = lambda v, *a: walked.append(v) or real(v, *a)
        ram.set("t", trie, {"execution_time": 60.0})
    finally:
        memory_backend.memory_footprint = real
    kept = ram._store["t"][1]
    assert type(kept) is memory_backend._Pickled
    assert ram.get_metadata("t")["size"] == kept.nbytes
    assert not walked

    _meta, first = ram.get("t")
    assert first is not trie and trie_module.shape(first) == trie_module.shape(trie)
    first.root.children.clear()  # a caller changing what it got
    first.root.end = True
    _meta, second = ram.get("t")
    assert trie_module.shape(second) == trie_module.shape(trie)
    trie.root.children.clear()  # and the caller's own object, after the store
    _meta, third = ram.get("t")
    assert trie_module.shape(third) == trie_module.shape(second)


def test_a_mutated_hit_does_not_reach_the_disk_copy(tmp_path, trie_module):
    trie = trie_module.Trie(trie_module.words(20_000))
    expected = trie_module.shape(trie)
    tiers = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path))])
    tiers.set("t", trie, {"execution_time": 60.0, "force_persist": True})
    _meta, served = tiers.get("t")
    served.root.children.popitem()
    tiers.backends[1].shutdown()
    again = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path))])
    _meta, restored = again.get("t")
    assert trie_module.shape(restored) == expected


def test_a_few_slotted_objects_are_not_called_big(trie_module):
    class Point:
        __slots__ = ("x", "y")

        def __init__(self, x, y):
            self.x, self.y = x, y

    assert not memory_backend._at_least([Point(i, i) for i in range(100)], 1 << 20)
    assert not memory_backend._at_least(trie_module.Trie(trie_module.words(50)), 1 << 20)
    assert memory_backend._at_least(trie_module.Trie(trie_module.words(5_000)), 1 << 20)


class Slotted:
    __slots__ = ("drop", "keep")


class Plain:
    pass


def test_names_apart_gives_the_same_answers_in_c_and_python():
    from cash import kept_state

    shared = "drop"
    values = []
    for kind in (Slotted, Plain):
        for i in range(3):
            obj = kind()
            obj.drop = shared  # the attribute name's own object as a value
            if i:
                obj.keep = [i, random.random()]
            values.append(obj)
    weird = Plain()
    weird.__dict__[1] = "not a name"
    values.append(weird)
    streams = set()
    for native in (True, False):
        original = kept_state._native.kernels
        try:
            if not native:
                kept_state._native.kernels = None
            streams.add(kept_state.dumps(values, protocol=5))
        finally:
            kept_state._native.kernels = original
    assert len(streams) == 1
    assert pickle.loads(streams.pop())[0].drop == "drop"
