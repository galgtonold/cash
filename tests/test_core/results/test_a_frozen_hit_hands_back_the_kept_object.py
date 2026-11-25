"""``frozen=True``: a hit hands back the object it handed out before, as
``functools.lru_cache`` does, even while something outside holds it.

Without ``frozen=True`` a hit may hand back its last copy only when nothing
outside holds it (`handed_copies`): a library that keeps what it is given
(PyThaiNLP's tokenizer keeps every word trie by its id) made every hit copy
the trie again. Declared frozen, the result is promised never to be
changed, so the "held" part of that proof is waived; the "unchanged" part
is not. A kept object that changed after all is never handed out again:
KEY-FROZEN-MUTATED names the producer, and the next hit gets a new copy
read out of the stored entry.
"""

from __future__ import annotations

import random
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

from cash import value_watch
from tests._scripts import run_python

cow = pytest.mark.skipif(
    int(pd.__version__.split(".", 1)[0]) < 3 and pd.options.mode.copy_on_write is not True,
    reason="a pandas table is proved unchanged only under copy-on-write; on pandas 2 without it a hit copies",
)

watched = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="an object graph is proved unchanged by the C extension's watchers",
)


class Trie:
    """A word trie of plain nodes, over the watchers' 64 KiB floor."""

    class Node:
        __slots__ = ("children", "end")

        def __init__(self):
            self.children = {}
            self.end = False

    def __init__(self, words):
        self.root = Trie.Node()
        for word in words:
            node = self.root
            for ch in word:
                node = node.children.setdefault(ch, Trie.Node())
            node.end = True

    def words(self):
        found, todo = [], [("", self.root)]
        while todo:
            prefix, node = todo.pop()
            if node.end:
                found.append(prefix)
            todo.extend((prefix + ch, child) for ch, child in node.children.items())
        return sorted(found)


def word_list(n=4000, seed=3):
    rnd = random.Random(seed)
    return ["".join(rnd.choice("abcdefg") for _ in range(rnd.randint(3, 9))) for _ in range(n)]


def make_trie():
    return Trie(word_list())


def make_dict():
    return {f"k{i}": [i, str(i) * 3] for i in range(6000)}


def make_array():
    return np.arange(100_000, dtype=np.float64)


def make_frame():
    n = 50_000
    return pd.DataFrame({"a": np.arange(n), "b": np.arange(n) * 0.5, "s": [f"w{i}" for i in range(n)]})


def same(kind, a, b):
    if kind == "array":
        return np.array_equal(a, b)
    if kind == "frame":
        return a.equals(b)
    if kind == "trie":
        return a.words() == b.words()
    return a == b


KINDS = {
    "trie": pytest.param("trie", make_trie, marks=watched, id="plain-object"),
    "dict": pytest.param("dict", make_dict, marks=watched, id="dict"),
    "array": pytest.param("array", make_array, id="numpy"),
    "frame": pytest.param("frame", make_frame, marks=cow, id="pandas"),
}


def mutated_warnings(seen):
    return [str(w.message) for w in seen if "KEY-FROZEN-MUTATED" in str(w.message)]


@pytest.mark.parametrize("kind, make", list(KINDS.values()))
def test_every_hit_while_the_result_is_held_is_the_same_object(cash_instance, kind, make):
    @cash_instance.cache(frozen=True)
    def produce():
        return make()

    held = [produce()]  # the miss
    held += [produce() for _ in range(5)]  # hits, each one kept by the caller
    hits = held[1:]
    assert len({id(h) for h in hits}) == 1, "a frozen hit copied a result it had handed out before"
    assert same(kind, hits[0], make())
    assert produce.cache_info()["hits"] == 5


@pytest.mark.parametrize(
    "kind, make",
    [
        pytest.param("trie", make_trie, marks=watched, id="plain-object"),
        pytest.param("array", make_array, id="numpy"),
    ],
)
def test_without_frozen_a_held_result_is_copied_on_every_hit(cash_instance, kind, make):
    @cash_instance.cache
    def produce():
        return make()

    produce()
    hits = [produce() for _ in range(5)]
    assert len({id(h) for h in hits}) == 5, "a hit handed back a copy the caller still holds"
    for h in hits:
        assert same(kind, h, make())


CHANGES = [
    pytest.param("trie", make_trie, lambda t: t.root.children.clear(), marks=watched, id="plain-object"),
    pytest.param(
        "trie", make_trie, lambda t: setattr(next(iter(t.root.children.values())), "end", "x"), marks=watched, id="slot"
    ),
    pytest.param("dict", make_dict, lambda d: d["k1"].append("x"), marks=watched, id="dict-nested-list"),
    pytest.param("array", make_array, lambda a: setattr(a, "shape", (1000, 100)), id="numpy-reshaped-in-place"),
    pytest.param("frame", make_frame, lambda f: f.loc.__setitem__((0, "a"), -1), marks=cow, id="pandas-cell"),
    pytest.param("frame", make_frame, lambda f: f.__setitem__("new", 1), marks=cow, id="pandas-new-column"),
    pytest.param("frame", make_frame, lambda f: setattr(f.index, "name", "x"), marks=cow, id="pandas-index-name"),
    pytest.param("frame", make_frame, lambda f: f.attrs.__setitem__("k", 1), marks=cow, id="pandas-attrs"),
]


@pytest.mark.parametrize("kind, make, change", CHANGES)
def test_a_kept_result_changed_after_all_is_never_handed_out_again(cash_instance, kind, make, change):
    @cash_instance.cache(frozen=True)
    def produce():
        return make()

    produce()
    first = produce()
    assert produce() is first
    change(first)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fresh = produce()
        again = produce()
        third = produce()
    assert fresh is not first, "the changed object was handed out again"
    assert same(kind, fresh, make()), "the new copy is not what the function returns"
    found = mutated_warnings(seen)
    assert len(found) == 1, found
    assert "produce" in found[0] and "frozen=True" in found[0]
    assert again is fresh and third is fresh, "the new copy is kept for the hits after"


@watched
def test_a_frozen_result_read_from_disk_is_decoded_once_per_process(tmp_path):
    # The trie's classes in a module of their own, as a library's are.
    (tmp_path / "wordtrie.py").write_text(
        textwrap.dedent(
            """
            class Node:
                __slots__ = ("children", "end")

                def __init__(self):
                    self.children = {}
                    self.end = False


            class Trie:
                def __init__(self, n):
                    self.root = Node()
                    for i in range(n):
                        node = self.root
                        for ch in str(i * 7919):
                            node = node.children.setdefault(ch, Node())
                        node.end = True
            """
        ),
        encoding="utf-8",
    )
    script = tmp_path / "walk.py"
    script.write_text(
        textwrap.dedent(
            """
            import cash
            from wordtrie import Trie


            @cash.cache(frozen=True)
            def trie():
                return Trie(20000)


            kept = [trie() for _ in range(4)]
            info = trie.cache_info()
            print("RUN", info["hits"], info["misses"], len({id(t) for t in kept[1:]}), id(kept[0]) == id(kept[1]))
            """
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / ".cash"
    first = run_python(script, cwd=tmp_path, cache_dir=cache_dir)
    assert "RUN 3 1 1" in first.stdout, first.stdout
    second = run_python(script, cwd=tmp_path, cache_dir=cache_dir)
    assert "RUN 4 0 1 True" in second.stdout, second.stdout + second.stderr


@pytest.mark.parametrize("kind, make", [KINDS["trie"], KINDS["array"], KINDS["frame"]])
def test_hits_from_many_threads_hand_back_one_unchanged_object(cash_instance, kind, make):
    from concurrent.futures import ThreadPoolExecutor

    @cash_instance.cache(frozen=True)
    def produce():
        return make()

    produce()
    kept = produce()
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda _i: produce(), range(40)))
    assert all(g is kept for g in got), "a hit from another thread copied the kept object"
    assert same(kind, kept, make())
