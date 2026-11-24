"""A hit hands back the copy the last hit handed out, but only when nothing
can tell it from a new one.

A function returning a big read-only structure (a dictionary trie) called
again and again had every hit unpickle a new copy of it from the RAM tier,
which the caller then dropped: slower than computing it, where
``functools.lru_cache`` hands back the one object. The tier now keeps the
copy it handed out and hands it back while the stored entry is the same,
nothing outside the copy holds any part of it, and the watchers prove
nothing in it changed. Anything else -- the caller still holding it,
having changed a node at the top or deep down, keeping a node of it, or a
new entry stored -- gets a new copy, equal to what plain Python returns.
"""

from __future__ import annotations

import random
from abc import ABC

import pytest

from cash import value_watch
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="the copy is handed back only where the C extension's watchers prove it unchanged",
)

#: How many copies were read out of pickle bytes (`Trie.__setstate__`).
LOADS: list[int] = []


class Trie(ABC):
    """A word trie as dictionary libraries build it: an ABC over nodes with
    ``__slots__`` and a dict of children each."""

    class Node:
        __slots__ = ("children", "end")

        def __init__(self):
            self.end = False
            self.children = {}

    def __init__(self, words):
        self.root = Trie.Node()
        for word in words:
            node = self.root
            for ch in word:
                node = node.children.setdefault(ch, Trie.Node())
            node.end = True

    def __setstate__(self, state):
        LOADS.append(1)
        self.__dict__.update(state)

    def words(self):
        found, todo = [], [("", self.root)]
        while todo:
            prefix, node = todo.pop()
            if node.end:
                found.append(prefix)
            todo.extend((prefix + ch, child) for ch, child in node.children.items())
        return sorted(found)


class OtherNode(Trie.Node):
    """A node class laid out as `Trie.Node`."""

    __slots__ = ()


def word_list(n=4000, seed=7):
    rnd = random.Random(seed)
    return ["".join(rnd.choice("abcdef") for _ in range(rnd.randint(3, 9))) for _ in range(n)]


def build():
    return Trie(word_list())


PLAIN = build().words()


@pytest.fixture
def word_trie(cash_instance):
    @cash_instance.cache
    def word_dict_trie():
        return build()

    word_dict_trie()  # the miss: stored
    first = word_dict_trie()  # the first hit: a copy, kept
    assert first.words() == PLAIN
    del first
    return word_dict_trie


def test_an_unchanged_copy_nobody_holds_is_handed_back(word_trie):
    before = len(LOADS)
    second = word_trie()
    third_id = id(second)
    del second
    third = word_trie()
    assert len(LOADS) == before, "no hit read a new copy"
    assert id(third) == third_id
    assert third.words() == PLAIN


def test_a_copy_the_caller_still_holds_is_not_handed_back(word_trie):
    first = word_trie()
    second = word_trie()
    assert second is not first
    assert second.words() == PLAIN
    second.root.children.clear()
    assert first.words() == PLAIN, "the copies are each the caller's own"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda t: setattr(t, "root", Trie.Node()), id="the-root-rebound"),
        pytest.param(lambda t: setattr(t, "extra", 1), id="an-attribute-added"),
        pytest.param(lambda t: t.root.children.pop("a"), id="a-top-node-removed"),
        pytest.param(lambda t: setattr(t.root.children["a"].children["b"].children["c"], "end", "x"), id="a-deep-slot"),
        pytest.param(lambda t: t.root.children["a"].children["b"].children.clear(), id="a-deep-dict"),
        pytest.param(lambda t: t.root.children["f"].children.update(z=Trie.Node()), id="a-node-added"),
        # The watchers look a class's slots up once for a run of its
        # instances: one among them made another class is still seen.
        pytest.param(
            lambda t: setattr(t.root.children["c"].children["d"], "__class__", OtherNode), id="a-node-reclassed"
        ),
    ],
)
def test_a_copy_the_caller_changed_is_not_handed_back(word_trie, change):
    before = len(LOADS)
    first = word_trie()
    change(first)
    del first
    second = word_trie()
    assert len(LOADS) == before + 1, "a new copy was read"
    assert second.words() == PLAIN
    assert not hasattr(second, "extra")


def test_a_copy_held_through_a_nested_node_is_not_handed_back(word_trie):
    first = word_trie()
    inner = first.root.children["a"].children["b"]
    inner_words = sorted(inner.children)
    del first
    second = word_trie()
    assert second.words() == PLAIN
    second.root.children["a"].children["b"].children.clear()
    assert sorted(inner.children) == inner_words, "the node kept is not part of the copy handed out"


def test_a_copy_held_through_a_nested_dict_is_not_handed_back(word_trie):
    first = word_trie()
    kept = first.root.children
    del first
    second = word_trie()
    assert second.root.children is not kept
    assert second.words() == PLAIN


def test_a_new_entry_under_the_key_gets_a_new_copy():
    backend = InMemoryBackend()
    backend.set("k", build(), {"size": 100_000})
    _meta, first = backend.get("k")
    del first
    other = Trie(word_list(3000, seed=11))
    backend.set("k", other, {"size": 100_000})
    _meta, second = backend.get("k")
    assert second.words() == other.words()
    del second
    _meta, third = backend.get("k")
    assert third.words() == other.words()


def test_the_backend_hands_back_the_copy_while_untouched():
    backend = InMemoryBackend()
    backend.set("k", build(), {"size": 100_000})
    _meta, first = backend.get("k")
    first_id = id(first)
    del first
    _meta, second = backend.get("k")
    assert id(second) == first_id
    assert second.words() == PLAIN
    second.root.end = "changed"
    del second
    _meta, third = backend.get("k")
    assert third.root.end is False
    assert third.words() == PLAIN
