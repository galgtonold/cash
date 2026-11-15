"""A hit while the caller still holds the last copy gets a new copy equal
to what plain Python returns, and its own.

A library that keeps the result it was given (a tokenizer cache holding
the dictionary trie) holds the copy the RAM tier handed out last, so the
next hit cannot hand that one back: it gets a new copy, made from the kept
one when the watchers prove it unchanged. Each copy is the caller's own:
changing one changes no other, a copy the caller changed is never the one
a later hit is made from, and objects referenced twice inside the value
(a node's link back to its parent) are one object in the copy too.
"""

from __future__ import annotations

import random
from abc import ABC

import pytest

from cash import value_watch

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="a held copy is copied anew only where the C extension's watchers prove it unchanged",
)


class Trie(ABC):
    """A word trie with ``__slots__`` nodes, each linked back to its parent."""

    class Node:
        __slots__ = ("children", "end", "parent")

        def __init__(self, parent=None):
            self.end = False
            self.children = {}
            self.parent = parent

    def __init__(self, words):
        self.root = Trie.Node()
        self.count = 0
        self.tags = [("trie", 1), ["mutable", "tag"]]
        self.alias = self.tags  # the same list twice
        for word in words:
            node = self.root
            for ch in word:
                node = node.children.setdefault(ch, Trie.Node(node))
            node.end = True
            self.count += 1

    def words(self):
        found, todo = [], [("", self.root)]
        while todo:
            prefix, node = todo.pop()
            if node.end:
                found.append(prefix)
            todo.extend((prefix + ch, child) for ch, child in node.children.items())
        return sorted(found)


def word_list(n=3000, seed=5):
    rnd = random.Random(seed)
    return ["".join(rnd.choice("abcdef") for _ in range(rnd.randint(3, 9))) for _ in range(n)]


PLAIN = Trie(word_list()).words()


@pytest.fixture
def word_trie(cash_instance):
    @cash_instance.cache
    def word_dict_trie():
        return Trie(word_list())

    word_dict_trie()  # the miss: stored
    return word_dict_trie


def test_each_hit_while_the_copy_is_held_is_a_new_equal_copy(word_trie):
    held = [word_trie() for _ in range(4)]
    assert len({id(t) for t in held}) == 4
    for trie in held:
        assert trie.words() == PLAIN
        assert trie.count == len(word_list())
        assert trie.alias is trie.tags, "an object held twice is one object in the copy"
        child = trie.root.children["a"]
        assert child.parent is trie.root, "the link back to the parent is kept"
    assert held[1].root is not held[2].root
    assert held[1].tags[1] is not held[2].tags[1]


def test_changing_one_held_copy_changes_no_other(word_trie):
    first = word_trie()
    second = word_trie()
    second.root.children["a"].children.clear()
    second.tags[1].append("changed")
    second.root.children["b"].end = "changed"
    assert first.words() == PLAIN
    assert first.tags[1] == ["mutable", "tag"]
    third = word_trie()
    assert third.words() == PLAIN
    assert third.tags[1] == ["mutable", "tag"]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda t: t.root.children.pop("a"), id="a-top-node-removed"),
        pytest.param(lambda t: setattr(t.root.children["a"].children["b"], "end", "x"), id="a-deep-slot"),
        pytest.param(lambda t: t.tags[1].append("x"), id="a-nested-list"),
        pytest.param(lambda t: setattr(t, "count", -1), id="an-attribute"),
    ],
)
def test_a_held_copy_the_caller_changed_is_not_what_the_next_hit_copies(word_trie, change):
    first = word_trie()
    held = word_trie()  # kept for the next hit, and held
    change(held)
    later = [word_trie(), word_trie()]
    for trie in later:
        assert trie.words() == PLAIN
        assert trie.count == len(word_list())
        assert trie.tags == [("trie", 1), ["mutable", "tag"]]
    assert first.words() == PLAIN
