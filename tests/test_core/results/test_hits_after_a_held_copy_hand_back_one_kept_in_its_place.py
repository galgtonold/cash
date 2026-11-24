"""Once something outside holds the copy a hit handed out, the RAM tier
keeps a copy made from it in its place, so the hits after whose callers
let go get that one back instead of a new copy every time.

A library that keeps the first result it is given (a tokenizer keeping
the dictionary trie) holds that copy for good. The hits after it get new
copies made from it; one of those is kept in its place, and from then on
a caller that drops its result gets the same object back on the next hit.
The copy held outside is never handed out again and never changed, and a
copy changed by its caller is never handed back: the next hit gets a new
one equal to what plain Python returns.
"""

from __future__ import annotations

import random

import pytest

from cash import handed_copies, value_watch
from cash.backends import memory_backend

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="a copy is handed back only where the C extension's watchers prove it unchanged",
)


class Trie:
    """A word trie of plain nodes."""

    class Node:
        __slots__ = ("children", "end")

        def __init__(self):
            self.end = False
            self.children = {}

    def __init__(self, words):
        self.root = Trie.Node()
        self.tags = [("trie", 1), ["mutable", "tag"]]
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


def word_list(n=3000, seed=11):
    rnd = random.Random(seed)
    return ["".join(rnd.choice("abcdef") for _ in range(rnd.randint(3, 9))) for _ in range(n)]


PLAIN = Trie(word_list()).words()

#: What the library keeps: the first result it was handed.
HOLDER: list = []


@pytest.fixture
def made(monkeypatch):
    """Copies made per hit: read out of the kept bytes or made from the
    copy held outside. A hit that makes none hands back the kept copy."""
    counted = []
    load = memory_backend._Pickled.load
    clone = handed_copies._CLONE_HELD

    def counting_load(self):
        counted.append("load")
        return load(self)

    def counting_clone(holder, index):
        counted.append("clone")
        return clone(holder, index)

    monkeypatch.setattr(memory_backend._Pickled, "load", counting_load)
    if clone is not None:
        monkeypatch.setattr(handed_copies, "_CLONE_HELD", counting_clone)
    return counted


@pytest.fixture
def word_trie(cash_instance, made):
    @cash_instance.cache
    def word_dict_trie():
        return Trie(word_list())

    word_dict_trie()  # the miss: stored
    HOLDER.clear()
    HOLDER.append(word_dict_trie())  # the first hit, held from now on
    yield word_dict_trie
    HOLDER.clear()


def released_hit(func):
    """A hit whose caller drops the result (no reference of any kind is
    left, a weak one included): the result's id, which stays its own while
    the tier keeps it."""
    result = func()
    assert result.words() == PLAIN
    return id(result)


def test_hits_whose_callers_let_go_get_one_copy_back(word_trie, made):
    held = HOLDER[0]
    held_root = held.root
    for _ in range(3):  # copies made from the held one, one kept in its place
        released_hit(word_trie)
    kept = released_hit(word_trie)
    made.clear()
    for _ in range(4):
        again = word_trie()
        assert id(again) == kept, "a released, unchanged copy is handed back"
        assert again is not held
        del again
    assert made == [], "no hit made a new copy"
    assert held.root is held_root
    assert held.words() == PLAIN, "the copy held outside is left as it was"


def test_the_held_copy_changed_reaches_no_later_hit(word_trie):
    for _ in range(3):
        released_hit(word_trie)
    held = HOLDER[0]
    held.root.children["a"].children.clear()
    held.tags[1].append("changed")
    for _ in range(3):
        later = word_trie()
        assert later is not held
        assert later.words() == PLAIN
        assert later.tags == [("trie", 1), ["mutable", "tag"]]
        del later


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda t: t.root.children.pop("a"), id="a-top-node-removed"),
        pytest.param(lambda t: setattr(t.root.children["b"], "end", "x"), id="a-slot"),
        pytest.param(lambda t: t.tags[1].append("x"), id="a-nested-list"),
    ],
)
def test_a_handed_back_copy_changed_is_not_handed_back(word_trie, made, change):
    for _ in range(3):
        released_hit(word_trie)
    kept = released_hit(word_trie)
    made.clear()
    reused = word_trie()
    assert id(reused) == kept and made == [], "handed back while unchanged"
    change(reused)
    fresh = word_trie()  # the changed copy still alive: not it, and not made from it
    assert fresh is not reused
    assert fresh.words() == PLAIN
    assert fresh.tags == [("trie", 1), ["mutable", "tag"]]
    del reused
    after = word_trie()
    assert after.words() == PLAIN
    assert after.tags == [("trie", 1), ["mutable", "tag"]]
    assert HOLDER[0].words() == PLAIN


def test_a_kept_copy_held_again_is_replaced_again(word_trie, made):
    # A library keeping each copy it is given by its id (PyThaiNLP's
    # longest-match tokenizers) holds the copy kept in the first one's
    # place too: the hits after, whose callers let go, still get one copy
    # back once a later copy is kept in its place.
    second_holder = []
    released_hit(word_trie)
    second_holder.append(word_trie())  # the copy kept in the held one's place, held too
    for _ in range(8):
        released_hit(word_trie)
    kept = released_hit(word_trie)
    made.clear()
    for _ in range(3):
        again = word_trie()
        assert id(again) == kept, "a released, unchanged copy is handed back"
        assert again is not HOLDER[0] and again is not second_holder[0]
        assert again.words() == PLAIN
        del again
    assert made == [], "no hit made a new copy"
    assert second_holder[0].words() == PLAIN
