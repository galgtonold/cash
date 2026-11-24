"""A value read from disk a second time goes into RAM as the bytes it was
read out of, and the copy just read is the one the next hit may hand back.

A big object of a class of its own (a dictionary trie) stays on disk until
its second read, which also keeps it in the RAM tier (`TieredBackend.get`).
That read pickled the value it had just unpickled, to keep it as pickle
bytes (0.8 s for a trie of 450,000 nodes, twice the read), and the next
hit unpickled a new copy and armed the watchers on it before hits became
fast. The disk's bytes are what the RAM tier would have made, so they are
kept as they are; the copy unpickled from them is kept as the copy handed
out, and the third read hands it back when nothing can tell it from a new
one -- never one its caller changed or still holds.
"""

from __future__ import annotations

import random

import pytest

from cash import stored_pickle, value_watch
from cash.backends import memory_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

#: Copies read out of pickle bytes (`Trie.__setstate__`).
LOADS: list[int] = []


class Trie:
    """A word trie of slotted nodes, as dictionary libraries build it."""

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


def word_list(n=16000, seed=3):
    rnd = random.Random(seed)
    return ["".join(rnd.choice("abcdefghijklmnop") for _ in range(rnd.randint(3, 9))) for _ in range(n)]


PLAIN = Trie(word_list()).words()


def _tiers(path) -> TieredBackend:
    return TieredBackend([InMemoryBackend(), FileBackend(str(path), flush_interval=0)])


@pytest.fixture
def reader(tmp_path):
    writer = _tiers(tmp_path)
    writer.set("k", Trie(word_list()), {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = _tiers(tmp_path)
    yield reader
    reader.shutdown()


@pytest.fixture
def pickles(monkeypatch):
    """The values pickled to be kept in the RAM tier."""
    made: list = []
    real = stored_pickle.pickle_value

    def counted(value, *args, **kwargs):
        made.append(type(value).__name__)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(stored_pickle, "pickle_value", counted)
    return made


def test_the_second_read_keeps_the_disk_bytes_without_pickling_again(reader, pickles):
    meta, first = reader.get("k")
    assert meta["source"] == "DISK" and reader.backends[0].get("k") == (None, None)
    meta, second = reader.get("k")
    assert meta["source"] == "DISK"
    assert pickles == [], "the value just unpickled is not pickled again"
    stored = reader.backends[0]._store["k"][1]
    assert type(stored) is memory_backend._Pickled
    with open(reader.backends[1]._get_path("k"), "rb") as fh:
        assert stored.stream in fh.read(), "kept as the bytes on disk"
    assert first.words() == second.words() == PLAIN


def test_a_ram_hit_after_it_equals_a_plain_run(reader):
    reader.get("k")
    reader.get("k")
    meta, third = reader.get("k")
    assert meta["source"] == "RAM"
    assert third.words() == PLAIN
    third.root.children.clear()
    _meta, fourth = reader.get("k")
    assert fourth.words() == PLAIN


@pytest.mark.skipif(
    not value_watch.enabled(), reason="a copy is handed back only where the watchers prove it unchanged"
)
class TestTheCopyTheSecondReadHandsOut:
    def test_is_handed_back_by_the_next_hit_when_untouched(self, reader):
        reader.get("k")
        _meta, second = reader.get("k")
        second_id = id(second)
        del second
        before = len(LOADS)
        meta, third = reader.get("k")
        assert meta["source"] == "RAM"
        assert len(LOADS) == before, "no new copy was read"
        assert id(third) == second_id
        assert third.words() == PLAIN

    def test_is_not_handed_back_once_its_caller_changed_it(self, reader):
        reader.get("k")
        _meta, second = reader.get("k")
        second.root.children["a"].children.clear()
        del second
        _meta, third = reader.get("k")
        assert third.words() == PLAIN

    def test_is_not_handed_back_while_its_caller_holds_it(self, reader):
        reader.get("k")
        _meta, second = reader.get("k")
        _meta, third = reader.get("k")
        assert third is not second
        third.root.children.clear()
        assert second.words() == PLAIN, "each caller's copy is its own"

    def test_is_not_handed_back_while_a_node_of_it_is_held(self, reader):
        reader.get("k")
        _meta, second = reader.get("k")
        node = second.root.children["b"]
        kept = sorted(node.children)
        del second
        _meta, third = reader.get("k")
        assert third.root.children["b"] is not node
        third.root.children["b"].children.clear()
        assert sorted(node.children) == kept
