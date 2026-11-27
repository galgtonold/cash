"""A hit while the last copy handed out is held, unchanged, makes the new
copy from it in C (`cash._cwatch.clone_held`) instead of unpickling the
kept bytes; what that copy cannot be made for is unpickled as before."""

from __future__ import annotations

import pickle

import pytest

from cash import value_watch
from cash._native import cwatch
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    not value_watch.enabled() or not hasattr(cwatch, "clone_held"),
    reason="needs the C extension",
)


class Node:
    __slots__ = ("children", "end")

    def __init__(self):
        self.end = False
        self.children = {}


class Tree:
    """Big enough (a 1 MB pickle) for the tier to keep its pickle bytes."""

    def __init__(self, n=60_000):
        self.root = Node()
        for i in range(n):
            node = self.root
            for ch in str(i):
                node = node.children.setdefault(ch, Node())
            node.end = True

    def words(self):
        found, todo = [], [("", self.root)]
        while todo:
            prefix, node = todo.pop()
            if node.end:
                found.append(prefix)
            todo.extend((prefix + ch, child) for ch, child in node.children.items())
        return sorted(found)


@pytest.fixture
def loads(monkeypatch):
    counted = []
    original = memory_backend._Pickled.load

    def load(self):
        counted.append(1)
        return original(self)

    monkeypatch.setattr(memory_backend._Pickled, "load", load)
    return counted


def test_a_held_unchanged_copy_is_copied_in_c(loads):
    backend = InMemoryBackend()
    backend.set("k", Tree(), {"size": 100_000})
    _meta, first = backend.get("k")  # read out, kept
    assert len(loads) == 1
    copies = [backend.get("k")[1] for _ in range(3)]
    assert len(loads) == 1, "no hit unpickled while the kept copy is held and unchanged"
    assert all(c.words() == Tree().words() for c in copies)
    assert len({id(c) for c in [first, *copies]}) == 4


def test_a_held_copy_changed_is_read_out_of_the_bytes_again(loads):
    backend = InMemoryBackend()
    backend.set("k", Tree(), {"size": 100_000})
    _meta, first = backend.get("k")
    first.root.children["1"].end = "changed"
    _meta, second = backend.get("k")
    assert len(loads) == 2
    assert second.root.children["1"].end is True
    assert first.root.children["1"].end == "changed"


class WithSetstate:
    def __init__(self):
        self.items = [1, 2]

    def __setstate__(self, state):
        self.__dict__.update(state)


class WithNew:
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        self.items = [1, 2]


class WithSetattr:
    def __init__(self):
        self.items = [1, 2]

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"a": {1, 2}}, id="a-set"),
        pytest.param({("a", Node()): 1}, id="a-key-not-an-atom"),
        pytest.param([bytearray(b"x")], id="a-bytearray"),
        pytest.param(WithSetstate(), id="a-setstate"),
        pytest.param(WithNew(), id="a-new"),
        pytest.param(WithSetattr(), id="a-setattr"),
    ],
)
def test_what_pickle_would_make_otherwise_is_not_copied(value):
    held = pickle.loads(pickle.dumps(value))
    assert cwatch.clone_held([held], 0) is None


def test_the_copy_holds_new_objects_and_the_shared_ones():
    shared = (1, "a", None)
    inner = [1, 2]
    value = pickle.loads(pickle.dumps({"x": [inner, inner], "t": shared, "n": Node()}))
    copy = cwatch.clone_held([value], 0)
    assert copy == {"x": [[1, 2], [1, 2]], "t": shared, "n": copy["n"]}
    assert copy is not value and copy["x"] is not value["x"]
    assert copy["x"][0] is copy["x"][1], "an object met twice is copied once"
    assert copy["x"][0] is not value["x"][0]
    assert copy["t"] is value["t"], "an immutable tuple of atoms is shared"
    assert copy["n"] is not value["n"] and copy["n"].children is not value["n"].children


def _classes(n):
    """*n* plain classes of slots."""
    return [type(f"C{i}", (), {"__slots__": ("link", "n")}) for i in range(n)]


def test_more_classes_than_the_call_remembers_are_each_copied():
    # `clone_held` asks whether a class may be copied once per class, for
    # the first few it meets: the classes past those are asked per object.
    chain = None
    for i, cls in enumerate(_classes(12) * 2):
        obj = cls()
        obj.link = chain
        obj.n = i
        chain = obj
    value = chain
    copy = cwatch.clone_held([value], 0)
    assert copy is not None
    a, b = value, copy
    while a is not None:
        assert type(b) is type(a) and b is not a
        assert b.n == a.n
        a, b = a.link, b.link
    assert b is None


def test_a_class_met_after_clonable_ones_is_still_refused():
    # Remembering the classes that passed lets none through that did not.
    nodes = [Node() for _ in range(20)]
    value = pickle.loads(pickle.dumps({"nodes": nodes, "last": WithSetstate()}))
    assert cwatch.clone_held([value], 0) is None


class Meta(type):
    pass


class Shell(metaclass=Meta):
    pass


class TupleKind(tuple):
    pass


def test_a_class_is_shared_and_a_tuple_subclass_is_not_copied():
    value = pickle.loads(pickle.dumps({"cls": Shell, "meta": Meta, "node": Node()}))
    copy = cwatch.clone_held([value], 0)
    assert copy["cls"] is Shell and copy["meta"] is Meta, "a class is shared"
    assert copy["node"] is not value["node"]
    # an instance of a class of the user's subclassing a builtin is no kind
    # the copy makes: the hit reads the bytes instead
    assert cwatch.clone_held([{"tup": TupleKind(([1],))}], 0) is None
