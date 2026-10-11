"""A value nested deeper than pickle follows is stored and copied whole.

The C pickler and ``copy.deepcopy`` recurse once or more per level of a value:
a linked list a few thousand nodes long raised RecursionError on Python 3.10
and 3.11, and on Windows 3.12 and 3.13, so no tier could store it; on Python
3.10 on Windows it ran out of C stack and took the process down. Such a value
is pickled flat (`kept_state.FlatValue`) and copied on a stack of its own
(`iterative_copy.deep_copy`), with the same sharing, cycles and attributes a
plain pickle round trip keeps. Bytes for a key are not flattened: a value too
deep to key still raises.
"""

from __future__ import annotations

import pickle

import pytest

from cash import kept_state
from cash.backends import iterative_copy
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import PickleSerializer

pytestmark = [pytest.mark.core]

#: Deeper than pickle follows on Python 3.10 to 3.13 (3.14 follows as deep
#: as the C stack allows, which can be deeper).
DEEP = 20_000


class Node:
    def __init__(self, v, nxt):
        self.v = v
        self.next = nxt


class Slotted:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a = a
        self.b = b


def _chain(n: int, tail=None) -> Node:
    head = tail
    for i in range(n):
        head = Node(i, head)
    return head


def _values(head) -> list:
    out = []
    while head is not None:
        out.append(head.v)
        head = head.next
    return out


def test_a_deep_chain_round_trips_through_the_disk_serializer():
    head = _chain(DEEP)
    back = PickleSerializer().deserialize(PickleSerializer().serialize(head))
    assert _values(back) == _values(head)


def test_a_deep_chain_round_trips_split_from_its_buffers():
    np = pytest.importorskip("numpy")
    big = np.arange(100_000.0)
    head = _chain(DEEP, Node(big, Node(big, None)))
    stream, buffers = PickleSerializer().serialize_split(head)
    assert len(buffers) == 1, "the array went out of band once"
    back = PickleSerializer().deserialize_split(stream, buffers)
    tail = back
    while tail.next.next is not None:
        tail = tail.next
    assert tail.v is tail.next.v, "two references to one array came back as two arrays"
    assert (tail.v == big).all()


def test_a_flat_value_keeps_sharing_cycles_slots_and_containers():
    shared = [1, 2]
    loop: list = [shared]
    loop.append((loop, shared))
    value = {
        "loop": loop,
        "set": {1, 2},
        "slots": Slotted([3], shared),
        "nested tuples": ((1, 2), (3, (4,))),
        "bytes": bytearray(b"ab"),
    }
    flat = kept_state.flatten(value, kept_state.reducers(pickle.HIGHEST_PROTOCOL))
    back = pickle.loads(kept_state.dumps(flat, pickle.HIGHEST_PROTOCOL))
    assert back["loop"][1][0] is back["loop"]
    assert back["loop"][0] is back["loop"][1][1] is back["slots"].b
    assert back["set"] == {1, 2} and back["slots"].a == [3]
    assert back["nested tuples"] == ((1, 2), (3, (4,))) and back["bytes"] == bytearray(b"ab")


def test_an_object_built_from_itself_is_refused_not_walked_forever():
    class Loop:
        def __reduce__(self):
            return (_identity, (self,))

    with pytest.raises(pickle.PicklingError):
        kept_state.flatten(Loop(), kept_state.reducers())
    with pytest.raises(pickle.PicklingError):
        iterative_copy.deep_copy(Loop(), {})


def _identity(x):
    return x


def test_a_key_is_never_taken_over_a_flattened_value(monkeypatch):
    """Where the pickler runs out of recursion, keyed bytes raise and are not
    pickled flat. How deep the pickler follows is not fixed: from Python 3.14
    it is the C stack left, and a CI runner's 3.14.8 pickled the whole
    DEEP chain. So the pickler is made to give up on it here, as it does on
    a smaller stack."""
    head = _chain(DEEP)
    real = kept_state._dumps

    def out_of_stack_on_the_chain(value, *args):
        if value is head:
            raise RecursionError("maximum recursion depth exceeded while pickling an object")
        return real(value, *args)

    monkeypatch.setattr(kept_state, "_dumps", out_of_stack_on_the_chain)
    with pytest.raises(RecursionError):
        kept_state.dumps(head, keyed=True)
    assert _values(pickle.loads(kept_state.dumps(head))) == _values(head), "a stored value is pickled flat"


def test_a_deep_chain_pickle_refuses_is_copied_by_the_ram_tier():
    """A lambda in it sends the copy to deepcopy, which refused the chain."""
    head = _chain(DEEP)
    head.fn = lambda: 1
    backend = InMemoryBackend()
    backend.set("k", head, {"copy_required": True})
    first, second = (backend.get("k")[1] for _ in range(2))
    assert first is not head and first is not second
    assert first.fn is head.fn, "a function is shared, as pickle and deepcopy share it"
    assert _values(first) == _values(head)


def test_the_iterative_copy_keeps_what_deepcopy_keeps():
    shared = [1]
    value = {"a": shared, "b": (shared, Slotted(shared, 2)), "c": Node(1, None)}
    copied = iterative_copy.deep_copy(value, {})
    assert copied["a"] is copied["b"][0] is copied["b"][1].a
    assert copied["a"] is not shared and copied["a"] == [1]
    assert copied["c"] is not value["c"] and copied["c"].v == 1
