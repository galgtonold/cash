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

import copy
import pickle
import sys

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


REDUCES: list[int] = []


class Counted:
    """Counted each time it is reduced; rebuilt with list items, which its
    reduce hands over as an iterator."""

    def __init__(self, v=None):
        self.v = v

    def __reduce_ex__(self, protocol):
        REDUCES.append(1)
        return (Counted, (), dict(vars(self)), iter([1, 2]))

    def extend(self, items):
        self.extended = list(items)


class AttrDict(dict):
    """The common ``config.key`` idiom: its state is the dict itself."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.__dict__ = self


class Rows(list):
    """A list whose ``__setstate__`` derives a field from its items."""

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.total = sum(self)


def _rows() -> Rows:
    rows = Rows([1, 2, 3])
    rows.total = 6
    return rows


def test_items_are_in_place_before_the_state_is_set():
    """pickle sets an object's items, then its state: a dict whose
    ``__dict__`` is itself, and a ``__setstate__`` that reads the items, see
    them. Flattened and copied iteratively alike."""
    value = [AttrDict(value=1, inner=AttrDict(id=7)), _rows()]
    flat = kept_state.flatten(value, kept_state.reducers(pickle.HIGHEST_PROTOCOL))
    for back in (pickle.loads(kept_state.dumps(flat, pickle.HIGHEST_PROTOCOL)), iterative_copy.deep_copy(value, {})):
        assert back[0].value == 1 and back[0].inner.id == 7
        assert back[1].total == 6 and back[1] == [1, 2, 3]


def test_a_flat_value_stored_in_the_old_step_order_is_refused():
    """A stream that set the state before the items rebuilt such a value
    without its attributes: it is not loaded (the entry is a miss)."""
    value = AttrDict(value=1)
    flat = kept_state.flatten(value, kept_state.reducers(pickle.HIGHEST_PROTOCOL))
    build = next(step for step in flat.steps if step[0] == "build")
    old = [step for step in flat.steps if step[0] != "build"]
    old.insert(old.index(next(step for step in old if step[0] == "setitems")), build)
    with pytest.raises(pickle.UnpicklingError):
        kept_state.load_flat(old, flat.root)
    assert kept_state.load_flat(flat.steps, flat.root).value == 1


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
    """A lambda in it sends the copy to deepcopy, which refused the chain.

    Twice the recursion limit is deeper than deepcopy follows (it takes a
    Python frame or more per level), and checked to be. `DEEP` levels cost
    over a second per copy, so under a loaded test run the test went past
    its timeout."""
    head = _chain(2 * sys.getrecursionlimit())
    head.fn = lambda: 1
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(head)
    with pytest.raises(RecursionError):
        copy.deepcopy(head)
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


class TestWhereThePicklerRunsOutOfStackFirst:
    """On Python 3.10 on Windows (`kept_state.FLAT_FIRST`) a value is found
    shallow before the pickler is let follow it. Finding that by flattening
    it cost seconds for every pickle of a list of 200,000 rows, keys
    included, and ran a test past its 30 s timeout on that CI job."""

    @pytest.fixture(autouse=True)
    def flat_first(self, monkeypatch):
        monkeypatch.setattr(kept_state, "FLAT_FIRST", True)

    def test_a_wide_shallow_value_is_pickled_without_flattening_it(self, monkeypatch):
        def no_flatten(*_args, **_kwargs):
            raise AssertionError("a shallow value was flattened")

        monkeypatch.setattr(kept_state, "flatten", no_flatten)
        rows = [(i, {"k": i, "tags": ["a", "b"]}) for i in range(2_000)]
        for keyed in (False, True):
            assert pickle.loads(kept_state.dumps(rows, keyed=keyed)) == rows
        # Every instance's reduce names its class: one object, met once per
        # instance, is not a value met again level after level.
        nodes = [Node(i, None) for i in range(2_000)]
        for keyed in (False, True):
            assert [n.v for n in pickle.loads(kept_state.dumps(nodes, keyed=keyed))] == list(range(2_000))

    def test_a_dict_that_is_its_own_state_keeps_its_attributes(self):
        """Its state is itself, a cycle the depth walk does not call shallow:
        a one-row list of it is flattened, and must come back whole."""
        rows = [AttrDict(id=0)]
        back = pickle.loads(kept_state.dumps(rows))
        assert back[0].id == 0 and back[0] == {"id": 0}

    def test_a_chain_its_list_holds_every_node_of_is_deep(self):
        """The list is two levels deep from each node, but the pickler goes
        down the chain from the first node it saves."""
        head = _chain(kept_state._SAFE_DEPTH * 2)
        nodes = []
        node = head
        while node is not None:
            nodes.append(node)
            node = node.next
        assert not kept_state._at_most_deep(nodes, kept_state._SAFE_DEPTH, kept_state.reducers(), lambda _o: False)
        back = pickle.loads(kept_state.dumps(nodes))
        assert _values(back[0]) == _values(head)

    def test_a_cycle_and_a_value_built_from_itself_are_left_to_flatten(self):
        loop: list = [1]
        loop.append(loop)
        back = pickle.loads(kept_state.dumps(loop))
        assert back[1] is back

        class Loop:
            def __reduce__(self):
                return (_identity, (self,))

        with pytest.raises(pickle.PicklingError):
            kept_state.dumps(Loop())

    def test_a_shallow_value_is_reduced_once_and_its_store_bytes_are_its_keyed_ones(self):
        """The walk before the pickle reduces each of the user's objects,
        and the pickle takes those reduces: one ``__reduce_ex__`` per object
        and pickle. The value is then pickled as on every other Python, its
        store's bytes also its keyed ones, so one pickle serves a digest and
        the store; every value was pickled twice there, once whole."""
        REDUCES.clear()
        value = {"a": Counted(1), "b": [Counted(2)]}
        stream, also_keyed = kept_state.dumps_also_keyed(value, protocol=5)
        assert len(REDUCES) == 2
        assert also_keyed
        back = pickle.loads(stream)
        assert back["a"].v == 1 and back["b"][0].v == 2
        assert back["a"].extended == [1, 2], "the items the walk read reached the pickle"

    def test_a_value_as_deep_as_the_pickler_follows_elsewhere_keys(self):
        """Flattened past `_SAFE_DEPTH`, a key goes as deep as the pickler
        goes on other Pythons, not half as deep."""
        value: list = []
        for _ in range(300):
            value = [("list", value)]
        assert kept_state.dumps(value, keyed=True)
