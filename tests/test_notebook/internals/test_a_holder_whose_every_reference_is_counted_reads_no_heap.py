"""A holder whose every reference is counted is found without reading the heap.

The share check climbs from an object up what refers to it,
``gc.get_referrers``, a read of every object the collector tracks per level.
At a loop's end the list it iterated is held only by cash's own frames and
its record of the loop: four reads a loop end, 12 ms each beside 360,000
objects. Where each reference the object has is a local of a running
function or an item of a builtin container a local is bound to, and their
sum is its reference count, the referrers are those containers. One
reference more -- a global list, a closure cell, an object's attribute, a
dict of another module, a suspended generator -- and the heap is read as
before. Pins the work, not the behaviour: a refactor may expect this to
fail.
"""

from __future__ import annotations

import gc
import types

import pytest

import cash.notebook.shared_objects as shared_objects

#: Counting a local bound to the object itself needs every local to hold a
#: reference of its own: 3.13. 3.14 lets a local borrow its caller's, and
#: before 3.13 reading a frame's locals copies them into a dict it keeps.
counts_locals = pytest.mark.skipif(
    not shared_objects._STRONG_LOCALS or not shared_objects._LOCALS_READABLE,
    reason="a local's own reference is counted only on 3.13 with the GIL",
)

HIDDEN: list = []


def _ids(objs) -> set[int]:
    return {id(obj) for obj in objs if not isinstance(obj, types.FrameType)}


@pytest.fixture(autouse=True)
def _clean():
    yield
    HIDDEN.clear()


@counts_locals
def test_an_object_held_by_locals_and_their_containers_is_answered_by_count():
    value = [1, 2]
    record = {"value": value}
    pair = (value, 3)
    asked = [value]
    found = shared_objects._referrers_by_count(asked)
    assert found is not None
    # What the heap read would have said.
    assert _ids(found) == _ids(gc.get_referrers(value))
    assert {id(record), id(pair), id(asked)} <= _ids(found)


@pytest.mark.skipif(not shared_objects._LOCALS_READABLE, reason="frames' locals are read on 3.13+ only")
def test_an_object_only_containers_hold_is_answered_by_count_from_3_13():
    record = {"value": [1, 2]}
    asked = [record["value"]]
    found = shared_objects._referrers_by_count(asked)
    assert found is not None
    assert {id(record), id(asked)} == _ids(found)


@pytest.mark.skipif(shared_objects._LOCALS_READABLE, reason="frames' locals are read there")
def test_before_3_13_the_heap_is_always_read():
    record = {"value": [1, 2]}
    assert shared_objects._referrers_by_count([record["value"]]) is None


def _hold_globally(value):
    HIDDEN.append(value)


def _hold_in_a_cell(value):
    return lambda: value


class _Box:
    pass


def _hold_as_attribute(value):
    box = _Box()
    box.value = value
    return box


def _hold_in_another_module(value):
    module = types.ModuleType("cash_test_holder_module")
    module.REGISTRY = {"value": value}
    return module


def _hold_in_a_suspended_generator(value):
    def gen(held):
        yield
        yield held

    suspended = gen(value)
    next(suspended)
    return suspended


@pytest.mark.parametrize(
    "hold",
    [_hold_globally, _hold_in_a_cell, _hold_as_attribute, _hold_in_another_module, _hold_in_a_suspended_generator],
)
def test_a_hidden_holder_leaves_the_answer_to_the_heap(hold):
    value = [1, 2]
    keep = hold(value)
    record = {"value": value}
    assert shared_objects._referrers_by_count([value]) is None
    del keep, record


def test_a_tuple_gc_does_not_track_held_by_a_hidden_holder_reads_the_heap():
    value = tuple(range(3))
    gc.collect()
    _hold_globally(value)
    assert shared_objects._referrers_by_count([value]) is None


def test_an_immortal_object_reads_the_heap():
    assert shared_objects._referrers_by_count([None]) is None


@pytest.fixture
def heap_reads(monkeypatch):
    reads = []
    real = shared_objects._referrers

    def counted(objs, stop):
        reads.append(len(objs))
        return real(objs, stop)

    monkeypatch.setattr(shared_objects, "_referrers", counted)
    return reads


def _climb(rows, user_ns):
    gave_up = shared_objects.GaveUp()
    found = shared_objects._holders_from_referrers([rows], set(), user_ns, {"rows"}, gave_up)
    return found, gave_up.names, gave_up.complete


@pytest.mark.parametrize("hold", [None, _hold_globally, _hold_in_a_cell, _hold_as_attribute, _hold_in_another_module])
def test_the_climb_finds_what_the_heap_read_finds(monkeypatch, hold):
    """``rows`` is held by ``box`` through a list in a dict; *hold* adds a
    holder no variable is bound to, which the count does not see: the heap
    is read, and the holder still stops the climb."""
    rows = [[1], [2]]
    inner = [rows]
    box = {"k": inner}
    user_ns = {"rows": rows, "box": box}
    keep = hold(rows) if hold is not None else None
    counted = _climb(rows, user_ns)
    monkeypatch.setattr(shared_objects, "_LOCALS_READABLE", False)
    assert counted == _climb(rows, user_ns)
    if hold is None or hold is _hold_as_attribute:
        # An object's attributes are climbed through, as a restore copies
        # them: found by the heap read, so not by the count.
        assert counted[0] == {"box"}
    else:
        assert counted[0] is None
        assert "box" in counted[1]
    del keep


@counts_locals
def test_a_variable_holding_the_output_in_a_container_is_found_with_no_heap_read(heap_reads):
    rows = [[1], [2]]
    inner = [rows]
    box = {"k": inner}
    user_ns = {"rows": rows, "box": box}
    assert _climb(rows, user_ns)[0] == {"box"}
    assert heap_reads == []
