"""The heap is read a block at a time when the objects a library object
holds are looked for up their referrers.

E6's SVR grid search, late in the session (430,000 objects), spent ~3.5 s
after its fit reading the heap object by object for who else holds the
fitted search's insides; with cash off that cell ran at plain speed. A
block of the heap is now read in one ``gc.get_referents`` call and only a
block that holds one of the objects is read object by object: the same
referrers, in the heap's order.
"""

from __future__ import annotations

import gc

import cash.notebook.shared_objects as shared_objects


def _one_at_a_time(objs: tuple) -> list:
    ids = {id(obj) for obj in objs}
    return [obj for obj in gc.get_objects() if any(id(child) in ids for child in gc.get_referents(obj))]


def test_the_referrers_are_those_of_a_read_object_by_object_in_heap_order():
    targets = tuple([] for _ in range(300))
    holders = [[t] for t in targets[::3]]
    pairs = [{"k": t} for t in targets[1::7]]
    filler = [(i, str(i)) for i in range(5_000)]

    gc.disable()  # the heap's order holds between the two reads
    try:
        found = shared_objects._referrers_of(targets, set())
        expected = _one_at_a_time(targets)
    finally:
        gc.enable()

    found_ids = [id(obj) for obj in found]
    # `_one_at_a_time` ran later and also sees the list `found` itself.
    assert found_ids == [id(obj) for obj in expected if obj is not found]
    for holder in holders + pairs:
        assert id(holder) in found_ids
    assert len(filler) == 5_000


def test_objects_nothing_else_holds_are_found_held_by_their_tuple_only():
    objs = ([], [])
    assert shared_objects._referrers_of(objs, set()) == [objs]
