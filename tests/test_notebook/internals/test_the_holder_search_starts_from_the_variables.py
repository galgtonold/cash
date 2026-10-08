"""The variable that holds a statement's object is found without walking the heap.

``train = data['train']`` then ``train['f'] = ...``: the frame is held by the
dict ``data`` is bound to too. Looking for that holder up
``gc.get_referrers`` walks every object the collector tracks, once per level
of containers, so with three million records elsewhere in the notebook each
such statement cost 0.3 s more. The variables are looked at first, from the
namespace down; the heap is walked only when the count still finds a holder
after that. These tests pin how the work is done (no heap walk) and that the
answers stay those of the walk up.
"""

from __future__ import annotations

import gc
import sys

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import share_group


@pytest.fixture
def referrer_calls():
    """Counts the calls of ``gc.get_referrers`` by a profile hook: a wrapper
    taking ``*objs`` would add a tuple holding the very objects searched."""
    calls = []
    real = gc.get_referrers

    def hook(frame, event, arg):
        if event == "c_call" and arg is real:
            calls.append(1)

    sys.setprofile(hook)
    try:
        yield calls
    finally:
        sys.setprofile(None)


def _share(names, ns):
    captured = {name: ns[name] for name in names}
    return share_group(names, captured, ns)


def test_a_dict_holding_the_object_is_found_without_a_heap_walk(referrer_calls):
    ns = {"data": {"train": [1.0, 2.0], "test": [3.0]}}
    ns["train"] = ns["data"]["train"]

    holders, shared = _share(["train"], ns)

    assert set(holders) == {"data"} and holders["data"] is ns["data"]
    assert shared == set()
    assert referrer_calls == [], "the holder search walked the heap"


def test_every_variable_bound_to_the_holder_joins(referrer_calls):
    ns = {"data": {"train": [1.0]}, "obj": None}
    ns["again"] = ns["data"]
    ns["nested"] = {"inner": [ns["data"]]}
    ns["train"] = ns["data"]["train"]

    holders, shared = _share(["train"], ns)

    assert set(holders) == {"data", "again", "nested"}
    assert shared == set()
    assert referrer_calls == []


def test_a_holder_that_is_no_variable_still_refuses(referrer_calls):
    """Positive control: a closure keeps the object too. The variables seen
    from above do not account for it, so the walk up runs and refuses."""
    ns = {"data": {"train": [1.0]}}
    ns["train"] = ns["data"]["train"]
    keep = ns["train"]
    ns["hook"] = lambda: keep

    holders, shared = _share(["train"], ns)

    assert holders == {} and shared == {"train"}
    assert referrer_calls, "a holder no variable accounts for must send the search up the heap"


def test_a_holder_deeper_than_the_look_from_above_is_found_by_the_walk_up(referrer_calls):
    ns = {"train": [1.0]}
    deep = ns["train"]
    for _ in range(shared_objects._DOWN_DEPTH + 2):
        deep = [deep]
    ns["deep"] = deep
    del deep  # a local of this frame would be one more holder

    holders, shared = _share(["train"], ns)

    assert set(holders) == {"deep"} and shared == set()
    assert referrer_calls


def test_a_container_wider_than_the_look_from_above_is_found_by_the_walk_up(referrer_calls):
    ns = {"train": [1.0]}
    ns["wide"] = [[i] for i in range(shared_objects._DOWN_WIDEST + 1)] + [ns["train"]]

    holders, shared = _share(["train"], ns)

    assert set(holders) == {"wide"} and shared == set()
    assert referrer_calls
