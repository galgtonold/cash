"""A loop over an object next to a big notebook does not walk the whole
notebook again before each statement.

Three small loops beside a list of 200,000 lists of words cost each run of
their cell over a second: before each, the notebook was walked for a weak
reference to, or the ``id()`` of, the object the loop assigns into, and
after each the other variables were walked for a weak reference although
only cash had made one. The route a loop takes is remembered for its
object (a hit asks again of the live objects), and the other variables are
walked only for a weak reference the user made.
"""

from __future__ import annotations

import weakref

import numpy as np
import pytest

from cash import _weak
from cash.notebook.statement import identity_holders, in_place


class _P:
    def __init__(self):
        self.n = 0
        self.seen = []


def _words(count=20_000):
    return [[f"w{j}" for j in range(10)] for _ in range(count)]


@pytest.fixture(autouse=True)
def _fresh_routes():
    in_place._ROUTED.clear()
    yield
    in_place._ROUTED.clear()


def _counting(monkeypatch):
    calls = []
    real = in_place._held_itself

    def counted(obj, namespace):
        calls.append(obj)
        return real(obj, namespace)

    monkeypatch.setattr(in_place, "_held_itself", counted)
    return calls


def test_a_loops_route_is_remembered_for_its_object(monkeypatch):
    calls = _counting(monkeypatch)
    arr = np.zeros(10)
    ns = {"arr": arr, "words": _words()}

    assert in_place.routed_as_a_copy(arr, ns)
    assert in_place.routed_as_a_copy(arr, ns)

    assert len(calls) == 1


def test_an_object_held_by_its_id_is_asked_each_time(monkeypatch):
    calls = _counting(monkeypatch)
    arr = np.zeros(10)
    ns = {"arr": arr, "reg": {id(arr): "a"}}

    assert not in_place.routed_as_a_copy(arr, ns)
    assert not in_place.routed_as_a_copy(arr, ns)

    assert len(calls) == 2


def test_a_forgotten_route_is_asked_again(monkeypatch):
    arr = np.zeros(10)
    ns = {"arr": arr}
    assert in_place.routed_as_a_copy(arr, ns)
    ns["reg"] = {id(arr): "a"}

    in_place.forget_routed(arr)

    assert not in_place.routed_as_a_copy(arr, ns)


def test_a_route_goes_with_its_object():
    arr = np.zeros(10)
    assert in_place.routed_as_a_copy(arr, {"arr": arr})
    key = id(arr)

    del arr

    assert key not in in_place._ROUTED


def test_the_reference_a_route_keeps_is_not_the_notebooks():
    arr = np.zeros(10)
    ns = {"arr": arr}
    in_place.routed_as_a_copy(arr, ns)

    assert weakref.getweakrefcount(arr)
    assert in_place.bound_as_a_copy(arr, ns)


def test_a_weak_reference_of_the_users_still_stops_the_copy():
    arr = np.zeros(10)
    ns = {"arr": arr}
    in_place.routed_as_a_copy(arr, ns)
    ns["wr"] = weakref.ref(arr)

    assert not in_place.bound_as_a_copy(arr, ns)


def test_an_id_as_a_text_key_in_a_big_list_of_dicts_is_found():
    """The garbage collector does not report a dict's text keys: one
    spelling an ``id()`` in a list too big to walk item by item went
    unseen."""
    p = _P()
    records = [{"i": i} for i in range(20_000)]
    records[777][hex(id(p.seen))] = "seen"

    assert "id()" in in_place.refusal(p, {"p": p, "records": records})


def test_an_id_in_a_big_list_of_lists_of_words_is_found():
    arr = np.zeros(10)
    words = _words()
    words[1234].append(str(id(arr)))

    assert not in_place.bound_as_a_copy(arr, {"arr": arr, "words": words})


def test_a_big_list_of_lists_of_words_does_not_stop_the_copy():
    arr = np.zeros(10)

    assert in_place.bound_as_a_copy(arr, {"arr": arr, "words": _words()})


def test_only_cashs_weak_references_do_not_walk_the_other_variables(monkeypatch):
    p = _P()
    kept = _weak.ref(p)

    def no_walk(self, roots):
        raise AssertionError("walked the other variables")

    monkeypatch.setattr(identity_holders._Walk, "run", no_walk)

    assert identity_holders._weakly_held("p", [p], {"p": p, "words": [["w"]]}) is None
    assert kept() is p


def test_a_weak_reference_of_the_users_beside_cashs_is_found():
    p = _P()
    kept = _weak.ref(p)
    ns = {"p": p, "box": {"ref": weakref.ref(p)}}

    found = identity_holders._weakly_held("p", [p], ns)

    assert found is not None and "'box'" in found
    assert kept() is p
