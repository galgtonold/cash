"""The RAM tier copies nested plain data one container at a time.

A parsed log of 87,000 sessions, each a name and a list of (action, datetime)
pairs, is 175,000 lists and tuples around two million leaves. ``deepcopy``
and a pickle round trip both pay per leaf, and a datetime is slow to pickle:
14 to 25 s on every store and every RAM hit, for a cell that runs in 0.5 s.
Only what can be written into needs a new object, which is the lists and the
tuples holding one; and a copy must still be one, with sharing kept as it was.
"""

from __future__ import annotations

import copy
import datetime
import gc
import pickle

import pytest

from cash import _plain_data
from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]


def _log(n: int = 50) -> list:
    stamp = datetime.datetime(2019, 10, 15, 18, 8, 2)
    return [
        (f"@User{i}", [(f"Action_{j}", stamp + datetime.timedelta(minutes=j)) for j in range(i % 7 + 1)])
        for i in range(n)
    ]


def _no_per_leaf_copy(monkeypatch) -> list[int]:
    """Make a pickle round trip fail the test, and count every ``deepcopy`` call.

    The recursive calls of ``deepcopy`` go through the module's own name, so
    the count includes the per-element ones: a walk over the log's leaves is
    hundreds of calls, a copy that found the log in the memo is a handful.
    """
    calls: list[int] = []
    real = copy.deepcopy

    def counting(x, memo=None, _nil=[]):
        calls.append(1)
        return real(x, memo)

    def refuse(*_a, **_k):
        raise AssertionError("pickled the whole value instead of copying per container")

    monkeypatch.setattr(copy, "deepcopy", counting)
    monkeypatch.setattr(_plain_data.pickle, "dumps", refuse)
    return calls


def test_a_parsed_log_is_stored_and_restored_without_a_per_leaf_copy(monkeypatch):
    b = InMemoryBackend()
    rows = _log()
    original = copy.deepcopy(rows)
    calls = _no_per_leaf_copy(monkeypatch)
    b.set("log", rows)
    first = b.get("log")[1]
    second = b.get("log")[1]
    assert first == original and second == original
    assert not calls, "a plain log went through deepcopy"
    assert first is not second
    first[3][1].append(("caller's", None))
    first[4] = ("caller's", [])
    assert b.get("log")[1] == original, "a caller's change reached the stored entry"
    rows[2][1].append(("the original's", None))
    assert b.get("log")[1] == original, "the stored entry shares the original's lists"


def test_a_log_inside_a_notebook_entry_is_copied_without_a_per_leaf_copy(monkeypatch):
    b = InMemoryBackend()
    rows = _log()
    original = copy.deepcopy(rows)
    entry = {"logs": rows, "n": 3}
    calls = _no_per_leaf_copy(monkeypatch)
    b.set("cell", entry)
    got = b.get("cell")[1]
    assert got == {"logs": original, "n": 3}
    assert len(calls) < 20, f"deepcopy walked the log: {len(calls)} calls"
    got["logs"][0][1].append(("caller's", None))
    assert b.get("cell")[1]["logs"] == original


def test_tuples_of_immutables_are_shared_and_every_list_is_not():
    rows = _log()
    copied = _plain_data.spine_copy(rows)
    assert copied == rows and copied is not rows
    for old, new in zip(rows, copied):
        assert new is not old, "a row holds a list, so it needs a new tuple"
        assert new[1] is not old[1]
        assert all(a is b for a, b in zip(old[1], new[1])), "a pair of immutables is shared"


def test_a_tuple_with_nothing_writable_below_is_returned_as_it_is():
    value = ((1, 2), ("a", (datetime.date(2020, 1, 1),)))
    assert _plain_data.spine_copy(value) is value


def test_every_level_is_copied_when_lists_and_tuples_are_mixed():
    value = [1, (2, [3, (4, [5])]), [6, [7, (8,)]], "x"]
    copied = _plain_data.spine_copy(value)
    assert copied == value
    assert copied[1][1] is not value[1][1] and copied[1][1][1][1] is not value[1][1][1][1]
    assert copied[2][1] is not value[2][1]
    assert copied[2][1][1] == (8,)  # shared or rebuilt: it is judged a level at a time, and a list sits beside it
    copied[1][1][1][1].append(99)
    copied[2][1].append(99)
    assert value == [1, (2, [3, (4, [5])]), [6, [7, (8,)]], "x"]


def test_a_list_held_twice_is_left_to_a_copy_that_keeps_it_one_list():
    shared = [1, 2]
    value = [("a", shared), ("b", shared), ("c", [shared])]
    assert _plain_data.spine_copy(value) is None
    b = InMemoryBackend()
    b.set("k", value)
    got = b.get("k")[1]
    assert got == value
    assert got[0][1] is got[1][1] is got[2][1][0], "the one list came back as three"
    assert got[0][1] is not shared


def test_a_bytearray_leaf_is_left_to_the_pickle_copy():
    value = [("a", [(1, bytearray(b"ab"))])]
    assert _plain_data.spine_copy(value) is None
    b = InMemoryBackend()
    b.set("k", value)
    b.get("k")[1][0][1][0][1][0] = ord("z")
    assert b.get("k")[1] == [("a", [(1, bytearray(b"ab"))])]


def test_anything_that_is_not_plain_data_is_refused():
    assert _plain_data.spine_copy({"a": [1]}) is None
    assert _plain_data.spine_copy([("a", {"b": 1})]) is None
    assert _plain_data.spine_copy([("a", [object()])]) is None
    assert _plain_data.spine_copy("text") is None


def test_the_memo_holds_what_was_copied_and_nothing_that_was_shared():
    rows = _log(10)
    memo: dict[int, object] = {}
    copied = _plain_data.spine_copy(rows, memo)
    assert memo[id(rows)] is copied
    for old, new in zip(rows, copied):
        assert memo[id(old)] is new and memo[id(old[1])] is new[1]
        assert id(old[1][0]) not in memo, "a shared pair needs no entry"


def test_a_name_bound_to_a_row_keeps_sharing_it_with_the_copy():
    """`first = rows[0][1]` next to `rows` in one entry: after a restore, a
    change through `first` is still a change to `rows`."""
    b = InMemoryBackend()
    rows = _log()
    b.set("cell", {"rows": rows, "first": rows[0][1]})
    got = b.get("cell")[1]
    assert got["first"] is got["rows"][0][1]
    assert got["rows"] is not rows and got["first"] is not rows[0][1]
    got["first"].append(("seen through rows", None))
    assert got["rows"][0][1][-1] == ("seen through rows", None)


@pytest.mark.parametrize("was_on", [True, False])
def test_the_collector_is_left_as_it_was_found(was_on):
    before = gc.isenabled()
    try:
        gc.enable() if was_on else gc.disable()
        _plain_data.spine_copy(_log(5))
        assert gc.isenabled() is was_on
        assert _plain_data.spine_copy([1, [2], {"x": 1}]) is None
        assert gc.isenabled() is was_on
    finally:
        gc.enable() if before else gc.disable()


def test_the_copy_equals_what_pickle_would_have_made():
    rows = _log(200)
    assert _plain_data.spine_copy(rows) == pickle.loads(pickle.dumps(rows))
