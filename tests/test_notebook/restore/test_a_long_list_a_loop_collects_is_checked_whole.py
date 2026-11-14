"""A long list a loop collects is checked whole, not given up on.

``deltas += list(group['t'].diff())`` over a grouped frame of times
collects a million pandas Timedeltas; ``list(arr)`` a million numpy floats;
``results.append(parse(line))`` over a file a dict per line. Each is
restored as a copy, so the identity walk must rule out a weak reference or
an ``id()`` anywhere in it. It counted each scalar and each dict against a
fixed count of objects and gave up past it, so the loop ran again on every
run, minutes for some. A numpy or pandas scalar holds a number and nothing
else, and is read at C speed with the leaves; the value's own walk goes on
past the count while the share check it runs in has time left.

Positive controls: a numpy integer that is the ``id()`` of a variable's
object is still found among the scalars and in the records, and outside a
share check's time the walk still stops at its count on records holding
objects; plain records are read whole (`_plain_tree_without_ids`).
"""

from __future__ import annotations

import types

import pytest

from cash.notebook import call_key
from cash.notebook.shared_objects import walk_budget
from cash.notebook.statement.identity_holders import identity_held_reason

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

#: More objects than the identity walk counts outside a share check.
N = 300_000


def test_a_list_of_numpy_floats_holds_no_id():
    deltas = list(np.arange(N, dtype=np.float64) / 7)
    assert identity_held_reason("deltas", deltas, {"deltas": deltas}) is None


def test_a_list_of_timedeltas_holds_no_id():
    deltas = list(pd.to_timedelta(np.arange(N), unit="s"))
    assert type(deltas[0]) is pd.Timedelta
    assert identity_held_reason("deltas", deltas, {"deltas": deltas}) is None


def test_a_numpy_integer_that_is_an_id_is_still_found():
    target = [1, 2, 3]
    ids = [np.int64(i) for i in range(N)] + [np.int64(id(target))]
    held = identity_held_reason("ids", ids, {"ids": ids, "target": target})
    assert held is not None and held.kind == "an id()", held


def _records(n: int) -> list[dict]:
    return [{"line": i, "value": i * 0.5, "name": f"name{i}"} for i in range(n)]


def test_records_past_the_count_are_read_whole_within_a_share_check():
    results = _records(N)
    with walk_budget(600.0):
        assert identity_held_reason("results", results, {"results": results}) is None


def test_plain_records_past_the_count_outside_a_share_check_are_read_whole():
    """Dicts of numbers and strings: plain data, read a level at a time
    whatever their count."""
    results = _records(N)
    assert identity_held_reason("results", results, {"results": results}) is None


def test_records_of_objects_past_the_count_outside_a_share_check_are_given_up_on():
    results = [{"line": i, "at": types.SimpleNamespace(i=i)} for i in range(N)]
    held = identity_held_reason("results", results, {"results": results})
    assert held is not None and "more objects than cash looks through" in held


def test_an_id_in_the_last_record_is_found_within_a_share_check():
    target = [1, 2, 3]
    results = _records(N)
    results[-1]["ref"] = id(target)
    with walk_budget(600.0):
        held = identity_held_reason("results", results, {"results": results, "target": target})
    assert held is not None and held.kind == "an id()", held


def test_the_state_check_reads_a_million_numpy_floats_and_timedeltas():
    """Pushed a value at a time, a million scalars passed the check's
    count of values: not known, not stored."""
    assert not call_key.holds_a_closure_with_state(list(np.arange(1_000_000, dtype=np.float64)))
    assert not call_key.holds_a_closure_with_state(list(pd.to_timedelta(np.arange(1_000_000), unit="s")))
