"""A wide list the share check reads by identity is read by the types of its
items first: one with no item of a wanted object's type, and no record (a
tuple), holds none of them, and its ids are not read.

student_7's ``data``, a list of 87,572 strings, was read id by id and type
by type for every statement that wrote a frame (40 ms each time, ~10 s over
the notebook). The answer is the same: an object found among the items, or
in a record among them, still names the list's variable.
"""

from __future__ import annotations

import pandas as pd
import pytest

import cash.notebook.shared_objects as shared_objects

WIDE = 5_000  # past `_DOWN_WIDEST`: read by identity, not item by item


class _Ids(set):
    """The wanted ids, noting each time they are compared with the items."""

    reads = 0

    def isdisjoint(self, other):
        type(self).reads += 1
        return super().isdisjoint(other)


@pytest.fixture(params=["kernel", "python"])
def type_reader(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(shared_objects, "_TYPE_SET", None)
    elif shared_objects._TYPE_SET is None:
        pytest.skip("this build has no C kernels")
    _Ids.reads = 0
    return request.param


def _holds(items, target):
    return shared_objects._holds_by_id(items, _Ids({id(target)}), {type(target)})


@pytest.mark.parametrize("container", [list, tuple])
def test_strings_only_are_not_read_by_id(type_reader, container):
    frame = pd.DataFrame({"a": [1, 2]})
    rows = container(f"line {n}" for n in range(WIDE))
    assert _holds(rows, frame) is False
    assert _Ids.reads == 0


@pytest.mark.parametrize("container", [list, tuple])
def test_the_object_among_the_items_is_found(type_reader, container):
    frame = pd.DataFrame({"a": [1, 2]})
    rows = container([*(f"line {n}" for n in range(WIDE)), frame])
    assert _holds(rows, frame) is True


def test_an_other_object_of_the_same_type_is_not_it(type_reader):
    frame = pd.DataFrame({"a": [1, 2]})
    rows = [*(f"line {n}" for n in range(WIDE)), pd.DataFrame({"a": [1, 2]})]
    assert _holds(rows, frame) is False
    assert _Ids.reads == 1


def test_the_object_in_a_record_is_found(type_reader):
    actions = ["Action_1", "Action_2"]
    logs = [*(f"line {n}" for n in range(WIDE)), ("User1", actions)]
    assert _holds(logs, actions) is True


def test_the_variable_holding_it_is_named(type_reader):
    frame = pd.DataFrame({"a": [1, 2]})
    user_ns = {"data": [f"line {n}" for n in range(WIDE)], "rows": [*(f"r{n}" for n in range(WIDE)), frame]}
    assert shared_objects._holders_from_names([frame], set(), user_ns, set()) == {"rows"}
