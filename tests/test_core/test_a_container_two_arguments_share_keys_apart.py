"""A container two arguments share keys apart from equal separate copies.

``edit(cfg, rows)`` with ``cfg["row"] is rows[0]``: written through one
argument, the change shows in the other, where two equal lists change in
one place only. Plain and JSON-like arguments are keyed by digests of their
own content, and only the top-level arguments were compared by identity, so
a list shared deeper, or a list shared by two records of one argument, keyed
like equal copies and the first caller's answer was served to the other.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cash import Cash, FileBackend


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda *args, **kwargs: c._args.hash_payload(args, kwargs)


def _shared_row():
    row = [1, 2]
    return ({"row": row}, [row]), ({"row": [1, 2]}, [[1, 2]])


def _shared_in_tuple():
    row = [1, 2]
    return ((row,), [row]), (([1, 2],), [[1, 2]])


def _argument_inside_another():
    row = [1, 2]
    return (row, [row]), (row, [[1, 2]])


def _records_sharing_a_list():
    tags = ["a"]
    return (
        ([{"id": 1, "tags": tags}, {"id": 2, "tags": tags}],),
        ([{"id": 1, "tags": ["a"]}, {"id": 2, "tags": ["a"]}],),
    )


def _record_list_passed_twice():
    tags = ["a"]
    return ([{"id": 1, "tags": tags}], tags), ([{"id": 1, "tags": ["a"]}], ["a"])


def _nested_records_sharing_a_dict():
    addr = {"city": "X"}
    return (
        ([{"a": {"addr": addr}}, {"a": {"addr": addr}}],),
        ([{"a": {"addr": {"city": "X"}}}, {"a": {"addr": {"city": "X"}}}],),
    )


def _an_array_twice_in_a_list():
    a = np.zeros(3)
    return ([a, a],), ([a, a.copy()],)


def _an_array_as_two_arguments():
    a = np.zeros(3)
    return (a, a), (a, a.copy())


@pytest.mark.parametrize(
    "make",
    [
        _shared_row,
        _shared_in_tuple,
        _argument_inside_another,
        _records_sharing_a_list,
        _record_list_passed_twice,
        _nested_records_sharing_a_dict,
        _an_array_twice_in_a_list,
        _an_array_as_two_arguments,
    ],
)
def test_shared_and_separate_key_apart(key, make):
    shared, separate = make()
    assert key(*shared) != key(*separate)
    assert key(*shared) == key(*make()[0])
    assert key(*separate) == key(*make()[1])


def test_the_edit_through_a_shared_list_gets_its_own_result(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def tags_after_edit(rows):
        snapshot = copy.deepcopy(rows)
        snapshot[0]["tags"].append("edited")
        return [r["tags"] for r in snapshot]

    tags = ["a"]
    assert tags_after_edit([{"id": 1, "tags": tags}, {"id": 2, "tags": tags}]) == [["a", "edited"]] * 2
    assert tags_after_edit([{"id": 1, "tags": ["a"]}, {"id": 2, "tags": ["a"]}]) == [["a", "edited"], ["a"]]


def test_a_record_held_elsewhere_is_not_shared(key):
    """Positive control: a variable naming one record, or a list inside it,
    only adds a reference; the value keys like a fresh copy."""
    rows = [{"id": i, "tags": [str(i)]} for i in range(50)]
    held, held_tags = rows[3], rows[4]["tags"]
    assert key(rows) == key([{"id": i, "tags": [str(i)]} for i in range(50)])
    assert held is rows[3] and held_tags is rows[4]["tags"]
