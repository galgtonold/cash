"""The identity check reads a big list of records -- ``SimpleNamespace``
objects, instances of a plain class of the notebook's, dicts holding sets
-- a level at a time, at C speed, rather than opening each object.

1.5 million ``SimpleNamespace`` records were opened an object at a time:
26 s for a statement whose value was going to be stored, and a share
check out of its time, and a warning, for one with less time to spare.
Read a level at a time, the same ids are found: an ``id()`` of the
notebook's object kept in a record's attribute or in a set still sends
the statement to run again, and a record a weak reference points at is
walked an object at a time, as before.
"""

from __future__ import annotations

import types
import weakref

import pytest

from cash.notebook.statement import identity_holders as ih


class Record:
    def __init__(self, i: int) -> None:
        self.i = i
        self.name = f"n{i}"


class Thing:
    pass


def _namespaces(n: int) -> list:
    return [types.SimpleNamespace(i=i, x=i * 0.5, name=f"n{i}") for i in range(n)]


def _with_sets(n: int) -> list:
    return [{"i": i, "tags": {i % 7, i % 11}} for i in range(n)]


def _instances(n: int) -> list:
    return [Record(i) for i in range(n)]


@pytest.fixture
def opened(monkeypatch) -> list[int]:
    """How many objects the walk opened one at a time."""
    count = [0]
    real = ih._Walk._open

    def counted(self, item, into):
        count[0] += 1
        return real(self, item, into)

    monkeypatch.setattr(ih._Walk, "_open", counted)
    return count


@pytest.mark.parametrize("make", [_namespaces, _with_sets, _instances], ids=["namespaces", "sets", "instances"])
def test_records_are_not_opened_one_at_a_time(opened, make):
    records = make(20_000)

    assert ih.identity_held_reason("records", records, {"records": records}) is None
    assert opened[0] < 1_000, opened[0]


@pytest.mark.parametrize(
    "keep",
    [
        lambda rows, n: setattr(rows[7], "owner", n),
        lambda rows, n: setattr(rows[9], "owner", [n]),
    ],
    ids=["an attribute", "a list in an attribute"],
)
@pytest.mark.parametrize("make", [_namespaces, _instances], ids=["namespaces", "instances"])
def test_an_id_kept_in_a_record_is_still_found(make, keep):
    thing = Thing()
    records = make(20_000)
    keep(records, id(thing))

    found = ih.identity_held_reason("records", records, {"records": records, "thing": thing})
    assert found is not None and found.kind == "an id()"


def test_an_id_kept_in_a_set_is_still_found():
    thing = Thing()
    records = _with_sets(20_000)
    records[11]["tags"].add(id(thing))

    found = ih.identity_held_reason("records", records, {"records": records, "thing": thing})
    assert found is not None and found.kind == "an id()"


def test_a_record_a_weak_reference_points_at_is_still_found():
    """Another variable finding one of the records by a weak reference
    keeps the statement from being stored, as the walk an object at a time
    found it."""
    records = _instances(20_000)
    ref = weakref.ref(records[5])

    found = ih.identity_held_reason("records", records, {"records": records, "ref": ref})
    assert found is not None and found.kind == "a weak reference"
