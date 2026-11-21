"""The identity check reads JSON-like data (exact dicts, lists and tuples
over strings, numbers and dates) for a number or a text that could be an
``id()`` in one pass at C speed, without making a list of each of its
levels.

A million ``{'i': i, 'v': [i, i + 1]}`` records were listed level by level
(`_plain_data.tree_walk`) and their ints picked out of each level in
Python: 0.85 s after the statement that made them, against 0.4 s read
in C. The same numbers and texts are found: an ``id()`` of the notebook's
object kept in a record, as a number, in hex, as a dict key or on a line
of its own in a longer text, still sends the statement to run again.
"""

from __future__ import annotations

import pytest

from cash import _native
from cash.notebook.statement import identity_holders as ih

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")


class Thing:
    pass


def _records(n: int = 20_000) -> list:
    return [{"i": i, "v": [i, i + 1], "tag": ("a", f"row{i}")} for i in range(n)]


def test_records_are_not_listed_a_level_at_a_time(monkeypatch):
    def refused(value):
        raise AssertionError("the records were listed level by level")

    monkeypatch.setattr(ih, "tree_walk", refused)
    records = _records()
    assert ih.holds_no_ids(records)
    assert ih.identity_held_reason("records", records, {"records": records}) is None


@pytest.mark.parametrize(
    "keep",
    [
        lambda rows, n: rows[7]["v"].append(n),
        lambda rows, n: rows[9].update(tag=("a", hex(n))),
        lambda rows, n: rows[11].update({n: "key"}),
        lambda rows, n: rows[13].update(note="seen at\n" + str(n) + "\nlater, " + "x" * 40),
    ],
    ids=["a number", "hex text", "a dict key", "a line of a long text"],
)
def test_an_id_kept_in_a_record_is_still_found(keep):
    thing = Thing()
    records = _records(2_000)
    keep(records, id(thing))
    assert not ih.holds_no_ids(records)
    found = ih.identity_held_reason("records", records, {"records": records, "thing": thing})
    assert found is not None and found.kind == "an id()"
