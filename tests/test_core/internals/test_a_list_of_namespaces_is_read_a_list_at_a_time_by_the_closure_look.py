"""The closure check reads a list of ``SimpleNamespace`` records a list at
a time, as it reads instances of a class of the user's, not one record at a
time: a million of them were read one by one, 11 s of a statement whose
value was stored. A function closing over what one record holds is still
found. Pins the work, not the behaviour: a refactor may expect this to
fail."""

from __future__ import annotations

import types

from cash import closure_sharing


def test_namespaces_are_not_read_one_at_a_time(monkeypatch):
    read: list = []
    parts = closure_sharing._object_parts

    def counted(item, kind):
        read.append(kind)
        return parts(item, kind)

    monkeypatch.setattr(closure_sharing, "_object_parts", counted)
    records = [types.SimpleNamespace(i=i, name=f"n{i}") for i in range(5_000)]
    found = closure_sharing.closures_sharing({"records": records, "f": lambda: records}, 5.0)

    assert found.finished
    assert read.count(types.SimpleNamespace) < 100, len(read)


def test_a_closure_over_what_a_record_holds_is_found():
    box = {"a": [1, 2]}
    records = [types.SimpleNamespace(i=i) for i in range(5_000)]
    records[7].box = box
    found = closure_sharing.closures_sharing({"records": records, "f": lambda: box["a"]}, 5.0)

    assert found.finished and found.names == ["f"]
