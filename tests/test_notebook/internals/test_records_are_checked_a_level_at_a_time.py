"""A statement that builds a million parsed records is checked for other
holders and for closures with state without a Python step per record.

Both checks walked the records one container at a time: seconds on top of a
first run that took two. A list of JSON-like records over plain values is
read a level at a time instead. These tests pin that the per-container walks
are not taken for such records, and are still taken for anything else.
"""

from __future__ import annotations

from cash.notebook import call_key, shared_objects


def _records(n: int) -> list:
    return [{"id": i, "tags": ["a", "b"], "meta": {"k": i}} for i in range(n)]


class _CountingSet(frozenset):
    calls = 0

    def issuperset(self, other):
        type(self).calls += 1
        return frozenset.issuperset(self, other)


def _children_asked(monkeypatch, namespace: dict) -> int:
    asked = []
    real = shared_objects.children_of

    def counting(value):
        asked.append(1)
        return real(value)

    monkeypatch.setattr(shared_objects, "children_of", counting)
    shared_objects.shared_names({"records": namespace["records"]}, (namespace,))
    return len(asked)


def test_the_holder_check_does_not_walk_each_record(monkeypatch):
    assert _children_asked(monkeypatch, {"records": _records(5_000)}) < 10


def test_the_holder_check_walks_records_another_name_reaches_into(monkeypatch):
    namespace = {"records": _records(5_000)}
    namespace["tags"] = namespace["records"][9]["tags"]

    assert _children_asked(monkeypatch, namespace) > 5_000


def test_the_closure_check_does_not_walk_each_record(monkeypatch):
    monkeypatch.setattr(_CountingSet, "calls", 0)
    monkeypatch.setattr(call_key, "_EXACT_ATOMS", _CountingSet(call_key._EXACT_ATOMS))

    assert not call_key.holds_a_closure_with_state(_records(5_000))
    assert _CountingSet.calls < 10


def test_the_closure_check_walks_records_holding_a_function(monkeypatch):
    monkeypatch.setattr(_CountingSet, "calls", 0)
    monkeypatch.setattr(call_key, "_EXACT_ATOMS", _CountingSet(call_key._EXACT_ATOMS))
    records = _records(5_000)
    records[-1]["fn"] = len

    assert not call_key.holds_a_closure_with_state(records)
    assert _CountingSet.calls > 5_000
