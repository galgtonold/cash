"""A statement that builds a million parsed records is checked for other
holders and for closures with state without a Python step per record.

Both checks walked the records one container at a time: seconds on top of a
first run that took two; so did, with numpy loaded, the check for a numpy view
inside an output. A list of JSON-like records over plain values is
read a level at a time instead. These tests pin that the per-container walks
are not taken for such records, and are still taken for anything else.
"""

from __future__ import annotations

import pytest

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


def test_the_holder_check_reads_records_another_name_reaches_into_a_level_at_a_time(monkeypatch):
    """Only the list another name holds is taken on its own: it keeps the
    records shared, as the walk found."""
    namespace = {"records": _records(5_000)}
    namespace["tags"] = namespace["records"][9]["tags"]

    assert _children_asked(monkeypatch, namespace) < 10
    assert shared_objects.shared_names({"records": namespace["records"]}, (namespace,)) == {"records"}


def test_the_holder_check_walks_records_holding_an_object(monkeypatch):
    import types

    namespace = {"records": _records(5_000)}
    namespace["records"][-1]["obj"] = types.SimpleNamespace(k=1)

    assert _children_asked(monkeypatch, namespace) > 5_000


def test_the_holder_check_walks_records_two_names_reach_into_one_inside_the_other(monkeypatch):
    """The record and a list inside it, each held by a name of the group:
    counted from both, the edge between them would count twice."""
    namespace = {"records": _records(5_000)}
    namespace["rec"] = namespace["records"][9]
    namespace["tags"] = namespace["records"][9]["tags"]
    asked = []
    real = shared_objects.children_of

    def counting(value):
        asked.append(1)
        return real(value)

    monkeypatch.setattr(shared_objects, "children_of", counting)
    roots = {name: namespace[name] for name in ("records", "rec", "tags")}

    assert shared_objects.shared_names(roots, (namespace,)) == set()
    assert len(asked) > 5_000


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


def _alias_walks(monkeypatch) -> list:
    from cash.notebook.statement import derivation_edges

    walks = []
    real = derivation_edges._aliases_in

    def counting(value, *args):
        walks.append(1)
        return real(value, *args)

    monkeypatch.setattr(derivation_edges, "_aliases_in", counting)
    return walks


def test_the_view_check_does_not_walk_records(monkeypatch):
    """With numpy loaded, the check for a numpy view held inside an output
    walked a million records a container at a time: 2.4 s."""
    pytest.importorskip("numpy")
    from cash.notebook.statement.derivation_edges import is_uncacheable_alias

    walks = _alias_walks(monkeypatch)
    records = _records(5_000)
    assert not is_uncacheable_alias(records, {"records": records})
    assert walks == []


def test_the_view_check_still_finds_a_view_inside_records(monkeypatch):
    np = pytest.importorskip("numpy")
    from cash.notebook.statement.derivation_edges import is_uncacheable_alias

    walks = _alias_walks(monkeypatch)
    base = np.arange(10)
    records = _records(5_000)
    records[-1]["window"] = base[:3]
    assert is_uncacheable_alias(records, {"records": records, "base": base})
    assert walks == [1]


def _counting_children(monkeypatch) -> list:
    asked: list = []
    real = shared_objects.children_of

    def counting(value):
        asked.append(1)
        return real(value)

    monkeypatch.setattr(shared_objects, "children_of", counting)
    return asked


def test_records_a_loop_variable_holds_one_of_are_not_walked(monkeypatch):
    """``for r in records:`` leaves ``r`` bound to the last record: the check
    of the loop's outputs walked every record, about 12 us each, on every run."""
    namespace = {"records": _records(5_000)}
    namespace["r"] = namespace["records"][-1]
    asked = _counting_children(monkeypatch)

    holders, shared = shared_objects.share_group(["r"], {"r": namespace["r"]}, namespace)

    assert (sorted(holders), shared) == (["records"], set())
    assert len(asked) < 50


def _parsed(n: int) -> list:
    import datetime

    when = datetime.datetime(2020, 1, 1)
    return [(f"u{i}", [("a", when), ("b", when)]) for i in range(n)]


def test_pairs_a_loop_variable_holds_a_list_of_are_not_walked(monkeypatch):
    """``for (name, act) in parsed:`` then ``parsed = sorted(parsed, ...)``:
    every pair was walked, twice, and once more for the closure check."""
    namespace = {"parsed": _parsed(5_000)}
    namespace["name"], namespace["act"] = namespace["parsed"][-1]
    asked = _counting_children(monkeypatch)

    holders, shared = shared_objects.share_group(["parsed"], {"parsed": namespace["parsed"]}, namespace)

    assert (sorted(holders), shared) == (["act"], set())
    assert len(asked) < 50


def test_the_rounds_of_the_holder_search_read_the_records_once(monkeypatch):
    from cash import _plain_data

    namespace = {"parsed": _parsed(5_000)}
    namespace["act"] = namespace["parsed"][-1][1]
    reads = []
    real = _plain_data.held_beyond_parents

    def counting(value, *args):
        if value is namespace["parsed"]:
            reads.append(1)
        return real(value, *args)

    monkeypatch.setattr(_plain_data, "held_beyond_parents", counting)

    holders, _shared = shared_objects.share_group(["parsed"], {"parsed": namespace["parsed"]}, namespace)

    assert sorted(holders) == ["act"]
    assert reads == [1]


def test_where_a_variable_holds_part_of_records_is_found_without_walking_them(monkeypatch):
    from cash.notebook import holder_patches

    parsed = _parsed(5_000)
    act = parsed[1234][1]
    steps = []
    real = holder_patches._steps

    def counting(value, *args):
        steps.append(1)
        return real(value, *args)

    monkeypatch.setattr(holder_patches, "_steps", counting)
    asked = _counting_children(monkeypatch)

    patches = holder_patches.holder_patches({"act": act}, {"parsed": parsed})

    assert patches["act"].places == (((), "parsed", (("item", 1234), ("item", 1))),)
    assert len(steps) + len(asked) < 50
