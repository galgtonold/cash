"""How the RAM tier keeps a big JSON-like value: as `marshal` bytes, written
once on the store and read once per hit, with no copy a container at a time.

`spine_copy` made a Python step per container on the store and again on each
hit (the hit through `copy_plan`), plus a walk to size the value: for a
million nested records 1.7 s, 2.3 s and 0.8 s. These pin that none of them
runs for such a value, and that a small one, or one marshal would not give
back as it was, still takes the copy it took.
"""

from __future__ import annotations

import datetime

import pytest

from cash import _plain_data
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend


def _records(n: int) -> list:
    return [{"id": i, "tags": ["a", "b"], "meta": {"k": i}} for i in range(n)]


def _reaches(value, target, depth: int = 0) -> bool:
    if value is target:
        return True
    if depth > 3 or type(value) not in (dict, list, tuple) or len(value) > 64:
        return False
    items = value.values() if type(value) is dict else value
    return any(_reaches(item, target, depth + 1) for item in items)


def _counting(monkeypatch, *names, reaching=None):
    """The calls of *names*, those whose value reaches *reaching* when given."""
    calls: list[str] = []
    for name in names:
        real = getattr(_plain_data, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            if reaching is None or (args and _reaches(args[0], reaching)):
                calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(_plain_data, name, counted)
    return calls


def test_a_big_value_is_stored_and_read_without_a_copy_a_container_at_a_time(monkeypatch):
    calls = _counting(monkeypatch, "_spine_copy", "_tree_walk", "copy_plan", "tree_size")
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"records": _records(5_000)}, "stdout": ""})
    for _ in range(2):
        assert backend.get("k")[1]["variables"]["records"] == _records(5_000)
    assert calls == []
    assert type(backend._store["k"][1]) is memory_backend._Marshalled


def test_its_size_is_the_size_of_the_value():
    """The control: sized from the one walk, it is the size the RAM tier gave it before."""
    value = {"variables": {"records": _records(5_000)}}
    backend = InMemoryBackend()
    backend.set("k", value)
    assert backend._store["k"][0]["size"] == _plain_data.tree_size(value)


def test_a_value_with_few_containers_is_copied_as_before(monkeypatch):
    calls = _counting(monkeypatch, "_spine_copy")
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"records": _records(100)}})
    assert calls == ["_spine_copy"]
    assert type(backend._store["k"][1]) is not memory_backend._Marshalled


def test_a_value_marshal_would_change_is_copied_as_before(monkeypatch):
    """marshal writes a bytearray as bytes and has no dates."""
    for leaf in (bytearray(b"ab"), datetime.date(2026, 10, 8)):
        backend = InMemoryBackend()
        backend.set("k", [{"id": i, "leaf": leaf} for i in range(5_000)])
        assert type(backend._store["k"][1]) is not memory_backend._Marshalled
        assert type(backend.get("k")[1][0]["leaf"]) is type(leaf)


def test_one_walk_inside_a_look_serves_every_check_and_the_store(monkeypatch):
    """`one_look`: the share check, the closure check and the store of the
    payload around the value walk the records' rows once between them."""
    records = _records(5_000)
    rows_walked = []
    real = _plain_data._keys_size

    def counting(dicts):
        if len(dicts) == len(records) and dicts[0] is records[0]:
            rows_walked.append(1)
        return real(dicts)

    monkeypatch.setattr(_plain_data, "_keys_size", counting)
    calls = _counting(monkeypatch, "tree_levels", "_tree_walk")
    named = {"records": records}
    with _plain_data.one_look(named):
        assert _plain_data.held_only_by_parents(records, (int, str))
        assert _plain_data.is_tree(records)
        InMemoryBackend().set("k", {"variables": dict(named), "stdout": ""})
    assert rows_walked == [1]
    assert calls == []


def test_outside_a_look_or_for_a_value_it_is_not_about_the_checks_walk_as_before(monkeypatch):
    calls = _counting(monkeypatch, "tree_levels")
    records = _records(100)
    assert _plain_data.is_tree(records)
    with _plain_data.one_look({"other": [1]}):
        assert _plain_data.held_only_by_parents(records, (int, str))
    assert calls == ["tree_levels", "tree_levels"]


def test_a_look_answers_for_the_object_its_name_binds_now():
    named = {"records": _records(10)}
    with _plain_data.one_look(named):
        assert _plain_data.is_tree(named["records"])
        named["records"] = [len]
        assert not _plain_data.is_tree(named["records"])


def test_a_look_holds_no_reference_to_the_values():
    import sys

    records = _records(10)
    before = sys.getrefcount(records)
    with _plain_data.one_look({"records": records}):
        _plain_data.tree_facts(records)
        assert sys.getrefcount(records) == before + 1  # the dict passed in


def test_records_beside_an_array_are_kept_as_bytes_and_sized_from_one_walk(monkeypatch):
    np = pytest.importorskip("numpy")
    records = _records(5_000)
    payload = {"variables": {"records": records}, "rng_state": {"numpy.random": np.random.get_state()}}
    calls = _counting(monkeypatch, "_spine_copy", "_tree_walk", "tree_size", reaching=records)
    backend = InMemoryBackend()
    backend.set("k", payload)
    assert calls == [], "the records are walked by tree_facts alone"
    assert "k" in backend._holds_bytes
    stored = backend._store["k"][1]["variables"]["records"]
    assert type(stored) is memory_backend._Marshalled
    assert backend.get("k")[1]["variables"]["records"] == records
    assert backend.peek_entry("k")[1]["variables"]["records"] == records


def test_records_another_name_reaches_into_are_not_kept_apart():
    np = pytest.importorskip("numpy")
    records = _records(5_000)
    payload = {"variables": {"records": records, "tags": records[3]["tags"]}, "arr": np.arange(3)}
    backend = InMemoryBackend()
    backend.set("k", payload)
    assert "k" not in backend._holds_bytes
