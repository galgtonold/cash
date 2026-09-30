"""A list of nested JSON-like records is keyed by its content at C speed.

Records with a list or a dict inside are neither plain rows nor flat dict
rows, so a warm hit walked every container in Python, three calls each:
about 20x the body for 100k records. Exact dicts, lists and tuples over the
leaves of plain data are now checked a level at a time and pickled whole.
"""

from __future__ import annotations

import time

import pytest

from cash import Cash, object_hashing
from cash.decorator import arg_hashing

pytestmark = [pytest.mark.core]


def _records(n):
    return [{"id": i, "name": f"user{i}", "tags": ["a", "b"], "addr": {"city": "X", "zip": str(i)}} for i in range(n)]


@pytest.fixture
def key(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    return lambda *args, **kwargs: c._args.hash_payload(args, kwargs)


def test_a_warm_hit_on_many_records_does_not_walk_them(tmp_path, monkeypatch):
    """Counted, not timed: none of the three walks is entered per record."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    rows = _records(20_000)

    @c.cache
    def count(rows):
        return len(rows)

    count(rows)
    calls = {"contains_set": 0, "stable_key_repr": 0, "carriers": 0}
    for name in ("contains_set", "stable_key_repr"):
        real = getattr(object_hashing, name)
        monkeypatch.setattr(
            object_hashing, name, lambda *a, _r=real, _n=name, **k: calls.__setitem__(_n, calls[_n] + 1) or _r(*a, **k)
        )
    real_iter = c._code_args.iter_code_carriers
    monkeypatch.setattr(
        c._code_args,
        "iter_code_carriers",
        lambda *a, **k: calls.__setitem__("carriers", calls["carriers"] + 1) or real_iter(*a, **k),
    )
    t0 = time.perf_counter()
    assert count(rows) == 20_000
    assert max(calls.values()) < 50, f"walked the records: {calls}"
    assert time.perf_counter() - t0 < 30  # a hang guard, not a benchmark


def test_records_key_by_content(key):
    rows = _records(100)
    assert arg_hashing.plain_census(rows)[0] == "tree"
    assert key(rows) == key(_records(100))
    changed = _records(100)
    changed[50]["addr"]["zip"] = "elsewhere"
    assert key(rows) != key(changed)
    reordered = _records(100)
    reordered[0] = {"name": "user0", "id": 0, "tags": ["a", "b"], "addr": {"city": "X", "zip": "0"}}
    assert key(rows) != key(reordered), "a dict's order is read by json.dumps and DataFrame"
    as_tuple = _records(100)
    as_tuple[0]["tags"] = ("a", "b")
    assert key(rows) != key(as_tuple)
    assert key({"a": [1]}) != key({"a": (1,)})
    assert key({1: "a"}) != key({"1": "a"})
    assert key({"a": 1}) != key({"a": True})
    assert key({"a": 1}) != key({"a": 1.0})


def test_what_is_not_json_like_takes_the_general_path(key):
    assert arg_hashing.plain_census([{"a": {1, 2}}]) is None
    assert arg_hashing.plain_census([{"a": object()}]) is None
    assert arg_hashing.plain_census([{("k",): 1}]) is None
    assert arg_hashing.plain_census([type("D", (dict,), {})(a=1)]) is None
    cyclic: dict = {}
    cyclic["self"] = cyclic
    assert arg_hashing.plain_census(cyclic) is None
    assert key([{"a": {1, 2}}]) == key([{"a": {2, 1}}])
