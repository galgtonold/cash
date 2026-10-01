"""A dict argument keys on its insertion order; a set does not.

A dict's order is part of what code reads: ``pd.DataFrame(d)`` orders its
columns by it, ``json.dumps`` and ``csv.DictWriter`` write in it. Sorting a
dict's items before keying made ``{"name": .., "score": ..}`` and its
reordering share an entry, and the second call was served the first one's
column order. Equal dicts built in two orders now cost one miss; a set, whose
order nothing can read back, still keys the same in any order.
"""

from __future__ import annotations

import pytest

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_a_reordered_dict_gets_its_own_result(tmp_path):
    pd = pytest.importorskip("pandas")
    c = _cash(tmp_path)

    @c.cache
    def frame_columns(data):
        return list(pd.DataFrame(data).columns)

    assert frame_columns({"name": ["x"], "score": [1]}) == ["name", "score"]
    assert frame_columns({"score": [1], "name": ["x"]}) == ["score", "name"]


def test_a_reordered_nested_dict_gets_its_own_result(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def keys(d):
        return [list(d), list(d["x"])]

    assert keys({"x": {"a": 1, "b": 2}, "y": 3}) == [["x", "y"], ["a", "b"]]
    assert keys({"x": {"b": 2, "a": 1}, "y": 3}) == [["x", "y"], ["b", "a"]]


def test_rows_of_reordered_dicts_get_their_own_header(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def header(rows):
        return ",".join(rows[0])

    assert header([{"id": 1, "name": "x"}]) == "id,name"
    assert header([{"name": "x", "id": 1}]) == "name,id"
    assert header([{"id": 1, "name": "x"}, {"name": "y", "id": 2}]) == "id,name"
    assert header([{"name": "y", "id": 2}, {"id": 1, "name": "x"}]) == "name,id"


def test_forwarded_keyword_arguments_keep_their_order(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def columns(**cols):
        return list(cols)

    assert columns(a=1, b=2) == ["a", "b"]
    assert columns(b=2, a=1) == ["b", "a"]


def test_named_arguments_in_any_order_hit(tmp_path):
    """Named parameters are bound to the signature, so the order a call
    writes them in is not an order the body can read."""
    c = _cash(tmp_path)

    @c.cache
    def f(rows, n):
        return rows[:n]

    assert f(rows=[(1, "a")], n=2) == [(1, "a")]
    assert f(n=2, rows=[(1, "a")]) == [(1, "a")]
    assert f.cache_info()["hits"] == 1


def test_an_equal_dict_in_the_same_order_hits(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def f(d):
        return sum(d.values())

    assert f({"a": 1, "b": 2}) == 3
    assert f({"a": 1, "b": 2}) == 3
    assert f.cache_info()["hits"] == 1


def test_different_dict_contents_still_miss(tmp_path):
    """Canonicalising order must NOT collapse genuinely different dicts."""
    c = _cash(tmp_path)

    @c.cache
    def f(d):
        return dict(d)

    assert f({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert f({"a": 1, "b": 3}) == {"a": 1, "b": 3}  # different value -> miss
    info = f.cache_info()
    assert info["misses"] == 2, "different contents must not collide"
    assert info["hits"] == 0


def test_list_order_is_semantic(tmp_path):
    """Lists are ordered data: [1, 2] and [2, 1] must remain distinct keys."""
    c = _cash(tmp_path)

    @c.cache
    def f(xs):
        return xs

    assert f([1, 2]) == [1, 2]
    assert f([2, 1]) == [2, 1]
    info = f.cache_info()
    assert info["misses"] == 2
    assert info["hits"] == 0


def test_mixed_type_keys_do_not_crash(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def f(d):
        return len(d)

    assert f({1: "a", "b": 2}) == 2
    assert f({1: "a", "b": 2}) == 2
    assert f.cache_info()["hits"] == 1


def test_set_argument_still_canonicalises(tmp_path):
    """Guard: broadening the dict path must not regress set canonicalisation -
    two equal sets in different iteration order still share a key."""
    c = _cash(tmp_path)

    @c.cache
    def f(s):
        return sum(s)

    assert f({1, 2, 3}) == 6
    f({3, 2, 1})
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1
