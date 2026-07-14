"""CAS-108: dict arguments equal but for insertion order must share a cache key.

Before the fix, ``f({'a': 1, 'b': 2})`` and ``f({'b': 2, 'a': 1})`` produced two
different cache keys (the stable-key canonicalisation of dict order only ran when
a set was present), so logically-identical calls missed the cache and recomputed.
Dict ordering is now canonicalised (recursively) whenever a dict argument value
is present, while list/tuple order stays semantic and set handling is untouched.
"""
from __future__ import annotations

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_permuted_flat_dict_hits(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def f(d):
        return sum(d.values())

    assert f({"a": 1, "b": 2}) == 3
    assert f({"b": 2, "a": 1}) == 3  # same dict, different insertion order
    info = f.cache_info()
    assert info["misses"] == 1, "permuted equal dicts must not recompute"
    assert info["hits"] == 1


def test_permuted_nested_dict_hits(tmp_path):
    c = _cash(tmp_path)

    @c.cache
    def f(d):
        return d

    assert f({"x": {"a": 1, "b": 2}, "y": 3}) == {"x": {"a": 1, "b": 2}, "y": 3}
    # Outer keys reordered AND inner dict keys reordered -> still a hit.
    f({"y": 3, "x": {"b": 2, "a": 1}})
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


def test_dict_inside_list_value_hits(tmp_path):
    """A dict nested inside a list argument is canonicalised; the list order
    itself is preserved (semantic)."""
    c = _cash(tmp_path)

    @c.cache
    def f(items):
        return len(items)

    assert f([{"a": 1, "b": 2}]) == 1
    f([{"b": 2, "a": 1}])
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


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


def test_unsortable_mixed_type_keys_do_not_crash(tmp_path):
    """Mixed-type dict keys (int + str) are not directly orderable; the
    pickle-bytes sort must handle them without crashing, and a permuted copy
    must still hit."""
    c = _cash(tmp_path)

    @c.cache
    def f(d):
        return len(d)

    assert f({1: "a", "b": 2}) == 2
    assert f({"b": 2, 1: "a"}) == 2  # permuted mixed-type keys -> hit, no crash
    info = f.cache_info()
    assert info["misses"] == 1
    assert info["hits"] == 1


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
