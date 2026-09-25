"""A registered hasher keys its type wherever the value sits in an argument.

It was consulted for each top-level argument only: a ``Store`` holding a
lock inside a list was pickled instead, the call never cached, and the
warning told the user to register the hasher they had registered. Built-in
content hashers now apply inside containers too, so a frame in a list keys
as a frame argument does.
"""

from __future__ import annotations

import threading
import warnings

import pytest

from cash import Cash, FileBackend


class Store:
    def __init__(self, url):
        self.url = url
        self._lock = threading.Lock()


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_a_hasher_keys_values_inside_a_list(tmp_path):
    c = _cash(tmp_path)
    c.register_hasher(Store, lambda s: s.url)

    @c.cache
    def fetch_all(stores):
        return [s.url for s in stores]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fetch_all([Store("db://a")]) == ["db://a"]
        assert fetch_all([Store("db://a")]) == ["db://a"]
        assert fetch_all({"main": [Store("db://b")]}.get("main")) == ["db://b"]
    assert fetch_all.cache_info()["hits"] == 1


def test_a_hasher_keys_values_inside_a_dict_and_a_subclass_state(tmp_path):
    c = _cash(tmp_path)
    c.register_hasher(Store, lambda s: s.url)

    @c.cache
    def names(stores):
        return sorted(stores)

    names({"a": Store("db://a")})
    names({"a": Store("db://a")})
    assert names.cache_info()["hits"] == 1
    names({"a": Store("db://other")})
    assert names.cache_info()["misses"] == 2


def test_a_frame_in_a_list_keys_by_content(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    import cash.object_hashing as object_hashing

    c = _cash(tmp_path)
    frame = pd.DataFrame({"a": [1, 2]})
    key = c._args.hash_payload(([frame],), {})
    assert key == c._args.hash_payload(([frame.copy()],), {})
    assert key != c._args.hash_payload(([frame.rename(columns={"a": "b"})],), {})
    calls = []
    real = object_hashing.hash_pandas

    def counting(value):
        calls.append(1)
        return real(value)

    monkeypatch.setattr(object_hashing, "hash_pandas", counting)
    c._args.hash_payload(([frame],), {})
    assert calls, "a frame inside a list was pickled, not content-hashed"
