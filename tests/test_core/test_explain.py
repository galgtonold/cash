"""Tests for ``f.explain(*args, **kwargs)`` — explain why a call hits or misses."""
from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

from cash import Cash, CacheExplanation


def test_explain_returns_cacheexplanation_instance(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x * 2

    e = f.explain(5)
    assert isinstance(e, CacheExplanation)
    assert e.func_name.endswith("f")
    assert e.reason in {"hit", "no_entry"}


def test_explain_first_call_is_no_entry(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x * 2

    e = f.explain(5)
    assert e.would_hit is False
    assert e.reason == "no_entry"
    assert e.cache_key is not None
    assert "hint" in e.details


def test_explain_after_call_is_hit(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x * 2

    f(5)
    e = f.explain(5)
    assert e.would_hit is True
    assert e.reason == "hit"
    assert e.cache_key is not None
    assert "cached_at" in e.details


def test_explain_does_not_call_function(tmp_path):
    """explain() is pure introspection — never invokes the function."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)
    calls = {"n": 0}

    @c.cache
    def f(x):
        calls["n"] += 1
        return x

    # Several explains, all before any real call.
    f.explain(1)
    f.explain(2)
    f.explain(3)
    assert calls["n"] == 0


def test_explain_does_not_mutate_stats(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x

    f.explain(5)
    info = f.cache_info()
    assert info["hits"] == 0
    assert info["misses"] == 0


def test_explain_unhashable_arg(tmp_path):
    """Unhashable args produce key_uncomputable with an arg_type hint."""
    class _Unpicklable:
        def __init__(self):
            self._lock = threading.Lock()

    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(obj):
        return 1

    e = f.explain(_Unpicklable())
    assert e.would_hit is False
    assert e.reason == "key_uncomputable"
    assert e.cache_key is None
    # _first_unhashable_arg_type uses __qualname__, which includes the
    # enclosing function for locally-defined classes — match the suffix.
    assert e.details.get("arg_type", "").endswith("_Unpicklable")


def test_explain_ttl_expired(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache(ttl=1)
    def f(x):
        return x * 2

    f(5)
    # Hit immediately
    e = f.explain(5)
    assert e.would_hit is True

    # Force expiry by rewriting metadata timestamp into the past.
    # The cache key is whatever the function used; fetch via cache_info path.
    # Easier: just sleep past the TTL.
    time.sleep(1.2)
    e = f.explain(5)
    assert e.would_hit is False
    assert e.reason == "ttl_expired"
    assert e.details["ttl_seconds"] == 1
    assert e.details["age_seconds"] > 1


def test_explain_file_changed(tmp_path):
    """If an auto-tracked file dep changed, explain reports file_changed."""
    data_file = tmp_path / "data.txt"
    data_file.write_text("v1")

    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)

    @c.cache
    def load():
        with open(data_file, "r") as fh:
            return fh.read()

    load()
    # Fresh: should hit.
    e = load.explain()
    assert e.would_hit is True

    # Touch the file with new content + new mtime.
    time.sleep(0.05)
    data_file.write_text("v2-with-more-chars")
    os.utime(data_file, None)

    e = load.explain()
    assert e.would_hit is False
    assert e.reason == "file_changed"
    assert any(str(data_file) in p or p.endswith("data.txt") for p in e.details["changed_files"])


def test_explain_str_format_for_hit(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x

    f(1)
    text = str(f.explain(1))
    assert "[HIT]" in text
    assert "cache_key:" in text


def test_explain_str_format_for_miss(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x

    text = str(f.explain(1))
    assert "[MISS]" in text
    assert "no_entry" in text


def test_explain_after_source_edit_returns_no_entry(tmp_path):
    """When the user redefines the function (different body, same name+args),
    explain() should report no_entry under the new source hash."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def f(x):
        return x * 2

    f(5)  # warm cache under the old source hash

    # Re-decorate a different body with the same qualname.
    @c.cache
    def f(x):  # noqa: F811 — intentional rebind to simulate source edit
        return x * 3

    e = f.explain(5)
    assert e.would_hit is False
    assert e.reason == "no_entry"


async def test_explain_on_async_wrapper(tmp_path):
    """explain() works synchronously on async-wrapped functions too."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    async def fetch(url):
        await asyncio.sleep(0)
        return f"data-{url}"

    # Before any call.
    e = fetch.explain("u")
    assert isinstance(e, CacheExplanation)
    assert e.reason == "no_entry"

    # After a call.
    await fetch("u")
    e = fetch.explain("u")
    assert e.would_hit is True
    assert e.reason == "hit"
