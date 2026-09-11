"""cash's scan of its own cache directory is nobody's dependency.

Round 19 (r19s2): in each worker of a process pool, the first cached call whose
nested cached call stored a result recomputed on the next run -- "file
changed: <cache dir> (size changed)". The nested store scanned the cache
directory (its size, what to evict) while the outer call's tracker was live,
and the directory itself became the outer entry's dependency.
"""
from __future__ import annotations

import os
import time

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


def test_a_nested_store_does_not_make_the_cache_dir_an_outer_dependency(tmp_path, monkeypatch):
    """In a pool worker a write runs inline, before the task's result goes
    back -- forced here, so the scan it makes happens on the calling thread."""
    from cash.backends import _base
    monkeypatch.setattr(_base, "_in_multiprocessing_child", lambda: True)
    cache_dir = tmp_path / "custom_cache_name"          # not ".cash": no name-based guard
    c = Cash(cache_dir=str(cache_dir), register_magic=False)
    runs = []

    @c.cache(assume_safe=True)
    def inner(i):
        time.sleep(0.12)                               # past the floor: it goes to disk
        return i * 2

    @c.cache(assume_safe=True)
    def outer(i):
        runs.append(i)
        time.sleep(0.12)
        return inner(i) + 1

    assert outer(1) == 3
    explanation = outer.explain(1)
    deps = (explanation.details or {}).get("file_deps") or {}
    norm = os.path.normcase(str(cache_dir)).replace("\\", "/").rstrip("/")
    assert not any(os.path.normcase(p).replace("\\", "/").rstrip("/") == norm for p in deps), deps
    assert explanation.would_hit
