"""explain() looks; it does not count as a use or rewrite anything.

Round 18 (r18s1 F2, r18s2): calling `f.explain(...)` moved USES and LAST USED
in `cash inspect`, and on a large entry the file backend rewrote the whole
entry file ~5 s later to persist the access stamp -- while the docs say
explain() writes nothing. It read through `backend.get`, the access path. It
now asks `peek_metadata`, which every backend answers without recording a use.
"""
from __future__ import annotations

import os
import time

import pytest

from cash import Cash, FileBackend

pytestmark = pytest.mark.core


def _entry_state(cache_dir):
    files = sorted(p for p in os.listdir(cache_dir) if p.endswith(".entry"))
    return [(name, os.stat(os.path.join(cache_dir, name)).st_mtime_ns,
             open(os.path.join(cache_dir, name), "rb").read()) for name in files]


@pytest.mark.parametrize("make", [
    lambda d: Cash(backend=FileBackend(cache_dir=d), register_magic=False),
    lambda d: Cash(cache_dir=d, register_magic=False, persist_all=True),
], ids=["file", "tiered"])
def test_explain_leaves_the_entry_and_its_use_count_alone(tmp_path, make):
    cache_dir = str(tmp_path / ".cash")
    c = make(cache_dir)

    @c.cache
    def f(x):
        time.sleep(0.15)
        return x * 2

    f(3)
    c.backend.flush() if hasattr(c.backend, "flush") else None
    time.sleep(0.05)
    key = f.explain(3).cache_key
    # A number, not the dict: a backend may hand back its cached metadata
    # object itself, and comparing that with itself proves nothing.
    uses_before = c.backend.peek_metadata(key).get("access_count", 0)
    before = _entry_state(cache_dir)
    for _ in range(5):
        assert f.explain(3).would_hit
    if hasattr(c.backend, "flush"):
        c.backend.flush()
    uses_after = c.backend.peek_metadata(key).get("access_count", 0)
    assert _entry_state(cache_dir) == before, "explain() rewrote the entry file"
    assert uses_after == uses_before, "explain() counted as a use"


def test_a_real_hit_still_counts_as_a_use(tmp_path):
    """Control: the ordinary read path still records the access."""
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path / ".cash")), register_magic=False)

    @c.cache
    def f(x):
        return x * 2

    f(3)
    key = f.explain(3).cache_key
    before = c.backend.peek_metadata(key).get("access_count", 0)
    f(3)
    assert c.backend.peek_metadata(key).get("access_count", 0) == before + 1
