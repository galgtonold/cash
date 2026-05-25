"""In-process helper redefinition invalidates the parent's cache.

Simulates the notebook flow: cell A defines ``helper``, cell B
caches ``main`` that calls ``helper``, cell C redefines ``helper``,
cell D calls ``main`` again — expects fresh recompute, not a stale
cache hit.

Achieved by re-resolving each captured helper from ``sys.modules``
inside ``_get_dependency_state_hash`` so the cache key reflects the
current source, not the analysis-time snapshot.
"""
from __future__ import annotations

import importlib

import pytest

from cash import Cash

from . import _purity_helper_module as hm


@pytest.fixture(autouse=True)
def _restore_helper():
    """Reset the module-level ``helper`` to its original definition
    after each test so suite ordering can't poison later runs."""
    original = hm.helper
    yield
    hm.helper = original


def test_in_process_helper_redefinition_invalidates_cache(tmp_path):
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def main(x):
        return hm.helper(x)

    # Original helper: x * 2
    assert main(5) == 10
    # Second call hits cache.
    assert main(5) == 10

    # Simulate notebook cell redefinition: rebind module attribute.
    def new_helper(x):
        return x * 3
    hm.helper = new_helper

    # Without per-call helper re-resolution, this would return the
    # stale 10. With the fix, the cache key includes the new helper's
    # source hash → miss → recompute → 15.
    assert main(5) == 15, (
        "in-process helper redefinition did not invalidate the cache; "
        "the parent cached the old helper's result. Per-call "
        "_current_helper_hashes() should pick up the new helper."
    )


def test_helper_redefinition_creates_new_cache_entry(tmp_path):
    """After redefinition, both old and new entries coexist; flipping
    back to the original helper should retrieve the original cached
    value (not recompute)."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def main(x):
        return hm.helper(x)

    original_helper = hm.helper

    # Warm cache with original (x * 2).
    assert main(7) == 14

    # Redefine.
    def new_helper(x):
        return x * 3
    hm.helper = new_helper

    # New helper → new cache key.
    assert main(7) == 21

    # Restore original helper → should hit the original cache entry,
    # not recompute. This proves the cache key truly tracks the
    # helper's identity, not just a "changed since last call" flag.
    hm.helper = original_helper
    n_calls_before = main.cache_info()["misses"]
    assert main(7) == 14
    n_calls_after = main.cache_info()["misses"]
    assert n_calls_after == n_calls_before, (
        "restoring original helper should hit the original cache entry, "
        "but a recompute happened (miss count increased)"
    )


def test_unchanged_helper_keeps_hitting_cache(tmp_path):
    """Per-call re-resolution must not falsely invalidate when the
    helper is genuinely unchanged."""
    c = Cash(cache_dir=str(tmp_path), register_magic=False)

    @c.cache
    def main(x):
        return hm.helper(x)

    main(1)  # miss
    main(1)  # hit
    main(1)  # hit
    main(1)  # hit

    info = main.cache_info()
    assert info["hits"] == 3, (
        f"Expected 3 hits after re-resolution overhead added — got "
        f"{info['hits']}. Re-resolution may be falsely invalidating."
    )
    assert info["misses"] == 1
