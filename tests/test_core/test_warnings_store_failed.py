"""When backend.set() raises, emit CashCacheStoreFailedWarning."""
from __future__ import annotations

import warnings

import pytest

from cash import Cash, CashCacheStoreFailedWarning


class _BrokenBackend:
    """Minimal in-memory backend whose set() always raises OSError."""

    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key, (None, None))

    def set(self, key, value, metadata, serializer=None):
        raise OSError("disk full (simulated)")

    def keys(self):
        return list(self._store.keys())

    def delete(self, key):
        self._store.pop(key, None)

    def lock(self, key):
        from contextlib import nullcontext
        return nullcontext()

    def cleanup_expired(self, predicate):
        return 0

    def shutdown(self):
        pass


def test_store_failure_emits_store_failed_warning():
    c = Cash(backend=_BrokenBackend(), register_magic=False)

    @c.cache
    def f(x):
        return x * 3

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = f(5)
        f(6)  # second call with different arg — store fails again,
              # but this is keyed (func, '') so dedup → no second warn

    assert result == 15  # compute still succeeds
    store_failed = [w for w in captured if issubclass(w.category, CashCacheStoreFailedWarning)]
    assert len(store_failed) == 1
    msg = str(store_failed[0].message)
    assert "_BrokenBackend" in msg or "BrokenBackend" in msg
    assert "OSError" in msg
