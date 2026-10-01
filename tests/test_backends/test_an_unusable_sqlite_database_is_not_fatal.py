"""A SQLite cache whose database cannot be opened turns itself off.

Like the file tier, it warns once and answers as an empty cache would,
instead of failing ``Cash()`` or the caller's program. And an entry it
receives from another tier keeps the age it had there, so a promotion does
not restart its ttl.
"""

from __future__ import annotations

import time
import warnings

import cash
from cash.backends.sqlite_backend import SQLiteBackend
from cash.exceptions import CashCacheStoreFailedWarning


def test_a_database_under_a_regular_file_warns_and_misses(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = SQLiteBackend(db_path=str(blocker / "sub" / "cache.db"))
        backend.set("k", 1, {})
        assert backend.get("k") == (None, None)
        assert backend.list_entries() == []
        backend.shutdown()
    assert any(issubclass(w.category, CashCacheStoreFailedWarning) for w in caught)


def test_a_cached_function_runs_over_an_unusable_sqlite_cache(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = cash.Cash(backend="sqlite", cache_dir=str(blocker / "sub"))

        @c.cache
        def f(x):
            return x + 1

        assert f(1) == 2


def test_a_written_entry_keeps_its_created_at(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "c.db"))
    written = time.time() - 100
    backend.set("k", 1, {"created_at": written})
    metadata, _ = backend.get("k")
    assert metadata["created_at"] == written
    backend.shutdown()
