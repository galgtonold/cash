"""Bare-backend storage/source label consistency.

Tier-string matching in the badge renderer assumes that the strings a
backend writes into ``metadata['storage']`` and ``metadata['source']``
match the backend's own ``source_label``. Without this, a configured-
tier list of ``['REDIS', ...]`` would never match a metric whose
``storage`` is ``['Redis']``.
"""
from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import pytest


# Build a list of bare backend factories. Each yields a (label, backend)
# pair, where the backend is fully constructed and ready for set/get.

def _bare_backends(tmp_path):
    from cash.backends.file_backend import FileBackend
    from cash.backends.memory_backend import InMemoryBackend
    from cash.backends.sqlite_backend import SQLiteBackend

    yield "RAM", InMemoryBackend()
    yield "DISK", FileBackend(str(tmp_path / "disk_cache"))
    yield "SQLITE", SQLiteBackend(str(tmp_path / "c.db"))

    # Redis with fakeredis
    try:
        import fakeredis
        import redis as _redis
    except ImportError:
        return
    with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
        from cash.backends.redis_backend import RedisBackend
        yield "REDIS", RedisBackend(prefix="cash:t:")


class TestStorageLabelMatchesSourceLabel:
    """The ``storage`` field on stored metadata must equal ``[source_label]``."""

    def test_all_bare_backends(self, tmp_path):
        for label, backend in _bare_backends(tmp_path):
            metadata = {"execution_time": 1.0}
            backend.set(f"k_{label}", "v", metadata)
            assert metadata.get("storage") == [label], (
                f"{type(backend).__name__}: expected ['{label}'], got {metadata.get('storage')!r}"
            )


class TestGetPopulatesSource:
    """After a bare-backend get(), metadata['source'] must equal source_label."""

    def test_all_bare_backends(self, tmp_path):
        for label, backend in _bare_backends(tmp_path):
            backend.set(f"k_src_{label}", "v")
            metadata, value = backend.get(f"k_src_{label}")
            assert value == "v", f"{type(backend).__name__}: value lost"
            assert metadata is not None and metadata.get("source") == label, (
                f"{type(backend).__name__}: expected source={label!r}, got {metadata.get('source') if metadata else None!r}"
            )
