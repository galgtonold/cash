"""Tests for FileBackend functionality."""
import os
from cash.backends import FileBackend


def test_set_get(file_backend):
    key = "test_key"
    value = b"test_value"
    file_backend.set(key, value)
    metadata, retrieved = file_backend.get(key)
    assert retrieved == value


def test_delete(file_backend):
    key = "test_key"
    value = b"test_value"
    file_backend.set(key, value)
    file_backend.delete(key)
    _, retrieved = file_backend.get(key)
    assert retrieved is None


def test_clear(file_backend, temp_cache_dir):
    file_backend.set("k1", b"v1")
    file_backend.set("k2", b"v2")
    file_backend.clear()
    _, v1 = file_backend.get("k1")
    _, v2 = file_backend.get("k2")
    assert v1 is None
    assert v2 is None
    assert os.path.exists(temp_cache_dir)


def test_persistence(temp_cache_dir):
    backend1 = FileBackend(temp_cache_dir)
    backend1.set("k1", b"v1")
    # Writes are async — shutdown drains pending writes so backend2
    # (which has its own PendingWrites) can see the result on disk.
    backend1.shutdown()
    backend2 = FileBackend(temp_cache_dir)
    _, value = backend2.get("k1")
    assert value == b"v1"
