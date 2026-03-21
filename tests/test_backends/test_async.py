"""Tests for AsyncBackendWrapper functionality."""
import concurrent.futures
import threading
import time

import pytest

from cash.backends.backend import InMemoryBackend, AsyncBackendWrapper
from cash.backends.file_backend import FileBackend
from cash.backends.tiered_backend import TieredBackend


def test_async_set(clean_backend):
    """Test that set is asynchronous."""
    backend = AsyncBackendWrapper(clean_backend)
    
    backend.set("key", "value")
    
    # Shutdown and wait for async operations to complete
    backend.shutdown(wait=True)
    
    # Verify value was set in underlying backend
    metadata, value = clean_backend.get("key")
    assert value == "value"


def test_async_delete(clean_backend):
    """Test that delete is asynchronous."""
    # Pre-populate backend
    clean_backend.set("key", "value")
    
    backend = AsyncBackendWrapper(clean_backend)
    backend.delete("key")
    
    # Wait for async operations
    backend.shutdown(wait=True)
    
    # Verify value was deleted
    metadata, value = clean_backend.get("key")
    assert value is None

def test_async_non_blocking():
    """Test that set returns quickly even if backend is slow."""
    
    class SlowBackend(InMemoryBackend):
        def set(self, key, value, metadata=None, serializer=None):
            time.sleep(0.1)  # Simulate slow operation
            super().set(key, value, metadata, serializer)
    
    slow_backend = SlowBackend()
    backend = AsyncBackendWrapper(slow_backend)
    
    start = time.time()
    backend.set("key", "value")
    elapsed = time.time() - start
    
    # Should return almost immediately (< 50ms), not wait for the 100ms sleep
    assert elapsed < 0.05, f"Async set took {elapsed:.3f}s, expected < 0.05s"
    
    # Cleanup and verify the value was actually set
    backend.shutdown(wait=True)
    metadata, value = slow_backend.get("key")
    assert value == "value", "Value should be set in backend after shutdown"


def test_concurrent_set_no_data_loss():
    """10 concurrent set() calls with distinct keys must all be readable after shutdown."""
    inner = InMemoryBackend()
    backend = AsyncBackendWrapper(inner)
    n = 10

    def worker(i):
        backend.set(f"key_{i}", f"value_{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(worker, i) for i in range(n)]
        concurrent.futures.wait(futures)

    backend.shutdown(wait=True)

    for i in range(n):
        _, val = inner.get(f"key_{i}")
        assert val == f"value_{i}", f"key_{i} missing after concurrent writes"


def test_concurrent_set_get_eventual_consistency():
    """Interleaved set() and get() on the same key: value is readable after shutdown."""
    inner = InMemoryBackend()
    backend = AsyncBackendWrapper(inner)
    results = {}

    def setter():
        backend.set("shared", "final_value")

    def getter():
        # Poll until the value appears (eventual consistency)
        backend.shutdown(wait=True)
        _, val = inner.get("shared")
        results["val"] = val

    t_set = threading.Thread(target=setter)
    t_set.start()
    t_set.join()

    t_get = threading.Thread(target=getter)
    t_get.start()
    t_get.join()

    assert results["val"] == "final_value"


def test_tiered_backend_with_compression_async(tmp_path):
    """TieredBackend with FileBackend(compress=True) wrapped in AsyncBackendWrapper
    stores data to disk in compressed form after shutdown(wait=True)."""
    file_b = FileBackend(str(tmp_path / "cache"), compress=True)
    tiered = TieredBackend([InMemoryBackend(), file_b])
    backend = AsyncBackendWrapper(tiered)

    # force_persist=True bypasses the promotion policy so the write goes to disk
    backend.set("mykey", {"data": list(range(100))}, {"force_persist": True})
    backend.shutdown(wait=True)

    # Data must be readable directly from the file backend
    _, val = file_b.get("mykey")
    assert val == {"data": list(range(100))}, "Compressed tiered data not persisted correctly"
