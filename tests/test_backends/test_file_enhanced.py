import time
import os
import pickle
from cash.backends import FileBackend
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry

class TestFileBackendEnhanced:
    
    def test_async_metadata_tracking(self, tmp_path):
        # Flush interval 1s
        backend = FileBackend(str(tmp_path), flush_interval=0.5)

        backend.set("key1", "value")
        backend._writes.wait_all()  # drain async data write before reading the meta file

        # Get initial metadata access time
        meta_path = backend._get_path("key1")
        meta_initial, _ = read_entry(meta_path, with_payload=False)
        
        # Wait a bit to ensure distinct timestamp
        time.sleep(1.0)
        
        # Access -> Should update last_access
        backend.get("key1")
        
        # Immediate check from disk (should be OLD or NEW depending on flush?)
        # Since flush is 0.5s, it might have happened or not.
        # But we want to ensure it DOES happen eventually.
        
        # In-memory metadata should be updated immediately
        assert backend._metadata_cache["key1"]["last_access"] > meta_initial["last_access"]
        
        # Wait for flush
        time.sleep(1.0)
        
        # Check disk. Reading the metadata region alone is what makes the
        # in-place flush observable: the flusher rewrites only that region,
        # never the payload.
        meta_updated, _ = read_entry(meta_path, with_payload=False)
            
        assert meta_updated["last_access"] > meta_initial["last_access"]
        
        backend.shutdown()

    def test_lru_eviction(self, tmp_path):
        # Max size: very small.
        # We need to know serialized size.
        # "value" string ~ 20 bytes? + metadata overhead.
        # Let's use 1500 bytes (1.5KB) limit.
        
        backend = FileBackend(str(tmp_path), max_size_bytes=1500, flush_interval=0) # No async flush for this test
        
        # Check empty size
        assert backend._current_size_bytes == 0
        
        # Insert items items approx 300-400 bytes each (Pickle overhead is significant)
        large_val = "x" * 400

        backend.set("k1", large_val)
        backend._writes.wait_all()  # async write — size only known after it lands
        s1 = backend._current_size_bytes
        assert s1 > 400
        
        time.sleep(0.1)
        backend.set("k2", large_val)
        
        # Access k1 to make it fresh.
        #
        # The sleep is load-bearing: without it this get() lands ~0.4ms after
        # the set("k2") above, and _check_and_evict sorts by last_access with a
        # STABLE sort. On a clock coarse enough to round both to the same tick
        # — Windows, especially under CI load — k1 and k2 tie, insertion order
        # wins, and eviction takes k1 instead of k2. The test then fails as
        # "k2 should have been evicted" on one job in five, at random.
        time.sleep(0.1)
        backend.get("k1")
        
        time.sleep(0.1)
        # k2 is now oldest accessed? No, k1 accessed recently. k2 is older.
        # k1 created t0, accessed t2.
        # k2 created t1.
        # LRU = k2.
        
        # Insert k3, forcing eviction if sum > 1000
        backend.set("k3", large_val)
        backend._writes.wait_all()  # let the async write + _check_and_evict settle

        # Total size without eviction would be s1 + (s2-s1) + ... ~ 3 * size
        # If 3 * size > 1000, eviction happens.

        # Verify k2 is gone (LRU)
        meta2, val2 = backend.get("k2")
        assert val2 is None, "k2 should have been evicted"
        
        # Verify k1 is still there (Recently Accessed)
        meta1, val1 = backend.get("k1")
        assert val1 == large_val, "k1 should be preserved"
        
        # Verify k3 is there
        meta3, val3 = backend.get("k3")
        assert val3 == large_val
        
        backend.shutdown()
        
    def test_size_init(self, tmp_path):
        """A second backend must account for what the first one left on disk.

        The total is established lazily now -- on the first write, the only
        thing that reads it -- so this triggers it with a ``set`` rather than
        a ``list_entries``. What is pinned is unchanged: without the
        pre-existing bytes, eviction would only ever see this process's own
        writes and the directory would grow past its cap forever.
        """
        # 1. Create backend, write stuff
        b1 = FileBackend(str(tmp_path))
        b1.set("k1", "v1")
        b1.shutdown()  # drain async write — size is only known once it lands
        size = b1._current_size_bytes

        assert size > 0

        # 2. Re-create backend over the same directory and make it need a total
        b2 = FileBackend(str(tmp_path), max_size_bytes=10 ** 9)
        b2.set("k2", "v2")
        b2._writes.wait_all()

        on_disk = sum(f.stat().st_size for f in tmp_path.iterdir()
                      if f.suffix == ENTRY_SUFFIX)
        assert b2._current_size_bytes == on_disk
        assert b2._current_size_bytes > size, "k1's bytes were not counted"
        b2.shutdown()


class TestCacheDirDeletedWhileLive:
    """The README suggests deleting ./.cash to wipe the cache. Doing that while a
    kernel is still running used to make the next write fail with
    CacheBackendError instead of simply recreating the directory."""

    def test_set_recreates_a_deleted_cache_dir(self, tmp_path):
        import shutil

        cache_dir = tmp_path / "cache"
        backend = FileBackend(str(cache_dir))
        backend.set("before", "v1")
        backend._writes.wait_all()
        assert cache_dir.exists()

        # User nukes ./.cash from outside while the backend is still live.
        shutil.rmtree(cache_dir)
        assert not cache_dir.exists()

        # Must not raise CacheBackendError — recreate and write.
        backend.set("after", "v2")
        backend._writes.wait_all()

        assert cache_dir.exists()
        _meta, value = backend.get("after")   # get() returns (metadata, value)
        assert value == "v2"
        backend.shutdown()
