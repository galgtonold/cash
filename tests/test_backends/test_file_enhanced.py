import time
import os
import pickle
from cash.backends import FileBackend

class TestFileBackendEnhanced:
    
    def test_async_metadata_tracking(self, tmp_path):
        # Flush interval 1s
        backend = FileBackend(str(tmp_path), flush_interval=0.5)
        
        backend.set("key1", "value")
        
        # Get initial metadata access time
        meta_path = os.path.join(tmp_path, backend._get_paths("key1")[0])
        with open(meta_path, 'rb') as f:
            meta_initial = pickle.load(f)
        
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
        
        # Check disk
        with open(meta_path, 'rb') as f:
            meta_updated = pickle.load(f)
            
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
        s1 = backend._current_size_bytes
        assert s1 > 400
        
        time.sleep(0.1)
        backend.set("k2", large_val)
        
        # Access k1 to make it fresh
        backend.get("k1")
        
        time.sleep(0.1)
        # k2 is now oldest accessed? No, k1 accessed recently. k2 is older.
        # k1 created t0, accessed t2.
        # k2 created t1.
        # LRU = k2.
        
        # Insert k3, forcing eviction if sum > 1000
        backend.set("k3", large_val)
        
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
        """Verify size initializes correctly from existing files."""
        # 1. Create backend, write stuff
        b1 = FileBackend(str(tmp_path))
        b1.set("k1", "v1")
        size = b1._current_size_bytes
        b1.shutdown()
        
        assert size > 0
        
        # 2. Re-create backend (lazy init: trigger with a cache operation)
        b2 = FileBackend(str(tmp_path))
        b2.list_entries()  # triggers _ensure_initialized()
        assert b2._current_size_bytes == size
        b2.shutdown()
