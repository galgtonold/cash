from unittest.mock import MagicMock
from cash.backends.tiered_backend import TieredBackend
from cash.backends import InMemoryBackend, FileBackend

class TestTieredBackend:
    
    def test_promotion_logic(self, tmp_path):
        l1 = InMemoryBackend()
        l2 = FileBackend(str(tmp_path))
        
        # Test Case 1: Fast execution -> L1 only
        # Mock policy or rely on default? Default: > 1.0s
        backend = TieredBackend([l1, l2])
        
        backend.set("fast_key", "value", metadata={"execution_time": 0.1, "size": 100})
        
        # Check L1
        meta, val = l1.get("fast_key")
        assert val == "value"
        
        # Check L2 (should be empty)
        meta, val = l2.get("fast_key")
        assert val is None
        
        # Test Case 2: Slow execution -> L1 and L2
        backend.set("slow_key", "value", metadata={"execution_time": 2.0, "size": 100})
        
        # Check L1
        meta, val = l1.get("slow_key")
        assert val == "value"
        
        # Check L2 (should present)
        meta, val = l2.get("slow_key")
        assert val == "value"
        
    def test_read_promotion(self, tmp_path):
        l1 = InMemoryBackend()
        l2 = FileBackend(str(tmp_path))
        backend = TieredBackend([l1, l2])
        
        # Pre-populate L2 manually
        l2.set("only_in_l2", "value", metadata={"execution_time": 5.0})
        
        # Check L1 empty
        assert l1.get("only_in_l2") == (None, None)
        
        # Get from TieredBackend
        meta, val = backend.get("only_in_l2")
        assert val == "value"
        
        # Verify promoted to L1
        meta, val = l1.get("only_in_l2")
        assert val == "value"

    def test_large_object_promotion_prevention(self, tmp_path):
        """Test that huge objects that are fast to compute but slow to read aren't cached on disk."""
        l1 = MagicMock()
        l2 = MagicMock()
        
        # When getting from L1, return empty (simulating set flow check or just to satisfy protocol)
        # But set() uses the passed metadata.
        
        backend = TieredBackend([l1, l2])
        
        size = 200 * 1024 * 1024
        backend.set("huge_key", "value", metadata={"execution_time": 1.5, "size": size})
        
        # Check L1 set called
        l1.set.assert_called_once()
        
        # Check L2 set NOT called
        l2.set.assert_not_called()
        
        # Counter-case: Small object, same time
        l1.reset_mock()
        l2.reset_mock()
        
        small_size = 100
        backend.set("small_key", "value", metadata={"execution_time": 1.5, "size": small_size})
        
        # Check L1 set called
        l1.set.assert_called_once()
        
        # Check L2 set CALLED (Promoted)
        l2.set.assert_called_once()

    def test_filebackend_uses_absolute_path(self, tmp_path):
        """Test that FileBackend resolves cache_dir to an absolute path at init time."""
        import os
        original_cwd = os.getcwd()
        try:
            # Create FileBackend with relative path while in tmp_path
            os.chdir(str(tmp_path))
            fb = FileBackend(".test_cache")
            assert os.path.isabs(fb.cache_dir), f"Expected absolute path, got: {fb.cache_dir}"
            expected = os.path.join(str(tmp_path), ".test_cache")
            assert fb.cache_dir == expected
        finally:
            os.chdir(original_cwd)

    def test_disk_promotion_survives_chdir(self, tmp_path):
        """Test that disk promotion works even after os.chdir() changes the working directory.
        
        This reproduces a real-world bug where notebooks call os.chdir() after
        Cash is initialized, causing FileBackend to write to the wrong directory
        when its cache_dir was stored as a relative path.
        """
        import os
        original_cwd = os.getcwd()
        try:
            # Create backends from the original directory (simulates Cash() init at import time)
            cache_dir = str(tmp_path / "cache")
            l1 = InMemoryBackend()
            l2 = FileBackend(cache_dir)

            backend = TieredBackend([l1, l2])

            # Change directory (simulates os.chdir() in a notebook cell)
            other_dir = str(tmp_path / "other")
            os.makedirs(other_dir, exist_ok=True)
            os.chdir(other_dir)

            # Store something that should be promoted to disk (30s execution)
            metadata = {'execution_time': 30.0, 'force_persist': False}
            backend.set("chdir_key", {"data": "hello"}, metadata)

            stored_meta, _ = backend.get("chdir_key")
            storage = stored_meta.get('storage', []) if stored_meta else []
            assert 'RAM' in storage, f"Expected RAM in storage, got: {storage}"
            assert 'DISK' in storage, f"Expected DISK in storage, got: {storage}"

            # Verify we can read it back from disk
            meta, val = l2.get("chdir_key")
            assert val is not None, "Expected value from FileBackend after chdir"
            assert val['data'] == 'hello'
        finally:
            os.chdir(original_cwd)
