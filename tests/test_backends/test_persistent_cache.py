"""Tests for persistent cache functionality across Cash instances."""
import os
from cash import Cash
from cash.backends.backend import FileBackend

def test_persistence_across_instances(temp_cache_dir):
    """Test that cached values persist across Cash instances."""
    # Create first instance using FileBackend directly to force persistence
    # (Default smart policy skips disk for fast items)
    backend1 = FileBackend(temp_cache_dir)
    app1 = Cash(backend=backend1)
    
    @app1.cache
    def compute(x):
        return x * 2
    
    result1 = compute(10)
    assert result1 == 20
    
    # Verify cache files were created
    files = os.listdir(temp_cache_dir)
    assert len(files) > 0, "Cache files should exist"
    
    # Create second instance pointing to same directory
    backend2 = FileBackend(temp_cache_dir)
    app2 = Cash(backend=backend2)
    
    # Define same function (same source code = same hash)
    @app2.cache
    def compute(x):
        return x * 2
    
    # This should hit the cache from app1
    # We can verify by checking backend entries
    entries = app2.backend.list_entries()
    assert len(entries) > 0, "Should have cached entries from app1"


def test_persistence_simple(temp_cache_dir):
    """Test that cache writes to disk."""
    backend = FileBackend(temp_cache_dir)
    app = Cash(backend=backend)
    
    @app.cache
    def add(a, b):
        return a + b
    
    result = add(1, 2)
    assert result == 3
    
    # Check if cache files exist
    files = os.listdir(temp_cache_dir)
    assert len(files) > 0, "Cache directory should not be empty"
    
    # Should have both .data and .meta files
    data_files = [f for f in files if f.endswith('.data')]
    meta_files = [f for f in files if f.endswith('.meta')]
    
    assert len(data_files) > 0, "Should have .data files"
    assert len(meta_files) > 0, "Should have .meta files"
