"""Tests for compression functionality in FileBackend."""
import os
from cash import Cash


from cash.backends.backend import FileBackend

def test_compression_enabled(temp_cache_dir):
    """Test that compression reduces file size for compressible data."""
    # Use FileBackend directly to bypass smart persistence policy
    backend = FileBackend(temp_cache_dir, compress=True)
    app = Cash(backend=backend)
    
    @app.cache
    def large_data():
        return b"0" * 10000
    
    large_data()
    entries = app.backend.list_entries()
    assert len(entries) == 1
    
    files = os.listdir(temp_cache_dir)
    data_file = [f for f in files if f.endswith('.data')][0]
    data_path = os.path.join(temp_cache_dir, data_file)
    file_size = os.path.getsize(data_path)
    
    assert file_size < 1000


def test_compression_disabled(temp_cache_dir):
    """Test that disabling compression keeps files uncompressed."""
    backend = FileBackend(temp_cache_dir, compress=False)
    app = Cash(backend=backend)
    
    @app.cache
    def large_data():
        return b"0" * 10000
    
    large_data()
    files = os.listdir(temp_cache_dir)
    data_file = [f for f in files if f.endswith('.data')][0]
    data_path = os.path.join(temp_cache_dir, data_file)
    file_size = os.path.getsize(data_path)
    
    assert file_size > 10000
