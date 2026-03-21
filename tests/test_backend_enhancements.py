"""Tests for backend enhancements: FileBackend TTL, SQLite backend."""

import time

from cash.backends.backend import FileBackend
from cash.backends.sqlite_backend import SQLiteBackend


class TestFileBackendTTL:
    """Test TTL expiration in FileBackend."""

    def test_no_ttl_by_default(self, tmp_path):
        fb = FileBackend(str(tmp_path / 'cache'))
        fb.set('key1', 'value1')
        meta, val = fb.get('key1')
        assert val == 'value1'
        assert 'ttl' not in meta or meta.get('ttl') is None

    def test_default_ttl_stored_in_metadata(self, tmp_path):
        fb = FileBackend(str(tmp_path / 'cache'), default_ttl=60)
        fb.set('key1', 'value1')
        meta, val = fb.get('key1')
        assert val == 'value1'
        assert meta['ttl'] == 60

    def test_ttl_expiration(self, tmp_path):
        fb = FileBackend(str(tmp_path / 'cache'), default_ttl=1)
        fb.set('key1', 'value1')
        meta, val = fb.get('key1')
        assert val == 'value1'
        # Wait for expiration
        time.sleep(1.1)
        meta, val = fb.get('key1')
        assert val is None

    def test_per_entry_ttl_overrides_default(self, tmp_path):
        fb = FileBackend(str(tmp_path / 'cache'), default_ttl=100)
        fb.set('key1', 'value1', metadata={'ttl': 1})
        time.sleep(1.1)
        meta, val = fb.get('key1')
        assert val is None  # Short TTL expired

    def test_entry_without_ttl_doesnt_expire(self, tmp_path):
        fb = FileBackend(str(tmp_path / 'cache'))  # No default TTL
        fb.set('key1', 'value1')
        # Entry should persist indefinitely
        meta, val = fb.get('key1')
        assert val == 'value1'

    def test_expired_entry_gets_deleted(self, tmp_path):
        fb = FileBackend(str(tmp_path / 'cache'), default_ttl=1)
        fb.set('key1', 'value1')
        time.sleep(1.1)
        fb.get('key1')  # Triggers deletion
        # Verify files are cleaned up
        entries = fb.list_entries()
        assert len([e for e in entries if e.get('key') == 'key1']) == 0


class TestSQLiteBackend:
    """Test SQLite backend."""

    def test_basic_set_get(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('key1', {'data': [1, 2, 3]})
        meta, val = db.get('key1')
        assert val == {'data': [1, 2, 3]}
        assert meta['key'] == 'key1'
        db.shutdown()

    def test_get_nonexistent(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        meta, val = db.get('nonexistent')
        assert meta is None
        assert val is None
        db.shutdown()

    def test_overwrite_existing(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('key1', 'old_value')
        db.set('key1', 'new_value')
        meta, val = db.get('key1')
        assert val == 'new_value'
        db.shutdown()

    def test_delete(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('key1', 'value1')
        db.delete('key1')
        meta, val = db.get('key1')
        assert val is None
        db.shutdown()

    def test_clear(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('key1', 'v1')
        db.set('key2', 'v2')
        db.set('key3', 'v3')
        db.clear()
        assert db.entry_count() == 0
        db.shutdown()

    def test_list_entries(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('k1', 'v1')
        db.set('k2', 'v2')
        entries = db.list_entries()
        assert len(entries) == 2
        keys = {e['key'] for e in entries}
        assert keys == {'k1', 'k2'}
        db.shutdown()

    def test_entry_count(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        assert db.entry_count() == 0
        db.set('k1', 'v1')
        assert db.entry_count() == 1
        db.set('k2', 'v2')
        assert db.entry_count() == 2
        db.delete('k1')
        assert db.entry_count() == 1
        db.shutdown()

    def test_total_size(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        assert db.total_size() == 0
        db.set('k1', 'x' * 1000)
        assert db.total_size() > 0
        db.shutdown()

    def test_access_count_updates(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('k1', 'v1')
        db.get('k1')  # access 1
        db.get('k1')  # access 2
        meta, _ = db.get('k1')  # access 3
        assert meta['access_count'] >= 2
        db.shutdown()

    def test_ttl_expiration(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'), default_ttl=1)
        db.set('k1', 'v1')
        meta, val = db.get('k1')
        assert val == 'v1'
        time.sleep(1.1)
        meta, val = db.get('k1')
        assert val is None
        db.shutdown()

    def test_per_entry_ttl(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('k1', 'v1', metadata={'ttl': 1})
        time.sleep(1.1)
        meta, val = db.get('k1')
        assert val is None
        db.shutdown()

    def test_no_ttl_no_expiration(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('k1', 'v1')
        meta, val = db.get('k1')
        assert val == 'v1'
        db.shutdown()

    def test_max_size_eviction(self, tmp_path):
        # Set very small max size
        db = SQLiteBackend(str(tmp_path / 'cache.db'), max_size_bytes=500)
        # Insert entries until eviction
        for i in range(20):
            db.set(f'k{i}', f'value_{"x" * 50}_{i}')
        # Should have evicted some
        assert db.entry_count() < 20
        db.shutdown()

    def test_cleanup_expired(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('k1', 'v1', metadata={'category': 'temp'})
        db.set('k2', 'v2', metadata={'category': 'perm'})
        count = db.cleanup_expired(lambda m: m.get('category') == 'temp')
        assert count == 1
        assert db.entry_count() == 1
        meta, val = db.get('k2')
        assert val == 'v2'
        db.shutdown()

    def test_complex_values(self, tmp_path):
        """Test storing various Python types."""
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        
        # dict
        db.set('dict', {'a': 1, 'b': [2, 3]})
        _, val = db.get('dict')
        assert val == {'a': 1, 'b': [2, 3]}
        
        # list
        db.set('list', [1, 2, 3, 4, 5])
        _, val = db.get('list')
        assert val == [1, 2, 3, 4, 5]
        
        # tuple
        db.set('tuple', (1, 'a', None))
        _, val = db.get('tuple')
        assert val == (1, 'a', None)
        
        # set
        db.set('set', {1, 2, 3})
        _, val = db.get('set')
        assert val == {1, 2, 3}
        
        db.shutdown()

    def test_storage_metadata(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        db.set('k1', 'v1')
        meta, _ = db.get('k1')
        assert 'SQLite' in meta.get('storage', [])
        db.shutdown()

    def test_concurrent_access(self, tmp_path):
        """Test thread safety."""
        import threading
        db = SQLiteBackend(str(tmp_path / 'cache.db'))
        errors = []
        
        def writer(start, count):
            try:
                for i in range(start, start + count):
                    db.set(f'k{i}', f'v{i}')
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=writer, args=(i * 10, 10)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert db.entry_count() == 50
        db.shutdown()
