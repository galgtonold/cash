"""Tests for metadata-only disk persistence.

When size-aware caching skips storing a large object, the execution metadata
(timing, lineages) should still be persisted to disk via set_metadata_only().
This ensures badge timing info survives kernel restarts.
"""
import os
import pickle
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry


class TestFileBackendMetadataOnly:
    """Test FileBackend.set_metadata_only() and get_metadata() for metadata-only entries."""

    def test_set_metadata_only_creates_meta_file(self, tmp_path):
        """set_metadata_only should create a .meta file but NO .data file."""
        from cash.backends import FileBackend

        backend = FileBackend(cache_dir=str(tmp_path))
        metadata = {
            'execution_time': 1.5,
            'outputs': ['df'],
            'code': 'df = df.sort_values(...)',
            'output_lineages': {'df': 'abc123'},
        }
        backend.set_metadata_only('test_key', metadata)

        path = backend._get_path('test_key')
        assert os.path.exists(path), "the entry file should exist"
        stored, payload = read_entry(path, with_payload=True)
        assert payload == b"", "a metadata-only entry must hold no payload"
        assert stored['metadata_only'] is True

    def test_set_metadata_only_marks_metadata_only_flag(self, tmp_path):
        """Stored metadata should have metadata_only=True."""
        from cash.backends import FileBackend

        backend = FileBackend(cache_dir=str(tmp_path))
        backend.set_metadata_only('test_key', {'execution_time': 2.0})

        stored, _ = read_entry(backend._get_path('test_key'), with_payload=False)

        assert stored['metadata_only'] is True
        assert stored['size'] == 0
        assert stored['execution_time'] == 2.0

    def test_get_metadata_returns_metadata_only_entry(self, tmp_path):
        """get_metadata should return metadata-only entries (no .data file needed)."""
        from cash.backends import FileBackend

        backend = FileBackend(cache_dir=str(tmp_path))
        metadata = {
            'execution_time': 3.14,
            'outputs': ['result'],
            'output_lineages': {'result': 'hash123'},
        }
        backend.set_metadata_only('key1', metadata)

        # get_metadata should work without .data file
        retrieved = backend.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 3.14
        assert retrieved['metadata_only'] is True

    def test_get_metadata_returns_none_for_missing_key(self, tmp_path):
        """get_metadata should return None for nonexistent keys."""
        from cash.backends import FileBackend

        backend = FileBackend(cache_dir=str(tmp_path))
        assert backend.get_metadata('nonexistent') is None

    def test_get_returns_none_for_metadata_only_entry(self, tmp_path):
        """get() (full data retrieval) should return None for metadata-only entries."""
        from cash.backends import FileBackend

        backend = FileBackend(cache_dir=str(tmp_path))
        backend.set_metadata_only('key1', {'execution_time': 1.0})

        # Full get() should fail (no .data file)
        metadata, value = backend.get('key1')
        assert value is None

    def test_metadata_survives_fresh_backend_instance(self, tmp_path):
        """Metadata-only entries should survive creating a new FileBackend on same dir."""
        from cash.backends import FileBackend

        # First instance: write metadata
        backend1 = FileBackend(cache_dir=str(tmp_path))
        backend1.set_metadata_only('key1', {
            'execution_time': 5.0,
            'output_lineages': {'x': 'hash_x'},
        })

        # Second instance: simulates kernel restart (fresh in-memory cache)
        backend2 = FileBackend(cache_dir=str(tmp_path))
        retrieved = backend2.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 5.0
        assert retrieved['output_lineages'] == {'x': 'hash_x'}


class TestCascadingBackendMetadataOnly:
    """Test CascadingBackend metadata-only operations."""

    def test_set_metadata_only_delegates_to_file_backend(self, tmp_path):
        """set_metadata_only should delegate to FileBackend in the chain."""
        from cash.backends import InMemoryBackend, FileBackend, CascadingBackend

        mem = InMemoryBackend()
        file = FileBackend(cache_dir=str(tmp_path))
        cascade = CascadingBackend([mem, file])

        cascade.set_metadata_only('key1', {'execution_time': 2.5})

        # FileBackend should have the metadata
        retrieved = file.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 2.5

        # InMemoryBackend should NOT have it (no set_metadata_only method)
        mem_meta, mem_val = mem.get('key1')
        assert mem_val is None

    def test_get_metadata_on_cascading_backend(self, tmp_path):
        """get_metadata on CascadingBackend should check all backends."""
        from cash.backends import InMemoryBackend, FileBackend, CascadingBackend

        mem = InMemoryBackend()
        file = FileBackend(cache_dir=str(tmp_path))
        cascade = CascadingBackend([mem, file])

        # Write metadata-only to file backend directly
        file.set_metadata_only('key1', {'execution_time': 7.0})

        # CascadingBackend.get_metadata should find it
        retrieved = cascade.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 7.0

    def test_get_metadata_returns_none_for_missing(self, tmp_path):
        """get_metadata on CascadingBackend should return None for missing keys."""
        from cash.backends import InMemoryBackend, FileBackend, CascadingBackend

        mem = InMemoryBackend()
        file = FileBackend(cache_dir=str(tmp_path))
        cascade = CascadingBackend([mem, file])

        assert cascade.get_metadata('nonexistent') is None

    def test_get_metadata_prefers_full_entry_over_metadata_only(self, tmp_path):
        """If a full cache entry exists, get_metadata should return its metadata."""
        from cash.backends import InMemoryBackend, FileBackend, CascadingBackend

        mem = InMemoryBackend()
        file = FileBackend(cache_dir=str(tmp_path))
        cascade = CascadingBackend([mem, file])

        # Store a full entry
        full_metadata = {'execution_time': 10.0, 'outputs': ['df']}
        cascade.set('key1', {'variables': {'df': 42}}, full_metadata)

        retrieved = cascade.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 10.0
        # Should NOT have metadata_only flag since it has full data
        assert not retrieved.get('metadata_only', False)

    def test_metadata_only_survives_simulated_restart(self, tmp_path):
        """Metadata-only entries should survive simulated kernel restart."""
        from cash.backends import InMemoryBackend, FileBackend, CascadingBackend

        # First session: write metadata-only
        file1 = FileBackend(cache_dir=str(tmp_path))
        cascade1 = CascadingBackend([InMemoryBackend(), file1])
        cascade1.set_metadata_only('key1', {
            'execution_time': 4.2,
            'output_lineages': {'df': 'lineage_hash'},
        })

        # Simulated restart: new backends, same disk dir
        file2 = FileBackend(cache_dir=str(tmp_path))
        cascade2 = CascadingBackend([InMemoryBackend(), file2])

        retrieved = cascade2.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 4.2
        assert retrieved['output_lineages'] == {'df': 'lineage_hash'}


class TestStatementProcessorMetadataPersistence:
    """Test StatementRestorer.persist_metadata_only and metadata persistence on skip."""

    def test_persist_metadata_only_calls_backend(self, tmp_path):
        """persist_metadata_only should call set_metadata_only on the backend."""
        from cash.notebook.statement.restore import StatementRestorer
        from cash.backends import FileBackend, CascadingBackend, InMemoryBackend

        file_backend = FileBackend(cache_dir=str(tmp_path))
        cascade = CascadingBackend([InMemoryBackend(), file_backend])

        StatementRestorer.persist_metadata_only(cascade, 'test_key', {
            'execution_time': 3.5,
            'outputs': ['df'],
        })

        # Verify it landed on disk
        retrieved = file_backend.get_metadata('test_key')
        assert retrieved is not None
        assert retrieved['execution_time'] == 3.5
        assert retrieved['metadata_only'] is True

    def test_persist_metadata_only_noop_for_unsupported_backend(self):
        """persist_metadata_only should be a no-op for backends without the method."""
        from cash.notebook.statement.restore import StatementRestorer

        # A backend that doesn't support set_metadata_only
        class DummyBackend:
            pass

        # Should not raise
        StatementRestorer.persist_metadata_only(DummyBackend(), 'key', {'execution_time': 1.0})


class TestTieredBackendMetadataOnly:
    """Test TieredBackend metadata-only operations (the actual default backend)."""

    def test_tiered_has_get_metadata(self, tmp_path):
        """TieredBackend should have get_metadata method."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered = TieredBackend([l1, l2])

        assert hasattr(tiered, 'get_metadata')

    def test_tiered_has_set_metadata_only(self, tmp_path):
        """TieredBackend should have set_metadata_only method."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered = TieredBackend([l1, l2])

        assert hasattr(tiered, 'set_metadata_only')

    def test_set_metadata_only_persists_to_file_backend(self, tmp_path):
        """set_metadata_only should write to FileBackend in the tier chain."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered = TieredBackend([l1, l2])

        tiered.set_metadata_only('key1', {'execution_time': 2.5, 'outputs': ['df']})

        # FileBackend should have the metadata
        retrieved = l2.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 2.5
        assert retrieved['metadata_only'] is True

    def test_get_metadata_finds_metadata_only_entry(self, tmp_path):
        """get_metadata should find metadata-only entries in FileBackend tier."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered = TieredBackend([l1, l2])

        tiered.set_metadata_only('key1', {'execution_time': 3.14})

        retrieved = tiered.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 3.14

    def test_get_metadata_returns_none_for_missing(self, tmp_path):
        """get_metadata should return None for nonexistent keys."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered = TieredBackend([l1, l2])

        assert tiered.get_metadata('nonexistent') is None

    def test_metadata_survives_restart(self, tmp_path):
        """Metadata-only entries should survive creating new TieredBackend (simulated restart)."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        # First session
        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered1 = TieredBackend([l1, l2])
        tiered1.set_metadata_only('key1', {
            'execution_time': 4.2,
            'output_lineages': {'df': 'hash123'},
        })

        # Simulated restart — new in-memory, same disk
        l1_new = InMemoryBackend()
        l2_new = FileBackend(cache_dir=str(tmp_path))
        tiered2 = TieredBackend([l1_new, l2_new])

        retrieved = tiered2.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 4.2

    def test_metadata_only_does_not_overwrite_full_entry(self, tmp_path):
        """set_metadata_only should NOT overwrite a full cache entry on disk."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))
        tiered = TieredBackend([l1, l2])

        # Store a full entry directly to L2 (simulating promotion)
        full_meta = {'execution_time': 10.0, 'outputs': ['df']}
        l2.set('key1', {'variables': {'df': 42}}, full_meta)

        # Now call set_metadata_only — should NOT overwrite
        tiered.set_metadata_only('key1', {'execution_time': 0.1, 'metadata_only': True})

        # Full entry should still be intact
        retrieved = l2.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 10.0
        assert not retrieved.get('metadata_only', False)

    def test_set_metadata_only_waits_for_inflight_full_write(self, tmp_path):
        """set_metadata_only must synchronise with an in-flight async set()
        for the same key before deciding whether a full entry exists.

        FileBackend.set() flushes to disk on a background executor. Without
        waiting, set_metadata_only's existence check races a not-yet-flushed
        full write, sees no files, and can clobber the full entry — the cause
        of the intermittent CI flake (`assert 0.1 == 10.0`). This pins the
        write in flight with an event so the race is deterministic.
        """
        import threading

        from cash.backends.file_backend import FileBackend

        writing = threading.Event()
        release = threading.Event()

        class BlockingWriteBackend(FileBackend):
            def _write_cache_files(self, *args, **kwargs):
                writing.set()
                release.wait(5)  # hold the full write in flight
                return super()._write_cache_files(*args, **kwargs)

        backend = BlockingWriteBackend(cache_dir=str(tmp_path))
        try:
            backend.set('key1', {'variables': {'df': 42}},
                        {'execution_time': 10.0, 'outputs': ['df']})
            assert writing.wait(2), "full write should have started"

            done = threading.Event()

            def call_set_metadata_only():
                backend.set_metadata_only(
                    'key1', {'execution_time': 0.1, 'metadata_only': True})
                done.set()

            worker = threading.Thread(target=call_set_metadata_only)
            worker.start()

            # The full write is still in flight, so set_metadata_only must
            # block (it waits) rather than return and clobber. Without the
            # fix it returns immediately and `done` is set.
            assert not done.wait(0.5), \
                "set_metadata_only must wait for the in-flight full write"

            release.set()
            worker.join(5)
            assert done.is_set()

            retrieved = backend.get_metadata('key1')
            assert retrieved is not None
            assert retrieved['execution_time'] == 10.0
            assert not retrieved.get('metadata_only', False)
        finally:
            release.set()
            backend.shutdown()

    def test_set_stores_metadata_even_without_promotion(self, tmp_path):
        """When TieredBackend.set() doesn't promote to disk (cheap stmt),
        _persist_metadata_only should still persist the metadata."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend
        from cash.notebook.statement.restore import StatementRestorer

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))

        # Use a promotion policy that rejects everything
        tiered = TieredBackend([l1, l2], promotion_policy=lambda t, s: False)

        # Simulate what statement_processor does: set() then persist_metadata_only()
        metadata = {'execution_time': 0.05, 'outputs': ['x'], 'size': 100}
        tiered.set('key1', {'variables': {'x': 42}}, metadata)

        # Full data should NOT be on disk (promotion rejected)
        full_meta, full_val = l2.get('key1')
        assert full_val is None, "Promotion should have been rejected"

        # Now persist metadata only
        StatementRestorer.persist_metadata_only(tiered, 'key1', metadata)

        # Metadata should be on disk
        retrieved = tiered.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 0.05

    def test_get_metadata_finds_promoted_full_entry(self, tmp_path):
        """get_metadata should also find metadata from fully-promoted entries."""
        from cash.backends import InMemoryBackend, FileBackend
        from cash.backends.tiered_backend import TieredBackend

        l1 = InMemoryBackend()
        l2 = FileBackend(cache_dir=str(tmp_path))

        # Promotion policy that accepts everything
        tiered = TieredBackend([l1, l2], promotion_policy=lambda t, s: True)

        metadata = {'execution_time': 5.0, 'outputs': ['df'], 'size': 1000}
        tiered.set('key1', {'variables': {'df': 42}}, metadata)

        # get_metadata should return metadata from the full entry
        retrieved = tiered.get_metadata('key1')
        assert retrieved is not None
        assert retrieved['execution_time'] == 5.0
