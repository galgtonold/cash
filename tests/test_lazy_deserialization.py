"""
Tests for lazy deserialization proxy and FileBackend.get_metadata.
"""

from cash.backends.lazy import LazyProxy, make_lazy_loader
from cash.backends import FileBackend, InMemoryBackend


class TestLazyProxy:
    """Tests for the LazyProxy class."""

    def test_deferred_resolution(self):
        """Value is not loaded until resolve() is called."""
        calls = []

        def loader():
            calls.append(1)
            return {"key": "value"}

        proxy = LazyProxy(loader, metadata={"size": 100})
        assert not proxy.is_resolved
        assert len(calls) == 0

        result = proxy.resolve()
        assert result == {"key": "value"}
        assert proxy.is_resolved
        assert len(calls) == 1

    def test_value_property(self):
        """The value property triggers lazy loading."""
        proxy = LazyProxy(lambda: 42, metadata={})
        assert not proxy.is_resolved
        assert proxy.value == 42
        assert proxy.is_resolved

    def test_single_load(self):
        """Loader is called only once even with multiple accesses."""
        calls = []

        def loader():
            calls.append(1)
            return [1, 2, 3]

        proxy = LazyProxy(loader)
        _ = proxy.value
        _ = proxy.value
        _ = proxy.resolve()
        assert len(calls) == 1

    def test_metadata_accessible_before_resolve(self):
        """Metadata is available without resolving the value."""
        meta = {"created_at": 123, "ttl": 3600}
        proxy = LazyProxy(lambda: "data", metadata=meta)
        assert proxy.metadata["ttl"] == 3600
        assert not proxy.is_resolved

    def test_repr_pending(self):
        proxy = LazyProxy(lambda: None, cache_key="stmt:abc")
        assert "pending" in repr(proxy)
        assert "abc" in repr(proxy)

    def test_repr_resolved(self):
        proxy = LazyProxy(lambda: 42)
        proxy.resolve()
        assert "resolved" in repr(proxy)
        assert "42" in repr(proxy)


class TestMakeLazyLoader:
    """Tests for the make_lazy_loader factory function."""

    def test_with_file_backend(self, tmp_path):
        """LazyProxy works with FileBackend.get_metadata."""
        backend = FileBackend(cache_dir=str(tmp_path / "cache"))
        backend.set("key1", {"data": [1, 2, 3]}, metadata={"type": "test"})

        proxy = make_lazy_loader(backend, "key1")
        assert proxy is not None
        assert proxy.metadata.get("type") == "test"
        # Data not yet deserialized via the metadata path
        # (FileBackend has get_metadata, so it should use that)

        value = proxy.resolve()
        assert value == {"data": [1, 2, 3]}
        backend.clear()

    def test_missing_key(self, tmp_path):
        """Returns None for non-existent key."""
        backend = FileBackend(cache_dir=str(tmp_path / "cache"))
        proxy = make_lazy_loader(backend, "nonexistent")
        assert proxy is None
        backend.clear()

    def test_with_inmemory_backend(self):
        """InMemoryBackend inherits get_metadata from the ABC."""
        backend = InMemoryBackend()
        backend.set("key1", 42, metadata={"info": "test"})

        proxy = make_lazy_loader(backend, "key1")
        assert proxy is not None
        # InMemoryBackend now inherits get_metadata from the ABC,
        # so the proxy starts lazy (unresolved) and resolves on access.
        assert not proxy.is_resolved
        assert proxy.value == 42
        assert proxy.is_resolved

    def test_inmemory_missing_key(self):
        """Returns None for non-existent key in InMemoryBackend."""
        backend = InMemoryBackend()
        proxy = make_lazy_loader(backend, "nonexistent")
        assert proxy is None


class TestFileBackendGetMetadata:
    """Tests for the new FileBackend.get_metadata method."""

    def test_returns_metadata_without_data(self, tmp_path):
        """get_metadata returns only metadata, no deserialization."""
        backend = FileBackend(cache_dir=str(tmp_path / "cache"))
        backend.set("key1", [1, 2, 3], metadata={"custom": "field"})

        meta = backend.get_metadata("key1")
        assert meta is not None
        assert meta.get("custom") == "field"
        assert meta.get("key") == "key1"
        backend.clear()

    def test_returns_none_for_missing(self, tmp_path):
        backend = FileBackend(cache_dir=str(tmp_path / "cache"))
        meta = backend.get_metadata("nonexistent")
        assert meta is None
        backend.clear()

    def test_respects_ttl(self, tmp_path):
        """Expired entries return None from get_metadata."""
        import time
        backend = FileBackend(cache_dir=str(tmp_path / "cache"), default_ttl=1)
        backend.set("key1", "value")
        meta = backend.get_metadata("key1")
        assert meta is not None

        time.sleep(1.1)
        meta = backend.get_metadata("key1")
        assert meta is None
        backend.clear()
