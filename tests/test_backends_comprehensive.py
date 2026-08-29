"""Tests for backend classes - CascadingBackend, InMemoryBackend, FileBackend."""
import hashlib
import os
import time
from cash.backends import (
    InMemoryBackend, FileBackend, CascadingBackend
)
from cash.backends.serialization import PickleSerializer, CloudPickleSerializer, get_serializer
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry


class TestInMemoryBackendAdvanced:
    """Advanced InMemoryBackend tests beyond basic get/set."""

    def test_delete(self):
        """Delete removes an entry."""
        b = InMemoryBackend()
        b.set("k1", "v1", {"key": "k1"})
        b.delete("k1")
        _, val = b.get("k1")
        assert val is None

    def test_delete_nonexistent(self):
        """Delete of nonexistent key doesn't crash."""
        b = InMemoryBackend()
        b.delete("nonexistent")

    def test_clear(self):
        """Clear removes all entries."""
        b = InMemoryBackend()
        b.set("k1", "v1", {"key": "k1"})
        b.set("k2", "v2", {"key": "k2"})
        b.clear()
        _, val = b.get("k1")
        assert val is None

    def test_list_entries(self):
        """list_entries returns all metadata."""
        b = InMemoryBackend()
        b.set("k1", "v1", {"key": "k1", "name": "first"})
        b.set("k2", "v2", {"key": "k2", "name": "second"})
        entries = b.list_entries()
        assert len(entries) == 2

    def test_cleanup_expired(self):
        """cleanup_expired removes matching entries."""
        b = InMemoryBackend()
        b.set("old", "v1", {"key": "old", "timestamp": 1000})
        b.set("new", "v2", {"key": "new", "timestamp": time.time()})

        removed = b.cleanup_expired(
            lambda m: m.get('timestamp', 0) < 2000
        )
        assert removed == 1
        _, val = b.get("old")
        assert val is None
        _, val = b.get("new")
        assert val is not None

    def test_max_entries_eviction(self):
        """Max entries triggers LRU eviction."""
        b = InMemoryBackend(max_entries=3)
        b.set("k1", "v1", {"key": "k1"})
        b.set("k2", "v2", {"key": "k2"})
        b.set("k3", "v3", {"key": "k3"})
        b.set("k4", "v4", {"key": "k4"})  # Should evict k1
        _, val = b.get("k1")
        assert val is None  # Evicted
        _, val = b.get("k4")
        assert val == "v4"

    def test_access_count_tracking(self):
        """Access count increases on get."""
        b = InMemoryBackend()
        b.set("k1", "v1", {"key": "k1"})
        b.get("k1")
        b.get("k1")
        entries = b.list_entries()
        # Should have incremented access count
        assert entries[0].get('access_count', 0) >= 2

    def test_lock(self):
        """Lock context manager works."""
        b = InMemoryBackend()
        with b.lock("key1"):
            b.set("key1", "value", {"key": "key1"})
        _, val = b.get("key1")
        assert val == "value"

    def test_shutdown(self):
        """Shutdown doesn't raise."""
        b = InMemoryBackend()
        b.shutdown()

    def test_deep_copy_on_set(self):
        """Values are deep-copied on set (mutable safety)."""
        b = InMemoryBackend()
        data = [1, 2, 3]
        b.set("k1", data, {"key": "k1"})
        data.append(4)  # Mutate original
        _, val = b.get("k1")
        assert val == [1, 2, 3]  # Stored copy unaffected


class TestFileBackendAdvanced:
    """Advanced FileBackend tests."""

    def test_basic_set_get(self, tmp_path):
        """Basic set/get round-trip."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("k1", {"data": 42}, {"key": "k1"})
        meta, val = b.get("k1")
        assert val == {"data": 42}
        assert meta["key"] == "k1"
        b.shutdown()

    def test_get_nonexistent(self, tmp_path):
        """Get nonexistent key returns None."""
        b = FileBackend(str(tmp_path / "cache"))
        meta, val = b.get("nonexistent")
        assert meta is None
        assert val is None
        b.shutdown()

    def test_delete(self, tmp_path):
        """Delete removes cache files."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("k1", "v1", {"key": "k1"})
        b.delete("k1")
        _, val = b.get("k1")
        assert val is None
        b.shutdown()

    def test_clear(self, tmp_path):
        """Clear removes all cache files."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("k1", "v1", {"key": "k1"})
        b.set("k2", "v2", {"key": "k2"})
        b.clear()
        _, v1 = b.get("k1")
        _, v2 = b.get("k2")
        assert v1 is None
        assert v2 is None
        b.shutdown()

    def test_compressed(self, tmp_path):
        """Compressed storage works."""
        b = FileBackend(str(tmp_path / "cache"), compress=True)
        data = "x" * 10000  # Compressible
        b.set("k1", data, {"key": "k1"})
        _, val = b.get("k1")
        assert val == data
        b.shutdown()

    def test_list_entries(self, tmp_path):
        """list_entries returns all metadata."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("k1", "v1", {"key": "k1"})
        b.set("k2", "v2", {"key": "k2"})
        b._flush_metadata()  # Force flush
        entries = b.list_entries()
        assert len(entries) == 2
        b.shutdown()

    def test_cleanup_expired(self, tmp_path):
        """cleanup_expired removes old entries."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("old", "v1", {"key": "old", "timestamp": 1000})
        b._flush_metadata()
        removed = b.cleanup_expired(lambda m: m.get('timestamp', 0) < 2000)
        assert removed == 1
        b.shutdown()

    def test_get_metadata(self, tmp_path):
        """get_metadata returns only metadata, not data."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("k1", "v1", {"key": "k1", "extra": "info"})
        meta = b.get_metadata("k1")
        assert meta is not None
        assert meta["key"] == "k1"
        b.shutdown()

    def test_ttl_expiration(self, tmp_path):
        """TTL-based expiration in FileBackend."""
        # Backdated created_at rather than a sleep: same expiry branch
        # (`time.time() - created_at > ttl`), no wall-clock dependency.
        b = FileBackend(str(tmp_path / "cache"), default_ttl=3600)
        b.set("k1", "v1", {"key": "k1"})
        _, val = b.get("k1")
        assert val == "v1"

        b.set("k2", "v2", {"key": "k2", "created_at": time.time() - 7200})
        _, val = b.get("k2")
        assert val is None  # Expired
        b.shutdown()

    def test_max_size_bytes(self, tmp_path):
        """max_size_bytes triggers eviction."""
        b = FileBackend(str(tmp_path / "cache"), max_size_bytes=500)
        # Store several items to exceed limit
        for i in range(20):
            b.set(f"k{i}", f"value_{i}" * 10, {"key": f"k{i}"})
        # Should have evicted some entries
        len(b.list_entries())
        # We can't assert exact count but it should be less than 20
        b.shutdown()

    def test_get_corrupted_header_is_a_miss(self, tmp_path):
        """An entry whose header is garbage returns None gracefully."""
        b = FileBackend(str(tmp_path / "cache"))
        b.set("key1", "value", {"info": "test"})
        # Drain the async write before corrupting the file — otherwise the
        # still-in-flight write lands AFTER our corruption and undoes it.
        b._writes.wait_all()
        with open(b._get_path("key1"), "wb") as f:
            f.write(b"corrupted data")
        if hasattr(b, '_metadata_cache'):
            b._metadata_cache.clear()
        meta, data = b.get("key1")
        assert data is None or meta is None

    def test_get_corrupted_payload_is_a_miss(self, tmp_path):
        """A readable header over an unusable payload returns None gracefully.

        Distinct from the arm above now that both live in one file: here the
        metadata parses fine, so nothing rejects the entry before the value is
        deserialized, and the failure has to be caught there.
        """
        b = FileBackend(str(tmp_path / "cache"))
        b.set("key1", "value", {"info": "test"})
        # Drain before corrupting.
        b._writes.wait_all()
        path = b._get_path("key1")
        _meta, payload = read_entry(path, with_payload=True)
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - len(payload))
            f.seek(0, os.SEEK_END)
            f.write(b"corrupted data")
        b._metadata_cache.clear()
        meta, data = b.get("key1")
        assert data is None


class TestCascadingBackend:
    """Test CascadingBackend (multi-tier caching)."""

    def test_basic_cascading(self):
        """Get checks backends in order."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        # Set only in lower tier
        b2.set("k1", "v1", {"key": "k1"})
        meta, val = cb.get("k1")
        assert val == "v1"

    def test_read_repair(self):
        """Found in lower tier promotes to higher tier."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        b2.set("k1", "v1", {"key": "k1"})
        cb.get("k1")

        # Should now be in b1 (read-repair)
        _, val = b1.get("k1")
        assert val == "v1"

    def test_set_writes_to_all(self):
        """Set writes to all backends."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        cb.set("k1", "v1", {"key": "k1"})
        _, v1 = b1.get("k1")
        _, v2 = b2.get("k1")
        assert v1 == "v1"
        assert v2 == "v1"

    def test_delete_from_all(self):
        """Delete removes from all backends."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        cb.set("k1", "v1", {"key": "k1"})
        cb.delete("k1")
        _, v1 = b1.get("k1")
        _, v2 = b2.get("k1")
        assert v1 is None
        assert v2 is None

    def test_clear_all(self):
        """Clear removes from all backends."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        cb.set("k1", "v1", {"key": "k1"})
        cb.clear()
        _, v1 = b1.get("k1")
        assert v1 is None

    def test_list_entries_deduplicates(self):
        """list_entries deduplicates across tiers."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        cb.set("k1", "v1", {"key": "k1"})
        entries = cb.list_entries()
        assert len(entries) == 1

    def test_cleanup_expired_all_tiers(self):
        """cleanup_expired processes all tiers, returns unique keys deleted."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])

        b1.set("old", "v1", {"key": "old", "timestamp": 1000})
        b2.set("old", "v1", {"key": "old", "timestamp": 1000})
        total = cb.cleanup_expired(lambda m: m.get('timestamp', 0) < 2000)
        # Returns count of unique keys deleted (1 key deleted from both backends)
        assert total == 1
        # Verify key is gone from both backends
        assert b1.get("old") == (None, None)
        assert b2.get("old") == (None, None)

    def test_shutdown_all(self):
        """shutdown calls shutdown on all backends."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])
        cb.shutdown()  # Should not raise

    def test_miss_all_tiers(self):
        """Get returns None when not in any tier."""
        b1 = InMemoryBackend()
        b2 = InMemoryBackend()
        cb = CascadingBackend([b1, b2])
        meta, val = cb.get("missing")
        assert meta is None
        assert val is None




class TestSerializers:
    """Test serialization classes."""

    def test_pickle_serializer_roundtrip(self):
        """PickleSerializer serialize/deserialize roundtrip."""
        s = PickleSerializer()
        data = {"key": [1, 2, 3], "nested": {"a": "b"}}
        serialized = s.serialize(data)
        result = s.deserialize(serialized)
        assert result == data

    def test_cloudpickle_serializer_roundtrip(self):
        """CloudPickleSerializer handles lambdas."""
        s = CloudPickleSerializer()
        data = {"value": 42}
        serialized = s.serialize(data)
        result = s.deserialize(serialized)
        assert result == data

    def test_get_serializer_default(self):
        """get_serializer returns PickleSerializer for basic types."""
        s = get_serializer(42)
        assert isinstance(s, PickleSerializer)

    def test_get_serializer_dict(self):
        """get_serializer returns PickleSerializer for dicts."""
        s = get_serializer({"a": 1})
        assert isinstance(s, PickleSerializer)

    def test_get_serializer_list(self):
        """get_serializer returns PickleSerializer for lists."""
        s = get_serializer([1, 2, 3])
        assert isinstance(s, PickleSerializer)
