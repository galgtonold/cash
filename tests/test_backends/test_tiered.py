"""Tests for TieredBackend functionality.

Covers promotion policy, read-repair, multi-tier operations,
cleanup, and edge cases.
"""
import pytest
from unittest.mock import MagicMock
from cash.backends.backend import InMemoryBackend, FileBackend
from cash.backends.tiered_backend import TieredBackend


@pytest.fixture
def memory_backend():
    return InMemoryBackend()


@pytest.fixture
def file_backend(tmp_path):
    return FileBackend(str(tmp_path / "cache"))


@pytest.fixture
def tiered(memory_backend, file_backend):
    return TieredBackend([memory_backend, file_backend])


class TestTieredBasicOperations:
    """Test basic get/set/delete/clear operations."""

    def test_set_and_get_from_memory(self, tiered, memory_backend):
        """Items should always be stored in tier 0 (memory)."""
        tiered.set("key1", {"val": 42}, {"execution_time": 0.01, "size": 100})
        meta, val = memory_backend.get("key1")
        assert val == {"val": 42}

    def test_get_returns_value(self, tiered):
        """Get should return stored value."""
        tiered.set("k", "hello", {"execution_time": 0.01, "size": 5})
        meta, val = tiered.get("k")
        assert val == "hello"

    def test_get_missing_key(self, tiered):
        """Get on missing key should return (None, None)."""
        meta, val = tiered.get("nonexistent")
        assert meta is None
        assert val is None

    def test_delete_removes_from_all_tiers(self, tiered, memory_backend, file_backend):
        """Delete should remove from all tiers."""
        # Force persist to both tiers
        tiered.set("k", "v", {"execution_time": 5.0, "size": 10})
        tiered.delete("k")
        assert memory_backend.get("k") == (None, None)
        assert file_backend.get("k") == (None, None)

    def test_clear_removes_all(self, tiered, memory_backend, file_backend):
        """Clear should clear all tiers."""
        tiered.set("a", 1, {"execution_time": 5.0, "size": 10})
        tiered.set("b", 2, {"execution_time": 5.0, "size": 10})
        tiered.clear()
        assert memory_backend.get("a") == (None, None)
        assert file_backend.get("a") == (None, None)


class TestPromotionPolicy:
    """Test the promotion policy that decides whether to persist to disk."""

    def test_default_policy_fast_no_promote(self, tiered, file_backend):
        """Fast operations (<1s) should NOT be promoted to disk."""
        tiered.set("fast", "data", {"execution_time": 0.5, "size": 100})
        meta, val = file_backend.get("fast")
        assert val is None  # Not promoted

    def test_default_policy_slow_small_promote(self, tiered, file_backend):
        """Slow operations (>1s) with small data should be promoted."""
        tiered.set("slow", "data", {"execution_time": 5.0, "size": 100})
        meta, val = file_backend.get("slow")
        assert val == "data"  # Promoted!

    def test_force_persist_overrides_policy(self, tiered, file_backend):
        """force_persist=True should always promote regardless of policy."""
        tiered.set("forced", "data", {
            "execution_time": 0.01,
            "size": 100,
            "force_persist": True,
        })
        meta, val = file_backend.get("forced")
        assert val == "data"  # Force-persisted

    def test_custom_promotion_policy(self, memory_backend, file_backend):
        """Custom promotion policy should be respected."""
        # Always promote
        always = TieredBackend(
            [memory_backend, file_backend],
            promotion_policy=lambda t, s: True,
        )
        always.set("k", "v", {"execution_time": 0.001, "size": 1})
        meta, val = file_backend.get("k")
        assert val == "v"

    def test_never_promote_policy(self, memory_backend, file_backend):
        """Never-promote policy should keep items in memory only."""
        never = TieredBackend(
            [memory_backend, file_backend],
            promotion_policy=lambda t, s: False,
        )
        never.set("k", "v", {"execution_time": 100, "size": 1})
        meta, val = file_backend.get("k")
        assert val is None

    def test_storage_metadata_updated(self, tiered):
        """Storage destinations should be recorded in stored metadata."""
        meta = {"execution_time": 5.0, "size": 10}
        tiered.set("k", "v", meta)
        stored_meta, _ = tiered.get("k")
        assert stored_meta is not None
        assert "storage" in stored_meta
        assert "RAM" in stored_meta["storage"]


class TestReadRepair:
    """Test read-repair / promotion from slow to fast tiers."""

    def test_read_repair_from_disk_to_memory(self, memory_backend, file_backend):
        """Reading from disk tier should promote to memory tier."""
        # Write directly to file backend only
        file_backend.set("key", "value", {"execution_time": 5.0, "size": 10})

        # Memory doesn't have it
        assert memory_backend.get("key") == (None, None)

        # Get via tiered should find it in file backend and promote
        tiered = TieredBackend([memory_backend, file_backend])
        meta, val = tiered.get("key")
        assert val == "value"

        # Now memory should have it (read-repair)
        meta2, val2 = memory_backend.get("key")
        assert val2 == "value"

    def test_source_metadata_ram(self, tiered):
        """Source metadata should say 'RAM' when served from memory."""
        tiered.set("k", "v", {"execution_time": 0.01, "size": 1})
        meta, val = tiered.get("k")
        assert meta.get("source") == "RAM"

    def test_source_metadata_disk(self, memory_backend, file_backend):
        """Source metadata should say 'DISK' when served from file backend."""
        file_backend.set("key", "value", {"execution_time": 5.0, "size": 10})
        tiered = TieredBackend([memory_backend, file_backend])
        # First get promotes to memory, but source should reflect where found
        meta, val = tiered.get("key")
        assert meta.get("source") == "DISK"


class TestListEntries:
    """Test listing entries across tiers."""

    def test_list_entries_deduplicates(self, tiered):
        """Same key in multiple tiers should appear only once."""
        tiered.set("k", "v", {"execution_time": 5.0, "size": 10})
        entries = tiered.list_entries()
        keys = [e["key"] for e in entries if "key" in e]
        assert keys.count("k") == 1

    def test_list_entries_combines_tiers(self, memory_backend, file_backend):
        """Entries from different tiers should all appear."""
        memory_backend.set("mem_only", "val", {"execution_time": 0.01, "size": 1})
        file_backend.set("disk_only", "val", {"execution_time": 5.0, "size": 1})
        tiered = TieredBackend([memory_backend, file_backend])
        entries = tiered.list_entries()
        keys = {e["key"] for e in entries if "key" in e}
        assert "mem_only" in keys
        assert "disk_only" in keys


class TestCleanupAndShutdown:
    """Test cleanup and shutdown operations."""

    def test_cleanup_expired(self, tiered):
        """Cleanup should propagate to all backends."""
        tiered.set("k1", "v1", {"execution_time": 0.01, "size": 1})
        # Cleanup with a policy that expires everything
        removed = tiered.cleanup_expired(lambda meta: True)
        assert removed >= 1

    def test_shutdown(self, tiered):
        """Shutdown should be called on all backends."""
        tiered.shutdown()
        # No error = success (backends don't have explicit shutdown tracking)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_backends_list(self):
        """TieredBackend with no backends should handle operations gracefully."""
        tiered = TieredBackend([])
        meta, val = tiered.get("k")
        assert meta is None and val is None
        tiered.set("k", "v", {"execution_time": 1, "size": 1})  # no error
        tiered.delete("k")  # no error
        tiered.clear()  # no error

    def test_single_backend(self):
        """TieredBackend with single backend should work like that backend."""
        mem = InMemoryBackend()
        tiered = TieredBackend([mem])
        tiered.set("k", 42, {"execution_time": 0.01, "size": 1})
        meta, val = tiered.get("k")
        assert val == 42

    def test_backend_failure_on_set(self, memory_backend):
        """If a tier fails on set, others should still work."""
        failing = MagicMock()
        failing.set.side_effect = Exception("disk error")
        tiered = TieredBackend([memory_backend, failing])
        tiered.set("k", "v", {"execution_time": 5.0, "size": 1})
        meta, val = memory_backend.get("k")
        assert val == "v"  # Memory still has it

    def test_three_tier_setup(self, tmp_path):
        """Three-tier setup should work (memory -> file -> file2)."""
        mem = InMemoryBackend()
        f1 = FileBackend(str(tmp_path / "cache1"))
        f2 = FileBackend(str(tmp_path / "cache2"))
        tiered = TieredBackend(
            [mem, f1, f2],
            promotion_policy=lambda t, s: True,  # always promote
        )
        tiered.set("k", "data", {"execution_time": 5.0, "size": 10})
        # All three tiers should have it
        assert mem.get("k")[1] == "data"
        assert f1.get("k")[1] == "data"
        assert f2.get("k")[1] == "data"
