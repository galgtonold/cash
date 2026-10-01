import sys
import time
from unittest.mock import patch

import pytest

from cash.backends import InMemoryBackend


class TestSmartMemoryBackend:
    @pytest.fixture(autouse=True)
    def _set_up(self):
        self.backend = InMemoryBackend(max_memory_percent=0.8, check_interval=1)

    def test_eviction_logic(self):
        """Under memory pressure the entry worth least per byte goes first:
        large, cheap and never read, not small, costly and read twice."""
        from types import SimpleNamespace

        from cash.backends import memory_backend

        reading = SimpleNamespace(percent=50.0, total=100_000_000)
        with (
            patch.object(memory_backend, "psutil", SimpleNamespace(virtual_memory=lambda: reading)),
            patch.object(InMemoryBackend, "_PRESSURE_KEEPS_BYTES", 0),
        ):
            self.backend.set("item1", "A" * 100_000, metadata={"execution_time": 0.1})
            self.backend.set("item2", "B" * 10, metadata={"execution_time": 1.0})
            self.backend.set("item3", "C" * 1_000, metadata={"execution_time": 0.5})
            self.backend.get("item2")
            self.backend.get("item2")

            # The next write's check finds the machine at 95%: this tier's
            # share of the overshoot is a few kB, so one entry goes.
            reading.percent = 95.0
            self.backend.set("item4", "D", metadata={"execution_time": 0.1})

            keys = [e["key"] for e in self.backend.list_entries()]
            assert "item1" not in keys
            assert "item2" in keys
            assert "item3" in keys

    def test_malloc_trim_called(self):
        if not sys.platform.startswith("linux"):
            return

        with patch("ctypes.CDLL") as mock_cdll:
            self.backend.clear()
            # Verify malloc_trim called
            mock_cdll.return_value.malloc_trim.assert_called_with(0)

    def test_max_entries_eviction(self):
        """Test that max_entries triggers LRU eviction."""
        backend = InMemoryBackend(max_entries=3)

        # Add 3 items
        backend.set("key1", "val1", metadata={"execution_time": 0.1})
        time.sleep(0.01)
        backend.set("key2", "val2", metadata={"execution_time": 0.1})
        time.sleep(0.01)
        backend.set("key3", "val3", metadata={"execution_time": 0.1})

        assert len(backend._store) == 3

        # Access key1 to make it recently used
        backend.get("key1")
        time.sleep(0.01)

        # Add 4th item - should evict the LRU (key2, since key1 was accessed more recently)
        backend.set("key4", "val4", metadata={"execution_time": 0.1})

        assert len(backend._store) == 3
        keys = {e["key"] for e in backend.list_entries()}
        assert "key1" in keys, "key1 was recently accessed, should not be evicted"
        assert "key4" in keys, "key4 was just added, should not be evicted"
        assert "key2" not in keys, "key2 was LRU, should be evicted"

    def test_max_entries_none_unlimited(self):
        """Test that max_entries=None allows unlimited entries."""
        backend = InMemoryBackend(max_entries=None)
        for i in range(100):
            backend.set(f"key_{i}", f"val_{i}")
        assert len(backend._store) == 100
