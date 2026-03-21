
import unittest
import time
from unittest.mock import MagicMock, patch
import sys
from cash.backends.backend import InMemoryBackend

class TestSmartMemoryBackend(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryBackend(max_memory_percent=0.8, check_interval=1)
        
    def test_eviction_logic(self):
        # We need to mock psutil.virtual_memory
        with patch('psutil.virtual_memory') as mock_mem:
            # Initial state: Memory is low (50%)
            mock_mem.return_value.percent = 50.0
            
            # Add 3 items
            # Item 1: High Size, Low Cost, Low Access
            self.backend.set("item1", "A" * 1000, metadata={'execution_time': 0.1})
            
            # Item 2: Small Size, High Cost, High Access
            self.backend.set("item2", "B" * 10, metadata={'execution_time': 1.0})
            
            # Item 3: Medium Size, Medium Cost, Medium Access
            self.backend.set("item3", "C" * 100, metadata={'execution_time': 0.5})
            
            # Simulate accesses
            self.backend.get("item2")
            self.backend.get("item2") # Access = 3 (set + 2 gets, wait set init access to 0)
            # Actually set init access to 0. get adds 1.
            # So currently: 
            # item1: access=0, score=0
            # item2: access=2, score=(1.0 * 2)/10 = 0.2
            # item3: access=0, score=0
            
            # Wait, access count is updated on get.
            
            # Trigger eviction: Set memory high (90%) initially, then drop to 50% after one call
            # We need a dynamic return value.
            # first call (check_and_evict): 90%
            # second call (inside _evict loop): 70% -> Stop evicting
            
            mock_vals = [MagicMock(), MagicMock(), MagicMock()]
            mock_vals[0].percent = 95.0 # Check triggers eviction (>90%)
            mock_vals[1].percent = 70.0 # Loop check shows sufficient drop to 70% -> Stop evicting
            
            mock_mem.side_effect = mock_vals
            
            # Add item 4 to trigger check (check_interval=1)
            self.backend.set("item4", "D", metadata={'execution_time': 0.1})
            
            # Now eviction should have happened.
            # Target is 0.8 * 0.9 = 72%. 
            # Item 1 should be evicted first (lowest score).
            # Scores:
            # item1: acc=0, size=1000ish. score=0.
            # item2: acc=2, size=small. score=high.
            # item3: acc=0, size=100ish. score=0.
            # item4: acc=0, size=small. score=small.
            
            # item1 and item3 have score 0. item1 is larger? 
            # Logic: score = (exec * acc) / size.
            # item1: (0.1 * 0) / 1000 = 0
            # item3: (0.5 * 0) / 100 = 0
            # Tie breaker: Last Access. 
            # All set around same time.
            
            entries = self.backend.list_entries()
            keys = [e['key'] for e in entries]
            
            # We expect item1 to be gone.
            self.assertNotIn("item1", keys)
            # item2 should definitely be there
            self.assertIn("item2", keys)

    def test_malloc_trim_called(self):
        if not sys.platform.startswith('linux'):
            return 
            
        with patch('ctypes.CDLL') as mock_cdll:
            self.backend.clear()
            # Verify malloc_trim called
            mock_cdll.return_value.malloc_trim.assert_called_with(0)

    def test_max_entries_eviction(self):
        """Test that max_entries triggers LRU eviction."""
        backend = InMemoryBackend(max_entries=3)
        
        # Add 3 items
        backend.set("key1", "val1", metadata={'execution_time': 0.1})
        time.sleep(0.01)
        backend.set("key2", "val2", metadata={'execution_time': 0.1})
        time.sleep(0.01)
        backend.set("key3", "val3", metadata={'execution_time': 0.1})
        
        self.assertEqual(len(backend._store), 3)
        
        # Access key1 to make it recently used
        backend.get("key1")
        time.sleep(0.01)
        
        # Add 4th item - should evict the LRU (key2, since key1 was accessed more recently)
        backend.set("key4", "val4", metadata={'execution_time': 0.1})
        
        self.assertEqual(len(backend._store), 3)
        keys = {e['key'] for e in backend.list_entries()}
        self.assertIn("key1", keys, "key1 was recently accessed, should not be evicted")
        self.assertIn("key4", keys, "key4 was just added, should not be evicted")
        self.assertNotIn("key2", keys, "key2 was LRU, should be evicted")

    def test_max_entries_none_unlimited(self):
        """Test that max_entries=None allows unlimited entries."""
        backend = InMemoryBackend(max_entries=None)
        for i in range(100):
            backend.set(f"key_{i}", f"val_{i}")
        self.assertEqual(len(backend._store), 100)

if __name__ == '__main__':
    unittest.main()
