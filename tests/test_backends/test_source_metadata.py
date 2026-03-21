from cash.backends.tiered_backend import TieredBackend
from cash.backends.backend import InMemoryBackend, FileBackend

class TestSourceMetadata:
    
    def test_source_ram(self):
        l1 = InMemoryBackend()
        backend = TieredBackend([l1])
        
        backend.set("key", "value")
        meta, val = backend.get("key")
        
        assert meta['source'] == 'RAM'
        assert val == "value"

    def test_source_disk(self, tmp_path):
        l1 = InMemoryBackend() # Empty L1
        l2 = FileBackend(str(tmp_path))
        backend = TieredBackend([l1, l2])
        
        # Manually populate L2 (Tier 2/Disk)
        l2.set("disk_key", "disk_val")
        
        # Get should come from Disk (and promote to RAM, but metadata reflects source of truth for THIS access?)
        # Logic: enumerate backends.
        # i=0 (L1): miss.
        # i=1 (L2): hit.
        # Inject source=DISK.
        # Promote to L1.
        # Return.
        
        meta, val = backend.get("disk_key")
        
        assert val == "disk_val"
        assert meta['source'] == 'DISK'
        
        # Next access should be RAM?
        # get() again.
        # i=0 (L1): hit (promoted).
        # Inject source=RAM.
        
        meta2, val2 = backend.get("disk_key")
        assert val2 == "disk_val"
        assert meta2['source'] == 'RAM'
