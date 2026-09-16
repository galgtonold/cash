"""Real InMemoryBackend: two eviction defects, each beside its control arm.

1. One value over 0.9 x the byte cap empties the tier. The new entry is in
   its own eviction candidate list, so it evicts everything older, then
   itself.
2. Pressure eviction reads HOST memory. When the pressure is not cash's, it
   empties the tier (``psutil`` is patched here so the result is
   deterministic).

    python benchmarks/eviction_sim/defect_ram_tier.py

Measured 2026-09-13:
  [flush]    oversized: 0/20 kept, big not kept.  Control: 20/20, big kept.
  [pressure] host 95%:  0/20 kept.                Control (50%): 20/20.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cash.backends import memory_backend  # noqa: E402
from cash.backends.memory_backend import InMemoryBackend  # noqa: E402

MB = 1_000_000


def oversized_write():
    for label, big in (("oversized (0.95 x cap)", int(0.95 * 100 * MB)),
                       ("control   (0.30 x cap)", int(0.30 * 100 * MB))):
        b = InMemoryBackend(max_size_bytes=100 * MB)
        for i in range(20):
            b.set(f"small-{i}", bytes(1 * MB), {"execution_time": 5.0})
            b.get(f"small-{i}")  # hot
        b.set("big", bytes(big), {"execution_time": 0.2})
        kept = sum(b.get(f"small-{i}")[0] is not None for i in range(20))
        print(f"[flush] {label}: hot 1MB entries kept {kept:2d}/20, "
              f"big kept={b.get('big')[0] is not None}")


def host_pressure():
    # System memory is high because of OTHER processes; cash's tier holds 20 MB.
    for label, pct in (("host at 95% (not cash)", 95.0), ("control: host at 50%", 50.0)):
        b = InMemoryBackend(max_size_bytes=None)
        fake = SimpleNamespace(virtual_memory=lambda pct=pct: SimpleNamespace(percent=pct))
        with mock.patch.object(memory_backend, "psutil", fake):
            for i in range(20):
                b.set(f"e-{i}", bytes(1 * MB), {"execution_time": 30.0})  # 30 s each to recompute
        kept = sum(f"e-{i}" in b._store for i in range(20))
        print(f"[pressure] {label}: entries kept {kept:2d}/20 after 20 writes")


if __name__ == "__main__":
    oversized_write()
    host_pressure()
