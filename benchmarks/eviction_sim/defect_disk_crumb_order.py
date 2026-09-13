"""Real FileBackend: are hot small entries evicted ahead of stale large ones?

Cap 100 MB, so the crumb threshold (``_EVICT_CRUMB_FRACTION`` = 0.1%) is 100 KB.

1. Write 80 x 1 MB entries and never read them again (stale).
2. Write 250 x 60 KB entries (crumbs), then read every one (hot).
3. Write 6 more 1 MB entries. The cache goes over its cap, and eviction needs ~11 MB.

Plain LRU evicts ~11 of the stale large entries and nothing hot.

    python benchmarks/eviction_sim/defect_disk_crumb_order.py            # as shipped
    python benchmarks/eviction_sim/defect_disk_crumb_order.py --control  # crumb split off

Measured 2026-09-13: as shipped, 80/80 stale kept and 82/250 hot kept.
Control: 69/80 stale kept and 250/250 hot kept.
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cash.backends.file_backend import FileBackend  # noqa: E402

MB = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="set _EVICT_CRUMB_FRACTION = 0 (plain LRU) for the control arm")
    a = ap.parse_args()
    if a.control:
        FileBackend._EVICT_CRUMB_FRACTION = 0.0

    d = tempfile.mkdtemp(prefix="crumb_repro_")
    b = FileBackend(d, max_size_bytes=100 * MB, flush_interval=3600)
    big = os.urandom(1 * MB)
    small = os.urandom(60_000)

    for i in range(80):
        b.set(f"stale-big-{i}", big + bytes([i]))
    b._writes.wait_all()
    time.sleep(0.05)
    for i in range(250):
        b.set(f"hot-small-{i}", small + i.to_bytes(2, "little"))
    b._writes.wait_all()
    time.sleep(0.05)
    for i in range(250):
        assert b.get(f"hot-small-{i}")[1] is not None
    time.sleep(0.05)
    for i in range(6):
        b.set(f"new-big-{i}", big + bytes([200 + i]))
    b._writes.wait_all()

    def alive(prefix, n):
        return sum(os.path.exists(b._get_path(f"{prefix}-{i}")) for i in range(n))

    print(f"arm: {'control (crumb split off)' if a.control else 'as shipped'}")
    print(f"stale big (never read)   survived: {alive('stale-big', 80):3d} / 80")
    print(f"hot small (just read)    survived: {alive('hot-small', 250):3d} / 250")
    print(f"new big (just written)   survived: {alive('new-big', 6):3d} / 6")
    b.shutdown()


if __name__ == "__main__":
    main()
