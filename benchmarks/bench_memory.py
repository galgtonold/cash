"""What does the cache hold in RAM, on top of what the user already has?

Reads, writes, disk and startup are all on a ledger. Memory was not, and it is
the dimension where cash's default is most surprising: the RAM tier
DEEP-COPIES on write and on read, so a downstream mutation cannot poison a
cached value. The copy is the right call -- a cache that hands out a shared
reference is a cache that corrupts silently -- but it means a cached value
exists twice, and in a notebook the cached things are frames and arrays.

Two numbers, because they answer different questions:

* **Large values** -- the multiple of the object's own size. This is the one
  that decides whether a 4GB frame fits in an 8GB machine.
* **Small values** -- the fixed cost per entry, which is what a cache of many
  small results costs regardless of payload.

Both are RSS deltas around a settled, garbage-collected process. ``getsizeof``
would miss the copy entirely, since it does not follow references.

Usage:
    python benchmarks/bench_memory.py
"""
from __future__ import annotations

import gc
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MB = 1024 * 1024


def _rss(proc) -> int:
    gc.collect()
    return proc.memory_info().rss


def large_values(proc, root) -> None:
    import numpy as np

    from cash import Cash

    print()
    print("  LARGE VALUES -- multiple of the object's own size")
    print()
    print(f"  {'object':<26}{'own size':>11}{'RSS held':>12}{'ratio':>8}")
    for label, n in (("64MB float64 array", 8 * 1024 * 1024),
                     ("256MB float64 array", 32 * 1024 * 1024)):
        d = tempfile.mkdtemp(dir=root)
        c = Cash(cache_dir=d, register_magic=False)

        @c.cache
        def build(k):
            return np.arange(n, dtype="float64")

        base = _rss(proc)
        arr = build(1)
        own = arr.nbytes
        _drain(c)
        held = _rss(proc) - base
        print(f"  {label:<26}{own / MB:>9.0f}MB{held / MB:>10.0f}MB"
              f"{held / own:>8.1f}x")
        del arr
        c.shutdown()


def small_values(proc, root) -> None:
    from cash import Cash

    print()
    print("  SMALL VALUES -- fixed cost per entry")
    print()
    print(f"  {'payload':<26}{'entries':>10}{'RSS held':>12}{'per entry':>12}")
    for payload_kb in (1, 8):
        d = tempfile.mkdtemp(dir=root)
        c = Cash(cache_dir=d, register_magic=False)
        blob = b"x" * (payload_kb * 1024)
        n = 2000
        base = _rss(proc)
        for i in range(n):
            c.backend.set(f"k{i}", blob,
                          {"size": len(blob), "execution_time": 1.0})
        _drain(c)
        held = _rss(proc) - base
        print(f"  {payload_kb}KB{'':<23}{n:>10}{held / MB:>10.1f}MB"
              f"{held / n:>10.0f}B")
        c.shutdown()


def _drain(c) -> None:
    for b in getattr(c.backend, "backends", [c.backend]):
        writes = getattr(b, "_writes", None)
        if writes is not None:
            writes.wait_all()


def main() -> int:
    try:
        import psutil
    except ImportError:
        print("bench_memory needs psutil (pip install psutil)")
        return 1
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("bench_memory needs numpy")
        return 1

    proc = psutil.Process()
    root = tempfile.mkdtemp(prefix="cash_memory_")
    try:
        large_values(proc, root)
        small_values(proc, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
