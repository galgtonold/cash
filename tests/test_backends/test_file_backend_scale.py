"""Opening a big cache directory must not cost one file read per entry.

``_init_stats`` runs once per process on the first cache operation. It used to
``open()`` and unpickle every ``.meta`` in the directory to prefill
``_metadata_cache`` -- linear in the entry count, and it dominated startup:
98ms at 1k entries, 490ms at 5k, 2.1s at 20k, so roughly 10s at 100k and 100s
at 1M. Every process paid it, however few keys it went on to touch.

It was pure prefetch. ``get`` already reads the ``.meta`` from disk when a key
is not cached, so nothing depended on it for correctness -- and holding every
entry's metadata costs ~1.7KB each, 1.7GB at a million entries.

The size cap only ever needed the sizes, which ``scandir``/``stat`` gives 31x
faster. The LRU genuinely does need the metadata, so eviction loads it on
demand: the cost moves to the processes that actually evict.

That left one cost still scaling with the directory -- the ``scandir`` walk
itself, measured linear at ~3.1us/entry from 1k to 100k (4.3ms, 16.3ms,
59.9ms, 164ms, 308ms), so ~3.1s at a million entries. It too was paid at open
time by every process, and it too has exactly one consumer: the size cap. So
it is now deferred to the first write as well, which means a read-only run --
a kernel restart replaying entirely from cache -- pays nothing for it, and a
writing run pays it on the background write worker instead of the caller's
first cell.

The performance half of this is measured in ``benchmarks/bench_cache_scale.py``.
What is pinned here is the behaviour that makes it safe.
"""
from __future__ import annotations

import os
import pickle
import threading
import time

import pytest

from cash.backends import FileBackend
from cash.backends.file_backend import CACHE_FORMAT_VERSION
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry


def _seed(cache_dir, n, payload=b"x" * 256):
    """Write *n* entries directly, as a previous session would have left them.

    The ``CACHE_VERSION`` stamp is not optional: without it the format check
    treats the directory as written by an incompatible build and deletes every
    entry on first use, so the fixture would seed into a cache that promptly
    wiped itself.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "CACHE_VERSION").write_text(str(CACHE_FORMAT_VERSION),
                                             encoding="utf-8")
    path_of = FileBackend(str(cache_dir))._get_path
    keys = []
    for i in range(n):
        key = f"mod.f:state:{i}:args"
        keys.append(key)
        meta = {"key": key, "size": len(payload),
                "created_at": time.time(),
                "last_access": time.time() + i,
                "access_count": 1, "storage": ["DISK"]}
        with open(path_of(key), "wb") as fh:
            fh.write(pack_entry(meta, payload))
    return keys


def test_opening_a_cache_does_not_read_every_entry(tmp_path):
    """The whole point: startup cost must not scale with the directory."""
    cache = tmp_path / "cache"
    _seed(cache, 25)

    backend = FileBackend(str(cache))
    backend.get("mod.f:state:0:args")          # forces _ensure_initialized

    assert len(backend._metadata_cache) <= 1, (
        f"init loaded {len(backend._metadata_cache)} metadata entries; it should "
        f"load none, and `get` should have cached only the key it was asked for"
    )
    assert backend._metadata_loaded is False


def test_the_size_total_is_still_right(tmp_path):
    """A cheaper scan is worthless if it stops counting correctly.

    Triggered by a WRITE, because the total is established on demand now and a
    read is precisely the case that must not establish it -- see
    ``test_a_read_only_process_never_walks_the_directory``.
    """
    cache = tmp_path / "cache"
    _seed(cache, 20)

    backend = FileBackend(str(cache), max_size_bytes=10 ** 9)
    backend.set("trigger", b"z" * 100,
                {"size": 100, "created_at": time.time(), "last_access": time.time()})
    backend._writes.wait_all()

    assert backend._current_size_bytes == _on_disk(cache)
def test_a_key_is_still_readable_without_being_preloaded(tmp_path):
    """`get` reads the metadata itself -- that is why the prefetch was optional.

    Seeded through ``set`` rather than by writing files: this arm reads a
    VALUE back, so the payload has to go through the serializer the way a real
    session would write it.
    """
    cache = tmp_path / "cache"
    writer = FileBackend(str(cache))
    for i in range(10):
        writer.set(f"mod.f:state:{i}:args", {"payload": i})
    writer._writes.wait_all()

    reader = FileBackend(str(cache))          # fresh: nothing preloaded
    metadata, value = reader.get("mod.f:state:7:args")
    assert value == {"payload": 7}
    assert metadata is not None and metadata["key"] == "mod.f:state:7:args"
    assert len(reader._metadata_cache) == 1, "reading one key preloaded others"


def test_eviction_reaches_entries_this_process_never_touched(tmp_path):
    """The correctness risk of loading lazily, and the reason for the on-demand load.

    A fresh process ranking only the handful of keys it happens to have read
    would free almost nothing and let the directory grow past its cap forever.
    """
    cache = tmp_path / "cache"
    _seed(cache, 60, payload=b"x" * 5_000)
    before = len(list(cache.glob(f"*{ENTRY_SUFFIX}")))

    tight = FileBackend(str(cache), max_size_bytes=100_000)
    tight.set("newcomer", b"y" * 5_000,
              {"size": 5_000, "created_at": time.time(), "last_access": time.time()})
    tight._writes.wait_all()

    after = len(list(cache.glob(f"*{ENTRY_SUFFIX}")))
    assert after < before, (
        "eviction freed nothing: it can only see keys this process touched"
    )
    assert tight._metadata_loaded is True, "eviction did not load the metadata"
    assert tight._current_size_bytes <= 100_000 * 1.1


def test_deleting_an_untouched_key_updates_the_size(tmp_path):
    """Delete used to take the size from the metadata cache.

    With metadata loaded lazily an uncached key would have subtracted 0 and
    drifted the total upward forever. Measuring the files also counts the
    ``.meta`` bytes, which the cached ``size`` never included.
    """
    cache = tmp_path / "cache"
    _seed(cache, 5)

    backend = FileBackend(str(cache))
    backend.get("mod.f:state:0:args")          # cache exactly one key
    before = backend._current_size_bytes

    victim = "mod.f:state:3:args"              # never touched by this process

    on_disk = os.path.getsize(backend._get_path(victim))

    backend.delete(victim)
    assert backend._current_size_bytes == before - on_disk, (
        "the size total did not drop by what was actually removed"
    )


def _count_scans(backend):
    """Wrap ``_scan_size_bytes`` so a test can count walks, not time them.

    Counted rather than timed on purpose. A wall-clock assertion here would be
    a threshold test, and every residual flake in this suite has been one --
    doubly so on a box where the first touch of a freshly written file costs
    ~13ms against ~0.13ms warm.
    """
    calls = []
    real = backend._scan_size_bytes

    def counted():
        calls.append(1)
        return real()

    backend._scan_size_bytes = counted
    return calls


def _on_disk(cache_dir):
    return sum(f.stat().st_size for f in cache_dir.iterdir()
               if f.suffix == ENTRY_SUFFIX)


def test_a_read_only_process_never_walks_the_directory(tmp_path):
    """Opening a cache and reading from it must not cost a directory walk.

    The byte total exists only for the size cap, and ``_check_and_evict`` is
    the only thing that reads it -- so a process that opens a 100k-entry cache
    and reads a few keys was paying ~0.3s (3.1us/entry, measured linear from
    1k to 100k) to compute a number it never consulted. That process is the
    kernel-restart replay: all hits, no writes, the case cash exists to make
    fast.

    The cap is set deliberately. Without one there is nothing to scan FOR, so
    an uncapped backend would pass this even with the eager scan restored.
    """
    cache = tmp_path / "cache"
    _seed(cache, 30)

    backend = FileBackend(str(cache), max_size_bytes=10 ** 9)
    scans = _count_scans(backend)

    backend.get("mod.f:state:0:args")
    backend.get("mod.f:state:1:args")
    backend.get_metadata("mod.f:state:2:args")
    backend.list_entries()

    assert scans == [], f"a read-only run walked the directory {len(scans)}x"


def test_an_uncapped_backend_never_walks_the_directory_at_all(tmp_path):
    """No cap means no consumer for the total, so not even a write needs it."""
    cache = tmp_path / "cache"
    _seed(cache, 10)

    backend = FileBackend(str(cache))          # max_size_bytes=None
    scans = _count_scans(backend)

    backend.set("newcomer", b"y" * 100,
                {"size": 100, "created_at": time.time(), "last_access": time.time()})
    backend._writes.wait_all()

    assert scans == []


def test_the_first_write_establishes_the_total_exactly_once(tmp_path):
    """The correctness half, and the reason the walk is deferred and not deleted.

    Eviction has to know what the directory holds, including entries this
    process never wrote. Deferring the walk must not turn the total into
    "only what I wrote", or a fresh process would evict nothing and the cache
    would grow past its cap forever.
    """
    cache = tmp_path / "cache"
    _seed(cache, 20)

    backend = FileBackend(str(cache), max_size_bytes=10 ** 9)
    scans = _count_scans(backend)

    backend.set("newcomer", b"y" * 1_000,
                {"size": 1_000, "created_at": time.time(), "last_access": time.time()})
    backend._writes.wait_all()

    assert len(scans) == 1, f"the first write walked the directory {len(scans)}x"
    assert backend._current_size_bytes == _on_disk(cache), (
        "the total counts only this process's own write, so eviction would "
        "never see the 20 entries that were already there"
    )

    backend.set("newcomer2", b"y" * 1_000,
                {"size": 1_000, "created_at": time.time(), "last_access": time.time()})
    backend._writes.wait_all()

    assert len(scans) == 1, "the walk repeated; it is supposed to latch"
    assert backend._current_size_bytes == _on_disk(cache)


def test_eviction_frees_what_it_needs_and_not_much_more(tmp_path):
    """Eviction used to measure each victim in the wrong unit.

    The loop worked down a shortfall computed from the TOTAL entry bytes --
    header, metadata and payload -- while crediting each eviction with only
    the victim's ``size``, which is the payload alone. Every eviction looked
    smaller than it was, so the loop kept taking entries after it already had
    enough.

    The payload here is deliberately small next to an entry's fixed overhead,
    which is what makes the discrepancy loud: crediting ~100 bytes for an
    eviction that frees ~300 over-evicts roughly threefold. With two files per
    entry the same gap existed and was small enough to hide.
    """
    cache = tmp_path / "cache"
    _seed(cache, 40, payload=b"x" * 100)
    entry_bytes = _on_disk(cache) / 40

    # Over the cap by about two entries' worth.
    cap = int(entry_bytes * 30)
    backend = FileBackend(str(cache), max_size_bytes=cap)
    backend.set("newcomer", b"y" * 100,
                {"size": 100, "created_at": time.time(), "last_access": time.time()})
    backend._writes.wait_all()

    target = cap * 0.9
    remaining = _on_disk(cache)
    assert remaining <= target, "eviction did not get under the cap"
    assert remaining > target - 2 * entry_bytes, (
        f"evicted down to {remaining:.0f} bytes when {target:.0f} would have "
        f"done -- about {(target - remaining) / entry_bytes:.0f} entries too many"
    )


@pytest.mark.timeout(60)
def test_eviction_never_waits_on_a_write_it_cannot_reach(tmp_path):
    """Evicting a key with a queued write used to hang the process forever.

    `_check_and_evict` runs ON the single PendingWrites worker, and evicting
    calls `delete`, which drains that key's pending write. A write QUEUED
    BEHIND the one currently executing can only run on that same worker -- so
    the worker waits for a task that cannot start until it returns. The write
    thread hangs permanently, and the atexit flush behind it hangs with it.

    Reachable without contriving anything: a key's cached `last_access` is
    still its PREVIOUS write's, because the queued write has not updated it
    yet. Re-computing a long-idle entry while the cache sits near its cap
    therefore makes that entry the obvious LRU victim.

    The timeout marker is the real assertion -- without the fix this test does
    not fail, it hangs -- but the arms below say what should have happened, so
    a future change that completes for the wrong reason still gets caught.
    """
    payload = b"x" * 4000
    cache = tmp_path / "cache"

    def meta(when):
        return {"size": len(payload), "created_at": when, "last_access": when}

    # Cap that two entries exceed, so the second write must evict the first.
    backend = FileBackend(str(cache), max_size_bytes=6_000, flush_interval=0)

    # Seed the victim: on disk, in the metadata cache, with the oldest access.
    backend.set("victim", payload, meta(1.0))
    backend._writes.wait_all()

    running = threading.Event()
    real_write = backend._write_cache_files

    def slow_for_newcomer(key, *args, **kwargs):
        if key == "newcomer":
            running.set()
            time.sleep(1.0)        # hold the worker so the resubmit queues up
        return real_write(key, *args, **kwargs)

    backend._write_cache_files = slow_for_newcomer

    backend.set("newcomer", payload, meta(50.0))
    assert running.wait(10), "the slow write never started; fixture is broken"
    backend.set("victim", payload, meta(2.0))   # queued BEHIND newcomer

    backend._writes.wait_all()                  # hangs forever without the fix

    assert backend.get("victim")[1] == payload, (
        "the key with a queued write was evicted; its write should have "
        "protected it, being the newest thing in the cache"
    )
    backend.shutdown()
