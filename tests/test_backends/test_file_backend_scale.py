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

import glob
import os
import pickle
from collections import deque
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
    assert not backend._evict_queue, (
        "opening a cache ranked it for eviction; only a write that trips the "
        "cap should pay for that"
    )


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
    """The correctness risk of ranking lazily, and the reason for the walk.

    A fresh process ranking only the handful of keys it happens to have read
    would free almost nothing and let the directory grow past its cap forever.
    Ranking comes from a directory walk precisely so it sees entries this
    process has never heard of -- entries whose keys it cannot even name,
    since a filename is a SHA-256 of the key.
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
    assert len(tight._metadata_cache) <= 1, (
        f"eviction pulled {len(tight._metadata_cache)} entries into memory; it "
        f"ranks from the directory now and should hold nothing extra"
    )
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


def test_ranking_reads_no_entry_files(tmp_path):
    """Eviction ranks from a directory walk, not from opening every entry.

    It used to open and unpickle each one to sort by ``last_access``: 44us an
    entry warm, ~4.4s at 100k, on the write worker with every queued write
    behind it -- and it kept all that metadata in RAM afterwards, ~1.7KB each.
    A walk answers the same question at 1.6us an entry.

    Counted, not timed: the claim is "does not open files", which a counter
    states exactly and a stopwatch only suggests.
    """
    cache = tmp_path / "cache"
    _seed(cache, 40, payload=b"x" * 2_000)

    backend = FileBackend(str(cache), max_size_bytes=40_000)
    opened = []
    import cash.backends.entry_format as ef
    real = ef.read_entry

    def counting_read(path, **kw):
        opened.append(path)
        return real(path, **kw)

    ef.read_entry = counting_read
    try:
        backend._rebuild_evict_queue()
    finally:
        ef.read_entry = real

    assert len(backend._evict_queue) + len(backend._evict_crumbs) == 40, (
        f"{len(backend._evict_queue)} + {len(backend._evict_crumbs)} ranked"
    )
    assert opened == [], f"ranking opened {len(opened)} entry files"
    assert backend._metadata_cache == {}, "ranking pulled metadata into memory"


def test_ranking_is_oldest_first(tmp_path):
    """mtime IS last access: recording a read rewrites the header in place."""
    cache = tmp_path / "cache"
    backend = FileBackend(str(cache), flush_interval=0)
    for name in ("old", "mid", "new"):
        backend.set(name, b"x" * 100, {"size": 100})
        backend._writes.wait_all()
        time.sleep(0.02)

    backend._rebuild_evict_queue()
    order = [backend._paths[p] for p, _size, _m in backend._evict_queue]
    assert order == ["old", "mid", "new"], order

    # Reading the oldest must move it to the back.
    time.sleep(0.02)
    backend.get("old")
    backend._rebuild_evict_queue()
    order = [backend._paths[p] for p, _size, _m in backend._evict_queue]
    assert order == ["mid", "new", "old"], (
        f"a read did not refresh the ranking: {order}"
    )
    backend.shutdown()


def test_the_ranking_is_reused_across_eviction_passes(tmp_path):
    """A full cache evicts on most writes; ranking per write would be worse.

    Measured in steady state: eviction is entered on ~100% of writes once the
    cache is full, and frees about one entry per write (conservation -- one
    goes in, one comes out). Walking the directory each time would put a
    directory walk on nearly every write, which is what the old load-once
    design was avoiding by never refreshing at all.
    """
    cache = tmp_path / "cache"
    payload = b"x" * 400
    backend = FileBackend(str(cache), flush_interval=0)
    backend.set("probe", payload, {"size": len(payload)})
    backend._writes.wait_all()
    entry_bytes = os.path.getsize(backend._get_path("probe"))
    backend.shutdown()

    cache2 = tmp_path / "cache2"
    backend = FileBackend(str(cache2), flush_interval=0,
                          max_size_bytes=entry_bytes * 40)
    rebuilds = []
    real = backend._rebuild_evict_queue

    def counted():
        rebuilds.append(1)
        return real()

    backend._rebuild_evict_queue = counted

    for i in range(200):
        backend.set(f"k{i}", payload, {"size": len(payload)})
    backend._writes.wait_all()

    assert len(rebuilds) < 20, (
        f"{len(rebuilds)} directory walks for 200 writes; the ranking is "
        f"supposed to be drained across passes, not rebuilt per eviction"
    )
    backend.shutdown()


def _seed_sized(cache_dir, sizes, oldest_first):
    """Write entries of the given sizes, with mtimes in *oldest_first* order.

    Sparse files: the size is real to ``stat`` without needing the bytes, which
    keeps a 64MB arm cheap.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "CACHE_VERSION").write_text(str(CACHE_FORMAT_VERSION), encoding="utf-8")
    path_of = FileBackend(str(cache_dir))._get_path
    base = time.time() - 100_000
    for rank, key in enumerate(oldest_first):
        p = path_of(key)
        with open(p, "wb") as fh:
            fh.write(pack_entry({"key": key, "size": sizes[key]}, b""))
            fh.truncate(sizes[key])
        os.utime(p, (base + rank, base + rank))
    return path_of


def test_crumbs_are_not_shredded_to_free_space_they_cannot_free(tmp_path):
    """Mixed sizes: many tiny entries must not be destroyed for nothing.

    Oldest-first eviction chews through the tiny entries, freeing almost
    nothing per delete, and reaches the huge one anyway -- so the tiny ones are
    destroyed AND the huge one goes too. Each one is a recompute. Measured
    before this rule on 3000 x 2KB plus one 64MB entry, needing 22.7MB freed:
    3001 entries evicted and the whole 69.9MB cache emptied, against 1
    eviction when the big entry happened to be the oldest.
    """
    cache = tmp_path / "cache"
    sizes = {f"small:{i}": 2 * 1024 for i in range(400)}
    sizes["big"] = 32 * 1024 * 1024
    # every crumb older than the big entry, so plain LRU reaches them first
    order = [f"small:{i}" for i in range(400)] + ["big"]
    path_of = _seed_sized(cache, sizes, order)

    total = sum(sizes.values())
    b = FileBackend(str(cache), max_size_bytes=int(total * 0.75), flush_interval=0)
    b._ensure_size_scanned()
    b._check_and_evict()

    survivors = {f.name for f in os.scandir(cache) if f.name.endswith(ENTRY_SUFFIX)}
    big_gone = os.path.basename(path_of("big")) not in survivors
    crumbs_left = len(survivors) - (0 if big_gone else 1)

    assert big_gone, "the big entry had to go -- it is the only way to free the space"
    assert crumbs_left == 400, (
        f"{400 - crumbs_left} crumbs were destroyed as well; between them they "
        f"hold {400 * 2 * 1024} bytes, which could never have closed the gap"
    )
    b.shutdown()


def test_crumbs_are_evicted_normally_when_they_can_close_the_gap(tmp_path):
    """The control, and the more common case: ordinary LRU is preserved.

    Without it, "always take the big one first" would pass the arm above while
    evicting an expensive entry every time a few cheap ones would have done.
    """
    cache = tmp_path / "cache"
    sizes = {f"small:{i}": 64 * 1024 for i in range(400)}
    sizes["big"] = 4 * 1024 * 1024
    order = [f"small:{i}" for i in range(400)] + ["big"]
    path_of = _seed_sized(cache, sizes, order)

    total = sum(sizes.values())
    b = FileBackend(str(cache), max_size_bytes=int(total * 0.95), flush_interval=0)
    b._ensure_size_scanned()
    b._check_and_evict()

    survivors = {f.name for f in os.scandir(cache) if f.name.endswith(ENTRY_SUFFIX)}
    assert os.path.basename(path_of("big")) in survivors, (
        "the big entry was evicted even though the crumbs could cover the gap"
    )
    assert len(survivors) < 401, "nothing was evicted at all"
    b.shutdown()


def test_uniform_sizes_are_plain_lru(tmp_path):
    """No crumbs, no split: the oldest go first, exactly as before."""
    cache = tmp_path / "cache"
    sizes = {f"e:{i}": 64 * 1024 for i in range(200)}
    order = [f"e:{i}" for i in range(200)]
    path_of = _seed_sized(cache, sizes, order)

    total = sum(sizes.values())
    b = FileBackend(str(cache), max_size_bytes=int(total * 0.8), flush_interval=0)
    b._ensure_size_scanned()
    b._check_and_evict()

    survivors = {f.name for f in os.scandir(cache) if f.name.endswith(ENTRY_SUFFIX)}
    # the oldest went, the newest stayed
    assert os.path.basename(path_of("e:0")) not in survivors, "oldest survived"
    assert os.path.basename(path_of("e:199")) in survivors, "newest was evicted"
    b.shutdown()


# ---------------------------------------------------------------------------
# The ranking is a snapshot. A read AFTER it was taken must still protect.
# ---------------------------------------------------------------------------

def _seed_equal(cache_dir, n, size=64 * 1024):
    """*n* entries of one size, oldest-first by name, written through the API."""
    b = FileBackend(str(cache_dir), max_size_bytes=None, flush_interval=0)
    base = time.time() - 10_000
    for i in range(n):
        b.set(f"e{i}", b"x" * size,
              {"size": size, "created_at": base + i, "last_access": base + i})
    b._writes.wait_all()
    for i in range(n):
        p = b._get_path(f"e{i}")
        os.utime(p, (base + i, base + i))       # deterministic age order
    b.shutdown()


def test_a_read_protects_an_entry_already_queued_for_eviction(tmp_path):
    """The regression the eviction QUEUE introduced.

    The queue is drained across many passes, so an entry can be read after it
    was ranked and before it is reached. Without a check at the point of use
    that read does not protect it: measured, an entry read moments earlier --
    with a strictly newer mtime than its neighbour -- was still evicted first,
    because the queue had it at the head from before the read.

    The design this replaced re-sorted from live ``last_access`` every pass, so
    it did not have this hole. The queue is 25x cheaper; this restores what it
    cost.
    """
    cache = tmp_path / "cache"
    size = 64 * 1024
    _seed_equal(cache, 6, size)

    b = FileBackend(str(cache), max_size_bytes=10 * size, flush_interval=0)
    b._ensure_size_scanned()
    b._rebuild_evict_queue()
    assert b._paths.get(b._evict_queue[0][0]) is None or True   # queue is built

    b.get("e0")                    # the oldest, and at the head of the queue
    b._flush_metadata()

    for i in range(4):             # force eviction
        b.set(f"new{i}", b"x" * size, {"size": size})
    b._writes.wait_all()

    alive = {f.name for f in os.scandir(cache) if f.name.endswith(ENTRY_SUFFIX)}
    assert os.path.basename(b._get_path("e0")) in alive, (
        "an entry read after it was queued was still evicted; the ranking is a "
        "snapshot and nothing re-checked it at the point of use"
    )
    assert os.path.basename(b._get_path("e1")) not in alive, (
        "e1 was never read and should have gone instead"
    )
    b.shutdown()


def test_an_unread_entry_is_still_evicted(tmp_path):
    """The control. A guard that skipped everything would pass the arm above
    while making the cap unenforceable."""
    cache = tmp_path / "cache"
    size = 64 * 1024
    _seed_equal(cache, 6, size)

    b = FileBackend(str(cache), max_size_bytes=6 * size, flush_interval=0)
    b._ensure_size_scanned()
    for i in range(3):
        b.set(f"new{i}", b"x" * size, {"size": size})
    b._writes.wait_all()

    assert b._current_size_bytes <= 6 * size, (
        "the cache stayed over its cap: nothing could be evicted"
    )
    b.shutdown()


def test_an_in_process_read_protects_before_the_flusher_runs(tmp_path):
    """`last_access` moves in memory at once; mtime only when flushed.

    With `flush_interval=0` the flusher never runs, so mtime alone would miss
    this read entirely.
    """
    cache = tmp_path / "cache"
    size = 64 * 1024
    _seed_equal(cache, 6, size)

    b = FileBackend(str(cache), max_size_bytes=10 * size, flush_interval=0)
    b._ensure_size_scanned()
    b._rebuild_evict_queue()
    before = os.path.getmtime(b._get_path("e0"))

    b.get("e0")                    # NO flush: only the in-memory signal moves
    assert os.path.getmtime(b._get_path("e0")) == before, (
        "fixture is wrong: the read reached mtime, so this arm is not testing "
        "the in-memory signal"
    )

    for i in range(4):
        b.set(f"new{i}", b"x" * size, {"size": size})
    b._writes.wait_all()

    alive = {f.name for f in os.scandir(cache) if f.name.endswith(ENTRY_SUFFIX)}
    assert os.path.basename(b._get_path("e0")) in alive
    b.shutdown()


def test_another_processs_read_protects_through_mtime(tmp_path):
    """The other half: a reader in a different process shows up only as mtime.

    Simulated by moving the file's mtime with nothing in this instance's
    metadata cache -- which is exactly what a sibling process leaves behind.
    """
    cache = tmp_path / "cache"
    size = 64 * 1024
    _seed_equal(cache, 6, size)

    b = FileBackend(str(cache), max_size_bytes=10 * size, flush_interval=0)
    b._ensure_size_scanned()
    b._rebuild_evict_queue()

    victim = b._get_path("e0")
    b._metadata_cache.clear()                  # nothing known in-process
    b._paths.clear()
    os.utime(victim, (time.time(), time.time()))   # "another process read it"

    for i in range(4):
        b.set(f"new{i}", b"x" * size, {"size": size})
    b._writes.wait_all()

    alive = {f.name for f in os.scandir(cache) if f.name.endswith(ENTRY_SUFFIX)}
    assert os.path.basename(victim) in alive, (
        "a read by another process left a newer mtime and was ignored"
    )
    b.shutdown()


# ---------------------------------------------------------------------------
# When the cache holds a handful of large results, say so
# ---------------------------------------------------------------------------

def _warning_text(backend):
    """Trigger the ineffective-cache warning and return its message."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend._warned_evict_after_write = False
        backend._warn_evict_after_write(1)
    msgs = [str(w.message) for w in caught if "evicting entries" in str(w.message)]
    return msgs[0] if msgs else ""


def test_dominant_size_is_byte_weighted_not_the_mean(tmp_path):
    """The mean misses exactly the shape this advice is for.

    3000 crumbs beside one 64MB entry: the mean is ~24KB and reports thousands
    fitting the cap, while that single entry is 91% of the cache. Real caches
    are mixtures -- notebook statement crumbs beside big frames -- so the mean
    would stay quiet precisely where the advice matters.
    """
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=80 * 1024 * 1024)
    b._evict_queue = deque([(f"/big", 64 * 1024 * 1024, 0.0)])
    b._evict_crumbs = deque([(f"/c{i}", 2 * 1024, 0.0) for i in range(3000)])

    dominant = b._dominant_entry_size()
    mean = (64 * 1024 * 1024 + 3000 * 2 * 1024) / 3001
    assert dominant == 64 * 1024 * 1024, (
        f"got {dominant}, expected the 64MB entry that IS the cache; the mean "
        f"would have said {mean:.0f}"
    )

    # And the difference is visible in the message, not just in the helper: a
    # mean of ~24KB reports ~3400 entries fitting, which would suppress the
    # advice outright.
    text = _warning_text(b)
    assert "summary of those results" in text, text
    assert "only one fits at once" in text, text
    b.shutdown()


def test_dominant_size_is_none_with_nothing_ranked(tmp_path):
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=10 ** 9)
    assert b._dominant_entry_size() is None
    b.shutdown()


def test_a_handful_of_large_entries_gets_the_advice(tmp_path):
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=12 * 1024 * 1024)
    b._evict_queue = deque([(f"/e{i}", 3 * 1024 * 1024, 0.0) for i in range(4)])

    text = _warning_text(b)
    assert "summary of those results" in text, text
    assert "only about 4 fit" in text, text
    b.shutdown()


def test_many_entries_fitting_gets_no_shape_advice(tmp_path):
    """The suppression branch, exercised directly.

    In practice the warning cannot fire when thousands of entries fit -- the
    evicted entry would be thousands of writes old, so `evicted_recent` is
    false and nothing is emitted at all. That makes this branch belt-and-braces
    rather than load-bearing, and it is tested by calling the warning directly
    rather than by contriving a thrash that cannot happen.
    """
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=1024 * 1024 * 1024)
    b._evict_queue = deque([(f"/e{i}", 64 * 1024, 0.0) for i in range(5000)])

    text = _warning_text(b)
    assert text, "the warning itself should still fire"
    assert "summary of those results" not in text, (
        f"advice was added for a cache that holds ~16000 entries: {text}"
    )
    b.shutdown()


def test_a_healthy_small_entry_cache_never_warns_at_all(tmp_path):
    """The noise control, and the reason no separate size warning was added.

    Small entries turning over under LRU are not in trouble: the evicted entry
    is many writes old, so nothing fires. Size alone is not a problem -- size
    relative to capacity is -- and this warning already fires exactly then.
    """
    import warnings

    cache = tmp_path / "c"
    b = FileBackend(str(cache), max_size_bytes=60 * 1024, flush_interval=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(600):
            b.set(f"k{i}", b"x" * 4096, {"size": 4096})
        b._writes.wait_all()

    assert not [w for w in caught if "evicting entries" in str(w.message)], (
        "a healthy small-entry cache warned; ~15 entries fit and turnover is "
        "ordinary LRU"
    )
    b.shutdown()


def test_eviction_breaks_mtime_ties_by_write_order(tmp_path):
    """A burst that lands inside one filesystem timestamp tick is still LRU.

    ``_rebuild_evict_queue`` ranks on mtime, and a sort on equal keys falls
    back to ``scandir`` order -- a hash, not a time. So a burst of small
    writes that shares one tick is ranked arbitrarily and the entry written
    moments ago is evicted as if it were the coldest.

    Not hypothetical, and not rare: on ext4 twelve 4KB writes landed on TWO
    distinct mtimes (the burst spans 0.7ms; the timestamp step is 0.57ms),
    which is why `test_a_healthy_small_entry_cache_never_warns_at_all` failed
    on every Linux runner while passing on macOS and Windows, where the same
    burst spread over five values.

    The tie is forced here rather than raced for, so this fails on any
    filesystem: what is under test is the ordering rule, not the clock.
    """
    cache = tmp_path / "c"
    b = FileBackend(str(cache), max_size_bytes=200 * 1024, flush_interval=0)
    for i in range(10):
        b.set(f"k{i}", b"x" * 4096, {"size": 4096})
    b._writes.wait_all()

    # Every entry now looks written at the same instant, which is what a fast
    # burst produces on a coarse-granularity filesystem.
    #
    # The stamp is in the FUTURE on purpose. Ranking prefers an in-memory
    # `last_access` when it is NEWER than mtime, and back-dating the files
    # would hand the sort that per-entry value and quietly rescue it -- which
    # is how the first version of this test passed against the unfixed code.
    # In the real failure mtime is recorded after `last_access`, so the
    # override never fires and the coarse mtime is all the sort gets.
    stamp = time.time() + 60
    for path in glob.glob(os.path.join(str(cache), "*.entry")):
        os.utime(path, (stamp, stamp))

    b._rebuild_evict_queue()
    order = [b._paths.get(p) for p, _s, _m in b._evict_queue]

    assert order == [f"k{i}" for i in range(10)], (
        "eviction ranked a tied-mtime burst by directory order, not by when "
        f"the entries were written; got {order}"
    )
    b.shutdown()
