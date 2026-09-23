"""The disk tier's cap evicts by value per byte (GreedyDual-Size-Frequency).

Same ordering as the RAM tier (see test_ram_tier_value_eviction): the lowest
``H = L + hits * execution_time / size`` goes first, and the clock L rises to
each victim's H. The disk tier cannot read every entry's header to rank --
measured at 111 us an entry warm and 5.7 ms cold on Windows, against 2.3 us
for the ``scandir`` it ranks from -- so each entry's H is kept in an advisory
sidecar, ``_rank.log``: one line appended per write and per access flush, read
once per ranking. It is a cache of what the headers say, never the truth: lose
it, and entries rank as if their cost were unknown.

10 MB cap and ~1 MB values throughout, so each docstring's arithmetic can be
checked by hand.
"""

from __future__ import annotations

import os

from cash.backends import FileBackend

MB = 1_000_000
CAP = 10 * MB
INDEX = "_rank.log"


def _backend(cache_dir, cap=CAP):
    return FileBackend(str(cache_dir), max_size_bytes=cap, flush_interval=0)


def _put(b, key, size, seconds):
    b.set(key, b"x" * size, {"execution_time": seconds})
    b._writes.wait_all()


def _held(b, key):
    return os.path.exists(b._get_path(key))


def test_a_costly_result_outlives_cheaper_newer_ones(tmp_path):
    """Break caught: the ranking ignores execution time (LRU).

    'costly' is the OLDEST entry, and 1% bigger than the rest: 1 MB that
    took 30 s. Ten 1 MB entries of 50 ms each put the cache over 10 MB, and
    about 1 MB must go. By value per byte that is cheap entries; by age or
    by size it is 'costly'. (Equal sizes would let a one-byte difference in
    the stored key decide it, and the test would pass with cost ignored.)
    """
    b = _backend(tmp_path / "c")
    _put(b, "costly", MB + 10_000, 30.0)
    for i in range(10):
        _put(b, f"cheap-{i}", MB, 0.05)

    assert _held(b, "costly")
    assert not _held(b, "cheap-0")
    b.shutdown()


def test_value_is_per_byte_so_a_big_value_goes_before_small_ones(tmp_path):
    """Break caught: the ranking ignores size.

    Six 1 MB entries, then one 4 MB entry, all 1 s to compute: over the cap.
    Per byte the 4 MB entry saves a quarter of what each small one does, so
    it goes, although it is the newest.
    """
    b = _backend(tmp_path / "c")
    for i in range(6):
        _put(b, f"small-{i}", MB, 1.0)
    _put(b, "big", 4 * MB, 1.0)

    assert not _held(b, "big")
    assert all(_held(b, f"small-{i}") for i in range(6))
    b.shutdown()


def test_a_restarted_process_still_knows_what_entries_are_worth(tmp_path):
    """Break caught: value lives only in the process that wrote it.

    A kernel restart is the normal case: the process that evicts is rarely
    the one that wrote. Process A writes 'costly' (30 s, the oldest) and
    eight cheap entries, under the cap. 'costly' is also 1% bigger than the
    rest, so by size and by age it is the first to go. Process B, knowing
    nothing in memory, writes two more and must evict; only the rank index
    can tell it that 'costly' is worth keeping.
    """
    cache = tmp_path / "c"
    a = _backend(cache)
    _put(a, "costly", MB + 10_000, 30.0)
    for i in range(8):
        _put(a, f"cheap-{i}", MB, 0.05)
    a.shutdown()

    b = _backend(cache)
    _put(b, "later-0", MB, 0.05)
    _put(b, "later-1", MB, 0.05)

    assert _held(b, "costly")
    assert not _held(b, "cheap-0")
    b.shutdown()


def test_without_the_index_a_restart_ranks_entries_as_unknown_cost(tmp_path):
    """Control arm for the test above: it is the index that carries value.

    Same sequence, but the index is deleted between the two processes. With
    nothing to go on, B ranks A's entries as costing an unknown (small)
    amount, below the new entries it knows, so it evicts among A's by size
    and age -- and 'costly', the biggest and oldest, goes first. The
    entries B wrote itself survive.
    """
    cache = tmp_path / "c"
    a = _backend(cache)
    _put(a, "costly", MB + 10_000, 30.0)
    for i in range(8):
        _put(a, f"cheap-{i}", MB, 0.05)
    a.shutdown()
    os.remove(cache / INDEX)

    b = _backend(cache)
    _put(b, "later-0", MB, 0.05)
    _put(b, "later-1", MB, 0.05)

    assert not _held(b, "costly")
    assert _held(b, "later-0") and _held(b, "later-1")
    b.shutdown()


def test_a_new_cheap_value_is_a_candidate_before_the_next_ranking(tmp_path):
    """Break caught: entries written after the ranking was taken are never
    considered until it is rebuilt.

    Under LRU that was harmless -- the newest entry is the last to go. Under
    GDSF it is not: a cheap 4 MB value can be the least valuable thing in
    the cache the moment it lands. Two near-free fillers and eight good
    entries fill the cache; the tenth write takes the ranking and evicts the
    fillers. Then 'bulky' (4 MB, 10 ms) arrives, and it is the one that goes
    on the next eviction -- not three good entries worth 400x more per byte,
    which is what a ranking that never saw it would take.
    """
    b = _backend(tmp_path / "c")
    for i in range(2):
        _put(b, f"filler-{i}", MB, 0.001)
    for i in range(8):
        _put(b, f"good-{i}", MB, 1.0)  # the 10th write takes the ranking
    assert not _held(b, "filler-0"), "precondition: the ranking has been taken"
    _put(b, "bulky", 4 * MB, 0.01)

    assert not _held(b, "bulky")
    assert all(_held(b, f"good-{i}") for i in range(8))
    b.shutdown()


def test_hot_small_entries_outlive_stale_large_ones(tmp_path):
    """Break caught: the old size split -- "entries under 0.1% of the cap go
    first whenever they alone can close the gap" -- which ignored recency
    across the two size classes.

    Cap 100 MB: 80 x 1 MB written and never read again, then 250 x 60 KB
    read once each, then 6 more 1 MB writes, needing ~11 MB. The split took
    168 of the hot 60 KB entries and kept all 80 stale 1 MB ones. No costs
    are recorded here, so value per byte is hits / size: every stale 1 MB
    entry ranks below every hot 60 KB one.
    """
    b = FileBackend(str(tmp_path / "c"), max_size_bytes=100 * MB, flush_interval=0)
    for i in range(80):
        b.set(f"stale-{i}", b"x" * MB)
    b._writes.wait_all()
    for i in range(250):
        b.set(f"hot-{i}", b"x" * 60_000)
    b._writes.wait_all()
    for i in range(250):
        assert b.get(f"hot-{i}")[1] is not None
    for i in range(6):
        b.set(f"new-{i}", b"x" * MB)
    b._writes.wait_all()

    assert all(_held(b, f"hot-{i}") for i in range(250))
    assert sum(_held(b, f"stale-{i}") for i in range(80)) < 80
    b.shutdown()


def test_reading_an_entrys_metadata_does_not_protect_it(tmp_path):
    """Break caught: a metadata lookup counts as a use.

    ``get_metadata`` loads an entry's metadata without reading its value --
    freshness checks do it to a statement's producers. It must not re-base
    the entry's priority as if it had just been used. That only shows once
    the clock has moved, so process A first writes 20 equal entries through
    a 10 MB cap: evictions raise the clock, and each survivor's priority
    records the clock it was written at. B looks at the oldest survivor's
    metadata, then writes more. The oldest survivor must still go first.
    """
    cache = tmp_path / "c"
    a = _backend(cache)
    for i in range(20):
        _put(a, f"e-{i}", MB, 1.0)
    survivors = [i for i in range(20) if _held(a, f"e-{i}")]
    a.shutdown()
    oldest = f"e-{survivors[0]}"

    b = _backend(cache)
    assert b.get_metadata(oldest) is not None
    for i in range(3):
        _put(b, f"later-{i}", MB, 1.0)

    assert not _held(b, oldest)
    b.shutdown()


def _twenty_through_the_cap(cache):
    """Process A's part of the two tests below: 20 equal entries through
    the cap. Returns the survivors' keys, oldest first."""
    a = _backend(cache)
    for i in range(20):
        _put(a, f"e-{i}", MB, 1.0)
    survivors = [f"e-{i}" for i in range(20) if _held(a, f"e-{i}")]
    a.shutdown()
    return survivors


def test_a_looked_at_entry_ranks_by_its_mtime_like_the_rest(tmp_path):
    """Break caught: the test above, made deterministic. It failed about 1
    run in 15 with the natural timing.

    An entry's header ``last_access`` is ``time.time()`` taken just before
    its write. Its mtime is the filesystem's clock, which on Linux steps in
    4 ms ticks behind that. So the stamp can be later than the mtimes of
    entries written a moment after it. B ranked the entries whose metadata
    it had looked at by that stamp and the rest by mtime. Here every
    survivor's mtime is below the oldest's stamp but still in write order,
    as one tick leaves them.
    """
    cache = tmp_path / "c"
    survivors = _twenty_through_the_cap(cache)
    oldest = survivors[0]

    b = _backend(cache)
    stamp_ns = int(b.get_metadata(oldest)["last_access"] * 1e9)
    for n, key in enumerate(survivors):
        mtime_ns = stamp_ns - 4_000_000 + n * 100_000
        os.utime(b._get_path(key), ns=(mtime_ns, mtime_ns))
    for i in range(3):
        _put(b, f"later-{i}", MB, 1.0)

    assert not _held(b, oldest)
    b.shutdown()


def test_a_burst_sharing_one_mtime_goes_in_write_order_after_a_restart(tmp_path):
    """Break caught: equal entries with equal mtimes go in ``scandir`` order.

    A burst of writes can share one mtime, and then only the write order
    tells them apart. The writing process knows it, but a new one does not.
    B evicted such a burst in hash order, here the oldest entry and then the
    sixth. With everything equal, the two that go must be the two oldest.
    """
    cache = tmp_path / "c"
    survivors = _twenty_through_the_cap(cache)

    b = _backend(cache)
    shared_ns = os.stat(b._get_path(survivors[-1])).st_mtime_ns
    for key in survivors:
        os.utime(b._get_path(key), ns=(shared_ns, shared_ns))
    for i in range(2):
        _put(b, f"later-{i}", MB, 1.0)

    assert [k for k in survivors if not _held(b, k)] == survivors[:2]
    b.shutdown()


def test_a_valuable_entry_nobody_reads_is_not_immortal(tmp_path):
    """Break caught: the clock never advances, so priorities stop aging.

    'valuable' saves twice as much per byte as the stream behind it and is
    never read; once evictions raise L past its priority, it goes.

    Control arm: plain LRU passes this too.
    """
    b = _backend(tmp_path / "c")
    _put(b, "valuable", MB, 2.0)
    for i in range(40):
        _put(b, f"stream-{i}", MB, 1.0)

    assert not _held(b, "valuable")
    assert _held(b, "stream-39")
    b.shutdown()


def test_the_clock_survives_a_restart(tmp_path):
    """Break caught: a new process starts its clock at zero.

    After process A has evicted a long stream, its surviving entries carry
    priorities well above zero. If B restarted the clock at zero, every
    entry B writes would rank below all of A's and go first, however recent.
    B writes one entry worth as much per byte as A's, and it must not be
    the first victim.
    """
    cache = tmp_path / "c"
    a = _backend(cache)
    for i in range(40):
        _put(a, f"a-{i}", MB, 1.0)
    a.shutdown()

    b = _backend(cache)
    _put(b, "b-new", MB, 1.0)
    for i in range(2):
        _put(b, f"b-more-{i}", MB, 1.0)

    assert _held(b, "b-new")
    b.shutdown()


def test_a_damaged_index_is_ignored_not_fatal(tmp_path):
    """Break caught: a malformed index line raises out of eviction.

    The index is advisory and shared between processes that may be killed
    mid-append. A garbage line must be skipped; eviction carries on.
    """
    cache = tmp_path / "c"
    b = _backend(cache)
    _put(b, "first", MB, 1.0)
    with open(cache / INDEX, "a", encoding="utf-8") as fh:
        fh.write("not a record\n\x00\x01garbage 1.2.3\n@L nan-ish\n")
    for i in range(12):
        _put(b, f"k-{i}", MB, 1.0)

    assert b._current_size_bytes <= CAP
    assert _held(b, "k-11")
    b.shutdown()


def test_the_index_does_not_grow_without_bound(tmp_path):
    """Break caught: the index is only ever appended to.

    Rewriting the same 12 keys 25 times over appends 300 records for 12
    live entries (fewer, after evictions). Compaction at ranking time must
    keep the file proportional to the directory, not to its history.
    """
    cache = tmp_path / "c"
    b = _backend(cache, cap=12 * 100_000)
    for rnd in range(25):
        for i in range(12):
            _put(b, f"k-{i}", 100_000 + rnd, 1.0)
    b._rebuild_evict_queue()

    with open(cache / INDEX, encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) <= 2 * 12 + 64, f"{len(lines)} index lines for 12 entries"
    b.shutdown()


def test_clear_removes_the_index(tmp_path):
    """Break caught: a stale index outlives the entries it describes --
    written or still buffered."""
    cache = tmp_path / "c"
    b = _backend(cache)
    _put(b, "k", MB, 1.0)
    b._rank_index.flush()
    assert (cache / INDEX).exists(), "precondition: a write records a rank"
    _put(b, "buffered", MB, 1.0)  # a record still in the buffer
    b.clear()
    b.shutdown()  # shutdown flushes the buffer

    assert not (cache / INDEX).exists()
