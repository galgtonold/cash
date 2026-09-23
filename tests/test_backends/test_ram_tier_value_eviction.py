"""The RAM tier's byte cap evicts by value per byte (GreedyDual-Size-Frequency).

Ranking by recency alone treats a 30-second result the same as a 50 ms one of
the same size, and a 4 MB value the same as a 1 MB one. GDSF keeps
``H = L + hits * execution_time / size`` per entry, evicts the lowest H, and
raises the clock L to each victim's H, so an entry that stops being read
eventually ages out however valuable it was. See benchmarks/eviction_sim for
the trace simulation behind the choice; the notebook-level case is
tests/test_notebook_integration/storage/test_ram_tier_keeps_loop_entries.py.

Every test uses a 10 MB cap and ~1 MB values, so the arithmetic in each
docstring can be checked by hand.
"""

from __future__ import annotations

from cash.backends.memory_backend import InMemoryBackend

MB = 1_000_000
CAP = 10 * MB


def _capped():
    """A byte-capped tier whose psutil pressure check can never fire.

    That check reads HOST memory, so on a busy machine (a parallel test run
    pushes it past 90%) it evicts on its own schedule and these tests would
    measure it instead of the byte cap.
    """
    return InMemoryBackend(max_size_bytes=CAP, max_memory_percent=1.0)


def _put(backend, key, size, seconds):
    backend.set(key, bytes(size), {"execution_time": seconds})


def _held(backend, key):
    return backend.peek_metadata(key) is not None


def test_a_costly_result_outlives_cheaper_newer_ones():
    """Break caught: the ranking ignores execution time (plain LRU).

    'costly' is the OLDEST entry, 1 MB that took 30 s. Ten 1 MB entries that
    took 50 ms each push the tier over 10 MB, and about 1 MB must go. By
    value per byte the victims are cheap ones; by recency it is 'costly'.
    """
    b = _capped()
    _put(b, "costly", MB, 30.0)
    for i in range(10):
        _put(b, f"cheap-{i}", MB, 0.05)

    assert _held(b, "costly")
    assert not _held(b, "cheap-0"), "the oldest cheap entry should be the first victim"


def test_value_is_per_byte_so_a_big_value_goes_before_small_ones():
    """Break caught: the ranking ignores size (cost or recency alone).

    Six 1 MB entries and then one 4 MB entry, all 1 s to compute: 10 MB plus
    overhead, over the cap. Per byte the 4 MB entry saves a quarter of what
    each small one does, so it is evicted, although it is the newest.
    """
    b = _capped()
    for i in range(6):
        _put(b, f"small-{i}", MB, 1.0)
    _put(b, "big", 4 * MB, 1.0)

    assert not _held(b, "big")
    assert all(_held(b, f"small-{i}") for i in range(6))


def test_a_read_entry_outranks_an_unread_twin():
    """Break caught: a read leaves no mark on the ranking (neither its hit
    count nor its recency is recorded).

    'read' and 'unread' are the two oldest of nine 1 MB entries; 'read' is
    then read. The tenth entry puts the tier over 10 MB and two entries must
    go (the eviction target is 90% of the cap). They are 'unread' and the
    oldest filler, never 'read'.

    Control arm: plain LRU passes this too, because the read also makes
    'read' the most recent. It pins the property through the change.
    """
    b = _capped()
    _put(b, "read", MB, 1.0)
    _put(b, "unread", MB, 1.0)
    for i in range(7):
        _put(b, f"filler-{i}", MB, 1.0)
    b.get("read")
    _put(b, "last", MB, 1.0)

    assert _held(b, "read")
    assert not _held(b, "unread")


def test_a_valuable_entry_nobody_reads_is_not_immortal():
    """Break caught: the clock L never advances, so priorities stop aging.

    'valuable' saves twice as much per byte as the stream behind it and is
    never read. After evictions raise L past its priority, later entries
    outrank it, and it goes. Without aging it would survive forever,
    holding its slot however long ago it was last useful.

    Control arm: plain LRU passes this too (it evicts 'valuable' first).
    """
    b = _capped()
    _put(b, "valuable", MB, 2.0)
    for i in range(40):
        _put(b, f"stream-{i}", MB, 1.0)

    assert not _held(b, "valuable")
    assert _held(b, "stream-39")


def test_pressure_eviction_ranks_an_unread_costly_result_above_a_cheap_read_one(monkeypatch):
    """Break caught: pressure eviction scores by ``exec * access_count / size``,
    so every entry not yet read scores zero, whatever it cost to compute.

    Host memory reads 95% once, then 50% after the first drop, so exactly
    one entry goes. 'costly' (30 s, never read) must outrank 'cheap' (1 ms,
    read once), since a hit on it saves 30,000 times as much per byte.
    """
    from types import SimpleNamespace

    from cash.backends import memory_backend

    readings = iter([95.0])
    fake = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=next(readings, 50.0)))
    b = InMemoryBackend(check_interval=3)
    monkeypatch.setattr(memory_backend, "psutil", fake)
    _put(b, "costly", MB, 30.0)
    _put(b, "cheap", MB, 0.001)
    b.get("cheap")
    _put(b, "trigger", 1_000, 0.5)  # third write: the pressure check runs

    assert _held(b, "costly")
    assert not _held(b, "cheap")


# ---------------------------------------------------------------------------
# A value the cap can never hold is refused, not allowed to empty the tier.
# ---------------------------------------------------------------------------


def _hot_tier():
    """Twenty 1 MB entries, each read once, under a 100 MB cap."""
    b = InMemoryBackend(max_size_bytes=100 * MB, max_memory_percent=1.0)
    for i in range(20):
        _put(b, f"small-{i}", MB, 5.0)
        b.get(f"small-{i}")
    return b


def test_a_value_over_the_eviction_target_is_refused_not_stored():
    """Break caught: an oversized write is accepted, then evicts everything
    older and itself.

    The byte cap evicts down to 90% of the cap, so a value above that can
    never stay. It used to be stored anyway: the eviction took all twenty
    hot entries and then the new value, leaving an empty tier.
    """
    b = _hot_tier()
    _put(b, "huge", 95 * MB, 1000.0)

    assert not _held(b, "huge")
    assert all(_held(b, f"small-{i}") for i in range(20))
    assert b._current_size_bytes < 25 * MB


def test_a_big_value_under_the_target_is_still_stored():
    """Control arm: the refusal is for values the cap cannot hold, not for
    big ones. 85 MB fits under the 90 MB target, and at 1000 s it is worth
    more per byte than the 1 MB entries, so they make room for it."""
    b = _hot_tier()
    _put(b, "big", 85 * MB, 1000.0)

    assert _held(b, "big")
    assert b._current_size_bytes <= 100 * MB


def test_refusing_a_replacement_drops_the_old_value_for_that_key():
    """Break caught: the refused value leaves the key's previous value in
    place, where a later read would return it as current."""
    b = _hot_tier()
    _put(b, "k", MB, 1.0)
    _put(b, "k", 95 * MB, 1000.0)

    assert not _held(b, "k")
    assert sum(_held(b, f"small-{i}") for i in range(20)) == 20


def test_restoring_a_big_disk_entry_does_not_empty_the_ram_tier(tmp_path):
    """Break caught: read-repair promotes a disk hit into RAM with no size
    gate, so every restore of a value over the RAM cap flushed the tier.

    Disk holds a 12 MB value; the RAM tier (10 MB cap) holds six hot 1 MB
    entries. Reading the big value serves it from disk and leaves RAM as
    it was.
    """
    from cash.backends import FileBackend, TieredBackend

    ram = InMemoryBackend(max_size_bytes=CAP, max_memory_percent=1.0)
    disk = FileBackend(str(tmp_path / "d"), flush_interval=0)
    tiered = TieredBackend([ram, disk], promotion_policy=lambda e, s: True)
    disk.set("huge", b"x" * (12 * MB), {"execution_time": 60.0})
    disk._writes.wait_all()
    for i in range(6):
        _put(ram, f"hot-{i}", MB, 5.0)

    meta, value = tiered.get("huge")

    assert value is not None and len(value) == 12 * MB
    assert all(_held(ram, f"hot-{i}") for i in range(6))
    assert not _held(ram, "huge")
    disk.shutdown()


def test_a_refused_value_is_not_reported_as_held_in_ram(tmp_path):
    """Break caught: the tiered write lists RAM among a value's destinations
    even when RAM refused it -- the badge and the miss explanation read that
    list."""
    from cash.backends import FileBackend, TieredBackend

    ram = InMemoryBackend(max_size_bytes=CAP, max_memory_percent=1.0)
    disk = FileBackend(str(tmp_path / "d"), flush_interval=0)
    tiered = TieredBackend([ram, disk], promotion_policy=lambda e, s: True)
    meta = {"execution_time": 60.0}
    tiered.set("huge", b"x" * (12 * MB), meta)
    disk._writes.wait_all()

    assert meta["storage"] == ["DISK"]
    disk.shutdown()


def test_an_entry_without_execution_time_is_still_evictable():
    """Break caught: a missing execution_time crashes the ranking or pins
    the entry. Raw backend users (and metadata-less writes) never set it.
    """
    b = _capped()
    for i in range(12):
        b.set(f"bare-{i}", bytes(MB))

    assert not _held(b, "bare-0")
    assert _held(b, "bare-11")
    assert b._current_size_bytes <= CAP
