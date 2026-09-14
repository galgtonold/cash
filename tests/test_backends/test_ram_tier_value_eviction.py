"""The RAM tier's byte cap evicts by value per byte (GreedyDual-Size-Frequency).

Ranking by recency alone treats a 30-second result the same as a 50 ms one of
the same size, and a 4 MB value the same as a 1 MB one. GDSF keeps
``H = L + hits * execution_time / size`` per entry, evicts the lowest H, and
raises the clock L to each victim's H, so an entry that stops being read
eventually ages out however valuable it was. A trace-driven simulation of
cash's eviction chose it; the notebook-level case is
tests/test_notebook_integration/test_ram_tier_keeps_loop_entries.py.

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
