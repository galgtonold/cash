"""Many distinct calls with small results do not grow the process without
a bound.

The RAM tier's byte limit counted the stored values alone: 28 bytes for a
small int, where the key, the metadata and the tier's own rows for the
entry take about 1,100. The limit of a fifth of the machine's memory was
then never reached. And the disk tier kept the metadata of every key the
process had written or read, 800 bytes a key, with no limit at all.
"""

from __future__ import annotations

import pytest

from cash.backends import memory_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.touched_entries import TouchedEntries

CAP = 1 << 20


def test_the_ram_tiers_byte_limit_bounds_small_entries_too():
    tier = InMemoryBackend(max_size_bytes=CAP)
    for i in range(5_000):
        tier.set(f"mod.f:state:dyn:{i:064d}", i, {"execution_time": 0.001})
    fit = CAP // memory_backend.ENTRY_OVERHEAD_BYTES
    assert len(tier._store) <= fit, f"{len(tier._store)} small entries are held under a 1 MiB limit"
    # As many as fit are kept: the limit is not reached by a few hundred.
    assert len(tier._store) >= fit // 2
    # The newest are the ones kept.
    assert tier.get(f"mod.f:state:dyn:{4_999:064d}")[1] == 4_999


def test_what_the_ram_tier_counts_goes_back_to_zero():
    tier = InMemoryBackend(max_size_bytes=CAP)
    for i in range(50):
        tier.set(f"k{i}", i, {"execution_time": 0.001})
    tier.set("k0", "another value", {"execution_time": 0.001})  # replaced, counted once
    assert tier._current_size_bytes >= 50 * memory_backend.ENTRY_OVERHEAD_BYTES
    for i in range(50):
        tier.delete(f"k{i}")
    assert tier._current_size_bytes == 0


def test_a_big_value_is_still_counted_by_its_size():
    tier = InMemoryBackend(max_size_bytes=64 << 20)
    tier.set("big", b"x" * (4 << 20), {"execution_time": 1.0})
    assert (4 << 20) <= tier._current_size_bytes <= (4 << 20) + 65_536


def _touched(monkeypatch, most=100):
    monkeypatch.setattr(TouchedEntries, "MAX_ENTRIES", most)
    return TouchedEntries()


def test_only_the_keys_touched_last_are_held(monkeypatch):
    touched = _touched(monkeypatch)
    let_go = []
    touched.on_let_go = let_go.append
    for i in range(1_000):
        touched.remember(f"k{i}", f"/c/{i}.entry", {"n": i}, ("id", i), (i, i, i))
        assert len(touched) <= 100 + 100 // 8 + 1
    assert touched.metadata("k999") == {"n": 999}
    assert touched.key_for("/c/999.entry") == "k999"
    assert touched.metadata("k0") is None
    assert touched.key_for("/c/0.entry") is None
    assert touched.identity("k0") is None
    assert len(touched._keys) == len(touched._origin) == len(touched)
    assert set(let_go) == {f"k{i}" for i in range(1_000)} - set(touched._metadata)


def test_a_key_whose_read_is_not_flushed_is_held_however_old(monkeypatch):
    touched = _touched(monkeypatch)
    for i in range(50):
        touched.remember(f"read{i}", f"/c/r{i}.entry", {"access_count": 1})
        touched.note_read(f"read{i}")
    for i in range(1_000):
        touched.remember(f"k{i}", f"/c/{i}.entry", {})
    assert all(touched.metadata(f"read{i}") == {"access_count": 1} for i in range(50))
    assert sorted(touched.take_unflushed()) == sorted(f"read{i}" for i in range(50))
    for i in range(1_000, 1_200):
        touched.remember(f"k{i}", f"/c/{i}.entry", {})
    assert touched.metadata("read0") is None, "flushed, it goes as any old key does"


def test_a_key_read_again_counts_as_touched_last(monkeypatch):
    touched = _touched(monkeypatch)
    touched.remember("early", "/c/early.entry", {})
    for i in range(1_000):
        touched.remember(f"k{i}", f"/c/{i}.entry", {})
        if i % 50 == 0:
            touched.note_read("early")
            touched.take_unflushed()
    assert touched.metadata("early") == {}


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(TouchedEntries, "MAX_ENTRIES", 200)
    return 200 + 200 // 8 + 1


def test_a_loop_of_distinct_calls_keeps_a_bounded_table_and_every_result(disk_cash, small_window):
    runs = []

    @disk_cash.cache
    def f(x):
        runs.append(x)
        return x + 1

    for i in range(1_000):
        assert f(i) == i + 1
    ram, disk = disk_cash.backend.backends
    assert isinstance(disk, FileBackend)
    disk._writes.wait_all()
    assert len(disk._touched) <= small_window, f"the disk tier holds the metadata of {len(disk._touched)} keys"
    assert len(disk.evictor.write_seq_by_key) <= small_window
    assert len(disk.evictor.base) <= small_window
    # A key the table let go is served from its file all the same, and
    # counted as read there.
    ram.clear()
    assert [f(i) for i in range(0, 1_000, 97)] == [i + 1 for i in range(0, 1_000, 97)]
    assert runs == list(range(1_000))
    disk._flush_metadata()
    key = next(key for key in disk._touched._metadata if disk._touched._metadata[key].get("access_count"))
    assert disk.get_metadata(key)["access_count"] == 1
