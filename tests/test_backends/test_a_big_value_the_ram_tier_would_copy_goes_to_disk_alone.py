"""A big value the RAM tier would copy goes to the disk tier alone.

The RAM tier copied each value it kept, so a caller's changes could not reach
it. For a big one the copy doubled the memory the value took: a retail
pipeline whose tables hold text peaked at 2.9 GB, plain Python at 1.2 GB,
and the fresh memory cost seconds of system time a cell. A value of which
the RAM tier would copy 64 MB or more now goes to disk alone when a disk
tier writes it: its bytes are made before the store returns, and each hit
reads it from disk. A table the RAM tier keeps uncopied stays there.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from cash.backends import memory_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import DISK_ALONE, TieredBackend

#: Rows of 8 bytes: 2 MiB, over the size these tests set.
ROWS = 1 << 18


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    """1 MiB in place of 64 MB, so the values here stay small."""
    monkeypatch.setattr(memory_backend, "DISK_ALONE_MIN_BYTES", 1 << 20)


@pytest.fixture
def copies(monkeypatch):
    """Calls of the RAM tier's two copies of a value: its own, and the one
    for a background write."""
    seen = {"tier": 0, "write": 0}
    real_tier = InMemoryBackend._safe_deep_copy
    real_write = memory_backend._write_copy

    def tier(*a, **k):
        seen["tier"] += 1
        return real_tier(*a, **k)

    def write(*a, **k):
        seen["write"] += 1
        return real_write(*a, **k)

    monkeypatch.setattr(InMemoryBackend, "_safe_deep_copy", staticmethod(tier))
    monkeypatch.setattr(memory_backend, "_write_copy", write)
    return seen


@pytest.fixture
def tiers(tmp_path):
    made = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    yield made
    made.shutdown()


def _store(tiers, value, key="k"):
    metadata = {"execution_time": 50.0}
    tiers.set(key, value, metadata)
    return metadata


def test_it_is_written_to_disk_alone_without_a_copy(tiers, copies):
    metadata = _store(tiers, np.arange(ROWS, dtype=np.float64))
    assert metadata["storage"] == ["DISK"], metadata["storage"]
    assert tiers.backends[0].peek_metadata("k") is None
    assert copies == {"tier": 0, "write": 0}


def test_a_change_after_the_store_does_not_reach_it(tiers):
    value = np.arange(ROWS, dtype=np.float64)
    _store(tiers, value)
    value[0] = -1.0  # the bytes were made before the store returned
    value *= 2
    read = tiers.get("k")[1]
    assert read[0] == 0.0 and read[-1] == ROWS - 1


def test_a_hit_is_a_copy_the_caller_can_change(tiers):
    _store(tiers, {"variables": {"x": np.arange(ROWS, dtype=np.float64)}})
    first = tiers.get("k")[1]["variables"]["x"]
    first[0] = 99.0
    second = tiers.get("k")[1]["variables"]["x"]
    assert second[0] == 0.0
    assert not np.shares_memory(first, second)


def test_hits_read_it_from_disk_and_leave_the_ram_tier_empty(tiers):
    _store(tiers, np.arange(ROWS, dtype=np.float64))
    for _ in range(3):
        metadata, read = tiers.get("k")
        assert metadata["source"] == "DISK" and metadata.get(DISK_ALONE)
        assert read[-1] == ROWS - 1
    assert tiers.backends[0].peek_metadata("k") is None


def test_with_the_callers_value_gone_a_new_session_reads_it_back(tmp_path):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    value = np.arange(ROWS, dtype=np.float64)
    expected = value.copy()
    _store(tiers, value)
    del value
    gc.collect()
    tiers.shutdown()  # the write finished
    later = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    read = later.get("k")[1]
    later.shutdown()
    np.testing.assert_array_equal(read, expected)


def test_a_table_the_ram_tier_keeps_uncopied_stays_there(tiers):
    pd = pytest.importorskip("pandas")
    from cash.backends import frame_sharing

    if not frame_sharing._checked_works():
        pytest.skip("needs pandas copy-on-write")
    rows = frame_sharing.CHECKED_MIN_BYTES // 16 + 1
    raw = pd.DataFrame({"f": np.arange(float(rows)), "n": np.arange(rows)})
    table = raw[raw["f"] >= 0]
    assert frame_sharing.keeps_uncopied(table)
    metadata = _store(tiers, {"variables": {"table": table}})
    assert metadata["storage"][0] == "RAM", metadata["storage"]


def test_a_table_the_ram_tier_would_copy_goes_to_disk_alone(tiers):
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame({"f": np.arange(float(ROWS)), "tags": [("a",)] * ROWS})
    expected = table.copy()
    metadata = _store(tiers, {"variables": {"table": table}})
    assert metadata["storage"] == ["DISK"], metadata["storage"]
    table.iloc[0, 0] = -5.0
    pd.testing.assert_frame_equal(tiers.get("k")[1]["variables"]["table"], expected)


def test_a_value_the_disk_tier_cannot_write_is_kept_in_ram(tiers, monkeypatch):
    def full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tiers.backends[1], "set", full)
    metadata = _store(tiers, np.arange(ROWS, dtype=np.float64))
    assert metadata["storage"] == ["RAM"], metadata["storage"]
    assert not tiers.backends[0].peek_metadata("k").get(DISK_ALONE)
    assert tiers.get("k")[1][-1] == ROWS - 1


def test_a_small_value_goes_to_the_ram_tier_as_before(tiers):
    metadata = _store(tiers, np.arange(1000, dtype=np.float64))
    assert metadata["storage"][0] == "RAM", metadata["storage"]
