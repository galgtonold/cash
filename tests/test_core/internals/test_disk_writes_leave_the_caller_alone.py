"""How the file tier's background writes divide the work with the caller.

* A value cash owns (the RAM tier's own copy, ``set(..., private=True)``) is
  serialized by the writer, from its own buffers: the caller neither pickles
  nor copies it.
* A caller's value is still copied before ``set`` returns, so a later change
  to it cannot reach the entry.
* Queued writes hold copies of values, so a run of large stores waits for
  older writes once more than ``MAX_QUEUED_BYTES`` would be held.
* A notebook cell ends with at most ``MAX_BACKLOG_S`` of writing left, by
  what the queue's writes took so far.
"""

from __future__ import annotations

import threading
import time
import tracemalloc
import zlib

import pytest

np = pytest.importorskip("numpy")

from cash.backends import FileBackend, InMemoryBackend, _writes, serialization
from cash.backends.tiered_backend import TieredBackend

N = 1_000_000  # 8 MB of float64


@pytest.fixture
def splits(monkeypatch):
    """Every ``serialize_split``: (thread name, copy=) per call."""
    calls = []
    real = serialization.PickleSerializer.serialize_split

    def spy(self, data, *, copy=True):
        calls.append((threading.current_thread().name, copy))
        return real(self, data, copy=copy)

    monkeypatch.setattr(serialization.PickleSerializer, "serialize_split", spy)
    return calls


def test_a_private_value_is_serialized_by_the_writer_without_a_copy(tmp_path, splits):
    backend = FileBackend(str(tmp_path), flush_interval=0)
    arr = np.arange(N, dtype=np.float64)
    backend.get_metadata("warm")  # the directory and the stamp, outside the measurement
    tracemalloc.start()
    try:
        backend.set("k", arr, {"size": arr.nbytes}, private=True)
        _current, caller_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    backend._writes.wait_all()
    assert splits == [("cash-cache-writer", False)]
    assert caller_peak < arr.nbytes / 8, f"set() of a private {arr.nbytes}-byte array allocated {caller_peak}"
    _meta, got = backend.get("k")
    np.testing.assert_array_equal(got, arr)
    backend.shutdown()


def test_a_callers_value_is_copied_before_set_returns(tmp_path, splits):
    backend = FileBackend(str(tmp_path), flush_interval=0)
    arr = np.arange(N, dtype=np.float64)
    backend.set("k", arr)
    arr[:] = -1.0  # changed after set: the entry holds what was set
    backend._writes.wait_all()
    assert splits == [(threading.main_thread().name, True)]
    assert backend.get("k")[1][5] == 5.0
    backend.shutdown()


def test_persisting_a_ram_entry_hands_the_disk_tier_the_ram_copy(tmp_path, splits):
    disk = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    tiered = TieredBackend([InMemoryBackend(), disk])
    meta = {
        "execution_time": 0.02,
        "cost_model_family": "_GENERIC",
        "cost_model_type_name": "ndarray",
        "cost_model_size_bytes": 8 * N,
    }
    tiered.set("k", np.arange(N, dtype=np.float64), meta)
    assert tiered.persist_from_memory("k", rebuild_seconds=60.0) is True
    disk._writes.wait_all()
    assert splits == [("cash-cache-writer", False)]
    np.testing.assert_array_equal(disk.get("k")[1], np.arange(N, dtype=np.float64))
    disk.shutdown()


def test_queued_writes_hold_at_most_max_queued_bytes(monkeypatch):
    monkeypatch.setattr(_writes, "MAX_QUEUED_BYTES", 100)
    queue = _writes.PendingWrites()
    gate = threading.Event()
    queue.submit_sized("a", 80, True, gate.wait, 10)

    accepted = threading.Event()
    second = threading.Thread(target=lambda: (queue.submit_sized("b", 80, True, lambda: None), accepted.set()))
    second.start()
    assert not accepted.wait(0.3), "a second 80-byte write was queued behind an unfinished one over a 100-byte bound"
    gate.set()
    assert accepted.wait(5)
    second.join(5)

    # One write is always taken, whatever its size.
    t0 = time.monotonic()
    queue.submit_sized("c", 10_000, True, lambda: None).result(5)
    assert time.monotonic() - t0 < 5
    queue.shutdown(wait=True, timeout=5)


def test_wait_for_backlog_leaves_about_that_much_writing(monkeypatch):
    queue = _writes.PendingWrites()
    for i in range(10):
        queue.submit_sized(f"k{i}", 100, True, time.sleep, 0.2)
    t0 = time.monotonic()
    assert queue.wait_for_backlog(0.5)
    waited = time.monotonic() - t0
    left = queue.pending_count()
    assert 1 <= left <= 4, f"{left} writes of 0.2 s left after waiting down to 0.5 s"
    assert waited >= 1.0, f"waited only {waited:.2f}s for 2 s of writes"
    assert queue.wait_for_backlog(0.0)
    assert queue.pending_count() == 0
    queue.shutdown(wait=True, timeout=5)


def test_a_private_write_counts_toward_the_backlog_not_the_memory_bound(monkeypatch):
    monkeypatch.setattr(_writes, "MAX_QUEUED_BYTES", 100)
    queue = _writes.PendingWrites()
    gate = threading.Event()
    queue.submit_sized("a", 10_000, False, gate.wait, 10)
    queue.submit_sized("b", 80, True, lambda: None)  # not held up by "a"
    assert queue.backlog_seconds() > 0
    gate.set()
    assert queue.wait_for_backlog(0.0)
    queue.shutdown(wait=True, timeout=5)
