"""Queueing a store costs its caller the same however long the queue is.

Every sized submit listed all unfinished writes under the queue's lock to
total the bytes they hold, so a burst of N small stores made N*N/2 checks,
and the writer thread waits for that lock to finish each write. Through the
queue a store then blocked its caller for longer than writing it directly:
a profile of 1,000 queued stores showed 277,000 ``Future.done`` calls.

The bytes held are a running total now, and the queue is listed only when a
submit really has to wait for room. Counted, not timed.
"""

from __future__ import annotations

import threading

import pytest

from cash.backends import _writes
from cash.backends._writes import PendingWrites
from tests._work_counts import method_calls

BURST = 400


@pytest.fixture
def queue():
    pending = PendingWrites()
    yield pending
    pending.shutdown(wait=True)


def _burst(queue, gate, count=BURST, nbytes=100, copied=True):
    """*count* sized writes queued behind one that waits for *gate*."""
    queue.submit_sized("first", nbytes, copied, gate.wait)
    return [queue.submit_sized(f"k{i}", nbytes, copied, lambda: None) for i in range(count)]


def test_a_burst_of_small_stores_never_lists_the_queue(queue):
    gate = threading.Event()
    with method_calls(PendingWrites, "_unfinished_sized") as listed:
        futures = _burst(queue, gate)
        assert sum(not f.done() for f in futures) == BURST, "precondition: the whole burst was queued at once"
    gate.set()
    queue.wait_all()

    assert listed.calls == 0, f"{listed.calls} listings of the queue for {BURST} submits"


def test_the_running_total_follows_the_queue(queue):
    gate = threading.Event()
    _burst(queue, gate, count=50, nbytes=1000)
    queue.submit_sized("not-a-copy", 5000, False, lambda: None)
    assert queue._copied_bytes == 51 * 1000, "copies are counted, bytes the caller still owns are not"
    gate.set()
    queue.wait_all()
    queue.submit_sized("last", 1, True, lambda: None).result()
    queue.wait_all()
    assert queue._copied_bytes == 0, "a finished write must leave the total"


def test_a_submit_over_the_bound_still_waits_for_room(queue, monkeypatch):
    """The control: the bound the total exists for still holds, oldest first."""
    monkeypatch.setattr(_writes, "MAX_QUEUED_BYTES", 250)
    gate = threading.Event()
    started = []
    queue.submit_sized("a", 100, True, lambda: (started.append("a"), gate.wait()))
    queue.submit_sized("b", 100, True, lambda: started.append("b"))

    release = threading.Timer(0.3, gate.set)
    release.start()
    try:
        queue.submit_sized("c", 100, True, lambda: started.append("c"))  # 300 > 250: waits for "a"
        assert gate.is_set(), "the submit returned while the queue held more than the bound"
    finally:
        release.join()
    queue.wait_all()
    assert started == ["a", "b", "c"]
    assert queue._copied_bytes == 0
