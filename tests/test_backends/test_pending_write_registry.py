"""A finished write must not stay in ``_pending`` forever.

``PendingWrites`` kept one future per distinct key for the life of the
backend: after 20,000 writes it held 20,000 futures with zero in flight.
``wait_all`` walks that dict, so it was O(every key ever written) rather than
O(in flight) -- profiling 20 writes against a 20k-entry cache showed 400,210
calls to ``Future.result``, about 20,010 per write. Removing the retention
took one write from 7.05ms to 2.51ms, against a measured raw-filesystem floor
of ~2.1ms for the same two-file write.

``wait`` already dropped FAILED futures for exactly this reason and its
docstring states the principle. The successful case never got the same
treatment.

Failures must still be retained: ``wait(key)`` reports one the next time that
key is looked up, and ``failed_writes()`` is the only record that a write was
discarded. ``test_a_failed_write_is_still_remembered`` is the arm that stops a
future cleanup from restoring the older bug where a failed write vanished
without a trace.
"""
from __future__ import annotations

import time

import pytest

from cash.backends._base import PendingWrites


@pytest.fixture
def writes():
    pending = PendingWrites()
    yield pending
    pending.shutdown() if hasattr(pending, "shutdown") else None


def test_successful_writes_are_not_retained(writes):
    """The leak itself."""
    for i in range(50):
        writes.submit(f"k{i}", lambda: None)
    writes.wait_all()
    assert len(writes._pending) == 0, (
        f"{len(writes._pending)} finished futures retained; wait_all walks this "
        f"dict, so it becomes O(every key ever written)"
    )


def test_in_flight_writes_are_still_tracked(writes):
    """The control. A cleanup that dropped everything would pass the arm above.

    Without the registry an in-flight write is invisible, and `get` could
    return a clean miss for a key that is mid-write.
    """
    started = []

    def slow():
        started.append(1)
        time.sleep(0.3)

    writes.submit("slow", slow)
    assert len(writes._pending) == 1, "an in-flight write was dropped"
    writes.wait_all()
    assert started == [1]
    assert len(writes._pending) == 0


@pytest.mark.expects_failed_writes
def test_a_failed_write_is_still_remembered(writes):
    """Failures are the reason `_pending` cannot simply be cleared on completion."""
    def boom():
        raise OSError("disk full")

    writes.submit("doomed", boom)
    writes.wait_all()
    assert len(writes._pending) == 1, (
        "the failed future was dropped, so failed_writes() and the warning on "
        "the next lookup of this key both lose their only record"
    )
    assert [k for k, _ in writes.failed_writes()] == ["doomed"]

    # The marker is what keeps this from being reported as a real leak: the
    # suite fails any test that leaves a silently-discarded write behind, which
    # is exactly the hole this registry exists to close.


def test_waiting_on_a_finished_key_is_a_no_op(writes):
    """`wait(key)` must not care that the future has been forgotten."""
    writes.submit("done", lambda: None)
    writes.wait_all()
    writes.wait("done")          # must not raise or block


def test_resubmitting_a_forgotten_key_still_orders_correctly(writes):
    """`submit` waits on a previous in-flight write for the same key.

    Dropping finished futures means that lookup now returns None for a
    completed write -- which is right, there is nothing to wait for, but the
    ordering guarantee has to survive it.
    """
    order = []
    writes.submit("k", lambda: order.append("first"))
    writes.wait_all()
    writes.submit("k", lambda: order.append("second"))
    writes.wait_all()
    assert order == ["first", "second"]


def test_the_registry_does_not_grow_across_many_keys(writes):
    """The memory half: a future and its key stayed reachable per entry."""
    for batch in range(4):
        for i in range(200):
            writes.submit(f"b{batch}_k{i}", lambda: None)
        writes.wait_all()
        assert len(writes._pending) == 0, (
            f"after batch {batch} the registry holds {len(writes._pending)}"
        )
