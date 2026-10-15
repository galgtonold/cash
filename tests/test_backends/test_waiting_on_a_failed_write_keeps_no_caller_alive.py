"""Waiting on a failed write that stays queued must keep nothing of the waiter.

A failed write stays in ``_pending`` until a read of its key drops it. Waiting
on it with ``future.result()`` raised its exception again, and every raise adds
the waiting frames to that exception's traceback, so the queue kept the
waiter's frames, and every local in them, alive. A child process starting
(``_finish_before_child``) waits for every queued write: a failed write left
by an earlier S3 test kept the frames of a ``ProcessPoolExecutor.submit``
alive, with the task's future and the value it returned.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from cash.backends._writes import PendingWrites

pytestmark = pytest.mark.expects_failed_writes


class Held:
    pass


def _fail() -> None:
    raise RuntimeError("bucket gone")


def _wait_holding(pw: PendingWrites) -> weakref.ref:
    held = Held()  # a local of the waiting frame
    pw.wait_all()
    return weakref.ref(held)


def test_a_waiter_s_locals_die_with_it():
    pw = PendingWrites()
    gc.collect()
    gc.disable()
    try:
        failed = pw.submit("k", _fail)
        failed.exception()  # it finished, and failed
        ref = _wait_holding(pw)
        assert ref() is None, "the failed write's traceback keeps the waiter's frame"
    finally:
        gc.enable()
        pw.shutdown(wait=True)
