"""A cache's writer thread ends once it has nothing to write.

Each backend's queue kept its writer thread until someone shut the backend
down, which nothing does for a cache dropped mid-session: one thread per
cache, for the rest of the process. A test worker on macOS that had made
a few thousand of them could not start another ("can't start new
thread"), and the next test needing one failed. An idle writer now ends,
and the next write starts another.
"""

from __future__ import annotations

import threading
import time

from cash.backends import _writes
from cash.backends._writes import PendingWrites


def _writers(pw: PendingWrites) -> list[threading.Thread]:
    return [t for t in pw._executor._threads if t.is_alive()]


def _until(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_an_idle_writer_ends_and_the_next_write_starts_one(monkeypatch):
    monkeypatch.setattr(_writes, "WRITER_IDLE_S", 0.05)
    pw = PendingWrites()
    try:
        assert pw.submit("a", lambda: 1).result(timeout=5) == 1
        assert _until(lambda: not _writers(pw)), "the idle writer thread is still running"
        assert pw.submit("b", lambda: 2).result(timeout=5) == 2
    finally:
        pw.shutdown(wait=True)


def test_a_writer_with_work_queued_does_not_end(monkeypatch):
    monkeypatch.setattr(_writes, "WRITER_IDLE_S", 0.05)
    pw = PendingWrites()
    try:
        gate = threading.Event()
        first = pw.submit("a", gate.wait, 5)
        rest = [pw.submit(f"k{i}", lambda i=i: i) for i in range(20)]
        time.sleep(0.2)  # past the idle time while the first write blocks
        gate.set()
        assert first.result(timeout=5) is True
        assert [f.result(timeout=5) for f in rest] == list(range(20))
    finally:
        pw.shutdown(wait=True)
