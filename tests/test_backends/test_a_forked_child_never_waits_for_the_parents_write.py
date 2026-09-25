"""A forked child never blocks on a cache write the parent had in flight.

Fork is the default start method on Linux before Python 3.14. The child
inherits the parent's write queue -- the unfinished future of a write still
running in the parent -- but not the thread that would finish it. Every read
of that key waits for it: a ``multiprocessing`` pool forked while a large
value was still being written hung with every worker blocked.

The write is the parent's to finish. The child reads what is on disk: the
entry, or a miss while it is not there yet.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from cash.backends._writes import PendingWrites
from cash.backends.file_backend import FileBackend

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")


def _in_child(body) -> int:
    """Run *body* in a forked child; its exit code, or -1 if it hung."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child
        code = 1
        try:
            code = 0 if body() else 3
        except BaseException:
            code = 2
        finally:
            os._exit(code)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.02)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return -1


@pytest.fixture
def held_write():
    """A write queue whose write of "k" does not finish until released."""
    release = threading.Event()
    started = threading.Event()

    def slow_write():
        started.set()
        release.wait(30)

    queue = PendingWrites()
    queue.submit("k", slow_write)
    assert started.wait(5)
    yield queue
    release.set()
    queue.shutdown(wait=True, timeout=5)


def test_a_child_does_not_wait_for_the_parents_write(held_write):
    def body():
        held_write.wait("k")
        held_write.submit("k", lambda: None).result(timeout=5)
        held_write.wait_all()
        held_write.drain("k")
        return True

    assert _in_child(body) == 0


def test_a_child_reads_a_file_entry_the_parent_is_still_writing(tmp_path):
    backend = FileBackend(str(tmp_path), flush_interval=0)
    backend.set("done", "stored")
    backend._writes.wait_all()

    release = threading.Event()
    writing = threading.Event()
    real_write = backend._write_cache_files

    def slow_write(*args):
        writing.set()
        release.wait(30)
        real_write(*args)

    backend._write_cache_files = slow_write
    try:
        backend.set("k", "value")
        assert writing.wait(5)

        def body():
            backend._write_cache_files = real_write
            assert backend.get("k") == (None, None)  # not on disk yet: a miss
            assert backend.get("done")[1] == "stored"
            backend.set("k", "from the child")
            assert backend.get("k")[1] == "from the child"
            backend.list_entries()
            return True

        assert _in_child(body) == 0
    finally:
        release.set()
        backend.shutdown()
