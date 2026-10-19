"""A file read in a thread the cached call starts is one of the call's inputs.

A thread starts with an empty context, so the tracker the ContextVar holds
was not there: ``Thread(target=lambda: open(p).read())`` in a cached function
read a file no tracker saw. A module-level target was rescued by the reads
its code is remembered for; a lambda or a nested function was not, and after
the file changed the old result was served with no warning. A thread the
call starts on code of the user's now runs in a copy of the call's context.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cash.tracking.tracker_context import active_tracker

pytestmark = pytest.mark.core


def _edit_and_call_again(load, path):
    first = load(str(path))
    path.write_text("new", encoding="utf-8")
    return first, load(str(path))


def test_a_lambda_target(disk_cash, tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("old", encoding="utf-8")

    @disk_cash.cache(assume_safe=True)
    def load(p):
        out = []
        t = threading.Thread(target=lambda: out.append(open(p, encoding="utf-8").read()))
        t.start()
        t.join()
        return out[0]

    assert _edit_and_call_again(load, path) == ("old", "new")


def test_a_nested_function_target(disk_cash, tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("old", encoding="utf-8")

    @disk_cash.cache(assume_safe=True)
    def load(p):
        out = []

        def read():
            with open(p, encoding="utf-8") as fh:
                out.append(fh.read())

        t = threading.Thread(target=read)
        t.start()
        t.join()
        return out[0]

    assert _edit_and_call_again(load, path) == ("old", "new")


def test_a_thread_subclass_of_the_users(disk_cash, tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("old", encoding="utf-8")

    class Reader(threading.Thread):
        def __init__(self, p):
            super().__init__()
            self.p, self.text = p, None

        def run(self):
            with open(self.p, encoding="utf-8") as fh:
                self.text = fh.read()

    @disk_cash.cache(assume_safe=True)
    def load(p):
        t = Reader(p)
        t.start()
        t.join()
        return t.text

    assert _edit_and_call_again(load, path) == ("old", "new")


def test_an_unchanged_file_still_hits(disk_cash, tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("old", encoding="utf-8")
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(p):
        runs.append(1)
        out = []
        t = threading.Thread(target=lambda: out.append(open(p, encoding="utf-8").read()))
        t.start()
        t.join()
        return out[0]

    assert load(str(path)) == load(str(path)) == "old"
    assert len(runs) == 1


def test_a_pool_worker_the_call_starts_keeps_no_tracker(disk_cash):
    """A library's thread is left as it was: a worker the pool starts during
    the call would otherwise hold the call's tracker for every later task."""
    pool = ThreadPoolExecutor(1)

    @disk_cash.cache(assume_safe=True)
    def warm(n):
        return pool.submit(lambda: n + 1).result()

    try:
        assert warm(1) == 2
        assert pool.submit(active_tracker.get).result() is None
    finally:
        pool.shutdown(wait=True)
