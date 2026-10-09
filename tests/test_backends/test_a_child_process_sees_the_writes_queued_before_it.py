"""A process this one starts reads every cache write queued before it started.

The file tier writes in the background, and nothing else waits for a write
until something reads it. A child process may read the cache folder itself:
a script run with ``subprocess``, a ``!python`` line in a notebook. So the
writes already queued land before the child starts. A write that never
finishes (a stalled cache folder) holds the child up only for so long; then
it starts and reads a miss for that entry.
"""

from __future__ import annotations

import threading
import time

from cash.backends import _writes
from cash.backends.file_backend import FileBackend
from tests._scripts import run_python

_READ = "import sys\nfrom cash import FileBackend\nprint(FileBackend(sys.argv[1], flush_interval=0).get('k')[1])\n"


def _slowed(backend, hold):
    real = backend._write_cache_files

    def slow(*args):
        hold()
        real(*args)

    backend._write_cache_files = slow


def test_a_child_reads_what_the_parent_queued_before_starting_it(tmp_path):
    backend = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    _slowed(backend, lambda: time.sleep(4.0))
    backend.set("k", "stored")
    assert backend._writes.pending_count() == 1

    child = run_python("-c", _READ, tmp_path / "cache", cwd=tmp_path)
    assert child.stdout.strip() == "stored"
    backend.shutdown()


def test_a_stalled_write_holds_a_child_up_only_so_long(tmp_path, monkeypatch):
    monkeypatch.setattr(_writes, "CHILD_WAIT_S", 0.5)
    release = threading.Event()
    backend = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    _slowed(backend, lambda: release.wait(60))
    backend.set("k", "stored")
    try:
        t0 = time.monotonic()
        first = run_python("-c", _READ, tmp_path / "cache", cwd=tmp_path)
        second = run_python("-c", _READ, tmp_path / "cache", cwd=tmp_path)
        assert first.stdout.strip() == second.stdout.strip() == "None"  # a miss
        assert time.monotonic() - t0 < 30
    finally:
        release.set()
        backend.shutdown()
