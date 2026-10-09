"""A big file a cached body reads is hashed beside the body, not before it.

The digest a stored entry records is the file as the body first opened it.
It was taken inside the ``open`` call, so the body waited for every byte to
be hashed before it read the first: 3.6 s against 0.5 s plain for a 1 GB
file. It is now taken on a thread of its own and waited for when the entry
is stored -- and before anything in the process opens the file to write, so
the digest is still the file the body read.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from cash.tracking import file_dep_snapshot, file_tracker
from cash.tracking.file_tracker import FileAccessTracker

BIG = 6 * 1024 * 1024


def _settled(path, body: bytes) -> str:
    path.write_bytes(body)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))
    return str(path)


@pytest.fixture(autouse=True)
def _nothing_remembered(monkeypatch):
    file_dep_snapshot._HASH_MEMO.clear()
    monkeypatch.setattr(file_dep_snapshot, "_HASH_MEMO_TTL_SECONDS", 0.0)
    yield
    file_dep_snapshot._HASH_MEMO.clear()


def _slow_digest(monkeypatch, seconds: float, threads: list[str]):
    real = file_tracker.file_content_hash

    def slow(path, *args, **kwargs):
        threads.append(threading.current_thread().name)
        time.sleep(seconds)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(file_tracker, "file_content_hash", slow)


def test_the_open_does_not_wait_for_the_digest(tmp_path, monkeypatch):
    path = _settled(tmp_path / "big.bin", b"x" * BIG)
    threads: list[str] = []
    _slow_digest(monkeypatch, 1.0, threads)
    with FileAccessTracker(hash_on_read=True) as tracker:
        started = time.perf_counter()
        with open(path, "rb") as fh:
            opened = time.perf_counter() - started
            fh.read(10)
    assert threads and threads[0] != threading.current_thread().name, threads
    assert opened < 0.8, f"the open waited {opened:.2f} s for the digest"
    expected = file_dep_snapshot.file_content_hash(path)
    assert tracker.read_digests[os.path.realpath(path)] == expected


def test_a_write_waits_for_the_digest_of_the_file_as_read(tmp_path, monkeypatch):
    """An ``np.memmap``-style write: same size, then the mtime put back. The
    entry must record the file the body READ, or the next call over the
    written file would be served a result computed from the old bytes."""
    path = _settled(tmp_path / "big.bin", b"x" * BIG)
    original = file_dep_snapshot.file_content_hash(path)
    threads: list[str] = []
    _slow_digest(monkeypatch, 0.5, threads)
    with FileAccessTracker(hash_on_read=True) as tracker:
        with open(path, "rb") as fh:
            fh.read(10)
        before = os.stat(path)
        with open(path, "r+b") as fh:
            fh.write(b"Z")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert tracker.read_digests[os.path.realpath(path)] == original, "the digest saw the write"


def test_a_small_file_is_hashed_at_the_open(tmp_path, monkeypatch):
    path = _settled(tmp_path / "small.bin", b"x" * 1000)
    threads: list[str] = []
    _slow_digest(monkeypatch, 0.0, threads)
    with FileAccessTracker(hash_on_read=True) as tracker:
        with open(path, "rb") as fh:
            fh.read()
    assert threads == [threading.current_thread().name]
    assert tracker.read_digests[os.path.realpath(path)]


def test_a_cached_body_over_a_big_file_hits_and_sees_an_edit(disk_cash, tmp_path):
    path = _settled(tmp_path / "big.bin", b"x" * BIG)
    runs: list[int] = []

    @disk_cash.cache(assume_safe=True)
    def load(p):
        runs.append(1)
        with open(p, "rb") as fh:
            return fh.read(3)

    assert load(path) == b"xxx"
    assert load(path) == b"xxx"
    assert len(runs) == 1
    _settled(tmp_path / "big.bin", b"y" * BIG)
    assert load(path) == b"yyy"
    assert len(runs) == 2
