"""A file whose size and timestamps did not move since it was hashed is not read again.

Every freshness check hashed its file's content, so an unchanged input was
read in full again on every cell run and after every restart: before each
cell of round 23's r23s2, 1,312 exports, about 5 s. Its metadata now answers
for it -- size, modification time, which file it is, and on Linux and macOS the
inode change time -- provided it had been left alone for a while before it was
hashed (a file written moments before can be written again within one
timestamp tick). The content still decides whenever the metadata moved.

What this gives up, on Windows, is an edit that keeps the size and puts the
modification time back, and a write that moves no timestamp (np.memmap): see
known-limitations.
"""
from __future__ import annotations

import os

import pytest

from cash.notebook import file_dep_snapshot
from cash.notebook.file_dep_snapshot import file_dep_is_fresh, snapshot_file_deps


@pytest.fixture(autouse=True)
def _no_digest_reuse(monkeypatch):
    """No in-process memo: what is under test is whether the file is read at all."""
    monkeypatch.setattr(file_dep_snapshot, "_HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_HASH_MEMO_TTL_SECONDS", 0.0)
    file_dep_snapshot._HASH_MEMO.clear()


@pytest.fixture
def reads(monkeypatch):
    real = file_dep_snapshot.file_content_hash
    seen: list[str] = []

    def counting(path, *a, **k):
        seen.append(path)
        return real(path, *a, **k)
    monkeypatch.setattr(file_dep_snapshot, "file_content_hash", counting)
    return seen


def _settled(tmp_path, text="a,b\n1,2\n"):
    p = tmp_path / "input.csv"
    p.write_text(text)
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))
    return str(p)


def test_an_unchanged_settled_file_is_not_read(tmp_path, reads):
    path = _settled(tmp_path)
    stored = snapshot_file_deps({path})[path]
    reads.clear()

    assert file_dep_is_fresh(path, stored) == (True, None)
    assert reads == [], "the file was read again though nothing about it moved"


def test_a_touch_is_read_and_stays_fresh(tmp_path, reads):
    """The metadata moved, so the content decides -- and it did not change."""
    path = _settled(tmp_path)
    stored = snapshot_file_deps({path})[path]
    os.utime(path)
    reads.clear()

    assert file_dep_is_fresh(path, stored) == (True, None)
    assert reads == [path]


def test_an_edit_is_caught(tmp_path):
    path = _settled(tmp_path)
    stored = snapshot_file_deps({path})[path]
    with open(path, "w") as fh:
        fh.write("a,b\n9,9\n")                     # same size, a new mtime

    assert file_dep_is_fresh(path, stored)[0] is False


def test_a_file_hashed_while_young_is_read(tmp_path, reads):
    """Written moments before its digest was taken, it could be written again
    within the same timestamp tick: its metadata does not answer for it."""
    p = tmp_path / "fresh.csv"
    p.write_text("a,b\n1,2\n")
    stored = snapshot_file_deps({str(p)})[str(p)]
    reads.clear()

    assert file_dep_is_fresh(str(p), stored) == (True, None)
    assert reads == [str(p)]


def test_a_snapshot_from_before_this_is_read(tmp_path, reads):
    path = _settled(tmp_path)
    stored = snapshot_file_deps({path})[path]
    stored.pop("hashed_at", None)
    reads.clear()

    assert file_dep_is_fresh(path, stored) == (True, None)
    assert reads == [path]


def test_another_file_under_the_same_path_is_read(tmp_path, reads):
    """Same size, same timestamps, different file (a re-pointed junction, a
    release copy): which file it is has to match too."""
    path = _settled(tmp_path)
    stored = snapshot_file_deps({path})[path]
    before = os.stat(path)
    other = tmp_path / "other.csv"
    other.write_text("a,b\n7,7\n")
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(other, path)
    reads.clear()

    assert file_dep_is_fresh(path, stored)[0] is False
    assert reads == [path]
