"""A file read inside a tracker is stat'ed once, as its path is resolved.

That ``lstat`` shows a regular file is there (`realpath_of_read_this_run`), so
the open event no longer asks the disk again whether it exists, and the
digest taken as the body reads it uses the same stat. Each stat on Windows
opens a handle to the file.
"""

from __future__ import annotations

import os

from cash._paths import normalize_path
from cash.tracking import file_dep_snapshot
from cash.tracking.file_tracker import FileAccessTracker


def test_a_read_is_stat_ed_once(tmp_path, monkeypatch):
    path = tmp_path / "data.txt"
    path.write_text("some data", encoding="utf-8")
    asked = []
    real_stat, real_lstat = os.stat, os.lstat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: asked.append(("stat", os.fspath(p))) or real_stat(p, *a, **k))
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: asked.append(("lstat", os.fspath(p))) or real_lstat(p, *a, **k))
    file_dep_snapshot.begin_file_state_epoch()
    try:
        tracker = FileAccessTracker(hash_on_read=True)
        with tracker:
            with open(path, encoding="utf-8") as fh:
                assert fh.read() == "some data"
    finally:
        file_dep_snapshot.end_file_state_epoch()
    monkeypatch.undo()
    resolved = os.path.realpath(path)
    assert [kind for kind, p in asked if os.path.realpath(p) == resolved] == ["lstat"]
    # The read is recorded (by its resolved path, in forward slashes on
    # Windows too), with the digest of what it read.
    assert tracker.read_digests[normalize_path(resolved)] == file_dep_snapshot.file_content_hash(resolved)
