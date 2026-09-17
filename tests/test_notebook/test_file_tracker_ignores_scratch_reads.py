"""Reads that are not the user's data do not become file dependencies.

Round 25 (r25s1): a ``cross_val_predict(n_jobs=4)`` loop recorded joblib's
memmap scratch files, which joblib deletes when the call returns, so the loop
read as stale on every later run; and ``open(3)`` -- a file descriptor --
was recorded as a read of ``<cwd>/3``, a directory every report written under
the cwd sits in.
"""
from __future__ import annotations

import os

from cash.notebook.file_tracker import FileAccessTracker


def _tracked(action):
    tracker = FileAccessTracker(user_ns={})
    with tracker:
        action()
    return {str(f).replace("\\", "/") for f in tracker.get_accessed_files()}


def test_a_joblib_memmap_read_is_not_a_dependency(tmp_path):
    scratch = tmp_path / "joblib_memmapping_folder_1234_abcd"
    scratch.mkdir()
    (scratch / "arr.pkl").write_bytes(b"x")
    (tmp_path / "data.csv").write_text("a\n1\n")

    def read():
        with open(scratch / "arr.pkl", "rb"):
            pass
        with open(tmp_path / "data.csv", "rb"):
            pass
    tracked = _tracked(read)
    assert not [t for t in tracked if "joblib_memmapping_folder_" in t], tracked
    assert any(t.endswith("data.csv") for t in tracked), tracked


def test_opening_a_file_descriptor_records_no_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("a\n1\n")
    fd = os.open(tmp_path / "data.csv", os.O_RDONLY)

    def read():
        with open(fd, "rb", closefd=True):
            pass
    tracked = _tracked(read)
    assert not tracked, tracked
