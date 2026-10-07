"""Cash's own cache lookups made inside a tracked block are dropped at once.

Each call a notebook statement keys looks its entry up on disk, inside the
statement's file tracker. The tracker resolved and probed every such path
before finding it was the cache's own (``is_cash_internal``): ~50 us on top of
a ~2 us miss, 3 ms of the first 50 calls of ``[f(i) for i in xs]``. A user's
own read is still resolved and recorded.
"""

from __future__ import annotations

import os

from cash.backends.file_backend import FileBackend
from cash.tracking import io_watch
from cash.tracking.file_tracker import FileAccessTracker


def _open_missing(path):
    try:
        open(path, "rb").close()
    except FileNotFoundError:
        pass


def test_a_lookup_of_a_cache_entry_is_not_resolved(tmp_path, monkeypatch):
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    assert backend.get("k" * 64) == (None, None)  # registers the directory
    resolved = []
    real = FileAccessTracker.track_path
    monkeypatch.setattr(FileAccessTracker, "track_path", lambda self, p: resolved.append(p) or real(self, p))
    user_file = tmp_path / "data.csv"
    io_watch.hold()
    try:
        with FileAccessTracker({}) as tracker:
            backend.get("k" * 64)
            _open_missing(user_file)
    finally:
        io_watch.release()
    assert resolved == [str(user_file)]
    assert os.path.realpath(user_file) in {os.path.realpath(p) for p in tracker.get_accessed_files() | tracker.absent_files}
