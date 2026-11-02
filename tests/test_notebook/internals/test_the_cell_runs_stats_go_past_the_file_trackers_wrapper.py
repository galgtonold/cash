"""The stats a cell run takes of the files above go past cash's own ``os.stat`` wrapper.

Before each cell the upstream check stats every file the cells above read
(``run_memo.stats_this_run``). Through the file tracker's wrapper of
``os.stat``, which looks at its caller to tell the user's questions from
cash's, each cost several times the stat: 24,000 documents a loop read,
two spellings each, took about 0.15 s a cell in the wrapper alone. The
wrapper records nothing cash asks, so the answers are the same.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import stats_this_run
from cash.tracking import file_dep_snapshot

N = 20


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()


@pytest.fixture
def wrapped_stat(monkeypatch):
    """``os.stat`` as the file tracker leaves it: a wrapper that names the
    function it wraps. Returns the paths it was called with."""
    through: list[str] = []
    real = os.stat

    def tracked_os_stat(path, *a, **k):
        through.append(str(path))
        return real(path, *a, **k)

    tracked_os_stat._is_file_tracker_patch = True
    tracked_os_stat._original_func = real
    monkeypatch.setattr(os, "stat", tracked_os_stat)
    return through


def test_the_run_stats_each_file_past_the_wrapper(tmp_path, wrapped_stat):
    paths = []
    for i in range(N):
        p = tmp_path / f"{i}_7.txt"
        p.write_text(f"review {i}\n", encoding="utf-8")
        paths.append(str(p))
    missing = str(tmp_path / "gone.txt")

    found = stats_this_run([*paths, missing])

    assert not set(wrapped_stat) & set(paths)
    for p in paths:
        resolved, st = found[p]
        assert resolved == p
        assert (st.st_size, st.st_mtime_ns) == (os.path.getsize(p), os.stat(p).st_mtime_ns)
    assert found[missing][1] is None
