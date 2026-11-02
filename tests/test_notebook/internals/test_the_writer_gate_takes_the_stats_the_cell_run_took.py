"""The writer gate's index of read files takes the stats the cell run took already.

Before a cell, the check of each upstream entry stats every file it read
(``run_memo.stats_this_run``), and the gate deciding whether a writer's
output is read stat-ed each of them again: 24,000 documents a loop read
made 24,000 more stats before each cell, and on Windows, where a stat
opens the file and the run's stats come from one listing a folder, about
2 s a cell. The index is the same either way.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import stats_this_run
from cash.notebook.upstream.file_writers import FileWriterScheduler
from cash.tracking import file_dep_snapshot

N = 30


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()


def _reads(tmp_path) -> set[str]:
    folder = tmp_path / "docs"
    folder.mkdir()
    paths = {str(folder)}
    for i in range(N):
        p = folder / f"{i}_7.txt"
        p.write_text(f"review {i}\n", encoding="utf-8")
        paths.add(str(p))
    paths.add(str(tmp_path / "gone.txt"))
    return paths


@pytest.fixture
def stats_of(monkeypatch, tmp_path):
    calls: list[str] = []
    root = os.path.normcase(os.path.normpath(str(tmp_path)))

    def counting(real):
        def call(path, *a, **k):
            if os.path.normcase(os.path.normpath(os.fspath(path))).startswith(root):
                calls.append(str(path))
            return real(path, *a, **k)

        return call

    monkeypatch.setattr(os, "stat", counting(os.stat))
    monkeypatch.setattr(os, "lstat", counting(os.lstat))
    return calls


def _index(paths):
    return FileWriterScheduler(None, None, None)._read_path_index(paths)


def _index_outside_a_run(paths):
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.end_file_state_epoch()
    try:
        return _index(set(paths))
    finally:
        file_dep_snapshot.begin_file_state_epoch()


def test_the_index_stats_no_file_the_run_has_looked_at(tmp_path, stats_of):
    paths = _reads(tmp_path)
    stats_this_run(paths)  # the check of the entries' files
    taken = len(stats_of)

    shared = _index(paths)

    assert len(stats_of) == taken
    own = _index_outside_a_run(paths)
    assert shared[0] == own[0]
    assert sorted(shared[1]) == sorted(own[1]) == [os.path.normcase(str(tmp_path / "docs")) + os.sep]
    assert sorted(shared[2]) == sorted(own[2])
