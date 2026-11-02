"""A file read by a relative path is stat-ed once a cell run, not once per spelling.

A file read as ``faces/p001/img7.npy`` is recorded under that spelling and
its absolute one (``FileAccessTracker.track_path``), and the check before
each cell stat-ed both: 40,000 stats for a folder of 20,000 images. The
relative spelling takes the stat of the absolute one in the working
directory. A path through ``..`` does not: spelled out, it goes up from the
folder a link is in, where the open goes up from where the link points.
"""

from __future__ import annotations

import os

import pytest

from cash._paths import normalize_path
from cash.notebook import run_memo
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
    run_memo._FILE_STATE_THIS_RUN.clear()


@pytest.fixture
def stats_taken(monkeypatch) -> list[str]:
    taken: list[str] = []
    real = os.stat

    def counting(path, *args, **kwargs):
        taken.append(os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting)
    return taken


@pytest.fixture
def faces(tmp_path, monkeypatch) -> list[str]:
    (tmp_path / "faces").mkdir()
    for i in range(N):
        (tmp_path / "faces" / f"img{i}.npy").write_bytes(bytes([i]) * 16)
    monkeypatch.chdir(tmp_path)
    return [f"faces/img{i}.npy" for i in range(N)]


@pytest.mark.parametrize("absolute_first", [True, False])
def test_both_spellings_of_each_file_take_one_stat(faces, stats_taken, absolute_first):
    spelled = [normalize_path(os.path.abspath(p)) for p in faces]
    paths = [*spelled, *faces] if absolute_first else [*faces, *spelled]

    stats = run_memo.stats_this_run(paths)

    assert len(stats_taken) == N
    for rel, absolute in zip(faces, spelled, strict=True):
        assert stats[rel] == (rel, os.stat(absolute))
        assert stats[absolute][1] == os.stat(absolute)


def test_a_path_through_dot_dot_is_stat_ed_as_spelled(faces, stats_taken):
    os.mkdir("sub")
    up = "sub/../faces/img0.npy"

    run_memo.stats_this_run([up, normalize_path(os.path.abspath("faces/img0.npy"))])

    assert up in stats_taken
