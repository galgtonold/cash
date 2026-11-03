"""Every freshness check of a cell run takes the run's stat of a file, not one of its own.

Before a trivial cell below a read of 20,000 images, the check of the
cells above stat-ed every image, and so did the simulation's check of the
entry that read them and the statement lookup's check of its own inherited
files: each file two to four times a cell, 7-12 s a cell on Windows. The
run keeps its stats (`run_memo.stats_this_run`), and each check takes them
(`FreshnessMemo.taken`). The statement lookup takes only a stat as recent as
its own answers may be (``_ANSWERS_LAST_S``), and takes an older one again.
"""

from __future__ import annotations

import os
import time
import types

import pytest

from cash.notebook import run_memo
from cash.notebook.statement import freshness
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_file_deps, snapshot_is_fresh

N = 70  # a snapshot of more files than the statement lookup keeps verdicts for whole


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
def docs(tmp_path) -> dict[str, dict]:
    paths = set()
    for i in range(N):
        p = tmp_path / f"img{i:03d}.npy"
        p.write_bytes(bytes([i % 251]) * 64)
        paths.add(str(p))
    return snapshot_file_deps(paths)


@pytest.fixture
def stats_taken(monkeypatch) -> list[str]:
    """Each ``stat`` or ``lstat`` taken, by path: the run's pass takes an
    ``lstat`` of anything that is no link (`run_memo.stats_this_run`)."""
    taken: list[str] = []

    def counting(real):
        def call(path, *args, **kwargs):
            taken.append(os.fspath(path))
            return real(path, *args, **kwargs)

        return call

    monkeypatch.setattr(os, "stat", counting(os.stat))
    monkeypatch.setattr(os, "lstat", counting(os.lstat))
    return taken


def _statement_lookup(deps: dict[str, dict]) -> bool:
    checker = CacheFreshnessChecker()
    checker.forget_file_answers(file_dep_snapshot.file_state_epoch())
    metadata = types.SimpleNamespace(file_dependencies=deps)
    return checker._invalidate_if_direct_file_changed(metadata, {"variables": {}}) is not None


def test_the_check_above_the_simulation_and_the_lookup_stat_each_file_once(docs, stats_taken):
    run_memo.stat_keys_this_run(docs)  # the check of the cells above
    run = run_memo.file_state_this_run()

    assert snapshot_is_fresh(docs, run["memo"]) == (True, None)  # the simulation's
    assert _statement_lookup(docs)

    assert sorted(stats_taken) == sorted(docs)


def test_the_lookup_stats_again_a_file_the_run_stat_ed_too_long_ago(docs, stats_taken, monkeypatch):
    run_memo.stats_this_run(docs)
    real = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real() + freshness._ANSWERS_LAST_S + 1)
    edited = next(iter(docs))
    with open(edited, "r+b") as f:
        f.write(b"\xff")  # same size, new content
    stats_taken.clear()

    assert not _statement_lookup(docs)

    assert edited in stats_taken


def test_a_stat_from_a_directory_listing_decides_only_by_content(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("1,2\n", encoding="utf-8")
    recorded = snapshot_file_deps({str(path)})[str(path)]
    real = os.stat(path)
    # As a Windows listing reports it: no file identity, a time it may lag on.
    listed = os.stat_result((real.st_mode, 0, real.st_dev, 1, 0, 0, real.st_size, 0, 0, 0))

    assert file_dep_is_fresh(str(path), recorded, taken=listed) == (True, None)
    assert file_dep_is_fresh(str(path), {k: v for k, v in recorded.items() if k != "hash"}, taken=listed)[0]
