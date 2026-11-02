"""A snapshot the upstream check found fresh is an answer for the rest of the run.

What the runtime's lookup takes from it (`run_memo.fresh_snapshot_this_run`):
only the same snapshot, only from answers as recent as it asks, and nothing
once a statement of the run wrote files.
"""

from __future__ import annotations

import time

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import (
    forget_file_state_this_run,
    fresh_snapshot_this_run,
    note_fresh_snapshot,
)
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_file_deps


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()


@pytest.fixture
def snap(tmp_path):
    paths = set()
    for i in range(5):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"document {i}\n", encoding="utf-8")
        paths.add(str(p))
    return snapshot_file_deps(paths)


def test_the_same_snapshot_is_known_fresh(snap):
    start = time.monotonic()
    note_fresh_snapshot(snap)

    assert fresh_snapshot_this_run({p: dict(entry) for p, entry in snap.items()}, start - 1.0)


def test_another_snapshot_is_not(snap):
    note_fresh_snapshot(snap)
    other = {p: dict(entry) for p, entry in snap.items()}
    first = sorted(other)[0]
    other[first]["size"] += 1

    assert not fresh_snapshot_this_run(other, time.monotonic() - 1.0)
    assert not fresh_snapshot_this_run(dict(list(snap.items())[1:]), time.monotonic() - 1.0)


def test_answers_older_than_asked_for_do_not_count(snap):
    run_memo.file_state_this_run()  # the run's answers begin
    time.sleep(0.05)
    asked_from = time.monotonic()
    note_fresh_snapshot(snap)  # dated by when the answers began, not now

    assert not fresh_snapshot_this_run(snap, asked_from)


def test_a_write_in_the_run_drops_it(snap):
    note_fresh_snapshot(snap)
    forget_file_state_this_run()  # what a statement that writes files does

    assert not fresh_snapshot_this_run(snap, time.monotonic() - 1.0)


def test_nothing_is_kept_outside_a_run(snap):
    file_dep_snapshot.end_file_state_epoch()
    try:
        note_fresh_snapshot(snap)
        assert not fresh_snapshot_this_run(snap, time.monotonic() - 1.0)
    finally:
        file_dep_snapshot.begin_file_state_epoch()


def test_the_next_run_starts_without_it(snap):
    note_fresh_snapshot(snap)
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()

    assert not fresh_snapshot_this_run(snap, time.monotonic() - 1.0)
