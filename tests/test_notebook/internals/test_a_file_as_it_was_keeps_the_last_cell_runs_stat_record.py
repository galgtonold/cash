"""A file as the last cell run found it keeps that run's stat record.

Before every cell the upstream check stats each file the cells above read
and keeps a record of each for the run (`run_memo.stats_this_run`). The
records live for the whole run, so each run's new ones reached the
collector's oldest generation: below a loop over 50,000 documents, 200,000
objects before every cell, and a full collection of the heap nearly every
cell. A file whose stat equals the last run's in every field keeps that
run's record; any change to the file gives a new one.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import stats_this_run
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


def _next_cell_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _inputs(tmp_path) -> list[str]:
    paths = []
    for i in range(N):
        p = tmp_path / f"doc{i:03d}.txt"
        p.write_text(f"document {i}\n", encoding="utf-8")
        paths.append(str(p))
    return paths


def test_an_unchanged_file_keeps_its_record(tmp_path):
    paths = _inputs(tmp_path)
    first = stats_this_run(paths)

    _next_cell_run()
    again = stats_this_run(paths)

    assert all(again[p] is first[p] for p in paths)
    assert all(run_memo.same_stat(again[p][1], os.stat(p)) for p in paths)


def test_a_changed_file_gets_a_new_record(tmp_path):
    paths = _inputs(tmp_path)
    first = stats_this_run(paths)
    edited, touched = paths[0], paths[1]
    with open(edited, "a", encoding="utf-8") as f:
        f.write("more\n")
    st = os.stat(touched)
    os.utime(touched, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000))  # a time alone

    _next_cell_run()
    again = stats_this_run(paths)

    for p in (edited, touched):
        assert again[p] is not first[p]
        assert again[p][1].st_mtime_ns == os.stat(p).st_mtime_ns
    assert again[edited][1].st_size == os.stat(edited).st_size
    assert all(again[p] is first[p] for p in paths[2:])


def test_a_write_in_the_run_drops_the_last_runs_records(tmp_path):
    paths = _inputs(tmp_path)
    first = stats_this_run(paths)

    _next_cell_run()
    run_memo.forget_file_state_this_run()
    again = stats_this_run(paths)

    assert all(again[p] is not first[p] for p in paths)
    assert all(run_memo.same_stat(again[p][1], first[p][1]) for p in paths)


def test_same_stat_sees_every_field():
    same_stat = run_memo.same_stat
    st = os.stat(os.curdir)
    fields = list(os.stat_result.__reduce__(st)[1][0])
    extra = dict(os.stat_result.__reduce__(st)[1][1])
    assert same_stat(st, os.stat_result(fields, extra))
    for name, value in extra.items():
        changed = dict(extra, **{name: value + 1})
        assert not same_stat(st, os.stat_result(fields, changed)), name
    for i in range(len(fields)):
        moved = [*fields[:i], fields[i] + 1, *fields[i + 1 :]]
        assert not same_stat(st, os.stat_result(moved, extra)), i
