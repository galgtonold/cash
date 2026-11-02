"""A file's freshness answer is kept under its path, once per check.

The answers a check reuses (`FreshnessMemo`) were keyed by the path and
every item of its recorded snapshot: nine tuples per file, kept for the
cell run. A statement inheriting a frame built from 24,000 documents
checked them before every cell, and the collections those tuples set off
were a large part of the check. The answer is now kept under the path,
with a copy of the snapshot it answers; another snapshot of the same path
is answered on its own.
"""

from __future__ import annotations

from cash.tracking.file_dep_snapshot import FreshnessMemo, snapshot_file_deps, snapshot_is_fresh

N = 20


def _snapshot(tmp_path):
    paths = set()
    for i in range(N):
        p = tmp_path / f"doc{i:02d}.txt"
        p.write_text(f"document {i}\n", encoding="utf-8")
        paths.add(str(p))
    return snapshot_file_deps(paths)


def test_one_answer_per_file_under_its_path(tmp_path):
    snap = _snapshot(tmp_path)
    memo = FreshnessMemo()

    assert snapshot_is_fresh(snap, memo)[0]
    assert snapshot_is_fresh(dict(snap), memo)[0]
    assert set(memo.answers) == set(snap)


def test_another_snapshot_of_the_same_file_is_answered_on_its_own(tmp_path):
    snap = _snapshot(tmp_path)
    memo = FreshnessMemo()
    assert snapshot_is_fresh(snap, memo)[0]
    path = sorted(snap)[0]
    older = {path: {**snap[path], "size": snap[path]["size"] + 1}}

    fresh, stale = snapshot_is_fresh(older, memo)
    assert not fresh
    assert stale.reason == "size"
    assert snapshot_is_fresh(snap, memo)[0]
    assert not snapshot_is_fresh(older, memo)[0]


def test_a_snapshot_changed_after_its_answer_is_answered_again(tmp_path):
    snap = _snapshot(tmp_path)
    memo = FreshnessMemo()
    assert snapshot_is_fresh(snap, memo)[0]
    path = sorted(snap)[0]
    snap[path]["size"] += 1  # the same dict, changed in place

    assert not snapshot_is_fresh(snap, memo)[0]
