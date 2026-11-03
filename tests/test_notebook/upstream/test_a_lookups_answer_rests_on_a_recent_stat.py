"""Every answer a statement lookup gives rests on a stat at most two seconds old.

The statement lookup shares its answers for a file between the lookups of a
cell run, and takes the cell run's own stats of the files. An answer is
given only while the stat it rests on was taken at most ``_ANSWERS_LAST_S``
before the lookup began, so an edit made earlier than that -- by another
process, or a thread, while the cell runs -- is seen. Dated by when the
answers were first given, a second lookup in the same cell took an answer
that rested on a stat taken before the first one began, and served a value
built from a file edited longer ago than that.
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
from cash.tracking.file_dep_snapshot import snapshot_file_deps


@pytest.fixture(autouse=True)
def _cell_runs(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    while file_dep_snapshot.file_state_epoch() is not None:
        file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()


@pytest.fixture
def clock(monkeypatch):
    """``time.monotonic`` moved on by hand: a slow cell run without the wait."""
    real = time.monotonic
    offset = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: real() + offset[0])

    def advance(seconds: float) -> None:
        offset[0] += seconds

    return advance


class _Backend:
    def __init__(self, deps):
        self._meta = {"key": "stmt:x", "file_dependencies": deps}

    def get(self, key):
        return dict(self._meta, key=key), "value"


@pytest.fixture(params=[20, 80], ids=["per-file", "whole-set"])
def docs(request, tmp_path) -> list[str]:
    """Few files (each answer kept on its own) and many (the set found fresh
    whole kept too), left alone for an hour: a stat as recorded answers for
    each without a read, so only a stat taken after the edit sees it."""
    paths = []
    settled = time.time() - 3600
    for i in range(request.param):
        p = tmp_path / f"d{i:03d}.txt"
        p.write_text(f"doc {i}\n", encoding="utf-8")
        os.utime(p, (settled, settled))
        paths.append(str(p))
    return paths


def _lookup(checker: CacheFreshnessChecker, key: str) -> tuple[bool, str | None]:
    state = types.SimpleNamespace(executed_file_deps={}, variable_sources={})
    _, data, _ = checker.check_cache(state, key, None, epoch=file_dep_snapshot.file_state_epoch())
    return data is not None, checker.last_miss_reason


def _edit(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("edited while the cell ran\n")


def test_an_edit_made_longer_ago_than_the_window_is_seen_by_the_second_lookup(docs, clock):
    window = freshness._ANSWERS_LAST_S
    checker = CacheFreshnessChecker(_Backend(snapshot_file_deps(set(docs))))
    run_memo.stats_this_run(docs)  # the cell run's pass, before the edit
    _edit(docs[3])
    clock(window * 0.25)
    _lookup(checker, "stmt:a")  # the edit is recent: either answer is allowed
    clock(window * 0.9)  # the edit is now older than the window; the first lookup is not

    served, reason = _lookup(checker, "stmt:b")

    assert not served
    assert "d003.txt" in (reason or "")


def test_an_edit_made_longer_ago_than_the_window_is_seen_by_the_first_lookup(docs, clock):
    window = freshness._ANSWERS_LAST_S
    checker = CacheFreshnessChecker(_Backend(snapshot_file_deps(set(docs))))
    run_memo.stats_this_run(docs[: len(docs) // 2])
    _edit(docs[3])  # in the half the pass stat-ed first
    clock(window * 0.6)
    run_memo.stats_this_run(docs[len(docs) // 2 :])
    clock(window * 0.6)  # the first half is too old now, the second is not

    served, reason = _lookup(checker, "stmt:a")

    assert not served
    assert "d003.txt" in (reason or "")


def test_an_edit_made_between_cells_is_seen(docs, clock):
    checker = CacheFreshnessChecker(_Backend(snapshot_file_deps(set(docs))))
    run_memo.stats_this_run(docs)
    assert _lookup(checker, "stmt:a")[0]
    file_dep_snapshot.end_file_state_epoch()
    _edit(docs[5])
    file_dep_snapshot.begin_file_state_epoch()  # the next cell
    run_memo.stats_this_run(docs)

    served, reason = _lookup(checker, "stmt:a")

    assert not served
    assert "d005.txt" in (reason or "")


def test_answers_on_recent_stats_are_shared_between_lookups(docs, clock, monkeypatch):
    """The control: within the window, a second lookup checks nothing again."""
    checker = CacheFreshnessChecker(_Backend(snapshot_file_deps(set(docs))))
    run_memo.stats_this_run(docs)
    clock(freshness._ANSWERS_LAST_S * 0.25)
    assert _lookup(checker, "stmt:a")[0]
    checks: list[str] = []
    real = file_dep_snapshot.file_dep_is_fresh
    monkeypatch.setattr(file_dep_snapshot, "file_dep_is_fresh", lambda *a, **k: checks.append(a[0]) or real(*a, **k))
    clock(freshness._ANSWERS_LAST_S * 0.25)

    assert _lookup(checker, "stmt:b")[0]
    assert checks == []
