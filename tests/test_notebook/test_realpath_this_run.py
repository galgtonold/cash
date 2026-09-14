"""A path is resolved once per cell run, not once per use.

``realpath`` is ~60us on Windows and a statement's files were resolved again at
every tracked read, lineage component and snapshot: 13% of a cell reading
3,000 files (round 23). The memo lasts one cell run; outside a run, and in the
next one, a path resolves afresh.
"""
from __future__ import annotations

import os

import pytest

from cash.notebook import file_dep_snapshot
from cash.notebook.file_dep_snapshot import realpath_this_run
from cash.utils import normalize_path


@pytest.fixture(autouse=True)
def _outside_any_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "_HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    file_dep_snapshot._REALPATH_MEMO.clear()


@pytest.fixture
def resolutions(monkeypatch):
    calls: list[str] = []
    real = os.path.realpath
    monkeypatch.setattr(file_dep_snapshot.os.path, "realpath", lambda p, *a, **k: calls.append(p) or real(p, *a, **k))
    return calls


def test_within_a_run_both_spellings_resolve_once(tmp_path, resolutions):
    f = tmp_path / "a.csv"
    f.write_text("x")
    file_dep_snapshot.begin_file_state_epoch()
    try:
        resolved = realpath_this_run(str(f))
        assert realpath_this_run(str(f)) == resolved
        # The tracker records the normalized form; the lineage resolves that.
        assert os.path.normcase(realpath_this_run(normalize_path(resolved))) == os.path.normcase(resolved)
    finally:
        file_dep_snapshot.end_file_state_epoch()
    assert len(resolutions) == 1


def test_outside_a_run_nothing_is_remembered(tmp_path, resolutions):
    f = tmp_path / "a.csv"
    f.write_text("x")
    realpath_this_run(str(f))
    realpath_this_run(str(f))
    assert len(resolutions) == 2


def test_the_next_run_resolves_afresh(tmp_path, resolutions):
    f = tmp_path / "a.csv"
    f.write_text("x")
    for _ in range(2):
        file_dep_snapshot.begin_file_state_epoch()
        try:
            realpath_this_run(str(f))
        finally:
            file_dep_snapshot.end_file_state_epoch()
    assert len(resolutions) == 2


def test_a_relative_path_follows_a_chdir(tmp_path, monkeypatch):
    for sub in ("one", "two"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "data.csv").write_text(sub)
    file_dep_snapshot.begin_file_state_epoch()
    try:
        monkeypatch.chdir(tmp_path / "one")
        first = realpath_this_run("data.csv")
        monkeypatch.chdir(tmp_path / "two")
        second = realpath_this_run("data.csv")
    finally:
        file_dep_snapshot.end_file_state_epoch()
    assert first != second
    assert second.endswith(os.path.join("two", "data.csv"))
