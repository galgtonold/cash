"""A file beside the code, recorded in another spelling of the same path,
takes the run's stat of it.

A file beside the code (an installed package's data) is checked in this
process's copy of that code (`dep_path_for_this_process`), whose path is
spelled with forward slashes on Windows, where the recorded one has
backslashes. The two differed, so each check took a stat of its own instead
of the run's, and no snapshot holding such a file was kept under the folder
watch: 24,751 stats in one cell of a notebook over an installed package.
The same place in another spelling is the recorded path.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from cash.notebook import run_memo
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.file_dep_snapshot import FreshnessMemo, dep_is_fresh, dep_path_for_this_process, snapshot_file_deps

_NAME = "fxbeside_spelled_pkg.core"


@pytest.fixture
def package(tmp_path, monkeypatch):
    pkg = tmp_path / "fxbeside_spelled_pkg"
    (pkg / "data").mkdir(parents=True)
    mod = types.ModuleType(_NAME)
    mod.__file__ = str(pkg / "core.py")
    monkeypatch.setitem(sys.modules, _NAME, mod)
    target = pkg / "data" / "a.csv"
    target.write_text("1,2\n", encoding="utf-8")
    return pkg, target


def _other_spelling(path: str) -> str:
    """*path* spelled otherwise: with the other separator on Windows, with a
    ``.`` component elsewhere."""
    if os.sep == "\\":
        return path.replace("/", "\\") if "/" in path else path.replace("\\", "/")
    head, tail = os.path.split(path)
    return os.path.join(head, ".", tail)


def _recorded(target) -> dict:
    entry = dict(snapshot_file_deps({str(target)})[str(target)])
    entry.update(code_rel="data/a.csv", code_mod=_NAME)
    return entry


def test_the_same_place_spelled_otherwise_is_the_recorded_path(package):
    _pkg, target = package
    here = dep_path_for_this_process("/elsewhere/a.csv", _recorded(target))
    spelled = _other_spelling(here)
    assert spelled != here
    assert dep_path_for_this_process(spelled, _recorded(target)) == spelled
    # Elsewhere, it is still this copy's file.
    assert dep_path_for_this_process("/elsewhere/a.csv", _recorded(target)) == here


def test_a_check_of_it_takes_the_stat_handed_in(package, monkeypatch):
    _pkg, target = package
    recorded = _recorded(target)
    spelled = _other_spelling(dep_path_for_this_process("/elsewhere/a.csv", recorded))
    st = os.stat(target)
    stats = []
    real = file_dep_snapshot.own_stat

    def counted(path, *args, **kwargs):
        stats.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(file_dep_snapshot, "own_stat", counted)
    resolved, fresh, _reason = dep_is_fresh(spelled, recorded, FreshnessMemo(), taken=st)
    assert fresh and resolved == spelled
    assert stats == []


@pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")
def test_a_snapshot_holding_it_is_kept_under_the_folder_watch(package, monkeypatch):
    _pkg, target = package
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._COVERED_SNAPSHOTS.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    try:
        spelled = _other_spelling(dep_path_for_this_process("/elsewhere/a.csv", _recorded(target)))
        snap = {spelled: _recorded(target)}
        run_memo.stats_this_run(snap)
        watch = file_watch.watcher()
        assert watch is not None
        assert run_memo._cover_of(watch, snap, snap) is not None

        # Loaded from another copy: the file is looked for there, so the cover lapses.
        cover = run_memo._cover_of(watch, snap, snap)
        other = types.ModuleType(_NAME)
        other.__file__ = os.path.join(os.path.dirname(os.path.dirname(str(target))), "..", "copy", "core.py")
        monkeypatch.setitem(sys.modules, _NAME, other)
        assert not watch.still(cover)
    finally:
        file_dep_snapshot.end_file_state_epoch()
        run_memo._FILE_STATE_THIS_RUN.clear()
        run_memo._COVERED_SNAPSHOTS.clear()
        file_watch.reset()
