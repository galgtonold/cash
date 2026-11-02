"""A folder the cell run's listing reported is stat-ed once a cell run, not once a check.

On Windows the run's stats of a crowded directory come from one listing
(`stats_from_listings`), and a listed stat stands in only where content
decides. A folder has no content digest: it is decided by its time, which a
listing may report late. So every check of the run -- the simulation's, the
statement lookup's, each file component -- took a stat of its own of every
folder: the 5,700 person folders of a photo set, before each trivial cell,
two to four times. The first check that needs it now takes the folder's own
stat for the run (`run_memo._Identifying`), and the others take that one.
"""

from __future__ import annotations

import os
import time
import types
from pathlib import Path

import pytest

from cash.notebook import run_memo
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_file_deps, snapshot_is_fresh

N = 24  # folders in one directory: more than a listing is taken for


def _listed(paths):
    """`stats_from_listings` as on Windows: no file identity, times as listed."""
    by_dir: dict[str, dict[str, str]] = {}
    for path in paths:
        directory, name = os.path.split(path)
        by_dir.setdefault(directory, {})[name] = path
    found = {}
    for directory, wanted in by_dir.items():
        if len(wanted) < file_dep_snapshot.LISTING_MIN_FILES:
            continue
        with os.scandir(directory) as entries:
            for entry in entries:
                path = wanted.get(entry.name)
                if path is not None:
                    r = entry.stat()
                    found[path] = os.stat_result(
                        (r.st_mode, 0, 0, 0, 0, 0, r.st_size, int(r.st_atime), int(r.st_mtime), int(r.st_ctime))
                        + (r.st_atime, r.st_mtime, r.st_ctime, r.st_atime_ns, r.st_mtime_ns, r.st_ctime_ns)
                    )
    return found


@pytest.fixture(autouse=True)
def _windows_listings(monkeypatch):
    monkeypatch.setattr(run_memo, "stats_from_listings", _listed)
    monkeypatch.setattr(file_dep_snapshot, "stats_from_listings", _listed)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    yield
    while file_dep_snapshot.HASH_EPOCH is not None:
        file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()


@pytest.fixture
def folders(tmp_path) -> dict[str, dict]:
    root = tmp_path / "lfw"
    paths = set()
    for i in range(N):
        d = root / f"person{i:03d}"
        d.mkdir(parents=True)
        (d / "img0.npy").write_bytes(b"x" * 16)
        paths.add(str(d))
    snap = snapshot_file_deps(paths)
    assert all("hash" not in entry for entry in snap.values()), "a folder is decided by its time"
    return snap


@pytest.fixture
def stats_of(monkeypatch, tmp_path):
    calls: list[str] = []
    real = os.stat

    def counting(path, *a, **k):
        if str(tmp_path) in os.fspath(path):
            calls.append(os.fspath(path))
        return real(path, *a, **k)

    monkeypatch.setattr(os, "stat", counting)
    return calls


def _statement_lookup(snap: dict) -> bool:
    checker = CacheFreshnessChecker()
    checker.forget_file_answers(file_dep_snapshot.file_state_epoch())
    metadata = types.SimpleNamespace(file_dependencies=snap)
    return checker._invalidate_if_direct_file_changed(metadata, {"variables": {}}) is not None


def _cell_run():
    """Start a cell run, ending the one before."""
    if file_dep_snapshot.HASH_EPOCH is not None:
        file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def test_every_check_of_a_run_takes_one_stat_of_a_listed_folder(folders, stats_of):
    _cell_run()
    run_memo.stat_keys_this_run(folders)  # the check of the cells above: one listing
    assert stats_of == [], "the folders were listed, not stat-ed"

    assert snapshot_is_fresh(folders, run_memo.file_state_this_run()["memo"]) == (True, None)  # the simulation's
    assert _statement_lookup(folders)
    assert _statement_lookup(folders)
    run_memo.file_component_this_run(set(folders))

    assert sorted(stats_of) == sorted(folders), "one stat of each folder for the run"


def test_a_folder_changed_after_the_listing_is_seen_by_the_run(folders, stats_of):
    """The listing may report a folder's time late: its own stat decides."""
    _cell_run()
    run_memo.stats_this_run(folders)
    changed = sorted(folders)[0]
    st = os.stat(changed)
    os.utime(changed, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # after the listing

    fresh, stale = snapshot_is_fresh(folders, run_memo.file_state_this_run()["memo"])
    assert not fresh and stale.path == changed
    assert not _statement_lookup(folders)


def test_a_folder_changed_between_cells_is_seen(folders):
    _cell_run()
    assert snapshot_is_fresh(folders, run_memo.file_state_this_run()["memo"]) == (True, None)
    assert _statement_lookup(folders)
    changed = sorted(folders)[3]
    (Path(changed) / "img1.npy").write_bytes(b"y")
    st = os.stat(changed)
    os.utime(changed, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000))  # whatever the clock's tick

    _cell_run()
    fresh, stale = snapshot_is_fresh(folders, run_memo.file_state_this_run()["memo"])
    assert not fresh and stale.path == changed
    assert not _statement_lookup(folders)


def test_the_lookup_takes_its_own_stats_when_the_run_s_are_too_old(folders, stats_of, monkeypatch):
    """The lookup's answers last ``_ANSWERS_LAST_S``: an older stat of the
    run, its own or listed, is taken again, and an edit since is seen."""
    from cash.notebook.statement import freshness

    _cell_run()
    run_memo.stats_this_run(folders)
    assert _statement_lookup(folders)
    run_memo.file_state_this_run()["stat_at"] = time.monotonic() - freshness._ANSWERS_LAST_S - 1
    changed = sorted(folders)[5]
    st = os.stat(changed)
    os.utime(changed, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    stats_of.clear()

    assert not _statement_lookup(folders)
    assert changed in stats_of
