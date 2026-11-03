"""A dependency in a watched folder is not looked at again by a later cell.

Before each cell of a notebook over installed packages, every check of the
cells above (the run's stats, the simulation's freshness check, the
statement lookup, the file component of a lineage) looked at each of its
~4,400 dependencies again: a stat of each file, and for each path the import
system had looked for and not found, an ``lstat``, the moved-project
fallbacks' ``os.path.exists`` of several more paths, and one more
``os.path.exists`` in the check itself. 57,000 stats, 40,000 lstats and
76,000 looks a cell on Windows. A path not there is now kept as such while
its folder is watched (`FileWatch.record_absent`), looked for only where it
was recorded, and a snapshot or component holding such paths is kept
under the watch like any other: the later cell takes one stat per watched
folder, which checks the folder is still the one watched.
"""

from __future__ import annotations

import collections
import os
import sys
import types

import pytest

from cash.notebook import run_memo
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.file_dep_snapshot import snapshot_absent_deps, snapshot_file_deps, snapshot_is_fresh

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FOLDERS, FILES, MISSING = 5, 40, 30


@pytest.fixture(autouse=True)
def _runs(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    file_watch.reset()


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _windows_listing(paths):
    """`stats_from_listings` as on Windows: entries of a crowded folder, with
    no file identity."""
    by_dir = collections.defaultdict(list)
    for p in paths:
        by_dir[os.path.dirname(p)].append(p)
    found = {}
    for ps in by_dir.values():
        if len(ps) < file_dep_snapshot.LISTING_MIN_FILES:
            continue
        for p in ps:
            try:
                st = os.lstat(p)
            except OSError:
                continue
            found[p] = os.stat_result((st.st_mode, 0, 0, 0, 0, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
    return found


@pytest.fixture
def snap(tmp_path) -> dict:
    """The dependencies of an import: modules read, and paths looked for
    and not found."""
    paths, missing = set(), []
    for d in range(FOLDERS):
        folder = tmp_path / f"pkg{d}"
        folder.mkdir()
        for i in range(FILES):
            p = folder / f"m{i:02d}.py"
            p.write_text(f"X = {i}\n", encoding="utf-8")
            paths.add(str(p))
        missing += [str(folder / f"m{i:02d}.cpython-314.so") for i in range(MISSING)]
    deps = snapshot_file_deps(paths)
    deps.update(snapshot_absent_deps(missing))
    return deps


@pytest.fixture
def looks(monkeypatch, tmp_path) -> collections.Counter:
    """Each stat, lstat and existence look at a path under *tmp_path*, by kind."""
    seen: collections.Counter = collections.Counter()
    root = str(tmp_path)

    def counting(kind, real):
        def call(path, *args, **kwargs):
            if os.fspath(path).startswith(root):
                seen[kind] += 1
            return real(path, *args, **kwargs)

        return call

    def start():
        monkeypatch.setattr(os, "stat", counting("stat", os.stat))
        monkeypatch.setattr(os, "lstat", counting("lstat", os.lstat))
        monkeypatch.setattr(os.path, "exists", counting("exists", os.path.exists))

    seen.start = start  # type: ignore[attr-defined]
    return seen


def _cell(deps: dict) -> None:
    """What a trivial cell's checks ask of the dependencies above it."""
    run_memo.stat_keys_this_run(deps)  # the check of the cells above
    run = run_memo.file_state_this_run()
    assert snapshot_is_fresh(deps, run["memo"]) == (True, None)  # the simulation's
    checker = CacheFreshnessChecker()
    checker.forget_file_answers(file_dep_snapshot.file_state_epoch())
    metadata = types.SimpleNamespace(file_dependencies=deps)
    assert checker._invalidate_if_direct_file_changed(metadata, {"variables": {}}) is not None
    run_memo.file_component_this_run(set(deps))  # a lineage's file component
    _next_run()


@pytest.mark.parametrize("listed", [False, True], ids=["stat-ed", "listed as on Windows"])
def test_a_later_cell_takes_one_stat_per_watched_folder(snap, looks, monkeypatch, listed):
    if listed:
        monkeypatch.setattr(file_dep_snapshot, "stats_from_listings", _windows_listing)
        monkeypatch.setattr(run_memo, "stats_from_listings", _windows_listing)
    _cell(snap)
    _cell(snap)
    looks.start()

    _cell(snap)

    # Each watched folder's own stat (`FileWatch.drain`), and nothing per file.
    assert looks["stat"] <= FOLDERS
    assert looks["lstat"] == 0
    assert looks["exists"] == 0


def test_a_path_made_where_one_was_not_is_seen(snap, tmp_path):
    _cell(snap)
    _cell(snap)
    made = tmp_path / "pkg2" / "m07.cpython-314.so"
    assert str(made) in snap
    made.write_bytes(b"\0")

    run_memo.stat_keys_this_run(snap)
    run = run_memo.file_state_this_run()
    fresh, stale = snapshot_is_fresh(snap, run["memo"])
    assert not fresh and stale.path == str(made) and stale.reason == "appeared"


def test_a_component_kept_under_the_watch_is_taken_before_any_stat(snap, looks):
    paths = set(snap)
    first = run_memo.file_component_this_run(paths)
    _next_run()
    looks.start()

    assert run_memo.file_component_this_run(paths) == first
    assert looks["stat"] <= FOLDERS and looks["lstat"] == 0 and looks["exists"] == 0
