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
    """Each stat, lstat and existence look at a path under *tmp_path*, by
    kind, and each answer about one dependency (``answers``)."""
    seen: collections.Counter = collections.Counter()
    root = str(tmp_path)

    def counting(kind, real):
        def call(path, *args, **kwargs):
            if os.fspath(path).startswith(root):
                seen[kind] += 1

            return real(path, *args, **kwargs)

        return call

    real_answer = file_dep_snapshot.dep_is_fresh

    def answer(path, *args, **kwargs):
        seen["answers"] += 1
        return real_answer(path, *args, **kwargs)

    def start():
        monkeypatch.setattr(file_dep_snapshot, "dep_is_fresh", answer)
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
    report = file_watch.watcher().report()
    assert looks["stat"] <= FOLDERS, report
    assert looks["lstat"] == 0, report
    assert looks["exists"] == 0, report


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
    assert looks["stat"] <= FOLDERS and looks["lstat"] == 0 and looks["exists"] == 0, file_watch.watcher().report()


# -- a folder that changes, a file no folder vouches for --------------------------
#
# The verdict on a snapshot lapsed whole when one of its folders reported a
# change, or when one file could not be vouched for: on Windows a snapshot
# of 8,000 files was checked file by file before every cell (113,948
# answers, 56,974 stats), its notebook's own folder being written by every
# save. Now only the files of the folders that changed, and the loose ones,
# are checked again.


def _snapshot_check(deps: dict) -> None:
    run_memo.stat_keys_this_run(deps)
    run = run_memo.file_state_this_run()
    assert snapshot_is_fresh(deps, run["memo"]) == (True, None)
    _next_run()


@pytest.mark.parametrize("listed", [False, True], ids=["stat-ed", "listed as on Windows"])
def test_a_change_in_one_folder_checks_again_only_the_files_in_it(snap, looks, monkeypatch, tmp_path, listed):
    if listed:
        monkeypatch.setattr(file_dep_snapshot, "stats_from_listings", _windows_listing)
        monkeypatch.setattr(run_memo, "stats_from_listings", _windows_listing)
    _snapshot_check(snap)
    _snapshot_check(snap)
    (tmp_path / "pkg3" / "notebook.ipynb").write_text("{}", encoding="utf-8")  # a save, in one folder
    looks.start()

    _snapshot_check(snap)

    # The folders' own stats, and pkg3's files and the paths not there in it:
    # each answered once, where every file of the snapshot was.
    report = file_watch.watcher().report()
    assert looks["stat"] + looks["lstat"] + looks["exists"] <= FOLDERS + 2 * (FILES + MISSING), report
    assert looks["answers"] == FILES + MISSING, report


def test_a_file_no_folder_vouches_for_is_checked_alone(snap, looks, tmp_path):
    # A file with a second link, stat-ed (a listing on Windows gives no link
    # count): a change through the other link is no change of its folder.
    lone = tmp_path / "lone"
    lone.mkdir()
    linked = lone / "linked.py"
    linked.write_text("Y = 1\n", encoding="utf-8")
    os.link(linked, tmp_path / "elsewhere.py")
    deps = {**snap, **snapshot_file_deps({str(linked)})}
    _snapshot_check(deps)
    _snapshot_check(deps)
    looks.start()

    _snapshot_check(deps)

    report = file_watch.watcher().report()
    assert looks["stat"] + looks["lstat"] + looks["exists"] <= FOLDERS + 1 + 4, report  # its stat, its real path's
    assert looks["answers"] == 1, report
    # And the watch says why (`FileWatch.report`).
    assert report["stats not kept, by why"]["a second hard link"] >= 1


def test_an_edit_in_the_folder_that_changed_is_seen(snap, tmp_path):
    _snapshot_check(snap)
    _snapshot_check(snap)
    edited = tmp_path / "pkg3" / "m11.py"
    edited.write_text("X = 'eleven'\n", encoding="utf-8")

    run_memo.stat_keys_this_run(snap)
    run = run_memo.file_state_this_run()
    fresh, stale = snapshot_is_fresh(snap, run["memo"])
    assert not fresh and stale.path == str(edited)


def test_a_restore_finds_its_files_with_the_runs_stats(snap, looks, tmp_path):
    # A statement restored after a restart records the files it read as
    # running it would: each found with a look of its own, 76,000
    # ``os.path.exists`` in the first cell after a restart on Windows.
    from cash.notebook.tracking_state import TrackingState
    from cash.notebook.upstream.cache_restore import CacheRestorer

    _cell(snap)
    state = TrackingState()
    restorer = CacheRestorer(None, state, None, None)
    looks.start()

    restorer._update_tracking_after_restore({"frame"}, {"file_dependencies": snap}, {})

    found = state.executed_file_deps["frame"]
    assert found == {p for p, entry in snap.items() if not entry.get("absent")}
    assert looks["exists"] == 0
    assert looks["stat"] + looks["lstat"] <= FOLDERS, file_watch.watcher().report()
