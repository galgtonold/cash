"""A file in a watched folder is not stat-ed again at the next cell run.

Before each cell, every file a cached statement above read was stat-ed
again: 19,000 photos in 300 folders, seconds a cell on Windows, for a
cell that changes no file. The folders are watched (`file_watch`) and a
stat taken while its folder was watched stands until the folder reports a
change. A folder that cannot be watched, a lost event queue, a file with a
second link or a link are stat-ed as before.
"""

from __future__ import annotations

import os
import stat
import struct
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import stats_this_run
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.file_dep_snapshot import FreshnessMemo, snapshot_file_deps, snapshot_is_fresh

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

N = 70  # a snapshot of more files than share a verdict (VERDICT_MIN_FILES)


@pytest.fixture(autouse=True)
def _runs(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    # Each file by its own stat, on Windows too, where a crowded folder is listed.
    monkeypatch.setattr(run_memo, "stats_from_listings", lambda paths: {})
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


@pytest.fixture
def photos(tmp_path) -> list[str]:
    folder = tmp_path / "photos"
    folder.mkdir()
    paths = []
    for i in range(N):
        p = folder / f"img{i:03d}.jpg"
        p.write_bytes(bytes([i]) * 64)
        paths.append(str(p))
    return paths


@pytest.fixture
def stats_of(monkeypatch, tmp_path) -> list[str]:
    """Each ``stat`` or ``lstat`` of a file under *tmp_path*, folders left out."""
    taken: list[str] = []
    root = str(tmp_path)

    def counting(real):
        def call(path, *args, **kwargs):
            st = real(path, *args, **kwargs)
            spelled = os.fspath(path)
            if spelled.startswith(root) and not stat.S_ISDIR(st.st_mode):
                taken.append(spelled)
            return st

        return call

    monkeypatch.setattr(os, "stat", counting(os.stat))
    monkeypatch.setattr(os, "lstat", counting(os.lstat))
    return taken


def test_the_next_cell_run_takes_no_stat_of_an_unchanged_watched_file(photos, stats_of):
    first = stats_this_run(photos)
    assert len(stats_of) >= N  # positive control: the first run stats each

    _next_run()
    stats_of.clear()
    again = stats_this_run(photos)
    assert stats_of == []
    assert {p: again[p][1] for p in photos} == {p: first[p][1] for p in photos}


def test_a_changed_file_is_stat_ed_again_and_the_rest_are_not(photos, stats_of):
    stats_this_run(photos)
    with open(photos[5], "ab") as f:
        f.write(b"more")

    _next_run()
    stats_of.clear()
    again = stats_this_run(photos)
    assert photos[5] in stats_of
    assert again[photos[5]][1].st_size == 68


def test_a_folder_that_cannot_be_watched_is_stat_ed_every_run(photos, stats_of, monkeypatch):
    watch = file_watch.watcher()
    assert watch is not None
    monkeypatch.setattr(watch._backend, "add", lambda path: None)
    stats_this_run(photos)

    _next_run()
    stats_of.clear()
    stats_this_run(photos)
    assert sorted(set(stats_of)) == sorted(photos)


def test_lost_events_drop_every_kept_stat(photos, stats_of, monkeypatch):
    watch = file_watch.watcher()
    assert watch is not None
    stats_this_run(photos)
    real_poll = watch._backend.poll
    monkeypatch.setattr(watch._backend, "poll", lambda: [*real_poll(), (None, False)])

    _next_run()
    stats_of.clear()
    stats_this_run(photos)
    assert sorted(set(stats_of)) == sorted(photos)


def test_a_folder_reported_gone_drops_its_stats(photos, stats_of, monkeypatch):
    watch = file_watch.watcher()
    stats_this_run(photos)
    keys = list(watch._by_key)
    real_poll = watch._backend.poll
    monkeypatch.setattr(watch._backend, "poll", lambda: [*real_poll(), *((k, True) for k in keys)])

    _next_run()
    stats_of.clear()
    stats_this_run(photos)
    assert sorted(set(stats_of)) == sorted(photos)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inotify")
def test_an_inotify_overflow_reads_as_every_folder_changed():
    overflow = struct.pack("iIII", -1, 0x4000, 0, 0)
    modify = struct.pack("iIII", 3, 0x2, 0, 16) + b"img000.jpg".ljust(16, b"\0")
    gone = struct.pack("iIII", 4, 0x8000, 0, 0)
    assert file_watch.parse_inotify_events(modify + overflow + gone) == [(3, False), (None, False), (4, True)]
    assert file_watch.parse_inotify_events(modify[:20]) == [(None, False)]  # torn: trust nothing


def test_a_file_with_a_second_link_is_stat_ed_every_run(photos, stats_of, tmp_path):
    try:
        os.link(photos[0], tmp_path / "other-name.jpg")
    except OSError:
        pytest.skip("no hard links here")
    stats_this_run(photos)

    _next_run()
    stats_of.clear()
    stats_this_run(photos)
    assert stats_of and set(stats_of) <= {photos[0]}


def test_a_symlink_is_stat_ed_every_run(photos, stats_of, tmp_path):
    link = tmp_path / "photos" / "link.jpg"
    try:
        os.symlink(photos[1], link)
    except OSError:
        pytest.skip("cannot make a symlink here")
    paths = [*photos, str(link)]
    stats_this_run(paths)

    _next_run()
    stats_of.clear()
    stats_this_run(paths)
    assert stats_of and set(stats_of) <= {str(link)}


def test_a_snapshot_found_fresh_is_not_checked_again_while_its_folder_is_quiet(photos, monkeypatch):
    deps = snapshot_file_deps(set(photos))
    checked: list[str] = []
    real = file_dep_snapshot.dep_is_fresh

    def counting(path, *args, **kwargs):
        checked.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(file_dep_snapshot, "dep_is_fresh", counting)

    def check() -> bool:
        run = run_memo.file_state_this_run()
        return snapshot_is_fresh(deps, run["memo"])[0]

    assert check()
    assert len(checked) >= N  # positive control: the first check goes through each file

    _next_run()
    checked.clear()
    assert check()
    assert checked == []

    with open(photos[9], "ab") as f:
        f.write(b"more")
    _next_run()
    assert snapshot_is_fresh(deps, FreshnessMemo(verdicts=run_memo.RUN_VERDICTS))[0] is False
    assert check() is False


def test_a_file_component_is_built_once_while_its_folder_is_quiet(photos, monkeypatch):
    from cash.notebook.statement import file_deps

    built: list[int] = []
    real = file_deps.compute_file_hash_component

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(file_deps, "compute_file_hash_component", counting)
    paths = set(photos)
    first = run_memo.file_component_this_run(paths)
    assert built == [1]

    _next_run()
    assert run_memo.file_component_this_run(paths) == first
    assert built == [1]

    with open(photos[2], "ab") as f:
        f.write(b"more")
    _next_run()
    assert run_memo.file_component_this_run(paths) != first
    assert built == [1, 1]
