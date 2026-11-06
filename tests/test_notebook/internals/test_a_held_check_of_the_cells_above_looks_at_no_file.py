"""The check of the cells above costs what changed, not what they read.

Before each cell the upstream check asks whether each cached cell's files
still have the stats it recorded. Below a cell that read 25,000 files every
trivial cell went through each of them in Python, ~0.2 s a cell on Linux and
more on Windows. Under the folder watch the verdict is now held across cells
(`run_memo.stat_keys_as_recorded`): a later check looks at no file of a
folder that reported nothing, and only at the files of one that did.
"""

from __future__ import annotations

import collections
import os
import struct
import sys
import time

import pytest

from cash.notebook import run_memo
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.file_dep_snapshot import LISTING_MIN_FILES
from cash.tracking.file_watch import _FolderIndex, _Volume
from cash.tracking.read_classification import stat_key

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FOLDERS, FILES = 4, 50


@pytest.fixture(autouse=True)
def _runs(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._HELD_KEYS.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._HELD_KEYS.clear()
    file_watch.reset()


def _ask(recorded: dict) -> bool:
    answer = run_memo.stat_keys_as_recorded(recorded)
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()
    return answer


@pytest.fixture
def recorded(tmp_path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    deps = {}
    for d in range(FOLDERS):
        folder = tmp_path / f"set{d}"
        folder.mkdir()
        for i in range(FILES):
            p = folder / f"f{i:02d}.txt"
            p.write_text(f"file {d} {i:02d}", encoding="utf-8")
            deps[f"set{d}/f{i:02d}.txt" if i % 2 else str(p)] = stat_key(os.stat(p))
    return deps


@pytest.fixture
def looks(monkeypatch, tmp_path) -> collections.Counter:
    """Each stat and lstat of a path under *tmp_path*, and each path the
    watch is asked about (``current``)."""
    seen: collections.Counter = collections.Counter()
    root = str(tmp_path)

    def counting(kind, real):
        def call(path, *args, **kwargs):
            if os.fspath(path).startswith(root) or not os.path.isabs(os.fspath(path)):
                seen[kind] += 1
                seen.paths.append((kind, os.fspath(path)))  # type: ignore[attr-defined]
            return real(path, *args, **kwargs)

        return call

    def start():
        monkeypatch.setattr(os, "stat", counting("stat", os.stat))
        monkeypatch.setattr(os, "lstat", counting("lstat", os.lstat))
        watch = file_watch.watcher()
        monkeypatch.setattr(watch, "current", counting("current", watch.current))

    seen.start = start  # type: ignore[attr-defined]
    seen.paths = []  # type: ignore[attr-defined]
    return seen


def test_a_later_check_looks_at_no_file(recorded, looks):
    assert _ask(recorded) and _ask(recorded)
    looks.start()

    assert _ask(recorded)

    # Each watched folder's own stat before the run (`FileWatch.drain`), no file.
    assert looks["stat"] <= FOLDERS, looks
    assert looks["lstat"] == 0, looks
    assert looks["current"] == 0, looks


def _listed(paths) -> dict:
    """`file_dep_snapshot.stats_from_listings` as on Windows, where a
    crowded folder's files are stat-ed by listing it: a relative path's
    folder is listed by its relative spelling."""
    return file_dep_snapshot.listing_stats(paths)


def test_a_later_check_looks_at_no_listed_file(recorded, looks, monkeypatch):
    """Files stat-ed by a listing are held too, a relative path's by its
    absolute spelling: kept by the relative one, which names no watched
    folder, every check before a cell on Windows listed them again."""
    assert len(recorded) // (2 * FOLDERS) >= LISTING_MIN_FILES, "no folder is listed: the case is not reached"
    monkeypatch.setattr(run_memo, "stats_from_listings", _listed)
    assert _ask(recorded) and _ask(recorded)
    looks.start()

    assert _ask(recorded)

    assert looks["stat"] <= FOLDERS, looks
    assert looks["lstat"] == 0, looks
    assert looks["current"] == 0, looks


def _reports_settled(quiet: float = 1.0, timeout: float = 8.0) -> None:
    """Wait until the watch takes no new report. Windows may report a write
    again later (its last-write time, when the file's cache is flushed): a
    report after the check below rightly has the folder's files checked
    again. Taken reports change folders' versions only, so a held verdict
    still sees the change."""
    watch = file_watch.watcher()
    deadline = time.monotonic() + timeout
    last, quiet_since = None, time.monotonic()
    while time.monotonic() < deadline:
        watch.drain()
        if watch.changes != last:
            last, quiet_since = watch.changes, time.monotonic()
        elif time.monotonic() - quiet_since >= quiet:
            return
        time.sleep(0.1)


def test_a_change_in_one_folder_looks_again_only_at_its_files(recorded, looks, tmp_path):
    assert _ask(recorded) and _ask(recorded)
    (tmp_path / "set2" / "notes.ipynb").write_text("{}", encoding="utf-8")  # a save beside the files
    _reports_settled()
    looks.start()

    assert _ask(recorded)
    assert looks["stat"] + looks["lstat"] <= FOLDERS + 2 * FILES, looks

    looks.clear()
    looks.paths.clear()  # type: ignore[attr-defined]
    assert _ask(recorded)
    # Which paths were looked at, so a failure on a platform says where.
    assert looks["stat"] <= FOLDERS and looks["lstat"] == 0 and looks["current"] == 0, (
        looks,
        looks.paths[:12],
        file_watch.watcher().report(),
    )


class _Kernel:
    """kernel32 as a volume's watch calls it, with no thread: `report` ends
    the pending read with the names given."""

    def __init__(self) -> None:
        self.ended: bytes | None = None

    def report(self, *records: tuple[int, str]) -> None:
        out = b""
        for at, (action, name) in enumerate(records):
            encoded = name.encode("utf-16-le")
            record = struct.pack("<II", action, len(encoded)) + encoded
            record += b"\0" * (-(len(record) + 4) % 4)
            out += struct.pack("<I", len(record) + 4 if at < len(records) - 1 else 0) + record
        self.ended = out

    def new_event(self):
        return "event"

    def new_read(self):
        return object(), object()

    def issue(self, handle, overlapped, buffer, event):
        return True

    def result(self, handle, overlapped):
        if self.ended is None:
            return False, 0, 996  # ERROR_IO_INCOMPLETE
        n = len(self.ended)
        return True, n, 0

    def data(self, buffer, n):
        data, self.ended = self.ended, None
        return data[:n]

    def cancel(self, handle, overlapped):
        self.ended = b""

    def close(self, handle):
        pass


class _AsOnWindows:
    """The folder watch's backend as on Windows (`file_watch._VolumeChanges`):
    one watch over the whole volume, whose reported names say which folders
    changed; the test makes the reports."""

    def __init__(self) -> None:
        self.kernel = _Kernel()
        self.volume = _Volume(self.kernel, os.sep, "root")
        self.volume.pending = True

    def add(self, path: str):
        key = _FolderIndex.spelled(path)
        self.volume.index.add(key, key)
        return key

    def remove(self, key) -> None:
        self.volume.index.remove(key)

    def poll(self):
        keys, gone, _dead = self.volume.poll()
        return [(key, True) for key in gone] + [(key, False) for key in keys]

    def close(self) -> None:
        self.volume.stop()

    def report(self, *records: tuple[int, os.PathLike]) -> None:
        self.kernel.report(*((action, _FolderIndex.spelled(os.fspath(p))) for action, p in records))


def test_a_folder_reported_modified_after_its_listing_is_not_looked_at_again(recorded, looks, tmp_path, monkeypatch):
    """Windows reports a folder modified when its parent's entry for it is
    brought up to date, which may be only when the folder is next opened:
    by the listing the check took of it after one of its files changed.
    That report said nothing changed below the folder, but every file in
    it was checked once more before the cell after."""
    backend = _AsOnWindows()
    monkeypatch.setattr(file_watch, "_backend", lambda: backend)
    file_watch.reset()
    monkeypatch.setattr(run_memo, "stats_from_listings", _listed)
    assert _ask(recorded) and _ask(recorded)
    set2 = tmp_path / "set2"
    (set2 / "notes.ipynb").write_text("{}", encoding="utf-8")
    backend.report((1, set2 / "notes.ipynb"), (3, set2 / "notes.ipynb"))
    looks.start()

    assert _ask(recorded)
    assert 0 < looks["stat"] + looks["lstat"] + looks["current"], "the changed folder's files were not checked"

    backend.report((3, set2))  # the listing above opened it
    looks.clear()
    looks.paths.clear()  # type: ignore[attr-defined]
    assert _ask(recorded)
    assert looks["stat"] <= FOLDERS and looks["lstat"] == 0 and looks["current"] == 0, (looks, looks.paths[:12])
