"""A folder above a watched one can be renamed, moved or removed while it is watched.

Watching a folder held a handle on it on Windows, and Windows then refused
to rename the folder above it (``PermissionError: Access is denied``): a
user could not rename a project folder in Explorer while a kernel ran.
Cash must never block a user's folder operations; on Windows the watch now
holds a handle only on the volume's root. After the rename, the moved
files' kept stats are dropped.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

from cash.notebook import run_memo
from cash.tracking import file_watch

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")


@pytest.fixture(autouse=True)
def _own_watch(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    yield
    file_watch.reset()


def _watched(tmp_path):
    """A file in ``w/sub``, its stat kept by the watch."""
    sub = tmp_path / "w" / "sub"
    sub.mkdir(parents=True)
    target = sub / "a.txt"
    target.write_text("first", encoding="utf-8")
    watch = file_watch.watcher()
    assert watch is not None
    token = watch.before_stat(str(target))
    st = os.lstat(target)
    watch.record(token, str(target), st)
    assert watch.current(str(target)) is st  # positive control: it is watched
    return watch, target


def test_the_folder_above_a_watched_one_can_be_renamed(tmp_path):
    watch, target = _watched(tmp_path)

    os.rename(tmp_path / "w", tmp_path / "w_renamed")  # raised PermissionError on Windows

    watch.drain(check_dirs=True)
    assert watch.current(str(target)) is None


def test_a_watched_folder_can_be_renamed(tmp_path):
    watch, target = _watched(tmp_path)

    os.rename(target.parent, tmp_path / "w" / "sub_renamed")

    watch.drain(check_dirs=True)
    assert watch.current(str(target)) is None


def test_the_folder_above_a_watched_one_can_be_removed_and_made_again(tmp_path):
    watch, target = _watched(tmp_path)

    shutil.rmtree(tmp_path / "w")
    target.parent.mkdir(parents=True)
    target.write_text("again", encoding="utf-8")

    watch.drain(check_dirs=True)
    assert watch.current(str(target)) is None


def test_an_edit_after_the_folder_is_renamed_back_is_seen(tmp_path):
    watch, target = _watched(tmp_path)
    os.rename(tmp_path / "w", tmp_path / "w_renamed")
    os.rename(tmp_path / "w_renamed", tmp_path / "w")
    watch.drain(check_dirs=True)
    if watch.current(str(target)) is None:  # the rename was taken as a change: watch it again
        token = watch.before_stat(str(target))
        watch.record(token, str(target), os.lstat(target))
    assert watch.current(str(target)) is not None

    target.write_text("SECOND", encoding="utf-8")
    watch.drain()
    assert watch.current(str(target)) is None
