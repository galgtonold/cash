"""``%cash_status`` says whether folders are watched for changes to the
files cells read (``file_watch``; ``CASH_FILE_WATCH``, on by default), how
many, and if not, why: the switch set to 0, no watch on this platform, or a
watch that could not start."""

from __future__ import annotations

import sys

import pytest

from cash.notebook import run_memo
from cash.tracking import file_watch
from tests._cell_driver import run_cash_cell


@pytest.fixture(autouse=True)
def _own_watch(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    yield
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()


def _line(magics) -> str:
    return magics.cash_status("dict")["file_watch"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inotify")
def test_on_it_counts_the_folders_watched(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("hello", encoding="utf-8")
    assert _line(cash_magics) == "on (inotify)"
    for _ in range(3):
        run_cash_cell(cash_magics, f"text = open({str(tmp_path / 'data.txt')!r}).read()")
    line = _line(cash_magics)
    assert line.startswith("on (inotify, "), line
    assert " watched)" in line, line
    assert file_watch.watcher().watched() >= 1


def test_switched_off_it_names_the_switch(cash_magics, monkeypatch):
    monkeypatch.setenv("CASH_FILE_WATCH", "0")
    assert _line(cash_magics) == "off (CASH_FILE_WATCH=0; unset it to watch folders for file changes)"


def test_where_no_folder_can_be_watched_it_says_so(cash_magics, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(file_watch.sys, "platform", "darwin")
        assert file_watch._platform_name() == "macOS"
    monkeypatch.setattr(file_watch, "_backend", lambda: None)
    monkeypatch.setattr(file_watch, "_platform_name", lambda: "macOS")
    assert _line(cash_magics) == (
        "off: not supported on macOS; every file a cached statement read is checked before each cell"
    )


def test_a_watch_that_cannot_start_says_why(cash_magics, monkeypatch):
    def refused():
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(file_watch, "_backend", refused)
    assert _line(cash_magics) == (
        "off: could not start (OSError: [Errno 24] Too many open files); "
        "every file a cached statement read is checked before each cell"
    )
