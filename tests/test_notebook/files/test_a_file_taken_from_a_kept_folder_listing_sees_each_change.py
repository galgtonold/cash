"""A file whose stat comes from a folder listing kept from an earlier cell
sees each change made since.

On Windows the check before a cell takes the stats of a crowded folder's
files from one listing of it, and the folder watch keeps that listing
whole while the folder reports no change: a later cell asking for other
files of the folder takes their entries from it instead of listing the
folder again. An edit, a new file, a write by a cell, or an edit through a
second hard link must each still be seen there, as a restart and Run All
would see it.

Here a listing on Linux is made to look like a Windows one: no link count,
and a file with a second link keeps the size and times its entry had when
first listed (NTFS keeps them per folder entry).
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import forget_file_state_this_run, stats_this_run
from cash.tracking import file_dep_snapshot, file_watch

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FILES = 120


@pytest.fixture(autouse=True)
def _runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    forget_file_state_this_run()
    file_watch.reset()
    if file_watch.watcher() is None:
        pytest.skip("no folder watch here")
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    forget_file_state_this_run()
    file_watch.reset()


class _ListedEntry:
    def __init__(self, entry: os.DirEntry, st: os.stat_result) -> None:
        self.name, self.path, self._entry, self._st = entry.name, entry.path, entry, st

    def is_symlink(self) -> bool:
        return self._entry.is_symlink()

    def stat(self) -> os.stat_result:
        return self._st


class _Listing:
    def __init__(self, entries) -> None:
        self._entries = entries

    def __enter__(self):
        return self._entries

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture
def listings(monkeypatch) -> list[str]:
    """Listings on every system, as Windows gives them; each folder listed
    is noted."""
    taken: list[str] = []
    frozen: dict[str, os.stat_result] = {}
    real = os.scandir

    def scandir(directory="."):
        taken.append(os.fspath(directory))
        with real(directory) as it:
            entries = []
            for e in it:
                st = e.stat(follow_symlinks=False)
                if st.st_nlink > 1:
                    st = frozen.setdefault(e.path, st)
                extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
                entries.append(_ListedEntry(e, os.stat_result((*st[:3], 0, *st[4:]), extra)))
        return _Listing(entries)

    monkeypatch.setattr(file_dep_snapshot.os, "scandir", scandir)
    monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    monkeypatch.setattr(file_dep_snapshot, "_LISTED_LINKS", {})
    return taken


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


@pytest.fixture
def photos(tmp_path) -> list[str]:
    folder = tmp_path / "photos"
    folder.mkdir()
    paths = []
    for i in range(FILES):
        p = folder / f"img{i:03d}.jpg"
        p.write_bytes(b"x" * 10)
        paths.append(str(p))
    return paths


def _sizes(paths: list[str]) -> dict[str, int]:
    found = stats_this_run(paths)
    return {p: found[p][1].st_size for p in paths}


def test_a_later_cell_takes_other_files_from_the_kept_listing(photos, listings):
    assert set(_sizes(photos[:40]).values()) == {10}
    assert listings == [os.path.dirname(photos[0])]  # positive control: listed once

    _next_run()
    assert set(_sizes(photos[40:80]).values()) == {10}
    assert listings == [os.path.dirname(photos[0])], "the quiet folder was listed again"


def test_an_edit_between_cells_is_seen(photos, listings):
    _sizes(photos[:40])
    with open(photos[60], "ab") as f:
        f.write(b"more")

    _next_run()
    assert _sizes(photos[40:80])[photos[60]] == 14
    assert len(listings) == 2  # the folder reported the edit: listed again


def test_a_new_file_between_cells_is_seen(photos, listings, tmp_path):
    _sizes(photos[:40])
    new = tmp_path / "photos" / "new.jpg"
    new.write_bytes(b"y" * 3)

    _next_run()
    assert _sizes([*photos[40:80], str(new)])[str(new)] == 3


def test_a_write_by_a_cell_in_the_same_run_is_seen(photos, listings):
    _sizes(photos[:40])
    with open(photos[70], "ab") as f:  # a statement of this cell writes it
        f.write(b"more")
    forget_file_state_this_run()  # as the runtime does after a write

    assert _sizes(photos[40:80])[photos[70]] == 14


def test_an_edit_through_a_second_hard_link_is_seen(photos, listings, tmp_path):
    elsewhere = tmp_path / "cache"
    elsewhere.mkdir()
    other = elsewhere / "img050.jpg"
    try:
        os.link(photos[50], other)
    except (OSError, NotImplementedError):
        pytest.skip("no hard links here")
    _sizes(photos[:40])
    with open(other, "ab") as f:  # the folder of photos reports nothing
        f.write(b"through the other link")

    _next_run()
    assert _sizes(photos[40:80])[photos[50]] == os.stat(photos[50]).st_size == 32
