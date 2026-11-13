"""A folder taken from a kept listing is not looked at again before each cell.

A notebook reading 13,000 photos by relative path records each of their
5,700 folders under both spellings. The check before each cell found the
relative spelling's stat in the kept listing of their parent
(`FileWatch.take_listed`), which kept the entry as the folder's stat
again: an ``lstat`` of each folder before every cell (`FileWatch.record`),
4,200 of the floor of every cell on Windows, and the folder's own stat,
with its file identity, put back by the listing's, which has none. A stat
the watch keeps already is now the answer.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import forget_file_state_this_run, stats_this_run
from cash.tracking import file_dep_snapshot, file_watch

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FOLDERS = 40


@pytest.fixture(autouse=True)
def _runs(monkeypatch):
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


@pytest.fixture(autouse=True)
def _listed_as_on_windows(monkeypatch):
    """Listings taken on every system, as on Windows: no link count and no
    file identity (``st_ino`` and ``st_dev`` 0). A folder's entry also keeps
    the times it had when first listed, as NTFS brings a parent's entry for
    a folder up to date only later (when the folder is next opened). And a
    folder has one link, as a stat on Windows says."""
    real = os.scandir

    class Entry:
        def __init__(self, e, st):
            self.name, self.path, self._e, self._st = e.name, e.path, e, st

        def is_symlink(self):
            return self._e.is_symlink()

        def is_dir(self, follow_symlinks=True):
            return self._e.is_dir(follow_symlinks=follow_symlinks)

        def stat(self):
            return self._st

    class Listing(list):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    first_listed: dict[str, os.stat_result] = {}

    def scandir(directory="."):
        out = Listing()
        with real(directory) as it:
            for e in it:
                st = e.stat()
                extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
                st = os.stat_result((st[0], 0, 0, 0, *st[4:]), extra)
                if e.is_dir():
                    st = first_listed.setdefault(e.path, st)
                out.append(Entry(e, st))
        return out

    monkeypatch.setattr(file_dep_snapshot.os, "scandir", scandir)
    monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    monkeypatch.setattr(file_dep_snapshot, "_LISTED_LINKS", {})
    links_of = file_dep_snapshot._links_of
    monkeypatch.setattr(
        file_dep_snapshot, "_links_of", lambda path, key: 1 if os.path.isdir(path) else links_of(path, key)
    )


@pytest.fixture
def people(tmp_path, monkeypatch) -> list[str]:
    """A folder per person, each holding two photos, read by relative path:
    every path recorded under both spellings."""
    monkeypatch.chdir(tmp_path)
    paths = []
    for i in range(FOLDERS):
        folder = tmp_path / "people" / f"p{i:02d}"
        folder.mkdir(parents=True)
        for name in ("a.jpg", "b.jpg"):
            (folder / name).write_bytes(b"x" * (i + 1))
        rel = f"people/p{i:02d}"
        paths += [rel, str(folder), f"{rel}/a.jpg", str(folder / "a.jpg"), f"{rel}/b.jpg", str(folder / "b.jpg")]
    return paths


@pytest.fixture
def lstats(monkeypatch) -> list[str]:
    taken: list[str] = []
    real = os.lstat

    def lstat(path, *args, **kwargs):
        taken.append(os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    return taken


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def test_the_folders_are_looked_at_once_not_before_each_cell(people, lstats):
    stats_this_run(people)
    first = len(lstats)
    assert first, "the first pass took no lstat: nothing is measured"

    later = []
    for _ in range(3):
        _next_run()
        lstats.clear()
        found = stats_this_run(people)
        later.append(len(lstats))
        for p in people:
            assert found[p][1] is not None and found[p][1].st_size == os.stat(p).st_size

    assert later == [0, 0, 0], f"each cell looked at the folders again: {later} lstats"
    watch = file_watch.watcher()
    folders = [p for p in people if os.path.isabs(p) and os.path.isdir(p)]
    assert [p for p in folders if watch.is_listed(p)] == [], "a folder's own stat was put back by a listing's"


def test_a_folder_changed_since_is_looked_at_again(people, lstats, tmp_path):
    """Positive control: the kept stat holds only while nothing changed."""
    stats_this_run(people)
    first_listed_mtime = os.stat(tmp_path / "people" / "p07").st_mtime_ns
    (tmp_path / "people" / "p07" / "c.jpg").write_bytes(b"new")
    _next_run()
    lstats.clear()
    found = stats_this_run(people)

    folder = str(tmp_path / "people" / "p07")
    assert os.stat(folder).st_mtime_ns != first_listed_mtime, "the write left the folder's time: nothing is measured"
    assert found[folder][1].st_mtime_ns == os.stat(folder).st_mtime_ns
    assert found["people/p07"][1].st_mtime_ns == os.stat(folder).st_mtime_ns
