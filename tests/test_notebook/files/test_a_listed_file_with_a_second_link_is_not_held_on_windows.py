"""A file a listing found is not held when it has a second hard link.

On Windows the upstream check takes the stats of a large set of files from
folder listings (`file_dep_snapshot.stats_from_listings`), and a listing's
stat has no link count (``st_nlink`` 0). The folder watch kept such a stat
as that of a file with one link, so an edit made through the second link,
in a folder it does not watch (a dataset cache beside the project), went
unseen, and a cell below the read was served the old result. Nor could the
listing itself be trusted for such a file: NTFS keeps a size and times per
folder entry, and a write through the other link leaves this entry's as
they were. A listed file now takes one stat for its link count, once, and
one with a second link is stat-ed by itself each time.

Here a listing on Linux is made to look like a Windows one.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.upstream import statement_lineage
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

#: Past what is held at all (`statement_lineage._HOLD_MIN_FILES`).
FILES = 300


@pytest.fixture(autouse=True)
def _runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    monkeypatch.setattr(statement_lineage, "_HELD_FILE_KEYS", run_memo.HeldPaths(kept=8))
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_watch.reset()


class _ListedEntry:
    """A folder entry as a Windows listing gives it: no link count, and the
    size and times it had when the folder was first listed (NTFS keeps them
    per entry, and a write through another link leaves them)."""

    def __init__(self, entry: os.DirEntry, st: os.stat_result) -> None:
        self.name, self.path, self._entry, self._st = entry.name, entry.path, entry, st

    def is_symlink(self) -> bool:
        return self._entry.is_symlink()

    def is_dir(self, follow_symlinks: bool = True) -> bool:
        return self._entry.is_dir(follow_symlinks=follow_symlinks)

    def stat(self) -> os.stat_result:
        return self._st


class _Listing:
    def __init__(self, entries) -> None:
        self._entries = entries

    def __enter__(self):
        return self._entries

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture(autouse=True)
def _listed_as_on_windows(monkeypatch):
    """Listings on every system, each entry as a Windows listing gives it."""
    first: dict[str, os.stat_result] = {}
    real = os.scandir

    def scandir(directory="."):
        with real(directory) as it:
            entries = []
            for e in it:
                if e.path not in first:
                    st = e.stat()
                    extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
                    first[e.path] = os.stat_result((*st[:3], 0, *st[4:]), extra)
                entries.append(_ListedEntry(e, first[e.path]))
        return _Listing(entries)

    monkeypatch.setattr(file_dep_snapshot.os, "scandir", scandir)
    monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    monkeypatch.setattr(file_dep_snapshot, "_LISTED_LINKS", {})


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _photos(tmp_path, link: bool) -> list[str]:
    photos, cache = tmp_path / "photos", tmp_path / "cache"
    photos.mkdir()
    cache.mkdir()
    found = []
    for i in range(FILES):
        p = photos / f"p{i:03d}.pgm"
        if link:
            original = cache / p.name
            original.write_text(f"photo {i:03d}", encoding="utf-8")
            try:
                os.link(original, p)
            except (OSError, NotImplementedError):
                pytest.skip("no hard links here")
        else:
            p.write_text(f"photo {i:03d}", encoding="utf-8")
        found.append(str(p))
    return found


def _keys(paths: list[str]) -> dict:
    _present, keys = statement_lineage._present_file_keys(set(paths))
    _next_run()
    return dict(keys)


def test_an_edit_through_the_other_link_is_seen(tmp_path):
    linked = _photos(tmp_path, link=True)
    if file_watch.watcher() is None:
        pytest.skip("no folder watch")
    for _ in range(3):
        _keys(linked)
    assert not statement_lineage._HELD_FILE_KEYS._entries, "a linked file was held"

    other = tmp_path / "cache" / "p007.pgm"
    st = os.stat(other)
    other.write_text("photo 007 EDITED", encoding="utf-8")
    os.utime(other, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    assert _keys(linked)[linked[7]] == stat_key(os.stat(linked[7]))


def test_files_with_one_link_are_still_held(tmp_path):
    single = _photos(tmp_path, link=False)
    if file_watch.watcher() is None:
        pytest.skip("no folder watch")
    for _ in range(3):
        assert _keys(single) == {p: stat_key(os.stat(p)) for p in single}
    assert statement_lineage._HELD_FILE_KEYS._entries, "files with one link were not held"
