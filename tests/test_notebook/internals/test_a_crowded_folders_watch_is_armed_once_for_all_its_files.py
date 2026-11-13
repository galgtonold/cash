"""A crowded folder's watch is armed once for all its files, and a quiet
folder is not listed again for other files of it.

Every file of a crowded folder shares its folder's change state, yet each
pass of the cell run's stats asked the folder watch for a token per file
before any listing (`FileWatch.before_stat`), and again for each file it
then stat-ed: two tokens per file, a quarter of a plain cell's time below a
read of 13,000 photos on Windows, where no listing is taken elsewhere and
the first set was not used at all. A listing now takes one token per
folder, just before the folder is listed (`FileWatch.before_listing`).

And a later cell asking for other files of a crowded folder listed all of
it again; the watch now keeps the listing while the folder is quiet.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import forget_file_state_this_run, stats_this_run
from cash.tracking import file_dep_snapshot, file_watch

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FILES = 100


@pytest.fixture(autouse=True)
def _runs(monkeypatch, tmp_path):
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


@pytest.fixture
def calls(monkeypatch) -> dict[str, int]:
    """How often the watch was asked for a token, per file and per folder."""
    counted = {"before_stat": 0, "before_listing": 0}
    for name in counted:
        real = getattr(file_watch.FileWatch, name)

        def call(self, path, _real=real, _name=name):
            counted[_name] += 1
            return _real(self, path)

        monkeypatch.setattr(file_watch.FileWatch, name, call)
    return counted


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


@pytest.fixture
def listings(monkeypatch) -> list[str]:
    """Listings taken on every system, as on Windows (no link count); each
    folder listed is noted."""
    taken: list[str] = []
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

    def scandir(directory="."):
        taken.append(os.fspath(directory))
        out = Listing()
        with real(directory) as it:
            for e in it:
                st = e.stat()
                extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
                out.append(Entry(e, os.stat_result((*st[:3], 0, *st[4:]), extra)))
        return out

    monkeypatch.setattr(file_dep_snapshot.os, "scandir", scandir)
    monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    monkeypatch.setattr(file_dep_snapshot, "_LISTED_LINKS", {})
    return taken


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def test_a_pass_with_no_listing_asks_one_token_per_file_it_stats(photos, calls, monkeypatch):
    monkeypatch.setattr(run_memo, "stats_from_listings", lambda paths: {})  # as on Linux
    stats_this_run(photos)
    assert calls["before_stat"] == FILES  # one for each stat, none beforehand

    _next_run()
    calls["before_stat"] = 0
    stats_this_run(photos)
    assert calls["before_stat"] == 0  # every stat held


def test_a_listed_folder_takes_one_token_for_all_its_files(photos, calls, listings):
    found = stats_this_run(photos)
    assert listings == [os.path.dirname(photos[0])]
    assert calls["before_stat"] == 0
    assert calls["before_listing"] == 1
    assert all(found[p][1].st_size == 10 for p in photos)

    _next_run()
    assert all(file_watch.watcher().current(p) is not None for p in photos)


def test_a_quiet_folder_is_not_listed_again_for_other_files(photos, listings):
    stats_this_run(photos[:30])
    assert len(listings) == 1

    _next_run()
    stats_this_run(photos[30:60])
    assert len(listings) == 1

    with open(os.path.join(os.path.dirname(photos[0]), "new.jpg"), "wb"):
        pass
    _next_run()
    stats_this_run(photos[60:90])
    assert len(listings) == 2  # positive control: a changed folder is listed again
