"""A change to a file in a watched folder is seen by the next cell.

Before each cell, a file a cached statement read is taken as unchanged while
its folder's change notifications report nothing (`file_watch`), instead of
being stat-ed again. Each test changes a file in a way the watch must report,
or must not be trusted for, and checks the next cell reads the file anew.
They pass without the watch too: the file was stat-ed before every cell.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.cache_status import CacheStatus
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


def _statuses(magics, cell: str) -> list:
    captured: list[list[dict]] = []
    real_render = magics.badges.render

    def capture(metrics, **kw):
        captured.append(list(metrics))
        return real_render(metrics, **kw)

    magics.badges.render = capture
    try:
        run_cash_cell(magics, cell)
    finally:
        magics.badges.render = real_render
    return [m["status"] for m in captured[-1]] if captured else []


def _cached(magics, cell: str, ns_name: str, expected) -> None:
    """Run *cell* until it is restored, as a positive control that a cached
    read is what the change below must invalidate."""
    run_cash_cell(magics, cell)
    assert _statuses(magics, cell)[-1] == CacheStatus.RESTORED
    assert magics.shell.user_ns[ns_name] == expected


@pytest.fixture
def photos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(20):
        p = folder / f"img{i:02d}.txt"
        p.write_text(f"photo {i:02d}", encoding="utf-8")
        # Settled long ago: a check takes the file as unchanged by its stat
        # alone, without reading it, so a stat trusted wrongly serves stale.
        os.utime(p, ns=(SETTLED_NS, SETTLED_NS))
    return folder


SETTLED_NS = 1_600_000_000 * 10**9


READ = "data = open({path!r}).read()"


def test_an_edit_to_a_watched_file_is_seen_at_the_next_cell(cash_magics, photos):
    target = photos / "img07.txt"
    cell = READ.format(path=str(target))
    _cached(cash_magics, cell, "data", "photo 07")

    target.write_text("PHOTO 07", encoding="utf-8")  # same size, no pause
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["data"] == "PHOTO 07"


def test_an_edit_made_right_after_the_files_stat_is_seen_at_the_next_cell(cash_magics, photos, monkeypatch):
    """The watch is armed before the stat it vouches for: an edit landing
    between the stat and the watch would otherwise be missed for good."""
    target = photos / "img03.txt"
    cell = READ.format(path=str(target))
    _cached(cash_magics, cell, "data", "photo 03")
    file_watch.reset()  # the next check starts the watch, and stats the file
    real_lstat = os.lstat
    edited: list[bool] = []

    def lstat_then_edit(path, *args, **kwargs):
        st = real_lstat(path, *args, **kwargs)
        if not edited and os.fspath(path) == str(target):
            edited.append(True)
            with open(target, "w", encoding="utf-8") as f:
                f.write("PHOTO 03")
        return st

    monkeypatch.setattr(os, "lstat", lstat_then_edit)
    run_cash_cell(cash_magics, cell)  # may still restore: the edit came after its check
    monkeypatch.setattr(os, "lstat", real_lstat)
    assert edited, "the check before the cell took no lstat of the file"

    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["data"] == "PHOTO 03"


def test_a_file_renamed_over_a_watched_file_is_seen(cash_magics, photos):
    target = photos / "img05.txt"
    cell = READ.format(path=str(target))
    _cached(cash_magics, cell, "data", "photo 05")

    replacement = photos.parent / "new.txt"
    replacement.write_text("other 05", encoding="utf-8")
    os.replace(replacement, target)
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["data"] == "other 05"


def test_a_watched_file_deleted_and_written_again_is_seen(cash_magics, photos):
    target = photos / "img06.txt"
    cell = READ.format(path=str(target))
    _cached(cash_magics, cell, "data", "photo 06")

    target.unlink()
    target.write_text("again 06", encoding="utf-8")
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["data"] == "again 06"


def test_a_new_file_in_a_listed_folder_is_seen(cash_magics, photos):
    cell = f"names = sorted(os.listdir({str(photos)!r}))"
    run_cash_cell(cash_magics, "import os")
    _cached(cash_magics, cell, "names", sorted(os.listdir(photos)))

    (photos / "img99.txt").write_text("photo 99", encoding="utf-8")
    run_cash_cell(cash_magics, cell)
    assert "img99.txt" in cash_magics.shell.user_ns["names"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_an_edit_to_a_symlinked_files_target_is_seen(cash_magics, photos, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = elsewhere / "real.txt"
    real.write_text("real one", encoding="utf-8")
    link = photos / "link.txt"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("cannot make a symlink here")
    cell = READ.format(path=str(link))
    _cached(cash_magics, cell, "data", "real one")

    real.write_text("REAL ONE", encoding="utf-8")
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["data"] == "REAL ONE"


def test_an_edit_in_a_folder_that_cannot_be_watched_is_seen(cash_magics, photos, monkeypatch):
    target = photos / "img08.txt"
    cell = READ.format(path=str(target))
    watch = file_watch.watcher()
    if watch is not None:
        monkeypatch.setattr(watch._backend, "usable", lambda path: False)
    _cached(cash_magics, cell, "data", "photo 08")
    assert watch is None or watch.watched() == 0

    target.write_text("PHOTO 08", encoding="utf-8")
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["data"] == "PHOTO 08"
