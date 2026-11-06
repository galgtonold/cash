"""What the upstream check holds of many files read still sees each change.

Below a loop over thousands of files, three things the check works out
from every path read are kept across cells while the folder watch vouches
for the files (`run_memo.HeldPaths`), and a folder that reports a change
has only its own paths looked at again:

* the files of a statement's outputs found where they were recorded, and
  their stat keys (`statement_lineage._present_file_keys`);
* each read path's comparable spellings, and which are folders
  (`FileWriterScheduler._read_path_index`);
* whether a writer's file is one of those its inputs were read from
  (`FileWriterScheduler._held_resolves_to`).

Each test changes a file in a way the answer must follow, and compares it
with the answer worked out from every path, as without the hold.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

from cash.notebook import run_memo
from cash.notebook.upstream import statement_lineage
from cash.notebook.upstream.file_writers import FileWriterScheduler, any_resolves_to
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

#: Past what is held at all (`_HOLD_MIN_FILES`), half by a relative path.
FOLDERS, FILES = 3, 120


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


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _settled() -> None:
    """Take the watch's reports of what the test changed."""
    file_watch.watcher().drain()


@pytest.fixture
def paths(tmp_path) -> list[str]:
    found = []
    for d in range(FOLDERS):
        folder = tmp_path / f"set{d}"
        folder.mkdir()
        for i in range(FILES):
            p = folder / f"f{i:03d}.txt"
            p.write_text(f"file {d} {i:03d}", encoding="utf-8")
            found.append(f"set{d}/f{i:03d}.txt" if i % 2 else str(p))
    found.append(str(tmp_path / "set0" / "later.txt"))
    return found


def _can_link(tmp_path) -> bool:
    try:
        os.symlink(tmp_path / "nowhere", tmp_path / "probe-link")
    except (OSError, NotImplementedError):
        return False
    os.unlink(tmp_path / "probe-link")
    return True


def _edit(p) -> None:
    st = os.stat(p)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(" EDITED")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


# -- the files of a statement's outputs ----------------------------------------


def _keys(paths: list[str]) -> dict:
    """One cell's keys of *paths*, then the next cell run begins."""
    present, keys = statement_lineage._present_file_keys(set(paths))
    assert present == frozenset(keys)
    _next_run()
    return dict(keys)


def _fresh_keys(paths: list[str]) -> dict:
    return {p: stat_key(os.stat(p)) for p in paths if os.path.exists(p)}


@pytest.fixture
def held_keys(paths) -> dict:
    first = _keys(paths)
    assert _keys(paths) == first == _fresh_keys(paths)
    assert statement_lineage._HELD_FILE_KEYS._entries, "nothing was held: the tests below would prove nothing"
    return first


@pytest.mark.parametrize("change", ["edit", "delete", "appear", "save beside"])
def test_the_files_of_a_statement_follow_each_change(paths, held_keys, tmp_path, change):
    if change == "edit":
        _edit(tmp_path / "set1" / "f003.txt")
    elif change == "delete":
        (tmp_path / "set2" / "f004.txt").unlink()
    elif change == "appear":
        (tmp_path / "set0" / "later.txt").write_text("here now", encoding="utf-8")
    else:
        (tmp_path / "set1" / "notes.ipynb").write_text("{}", encoding="utf-8")
    _settled()

    after = _keys(paths)

    assert after == _fresh_keys(paths)
    assert (after == held_keys) is (change == "save beside")
    # And again, once held anew.
    assert _keys(paths) == after


@pytest.fixture(params=["stat-ed", "listed"])
def stats(request, monkeypatch):
    """How the files of a crowded folder are stat-ed: each by itself, or by
    a listing of the folder as on Windows (`stats_from_listings`), whose
    entry for a link would give the stat of what it points to."""
    if request.param == "listed":
        monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    return request.param


def test_the_files_of_a_statement_follow_a_file_replaced_by_a_link(stats, paths, held_keys, tmp_path):
    if not _can_link(tmp_path):
        pytest.skip("this system makes no symbolic links")
    target = tmp_path / "elsewhere.txt"
    target.write_text("another file, longer than the rest", encoding="utf-8")
    p = tmp_path / "set2" / "f006.txt"
    p.unlink()
    p.symlink_to(target)
    _settled()

    assert _keys(paths) == _fresh_keys(paths) != held_keys
    _edit(target)  # a link's target lies where no watch vouches for it
    _settled()
    assert _keys(paths) == _fresh_keys(paths)


# -- the spellings of the paths read -------------------------------------------


def _scheduler() -> FileWriterScheduler:
    return FileWriterScheduler(MagicMock(), MagicMock(), MagicMock())


def _comparable(index) -> tuple:
    forms, dirs, places = index
    return set(forms), sorted(dirs), sorted(places)


def _index(scheduler: FileWriterScheduler, paths: list[str]) -> tuple:
    """One cell's index of *paths* (each cell asks with a new set), then the next run."""
    index = _comparable(scheduler._read_path_index(set(paths)))
    _next_run()
    return index


def _fresh_index(paths: list[str]) -> tuple:
    index = _comparable(_scheduler()._build_read_index(set(paths)))
    _next_run()
    return index


@pytest.mark.parametrize("change", ["file to folder", "delete", "appear", "save beside"])
def test_the_index_of_paths_read_follows_each_change(paths, tmp_path, change):
    scheduler = _scheduler()
    first = _index(scheduler, paths)
    assert _index(scheduler, paths) == first
    assert scheduler._held_reads._entries, "nothing was held: the test would prove nothing"

    if change == "file to folder":
        p = tmp_path / "set1" / "f004.txt"
        p.unlink()
        p.mkdir()
    elif change == "delete":
        (tmp_path / "set2" / "f004.txt").unlink()
    elif change == "appear":
        (tmp_path / "set0" / "later.txt").write_text("here now", encoding="utf-8")
    else:
        (tmp_path / "set1" / "notes.ipynb").write_text("{}", encoding="utf-8")
    _settled()

    after = _index(scheduler, paths)
    assert after == _fresh_index(paths)
    if change == "file to folder":
        assert after != first
    assert _index(scheduler, paths) == after


# -- whether a writer's file is among those read -------------------------------


def _resolves(scheduler: FileWriterScheduler, paths: list[str], targets: frozenset[str]) -> bool:
    answer = scheduler._held_resolves_to(set(paths), targets)
    _next_run()
    return answer


def test_a_read_file_that_becomes_a_link_to_the_written_one_is_seen(paths, tmp_path):
    if not _can_link(tmp_path):
        pytest.skip("this system makes no symbolic links")
    written = tmp_path / "movie_data.csv"
    written.write_text("review,sentiment\n", encoding="utf-8")
    targets = frozenset({os.path.realpath(written)})
    scheduler = _scheduler()
    assert not _resolves(scheduler, paths, targets)
    assert not _resolves(scheduler, paths, targets)
    assert scheduler._held_sources._entries, "nothing was held: the test would prove nothing"

    p = tmp_path / "set1" / "f007.txt"
    p.unlink()
    p.symlink_to(written)
    _settled()
    assert _resolves(scheduler, paths, targets) is any_resolves_to(paths, targets) is True

    p.unlink()
    p.write_text("a review again", encoding="utf-8")
    _settled()
    assert _resolves(scheduler, paths, targets) is any_resolves_to(paths, targets) is False
    assert not _resolves(scheduler, paths, targets)


def test_a_written_file_among_those_read_is_found_each_run(paths, tmp_path):
    scheduler = _scheduler()
    targets = frozenset({os.path.realpath(tmp_path / "set2" / "f010.txt")})
    for _ in range(3):
        assert _resolves(scheduler, paths, targets) is True
    other = frozenset({os.path.realpath(tmp_path / "set2" / "other.txt")})
    assert _resolves(scheduler, paths, other) is False
    (tmp_path / "set2" / "notes.ipynb").write_text("{}", encoding="utf-8")
    _settled()
    assert _resolves(scheduler, paths, other) is False
    assert _resolves(scheduler, paths, targets) is True
