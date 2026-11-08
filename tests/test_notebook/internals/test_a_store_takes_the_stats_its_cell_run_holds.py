"""A statement stored takes the stats its cell run already holds of its files.

The first statement stored below a read whose entry holds no file records
records the read's files (``TrackingState.records_as_run``). The check
before its cell had just stat-ed each of them, and their folders, to
validate the entries above; the store stat-ed every one again: below a
read of 13,000 photos in 5,700 folders, 19,000 stats more for the cell.

It takes the run's stat where that says what one taken now would: a
folder, or a file whose digest from the read is still its own. Not a
stat from a folder listing (on Windows, a listed file with a second hard
link has no link count and can predate a write through the other link),
and none at all once a statement of the cell wrote files. What the
entry then records is pinned in
``tests/test_notebook/files/test_a_file_changed_before_the_first_store_below_is_seen.py``.
"""

from __future__ import annotations

import itertools
import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.file_records import entry_file_records
from cash.notebook.statement.store import StatementStore
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_file_deps, snapshot_is_fresh
from cash.tracking.read_classification import stat_key
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 300
FOLDERS = 30

WAIT = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CELLS = [
    "import os, time\nimport photo_fetcher",
    "def photos():\n"
    "    for person in sorted(os.listdir('people')):\n"
    "        for name in sorted(os.listdir(os.path.join('people', person))):\n"
    "            yield os.path.join('people', person, name)",
    # Stored: the check before each cell below stats the files it read.
    f"faces = ({WAIT}, [open(p).read() for p in photos()])[1]",
    # Not stored, as it writes a file: no entry holds its records of the files it read.
    "X = photo_fetcher.Fetcher().load(list(photos()))",
    f"a = ({WAIT}, [len(x) for x in X])[1]",
]

# Reads the photos and writes beside them, as a dataset fetcher writes its
# cache: the statement calling it is not stored, which is found as it ran,
# so its reads were hashed (``FileAccessTracker.known_reads``).
FETCHER = """
import os

class Fetcher:
    def load(self, paths):
        data = []
        for p in paths:
            with open(p) as fh:
                data.append(fh.read().upper())
        with open('summary.txt', 'w') as fh:
            fh.write(str(len(data)))
        return data
"""


@pytest.fixture
def people(tmp_path, monkeypatch):
    """Photos in a folder each, read by relative path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "photo_fetcher", raising=False)
    (tmp_path / "photo_fetcher.py").write_text(FETCHER, encoding="utf-8")
    for i in range(FILES):
        folder = tmp_path / "people" / f"person{i % FOLDERS:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"p{i:04d}.txt").write_text(f"photo {i:04d}", encoding="utf-8")
    return tmp_path / "people"


@pytest.fixture
def store_stats(monkeypatch):
    """How many stats the stores took: ``store_stats()`` reads and resets."""
    counter = [itertools.count()]
    inside = [False]
    real_stat, real_lstat, real_save = os.stat, os.lstat, StatementStore.save

    def stat(*args, **kwargs):
        if inside[0]:
            next(counter[0])
        return real_stat(*args, **kwargs)

    def lstat(*args, **kwargs):
        if inside[0]:
            next(counter[0])
        return real_lstat(*args, **kwargs)

    def save(self, *args, **kwargs):
        inside[0] = True
        try:
            return real_save(self, *args, **kwargs)
        finally:
            inside[0] = False

    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(StatementStore, "save", save)

    def taken() -> int:
        n = next(counter[0])
        counter[0] = itertools.count()
        return n

    return taken


def test_the_first_store_below_an_unstored_read_takes_the_runs_stats(cash_magics, people, store_stats):
    for cell in CELLS[:-1]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    store_stats()

    run_cash_cell(cash_magics, CELLS[-1], cells=CELLS)
    taken = store_stats()

    assert cash_magics.shell.user_ns["a"] == [len(f"photo {i:04d}") for i in range(FILES)]
    backend = cash_magics._cash_instance.backend
    meta = backend.peek_metadata(cash_magics.tracking_state.variable_sources["a"])
    assert meta, "the statement below was not stored"
    records = entry_file_records(meta, backend)
    photo = str(people / "person03" / "p0123.txt").replace("\\", "/")
    folder = str(people / "person03").replace("\\", "/")
    assert photo in records and folder in records, sorted(records)[:5]
    assert snapshot_is_fresh(records)[0]
    # Each file and folder was stat-ed by the check before the cell: the
    # store stat-ed all of them again.
    assert taken <= (FILES + FOLDERS) // 4, f"the store stat-ed the files the cell run had stat-ed: {taken}"


def test_the_runs_stats_handed_on_leave_out_listed_ones_and_end_with_a_write(tmp_path, monkeypatch):
    """A stat from a folder listing is never handed on, and none is after
    a statement of the cell wrote files."""
    monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    crowded = tmp_path / "crowded"
    crowded.mkdir()
    listed = []
    for i in range(file_dep_snapshot.LISTING_MIN_FILES + 4):
        (crowded / f"f{i:02d}.txt").write_text(str(i), encoding="utf-8")
        listed.append(str(crowded / f"f{i:02d}.txt"))
    lone = tmp_path / "lone.txt"
    lone.write_text("lone", encoding="utf-8")
    file_dep_snapshot.begin_file_state_epoch()
    try:
        assert run_memo.own_stats_this_run() is None
        run_memo.stats_this_run([str(lone), *listed])
        handed = run_memo.own_stats_this_run()
        assert handed is not None
        assert handed.get(str(lone)) is not None, "a stat of the file itself is handed on"
        assert run_memo.stats_this_run(listed)[listed[0]][1] is not None
        assert [p for p in listed if handed.get(p) is not None] == [], "a listed stat was handed on"

        run_memo.forget_file_state_this_run()
        assert run_memo.own_stats_this_run() is None, "a stat taken before a write was handed on"
    finally:
        file_dep_snapshot.end_file_state_epoch()
    assert run_memo.own_stats_this_run() is None


def test_a_snapshot_takes_a_stat_handed_to_it_only_where_no_digest_is_taken(tmp_path):
    """A folder's stat handed over is taken, as is a file's whose digest
    from the read goes with it. Any other file is stat-ed and hashed by
    itself: an older stat beside a newer digest would vouch for content
    the value was not built from."""
    f = tmp_path / "f.txt"
    f.write_text("one", encoding="utf-8")
    folder, path = str(tmp_path), str(f)
    old, old_folder = os.stat(path), os.stat(folder)
    f.write_text("two", encoding="utf-8")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 5_000_000_000))
    (tmp_path / "new.txt").write_text("new", encoding="utf-8")
    os.utime(folder, ns=(old_folder.st_atime_ns, old_folder.st_mtime_ns + 5_000_000_000))
    now = os.stat(path)

    snap = snapshot_file_deps({path, folder}, taken={path: old, folder: old_folder})
    assert snap[folder]["mtime_ns"] == old_folder.st_mtime_ns, "the folder's stat handed over was not taken"
    assert snap[path]["mtime_ns"] == now.st_mtime_ns, "a file hashed now took an older stat"
    assert snap[path]["hash"] == file_dep_snapshot.file_content_hash(path)

    known = {path: (stat_key(old), "digest-as-read")}
    snap = snapshot_file_deps({path}, known=known, taken={path: old})
    assert (snap[path]["mtime_ns"], snap[path]["hash"]) == (old.st_mtime_ns, "digest-as-read")
