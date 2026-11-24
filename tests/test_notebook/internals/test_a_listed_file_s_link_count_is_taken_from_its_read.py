"""A listed file's link count is taken from the stat its read took.

On Windows a folder listing gives no link count, so a file found by one is
kept as its stat only once a stat of the file itself said it has a single
link (`file_dep_snapshot.note_links`). That count came from the snapshot
taken as a read's statement ran; since the records of a statement not
stored are taken only at a store below (`_PendingRecords`), the first
listing after the read took a stat of every file and folder it had read:
18,000 stats in the cell below a read of 13,000 photos. The read's own
``lstat`` of each (`realpath_of_read_this_run`) tells the count already.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import traceback

import pytest

from cash.notebook import run_memo
from cash.notebook.statement import store
from cash.notebook.statement.rebuild_cost import RebuildCostLedger
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key
from tests._cell_driver import run_cash_cell

FILES = 60

CELLS = [
    "import os\nimport photo_fetcher",
    # Not stored: it writes its cache beside the photos.
    "faces = photo_fetcher.fetch('photos')\npositive = faces",
    "shape = len('camera')",
    "n = len(positive)",
]

FETCHER = """
import os

def fetch(folder):
    data = []
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name)) as fh:
            data.append(fh.read().upper())
    with open('fetch_cache.txt', 'w') as fh:
        fh.write(str(len(data)))
    return data
"""


@pytest.fixture(autouse=True)
def _listed_as_on_windows(monkeypatch):
    """Listings taken on every system, as on Windows: no link count, which
    a stat of the file itself is noted for, and no file identity
    (``st_ino`` and ``st_dev`` 0, as ``DirEntry.stat`` gives there)."""
    real = os.scandir

    class Entry:
        def __init__(self, e, st):
            self.name, self.path, self._e, self._st = e.name, e.path, e, st

        def is_symlink(self):
            return self._e.is_symlink()

        def is_dir(self, **kwargs):
            return self._e.is_dir(**kwargs)

        def is_file(self, **kwargs):
            return self._e.is_file(**kwargs)

        def stat(self, **kwargs):
            return self._st

    class Listing:
        """An iterator, as ``os.scandir`` gives: ``os.walk`` on Python 3.10
        calls ``next`` on it (the patch reaches every ``os.scandir``)."""

        def __init__(self, entries):
            self._entries = iter(entries)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._entries)

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def scandir(directory="."):
        out = []
        with real(directory) as it:
            for e in it:
                st = e.stat()
                extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
                out.append(Entry(e, os.stat_result((st[0], 0, 0, 0, *st[4:]), extra)))
        return Listing(out)

    monkeypatch.setattr(file_dep_snapshot.os, "scandir", scandir)
    monkeypatch.setattr(run_memo, "stats_from_listings", file_dep_snapshot.listing_stats)
    monkeypatch.setattr(file_dep_snapshot, "stats_from_listings", file_dep_snapshot.listing_stats)
    monkeypatch.setattr(file_dep_snapshot, "_NOTE_LINKS", True)
    monkeypatch.setattr(file_dep_snapshot, "_LISTED_LINKS", {})


@pytest.fixture
def photos(tmp_path, monkeypatch):
    """Settled photos in one crowded folder, read by relative path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "photo_fetcher", raising=False)
    (tmp_path / "photo_fetcher.py").write_text(FETCHER, encoding="utf-8")
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(FILES):
        p = folder / f"p{i:03d}.txt"
        p.write_text(f"photo {i:03d}", encoding="utf-8")
        os.utime(p, (1_700_000_000, 1_700_000_000))
    return folder


class _Stats(list):
    """The paths stat-ed, and which thread and code of cash took each: what
    a failure on a CI machine, which cannot be stepped through, says."""

    def __init__(self) -> None:
        super().__init__()
        self.by: collections.Counter = collections.Counter()

    def clear(self) -> None:
        super().clear()
        self.by.clear()


@pytest.fixture
def photo_stats(monkeypatch, photos):
    """The ``stat`` calls of a photo, by path."""
    taken = _Stats()
    real = os.stat

    def stat(path, *args, **kwargs):
        p = os.fspath(path) if not isinstance(path, int) else ""
        if isinstance(p, str) and os.path.basename(p).startswith("p") and p.endswith(".txt"):
            taken.append(p)
            frames = traceback.extract_stack(limit=12)[:-1]
            mine = [f"{os.path.basename(f.filename)}:{f.name}" for f in frames if f"{os.sep}cash{os.sep}" in f.filename]
            taken.by[(threading.current_thread().name, " < ".join(reversed(mine[-4:])))] += 1
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    return taken


@pytest.mark.parametrize("stored_below", [False, True], ids=["not-stored-below", "stored-below"])
def test_the_cells_below_a_read_take_no_stat_for_a_link_count(
    cash_magics, photos, photo_stats, monkeypatch, stored_below
):
    """Whether ``n = len(positive)`` is stored hangs on how long the read
    took: it is stored when what is not on disk behind ``positive`` took
    `RebuildCostLedger.COSTLY_INPUTS_S` (0.1 s), as the read of 60 files
    did on a loaded Windows runner. Stored, its entry takes over the read's
    records (`_RecordsAsRun`): the read, not cacheable (it writes a file),
    took no digest and no stat at its end, and the run holds only listed
    stats of the photos, with no file identity on Windows; each photo was
    stat-ed again (60 stats), where the stat its read took has the digest
    the read's lineage remembered under it. Both ways pinned here."""
    monkeypatch.setattr(RebuildCostLedger, "COSTLY_INPUTS_S", 1e-9 if stored_below else float("inf"))
    recorded = []  # the photos each records taken name
    snapshot = store._snapshot_with_inherited

    def records(files, *args, **kwargs):
        recorded.append(sum(1 for f in files if os.path.basename(f).startswith("p")))
        return snapshot(files, *args, **kwargs)

    monkeypatch.setattr(store, "_snapshot_with_inherited", records)
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    assert len(cash_magics.shell.user_ns["faces"]) == FILES

    below = []
    by = []
    recorded.clear()
    for cell in CELLS[2:]:
        photo_stats.clear()
        run_cash_cell(cash_magics, cell, cells=CELLS)
        below.append(len(photo_stats))
        by.append(dict(photo_stats.by))

    assert cash_magics.shell.user_ns["n"] == FILES
    # The way it is pinned is the way it went: stored, the read's records were taken.
    assert any(recorded) is stored_below, recorded
    watch = file_watch.running()
    assert sum(below) <= FILES // 10, (
        f"the cells below took a stat of each photo again: {below}; taken by {by}; "
        f"folder watch: {watch.report() if watch is not None else None}"
    )


def test_a_file_with_a_second_link_is_still_stat_ed_by_itself(photos):
    """What the read told is the count: a file read with a second link is
    never taken from a listing (its listed size and times can lag a write
    through the other link)."""
    linked = photos / "p007.txt"
    try:
        os.link(linked, photos.parent / "elsewhere.txt")
    except (OSError, NotImplementedError):
        pytest.skip("no hard links here")
    paths = [str(p) for p in sorted(photos.iterdir())]
    file_dep_snapshot.begin_file_state_epoch()
    try:
        for p in paths:  # as each read resolves its path
            file_dep_snapshot.realpath_of_read_this_run(p)
        listed = file_dep_snapshot.listing_stats(paths)
    finally:
        file_dep_snapshot.end_file_state_epoch()
    assert str(linked) not in listed
    assert len(listed) == FILES - 1


def test_records_taken_by_the_read_s_stat_describe_the_file_as_read(tmp_path):
    """Records taken a while after the read, by the stat the read took
    (``read_lstats``), are of the file as read, digest and identity alike:
    a file written since is not vouched for as it is now."""
    photo = tmp_path / "p000.txt"
    photo.write_text("photo as read", encoding="utf-8")
    os.utime(photo, (1_700_000_000, 1_700_000_000))
    path = str(photo)
    read = os.lstat(path)
    as_read_digest = file_dep_snapshot.file_content_hash(path, read.st_size, read)  # remembered, as a lineage does
    photo.write_text("photo written since the read", encoding="utf-8")

    snapshot = file_dep_snapshot.snapshot_file_deps({path}, as_read={path: stat_key(read)}, read_lstats={path: read})

    entry = snapshot[path]
    assert (entry["size"], entry["mtime_ns"], entry["ino"]) == (read.st_size, read.st_mtime_ns, read.st_ino)
    assert entry["hash"] == as_read_digest
    assert file_dep_snapshot.file_dep_is_fresh(path, entry)[0] is False
