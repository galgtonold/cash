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

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.tracking import file_dep_snapshot
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
    a stat of the file itself is noted for."""
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

    class Listing(list):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    def scandir(directory="."):
        out = Listing()
        with real(directory) as it:
            for e in it:
                st = e.stat()
                extra = {f: getattr(st, f) for f in dir(st) if f.startswith("st_") and f.endswith("_ns")}
                out.append(Entry(e, os.stat_result((*st[:3], 0, *st[4:]), extra)))
        return out

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


@pytest.fixture
def photo_stats(monkeypatch, photos):
    """The ``stat`` calls of a photo, by path."""
    taken: list[str] = []
    real = os.stat

    def stat(path, *args, **kwargs):
        p = os.fspath(path) if not isinstance(path, int) else ""
        if isinstance(p, str) and os.path.basename(p).startswith("p") and p.endswith(".txt"):
            taken.append(p)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    return taken


def test_the_cells_below_a_read_take_no_stat_for_a_link_count(cash_magics, photos, photo_stats):
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    assert len(cash_magics.shell.user_ns["faces"]) == FILES

    below = []
    for cell in CELLS[2:]:
        photo_stats.clear()
        run_cash_cell(cash_magics, cell, cells=CELLS)
        below.append(len(photo_stats))

    assert cash_magics.shell.user_ns["n"] == FILES
    assert sum(below) <= FILES // 10, f"the cells below took a stat of each photo again: {below}"


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
