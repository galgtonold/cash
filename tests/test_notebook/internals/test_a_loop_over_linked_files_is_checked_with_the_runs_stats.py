"""The check after a loop over files the watch cannot vouch for takes the run's stats.

Photos with a second hard link (a dataset cache beside the project) are
stat-ed again before every cell: the watch keeps no stat of them. That one
stat a file each cell run is all a check needs. The first check after a loop
over values read from them took more (E5, 13,000 LFW photos, on a first run):

* the simulation of the loop stat-ed each of its files' two spellings once
  more, with an ``exists`` before each (`ControlSimulation._stats_of`): four
  stats a file the run had taken already;
* the file component of the loop's files, built when the loop ran, was built
  again from those unchanged stats: a real path, a version and a line of
  text a file (`run_memo.file_component_this_run`).

The simulation now takes the run's stats, and a component built in an earlier
run is taken again while every file's fresh stat is the one it was built
from. A file edited between two cell runs, or written by a cell, still gives
the cells below their new values.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.statement import file_deps
from cash.tracking import file_dep_snapshot, file_watch
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 200

WAIT = f"import time\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"

CELLS = [
    "import os\nnames = sorted(os.listdir('photos'))",
    WAIT + "faces = [open('photos/' + n).read() for n in names]",
    "lengths = []\nfor f in faces:\n    lengths.append(len(f))",
    "x = 1",
    "with open('photos/p007.pgm', 'w') as fh:\n    fh.write('photo 007, rewritten')",
    "again = [open('photos/' + n).read() for n in names]\nlengths2 = []\nfor a in again:\n    lengths2.append(len(a))",
    "total2 = sum(lengths2)",
    "total = sum(lengths) + x",
]

#: The cells whose run neither reads nor writes a photo: one stat a photo at most.
PLAIN = (2, 3, 6, 7)

NAMES = ("lengths", "x", "lengths2", "total2", "total")


@pytest.fixture(autouse=True)
def _own_watch(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    run_memo._SEEN_COMPONENTS.clear()
    yield
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    run_memo._SEEN_COMPONENTS.clear()


def _make(root) -> None:
    """Photos under ``root/photos``, each with a second link in ``root/cache``."""
    folder, cache = root / "photos", root / "cache"
    folder.mkdir()
    cache.mkdir()
    for i in range(FILES):
        original = cache / f"p{i:03d}.pgm"
        original.write_text(f"photo {i:03d}", encoding="utf-8")
        try:
            os.link(original, folder / f"p{i:03d}.pgm")
        except (OSError, NotImplementedError):
            pytest.skip("no hard links here")


@pytest.fixture
def photos(tmp_path, monkeypatch):
    """The photos, in the working directory."""
    _make(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def work(monkeypatch, photos) -> dict[str, int]:
    """Stats of a photo, under either link, and components built of them all."""
    counts = {"stats": 0, "components": 0}
    root = os.fspath(photos)

    def counting(real):
        def call(path, *args, **kwargs):
            where = os.fspath(path) if isinstance(path, (str, os.PathLike)) else ""
            if where.endswith(".pgm") and (where.startswith(("photos", "cache")) or where.startswith(root)):
                counts["stats"] += 1
            return real(path, *args, **kwargs)

        return call

    real_component = file_deps.compute_file_hash_component

    def component(files, *args, **kwargs):
        if len(files) >= FILES:
            counts["components"] += 1
        return real_component(files, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting(os.stat))
    monkeypatch.setattr(os, "lstat", counting(os.lstat))
    monkeypatch.setattr(file_deps, "compute_file_hash_component", component)
    return counts


def _run(magics, at: int) -> None:
    run_cash_cell(magics, CELLS[at], cells=CELLS)


def _plain(root, names=NAMES) -> dict:
    """What restart and Run All gives in plain Python, run in *root*."""
    here = os.getcwd()
    os.chdir(root)
    try:
        ns: dict = {}
        for cell in CELLS:
            exec(cell, ns)
    finally:
        os.chdir(here)
    return {name: ns[name] for name in names}


def _from_the_start(tmp_path_factory, names=NAMES) -> dict:
    """`_plain` over the photos as the test began."""
    root = tmp_path_factory.mktemp("plain")
    _make(root)
    return _plain(root, names)


def _ns(magics, names=NAMES) -> dict:
    return {name: magics.shell.user_ns[name] for name in names}


def test_a_cell_below_the_loop_stats_each_photo_once(cash_magics, work, tmp_path_factory):
    seen: dict[int, dict[str, int]] = {}
    for at in range(len(CELLS)):
        for key in work:
            work[key] = 0
        _run(cash_magics, at)
        seen[at] = dict(work)

    assert _ns(cash_magics) == _from_the_start(tmp_path_factory)
    for at in PLAIN:
        # The run's one stat a photo (and a few of the one a cell wrote, as
        # the checks of the cells above find it changed by the notebook);
        # the check after a loop took four more a photo.
        assert seen[at]["stats"] <= FILES + 10, (at, seen)
    # Built when each loop ran, from the stats of that run; not again by
    # the check of the cell below it, from the same stats.
    assert seen[3]["components"] == 0, seen
    assert seen[6]["components"] == 0, seen


def test_an_edit_between_cell_runs_reaches_the_cells_below(cash_magics, photos, tmp_path_factory):
    for at in range(len(CELLS)):
        _run(cash_magics, at)
    assert _ns(cash_magics) == _from_the_start(tmp_path_factory)
    before = _ns(cash_magics)

    # Through the cache's link: no folder the notebook read from reports it.
    target = photos / "cache" / "p123.pgm"
    st = os.stat(target)
    target.write_text("photo 123, edited in the cache", encoding="utf-8")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    _run(cash_magics, 6)
    _run(cash_magics, 7)
    assert _ns(cash_magics) == _plain(photos)
    assert _ns(cash_magics)["total"] != before["total"], "the edit went unseen"


def test_a_write_by_a_cell_reaches_the_cell_below_it(cash_magics, photos, tmp_path_factory):
    for at in range(4):
        _run(cash_magics, at)
    lengths = list(cash_magics.shell.user_ns["lengths"])

    _run(cash_magics, 4)  # rewrites photo 7
    _run(cash_magics, 5)
    _run(cash_magics, 6)

    names = ("lengths", "x", "lengths2", "total2")
    assert _ns(cash_magics, names) == _from_the_start(tmp_path_factory, names)
    assert cash_magics.shell.user_ns["lengths2"][7] != lengths[7], "the write went unseen"


def test_the_component_of_the_photos_is_built_again_only_when_one_changed(photos, work, monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    paths = {f"photos/p{i:03d}.pgm" for i in range(FILES)} | {
        str(photos / "photos" / f"p{i:03d}.pgm") for i in range(FILES)
    }

    def run() -> str:
        file_dep_snapshot.begin_file_state_epoch()
        try:
            return run_memo.file_component_this_run(paths)
        finally:
            file_dep_snapshot.end_file_state_epoch()
            run_memo._FILE_STATE_THIS_RUN.clear()

    def built_afresh() -> str:
        work["components"] -= 1  # not one of the runs'
        return file_deps.compute_file_hash_component(paths)

    first = run()
    assert [run() for _ in range(3)] == [first] * 3
    assert work["components"] == 1, work

    # Through the cache's link: the folder the photos are read from does not
    # report it.
    target = photos / "cache" / "p123.pgm"
    st = os.stat(target)
    target.write_text("photo 123, edited in the cache", encoding="utf-8")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    edited = run()
    assert edited != first
    assert edited == built_afresh()

    os.unlink(photos / "photos" / "p050.pgm")
    removed = run()
    assert removed not in (first, edited)
    assert removed == built_afresh()
    assert work["components"] == 3, work
