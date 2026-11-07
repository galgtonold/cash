"""The stores below a read whose entry holds no file records stat its files once.

A statement below a folder read inherits the read's files and records them
with its entry. When the read's entry holds no records (it was not stored:
uncacheable, refused, too cheap), each stored statement below took a stat of
every one of those files again at its store: below 13,000 photos read by
relative path, 42,000 stats per stored statement, 84,000 for a cell storing
two.

The records the read made of its files as it ran are kept for the session
(``TrackingState.records_as_run``) and taken once, at the first store below;
the first entry stored with them holds them on disk, and the entries further
below refer to its records rather than copy them. An entry stored so still
goes stale when an inherited file changes, also after a restart.
"""

from __future__ import annotations

import itertools
import os

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook import file_records
from cash.notebook.cache_status import CacheStatus
from cash.notebook.file_records import UNRESOLVED_PREFIX, entry_file_records
from cash.notebook.statement.store import StatementStore
from cash.tracking.file_dep_snapshot import snapshot_is_fresh
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 400

WAIT = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CELLS = [
    "import glob, time",
    # Not stored: its entry holds no records of the files.
    "X = [open(p).read() for p in sorted(glob.glob('photos/*.txt'))]  # @cash:no-cache",
    f"a = ({WAIT}, [len(x) for x in X])[1]",
    f"b = ({WAIT}, [x[:3] for x in X])[1]\nc = ({WAIT}, [x[-3:] for x in X])[1]",
    f"d = ({WAIT}, [x.upper() for x in X[:5]])[1]",
]
NAMES = ["glob", "time", "X", "a", "b", "c", "d"]
STORED = ["a", "b", "c", "d"]


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def clean_backend(cache_dir):
    """RAM in front of disk, as a notebook's cash is."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=cache_dir)])
    yield backend
    backend.clear()


@pytest.fixture
def photos(tmp_path, monkeypatch):
    """Files read by relative path, each with a second link in a cache beside it."""
    monkeypatch.chdir(tmp_path)
    folder, cache = tmp_path / "photos", tmp_path / "dataset_cache"
    folder.mkdir()
    cache.mkdir()
    for i in range(FILES):
        original = cache / f"p{i:04d}.txt"
        original.write_text(f"photo {i:04d}", encoding="utf-8")
        try:
            os.link(original, folder / f"p{i:04d}.txt")
        except (OSError, NotImplementedError):
            (folder / f"p{i:04d}.txt").write_text(f"photo {i:04d}", encoding="utf-8")
    return folder


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


def _run_all(magics) -> None:
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)


def _plain() -> dict:
    """What restart and Run All gives in plain Python."""
    ns: dict = {}
    for cell in CELLS:
        exec(cell, ns)
    return {name: ns[name] for name in STORED}


def _values(magics) -> dict:
    return {name: magics.shell.user_ns[name] for name in STORED}


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables,
    nothing read back remembered, every write landed."""
    magics._cash_instance.backend.backends[1]._wait_for_all_writes()
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    file_records._OWN.clear()
    file_records._MERGED.clear()
    for name in NAMES:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> dict[str, CacheStatus]:
    return {m["code"]: m["status"] for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")}


def test_the_stores_below_stat_the_inherited_files_once(cash_magics, photos, store_stats):
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    store_stats()

    taken = []
    for cell in CELLS[2:]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
        taken.append(store_stats())

    assert _values(cash_magics) == _plain()
    keys = [cash_magics.tracking_state.variable_sources[name] for name in STORED]
    assert all(cash_magics._cash_instance.backend.peek_metadata(key) for key in keys), "a statement was not stored"
    # Four stores below, each of which stat'ed every file before.
    assert sum(taken) <= FILES * 1.5, taken
    assert taken[1] <= FILES // 4, f"the cell storing two statements stat'ed the files again: {taken}"


def test_an_entry_below_goes_stale_when_an_inherited_file_changes(cash_magics, clean_backend, photos):
    _run_all(cash_magics)
    assert _values(cash_magics) == _plain()
    assert cash_magics.shell.user_ns["b"][123] == "pho"
    metas = {
        name: clean_backend.backends[1].peek_metadata(cash_magics.tracking_state.variable_sources[name])
        for name in STORED
    }
    assert all(metas.values()), "an entry below was not stored on disk"
    target = str(photos / "p0123.txt").replace("\\", "/")
    for meta in metas.values():
        records = entry_file_records(meta, clean_backend)
        assert not [p for p in records if p.startswith(UNRESOLVED_PREFIX)], records
        assert target in records
        assert snapshot_is_fresh(records)[0]

    # After a restart, unchanged: every entry below is served.
    _restart(cash_magics)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
        if cell is CELLS[3]:
            assert set(_statuses(cash_magics).values()) == {CacheStatus.RESTORED}, _statuses(cash_magics)
    assert _values(cash_magics) == _plain()

    # An inherited file changed after the store, same size: stale, also as
    # a later kernel reads the entries.
    path = photos / "p0123.txt"
    st = os.stat(path)
    path.write_text("PHOTO 0123", encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    _restart(cash_magics)
    for name, meta in metas.items():
        fresh, _ = snapshot_is_fresh(entry_file_records(meta, cash_magics._cash_instance.backend))
        assert not fresh, f"{name}'s entry is fresh after a file it was built from changed"
    _run_all(cash_magics)
    assert _values(cash_magics) == _plain()
    assert cash_magics.shell.user_ns["b"][123] == "PHO", "the edit was not seen"


def test_a_statement_too_cheap_to_store_below_takes_no_stat_of_the_files(cash_magics, photos, store_stats):
    """``positive_patches = faces.images`` below the unstored read stores
    no entry: the records of the read's files are not taken for it, which
    cost the cell that read 13,000 photos a stat of each."""
    for cell in CELLS[:2]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    store_stats()

    cells = [*CELLS[:2], "Y = X[:]", f"e = ({WAIT}, [len(y) for y in Y])[1]"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert store_stats() <= FILES // 4, "the store of a statement too cheap to keep stat'ed the files"
    assert not cash_magics._cash_instance.backend.peek_metadata(cash_magics.tracking_state.variable_sources["Y"])

    # The first entry stored below takes them, once.
    run_cash_cell(cash_magics, cells[3], cells=cells)
    assert store_stats() <= FILES * 1.5
    key = cash_magics.tracking_state.variable_sources["e"]
    records = entry_file_records(
        cash_magics._cash_instance.backend.peek_metadata(key), cash_magics._cash_instance.backend
    )
    assert str(photos / "p0123.txt").replace("\\", "/") in records
    assert snapshot_is_fresh(records)[0]
