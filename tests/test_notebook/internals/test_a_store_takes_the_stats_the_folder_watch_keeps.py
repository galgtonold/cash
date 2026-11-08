"""A store below an unstored read takes the stats the folder watch keeps.

The records of a statement not stored are taken at the first store below
(`_PendingRecords`), in place of the stats its cell run took
(`run_memo.own_stats_this_run`). Where no entry above holds those files'
records, the check before that cell stat-ed none of them, and the store
stat-ed every file and folder read: 22,000 below a read of 13,000 photos.
The cells between had stat-ed them, and the folder watch keeps those
stats while their folders report no change (`FileWatch.current`): taken
for a folder, and for a file whose digest under that very stat is
remembered, so the stat and the digest describe the same bytes.

A file changed since is not taken so: the change reaches the watch, and
the store stats it by itself, which finds it moved since its read.
"""

from __future__ import annotations

import itertools
import os
import sys

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook import file_records
from cash.notebook.file_records import UNRESOLVED_PREFIX, entry_file_records
from cash.notebook.statement.store import StatementStore
from cash.tracking import file_watch
from cash.tracking.file_dep_snapshot import snapshot_is_fresh
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FILES = 300
FOLDERS = 100

WAIT = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
CELLS = [
    "import os, time\nimport photo_fetcher",
    # Not stored: it writes its cache beside the photos.
    "faces = photo_fetcher.fetch('people')\npositive = faces",
    "shape = len('camera')",
    f"negative = ({WAIT}, [len(x) for x in positive])[1]",
]

FETCHER = """
import os

def fetch(root):
    data = []
    for person in sorted(os.listdir(root)):
        for name in sorted(os.listdir(os.path.join(root, person))):
            with open(os.path.join(root, person, name)) as fh:
                data.append(fh.read())
    with open('fetch_cache.txt', 'w') as fh:
        fh.write(str(len(data)))
    return data
"""


@pytest.fixture(autouse=True)
def _watched(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    if file_watch.watcher() is None:
        pytest.skip("no folder watch here")
    yield
    file_watch.reset()


@pytest.fixture
def clean_backend(tmp_path):
    """RAM in front of disk, as a notebook's cash is: a restart reads the disk."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path / "cache"))])
    yield backend
    backend.clear()


@pytest.fixture
def people(tmp_path, monkeypatch):
    """Settled photos, a few per folder, read by relative path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "photo_fetcher", raising=False)
    (tmp_path / "photo_fetcher.py").write_text(FETCHER, encoding="utf-8")
    for i in range(FILES):
        folder = tmp_path / "people" / f"person{i % FOLDERS:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"p{i:04d}.txt"
        p.write_text(f"photo {i:04d}", encoding="utf-8")
        os.utime(p, (1_700_000_000, 1_700_000_000))
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


def _records(magics, name: str) -> dict:
    backend = magics._cash_instance.backend
    meta = backend.peek_metadata(magics.tracking_state.variable_sources[name])
    assert meta, f"{name} was not stored"
    return entry_file_records(meta, backend)


def test_the_store_takes_the_kept_stats_of_the_files_read_above(cash_magics, people, store_stats):
    for cell in CELLS[:-1]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    store_stats()

    run_cash_cell(cash_magics, CELLS[-1], cells=CELLS)
    taken = store_stats()

    assert cash_magics.shell.user_ns["negative"] == [len(f"photo {i:04d}") for i in range(FILES)]
    records = _records(cash_magics, "negative")
    photo = str(people / "person023" / "p0123.txt").replace("\\", "/")
    folder = str(people / "person023").replace("\\", "/")
    assert photo in records and folder in records, sorted(records)[:5]
    assert not [p for p in records if p.startswith(UNRESOLVED_PREFIX)]
    assert snapshot_is_fresh(records)[0]
    # Every file and folder read was stat-ed again by the store.
    assert taken <= (FILES + FOLDERS) // 10, f"the store stat-ed the files the watch keeps: {taken}"


EDIT = "with open(os.path.join('people', 'person023', 'p0123.txt'), 'w') as fh:\n    fh.write('photo 9999!')"
# Behind cash's back: another process writes the photo.
ELSEWHERE = (
    "import subprocess, sys\n"
    "done = subprocess.run([sys.executable, '-c', "
    "\"open('people/person023/p0123.txt', 'w').write('photo 9999!')\"]).returncode"
)


def _plain(cells: list[str]) -> list[int]:
    """What restart and Run All gives in plain Python, over the files as they are."""
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns["negative"]


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the variables,
    nothing read back remembered."""
    magics._cash_instance.backend.backends[1]._wait_for_all_writes()
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    file_records._OWN.clear()
    file_records._MERGED.clear()
    for name in list(magics.shell.user_ns):
        if name in {"os", "time", "photo_fetcher", "faces", "positive", "shape", "negative", "fh", "done"}:
            magics.shell.user_ns.pop(name)


@pytest.mark.parametrize(
    "where",
    [
        pytest.param("cell above", id="edited-between-the-cells"),
        pytest.param("same cell", id="written-by-the-storing-cell"),
        pytest.param("elsewhere", id="written-by-another-process-in-the-storing-cell"),
    ],
)
def test_a_photo_changed_after_the_read_is_not_recorded_as_read(cash_magics, clean_backend, people, where):
    if where == "cell above":
        cells = [*CELLS[:-1], EDIT, CELLS[-1]]
    else:
        cells = [*CELLS[:-1], f"{EDIT if where == 'same cell' else ELSEWHERE}\n{CELLS[-1]}"]
    as_read = _plain(CELLS)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    negative = cash_magics.shell.user_ns["negative"]
    now = _plain(CELLS)  # over the photos as they are now
    assert as_read != now
    # Built from the photo as the read found it, or as it is now where the
    # read ran again below the change: the entry is fresh only for the second.
    assert negative in (as_read, now)
    records = _records(cash_magics, "negative")
    photo = str(people / "person023" / "p0123.txt").replace("\\", "/")
    assert photo in records
    assert snapshot_is_fresh(records)[0] == (negative == now), (
        "the entry's records do not describe the photo the value was built from"
    )

    _restart(cash_magics)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["negative"] == now, "the change was not seen after a restart"
