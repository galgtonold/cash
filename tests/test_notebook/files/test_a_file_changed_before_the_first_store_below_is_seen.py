"""A file changed before the first entry stored below an unstored read is seen.

The first statement stored below a read whose entry holds no file records
records the read's files, and takes the stats the check before its cell
took where they say what a stat taken now would. A file changed after the
read, between the cells or by a statement of the storing cell itself, must
not be recorded as the value was built from it: the entry is never fresh
for it, and restart and Run All gives what plain Python gives over the
files as they are.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook import file_records
from cash.notebook.file_records import entry_file_records
from cash.tracking.file_dep_snapshot import snapshot_is_fresh
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 120
FOLDERS = 12

WAIT = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
HEAD = [
    "import os, time\nimport photo_fetcher",
    "def photos():\n"
    "    for person in sorted(os.listdir('people')):\n"
    "        for name in sorted(os.listdir(os.path.join('people', person))):\n"
    "            yield os.path.join('people', person, name)",
    # Stored: the check before each cell below stats the files it read.
    f"faces = ({WAIT}, [open(p).read() for p in photos()])[1]",
    # Not stored, as it writes a file: no entry holds its records of the files it read.
    "X = photo_fetcher.Fetcher().load(list(photos()))",
]
STORE = f"a = ({WAIT}, [x[-4:] for x in X])[1]"
EDIT = (
    "with open(os.path.join('people', 'person03', 'p0003.txt'), 'w') as fh:\n"
    "    fh.write('photo 9999')\n"
    "os.utime(os.path.join('people', 'person03', 'p0003.txt'), (time.time() + 5, time.time() + 5))"
)
NAMES = ["os", "time", "photo_fetcher", "photos", "faces", "X", "fh", "a"]

FETCHER = """
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
def clean_backend(tmp_path):
    """RAM in front of disk, as a notebook's cash is: a restart reads the disk."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path / "cache"))])
    yield backend
    backend.clear()


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


def _plain(cells: list[str]) -> list[str]:
    """What restart and Run All gives in plain Python, over the files as they are."""
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns["a"]


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


def _entry_of_a_is_fresh(magics) -> bool:
    backend = magics._cash_instance.backend
    meta = backend.peek_metadata(magics.tracking_state.variable_sources["a"])
    assert meta, "the statement below was not stored"
    records = entry_file_records(meta, backend)
    assert str(os.path.abspath(os.path.join("people", "person03", "p0003.txt"))).replace("\\", "/") in records
    return snapshot_is_fresh(records)[0]


@pytest.mark.parametrize(
    "where",
    [
        pytest.param("cell above", id="edited-between-the-cells"),
        pytest.param("same cell", id="written-by-the-storing-cell"),
    ],
)
def test_a_photo_changed_after_the_read_is_not_recorded_as_read(cash_magics, clean_backend, people, where):
    cells = [*HEAD, EDIT, STORE] if where == "cell above" else [*HEAD, f"{EDIT}\n{STORE}"]
    as_read = _plain([*HEAD, STORE])
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    a = cash_magics.shell.user_ns["a"]
    now = _plain([*HEAD, STORE])  # over the photos as they are now
    assert "9999" not in as_read and "9999" in now
    # Built from the photo as the read found it, or from the photo as it is
    # now where the read ran again below the change (a change between the
    # cells): the entry is fresh only for the second.
    assert a in (as_read, now)
    if where == "same cell":
        assert a == as_read
    assert _entry_of_a_is_fresh(cash_magics) == (a == now), (
        "the entry's records do not describe the photo the value was built from"
    )

    _restart(cash_magics)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["a"] == _plain(cells)
    assert "9999" in cash_magics.shell.user_ns["a"], "the change was not seen after a restart"


def test_an_unchanged_photo_is_recorded_fresh(cash_magics, clean_backend, people):
    """The positive control: nothing changed, the entry is fresh and served
    after a restart."""
    cells = [*HEAD, STORE]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert _entry_of_a_is_fresh(cash_magics)
    _restart(cash_magics)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["a"] == _plain(cells)
