"""An entry below an unstored read records its files as read, not as written later.

A statement whose entry holds no records of the files it read (not stored)
leaves them to the first entry stored below, which records them as the
read found them. They are taken only once an entry needs them, not for each
statement below too cheap to store. A statement in between that writes one
of those files must not have its write recorded in their place: the entry
below, computed from what the read found, would be served after a restart
that reads the new content. They are taken before that write.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook import file_records
from cash.notebook.file_records import entry_file_records
from cash.tracking.file_dep_snapshot import file_dep_is_fresh
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 40

CELLS = [
    "import glob, time",
    # Not stored: its entry holds no records of the files.
    "X = [open(p).read() for p in sorted(glob.glob('docs/*.txt'))]  # @cash:no-cache",
    "Y = X",
    "with open('docs/d07.txt', 'w') as fh:\n    fh.write('EDITED')",
    f"e = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), [y.upper() for y in Y])[1]",
]
NAMES = ["glob", "time", "X", "Y", "fh", "e"]


@pytest.fixture
def clean_backend(tmp_path):
    """RAM in front of disk, as a notebook's cash is: a restart reads the disk."""
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path / "cache"))])
    yield backend
    backend.clear()


@pytest.fixture
def docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(FILES):
        (folder / f"d{i:02d}.txt").write_text(f"doc {i:02d}", encoding="utf-8")
    return folder


def _plain() -> list[str]:
    """What restart and Run All gives in plain Python, over the files as they are."""
    ns: dict = {}
    for cell in CELLS:
        exec(cell, ns)
    return ns["e"]


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


def test_a_write_between_the_read_and_the_entry_below_is_not_recorded_as_read(cash_magics, clean_backend, docs):
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    # Computed from the file as the read found it.
    assert cash_magics.shell.user_ns["e"][7] == "DOC 07"
    key = cash_magics.tracking_state.variable_sources["e"]
    meta = clean_backend.backends[1].peek_metadata(key)
    assert meta, "the entry below was not stored on disk"
    records = entry_file_records(meta, clean_backend)
    written = str(docs / "d07.txt").replace("\\", "/")
    kept = str(docs / "d08.txt").replace("\\", "/")
    assert written in records and kept in records, sorted(records)[:5]
    assert file_dep_is_fresh(kept, records[kept])[0], "a file left as read is not fresh"
    for spelling in (written, "docs/d07.txt"):
        assert not file_dep_is_fresh(spelling, records[spelling])[0], f"{spelling}: recorded as written below the read"

    # A restart reads the file as written: the entry below is not served.
    _restart(cash_magics)
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    assert cash_magics.shell.user_ns["e"] == _plain()
    assert cash_magics.shell.user_ns["e"][7] == "EDITED"
