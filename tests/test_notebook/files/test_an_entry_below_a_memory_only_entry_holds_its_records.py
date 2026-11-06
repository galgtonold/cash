"""An entry below one kept in memory only holds its file records itself.

An entry refers to the file records of the entry its input came from by
that entry's key and a token naming them (``cash.notebook.file_records``).
``faces = fetch_lfw_people(...)`` was kept in memory only, and after a
restart its key held other records or none: ``X_train`` (14 s) and
``grid.fit`` (61 s) below it found the reference unresolved and ran again,
"file changed: <cash: file records of ...>", with no file changed. Records
of an entry that a restart does not find are copied, as before references;
those of an entry on disk are still referred to. A real change to the file
is still a change.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.notebook import file_records
from cash.notebook.file_records import UNRESOLVED_PREFIX, entry_file_records
from cash.tracking.file_dep_snapshot import snapshot_is_fresh
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = "import time\nimport numpy as np"
# 40 MB in a fraction of a second: not worth the disk, kept in memory only.
BIG = (
    f"raw = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), "
    "np.tile(np.frombuffer(open('data.bin', 'rb').read(), dtype=np.uint8), 40_000))[1]"
)
# 1 KB: on disk.
SMALL = (
    f"raw = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), np.frombuffer(open('data.bin', 'rb').read(), dtype=np.uint8))[1]"
)
BELOW = f"s = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), raw[::1000] + 1)[1]"
FURTHER = f"t = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), s * 2)[1]"


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def clean_backend(cache_dir):
    # RAM in front of disk, as a notebook's cash is.
    backend = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=cache_dir)])
    yield backend
    backend.clear()


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x01" * 1000)
    return path


def _key(cash_magics, name: str) -> str:
    return cash_magics.tracking_state.variable_sources[name]


def _restarted(cache_dir, clean_backend) -> TieredBackend:
    """What a restarted kernel reads the cache through: an empty RAM tier
    in front of the same folder, every write landed."""
    clean_backend.backends[1]._wait_for_all_writes()
    # A new process: nothing read back is remembered.
    file_records._OWN.clear()
    file_records._MERGED.clear()
    return TieredBackend([InMemoryBackend(), FileBackend(cache_dir=cache_dir)])


def _run(cash_magics, *cells: str) -> None:
    for cell in (SETUP, *cells):
        run_cash_cell(cash_magics, cell)


def _resolves(meta, backend, path) -> None:
    records = entry_file_records(meta, backend)
    assert not [p for p in records if p.startswith(UNRESOLVED_PREFIX)], records
    assert str(path).replace("\\", "/") in records, records
    fresh, stale = snapshot_is_fresh(records)
    assert fresh, stale


@pytest.mark.parametrize(
    "lost",
    [
        pytest.param("dropped", id="its key holds nothing after the restart"),
        pytest.param("records", id="its key holds other records"),
        pytest.param("metadata", id="its key holds metadata without records"),
    ],
)
def test_the_entry_below_holds_while_the_file_is_as_read(cash_magics, clean_backend, cache_dir, data, lost):
    _run(cash_magics, BIG, BELOW, FURTHER)
    raw_key = _key(cash_magics, "raw")
    assert clean_backend.peek_metadata(raw_key)["storage"] == ["RAM"]  # the positive control
    below = clean_backend.backends[1].peek_metadata(_key(cash_magics, "s"))
    further = clean_backend.backends[1].peek_metadata(_key(cash_magics, "t"))
    assert below is not None and further is not None, "the entries below were not stored on disk"

    backend = _restarted(cache_dir, clean_backend)
    disk = backend.backends[1]
    if lost == "dropped":
        disk.delete(raw_key)
    elif lost == "records":
        other = data.parent / "other.bin"
        other.write_bytes(b"\x02")
        disk.set_metadata_only(raw_key, {"key": raw_key, "file_dependencies": {str(other): {}}, "files_token": "x"})
    else:
        disk.delete(raw_key)
        disk.set_metadata_only(raw_key, {"key": raw_key, "skipped_reason": "refused"})
    for meta in (below, further):
        _resolves(meta, backend, data)
    # Its own records: none referred to the reader's entry.
    assert raw_key not in [ref[0] for ref in below.get("file_refs") or ()], "it refers to an entry kept in RAM"
    # The entry further below refers to the one below, which is on disk.
    assert [ref[0] for ref in further.get("file_refs") or ()] == [_key(cash_magics, "s")]

    data.write_bytes(b"\x02" * 1000)
    for meta in (below, further):
        fresh, _ = snapshot_is_fresh(entry_file_records(meta, backend))
        assert not fresh, "a change to the file read is a change"


def test_an_entry_below_one_on_disk_still_refers_to_its_records(cash_magics, clean_backend, cache_dir, data):
    _run(cash_magics, SMALL, BELOW, FURTHER)
    raw_key = _key(cash_magics, "raw")
    above = clean_backend.backends[1].peek_metadata(raw_key)
    assert above and not above.get("metadata_only"), "the read was not stored on disk"  # the positive control
    below = clean_backend.peek_metadata(_key(cash_magics, "s"))
    assert below["file_refs"] == [[raw_key, above["files_token"]]]
    assert not below.get("file_dependencies"), "the records were copied"

    backend = _restarted(cache_dir, clean_backend)
    _resolves(backend.peek_metadata(_key(cash_magics, "s")), backend, data)
    _resolves(backend.peek_metadata(_key(cash_magics, "t")), backend, data)
