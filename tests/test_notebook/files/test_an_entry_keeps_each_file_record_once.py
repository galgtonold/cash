"""A statement's entry keeps each record of a file it depends on once.

Below a read of 15,000 files every statement's entry held a record of each,
copied from the entry its input came from. A trivial cell wrote 5 MB, and a
restart read it back to decide to run the cell again.

* An entry refers to the records of the entry its input came from, by its key
  and the token naming them, rather than copying them
  (``cash.notebook.file_records``); a reference that no longer names the
  records stored there is never fresh.
"""

from __future__ import annotations

import os
import uuid

import pytest

from cash.notebook.file_records import entry_file_records
from cash.tracking.file_dep_snapshot import snapshot_file_deps, snapshot_is_fresh
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

FILES = 12

READ = "import glob, time\nX = [open(p).read() for p in sorted(glob.glob('docs/*.txt'))]"


@pytest.fixture
def docs(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(FILES):
        (folder / f"d{i:02d}.txt").write_text(f"text {i}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return folder


def _metadata(cash_magics, backend, name: str) -> dict:
    key = cash_magics.tracking_state.variable_sources[name]
    meta = backend.peek_metadata(key)
    assert meta is not None, f"{name}'s statement stored no entry"
    return meta


def _doc_paths(folder) -> set[str]:
    return {str(folder / name).replace("\\", "/") for name in os.listdir(folder)}


def test_an_entry_below_a_read_refers_to_its_records(cash_magics, cash_instance, docs):
    backend = cash_instance.backend
    run_cash_cell(cash_magics, READ)
    run_cash_cell(cash_magics, f"n = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), len(X))[1]")
    above, below = _metadata(cash_magics, backend, "X"), _metadata(cash_magics, backend, "n")
    absolute = _doc_paths(docs)
    assert absolute <= set(above["file_dependencies"])  # the positive control

    assert not absolute & set(below.get("file_dependencies") or ()), "the records were copied"
    assert below["file_refs"] == [[cash_magics.tracking_state.variable_sources["X"], above["files_token"]]]
    # Read through the reference, they are the records the read made.
    records = entry_file_records(below, backend)
    assert records == entry_file_records(above, backend)
    assert absolute <= set(records)


def test_a_reference_to_other_records_is_never_fresh(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("x", encoding="utf-8")
    token = uuid.uuid4().hex
    above = {"key": "stmt:above", "files_token": token, "file_dependencies": snapshot_file_deps({str(data)})}

    class Backend:
        def peek_metadata(self, key):
            return above if key == "stmt:above" else None

    fresh, _ = snapshot_is_fresh(entry_file_records({"file_refs": [["stmt:above", token]]}, Backend()))
    assert fresh  # the positive control
    for refs in ([["stmt:above", uuid.uuid4().hex]], [["stmt:gone", token]]):
        records = entry_file_records({"file_refs": refs}, Backend())
        fresh, stale = snapshot_is_fresh(records)
        assert not fresh and stale.reason == "unresolved", (refs, records)
