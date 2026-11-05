"""A statement's entry keeps each record of a file it depends on once.

Below a read of 15,000 files every statement's entry held a record of each,
twice: copied from the entry its input came from, and once more under the
relative spelling the read was also recorded under. A trivial cell wrote
5 MB, and a restart read it back to decide to run the cell again.

* An entry refers to the records of the entry its input came from, by its key
  and the token naming them, rather than copying them
  (``cash.notebook.file_records``); a reference that no longer names the
  records stored there is never fresh.
* A cheap statement whose inputs were built from files, and which reads none
  itself, is not stored: run again, it costs what it did. One whose input is
  an open file or another stream is not judged so: run again, it would find
  the stream where its last run left it.
* A relative read's second spelling is folded into its absolute twin's
  record when stored, and read back as its own record, so a check under
  another working folder still looks for it where its name leads now.
"""

from __future__ import annotations

import os
import uuid

import pytest

from cash.notebook.file_records import entry_file_records
from cash.tracking.file_dep_snapshot import ALSO_SPELLED, snapshot_file_deps, snapshot_is_fresh
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


def _stored_keys(backend) -> int:
    return backend.entry_count()


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


def test_a_cheap_statement_below_a_read_is_not_stored(cash_magics, cash_instance, docs):
    backend = cash_instance.backend
    run_cash_cell(cash_magics, READ)
    stored = _stored_keys(backend)
    run_cash_cell(cash_magics, "m = X[:3]")
    assert cash_magics.shell.user_ns["m"] == ["text 0", "text 1", "text 2"]
    assert set(_doc_paths(docs)) <= cash_magics.tracking_state.executed_file_deps["m"]
    assert _stored_keys(backend) == stored, "it was stored with the records of the files X was read from"


def test_a_cheap_reader_of_a_file_handle_reads_it_again_on_a_second_run(cash_magics, docs):
    cells = ["fh = open('docs/d00.txt')", "lines = [line for line in fh]"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell)
        assert cash_magics.shell.user_ns["lines"] == ["text 0"]


def test_a_relative_read_is_recorded_once(cash_magics, cash_instance, docs):
    backend = cash_instance.backend
    run_cash_cell(cash_magics, READ)
    stored = _metadata(cash_magics, backend, "X")["file_dependencies"]
    assert not [path for path in stored if not os.path.isabs(path)], "a relative spelling has a record of its own"
    for name in os.listdir(docs):
        absolute = str(docs / name).replace("\\", "/")
        assert stored[absolute][ALSO_SPELLED] == [f"docs/{name}"]
    # Read back, each spelling has its record.
    records = entry_file_records(_metadata(cash_magics, backend, "X"), backend)
    for name in os.listdir(docs):
        assert records[f"docs/{name}"]["hash"] == records[str(docs / name).replace("\\", "/")]["hash"]


def test_a_relative_read_under_another_working_folder_is_read_again(cash_magics, cash_instance, tmp_path, monkeypatch):
    backend = cash_instance.backend
    for place, text in (("a", "from a"), ("b", "from b")):
        (tmp_path / place / "docs").mkdir(parents=True)
        (tmp_path / place / "docs" / "d.txt").write_text(text, encoding="utf-8")
    cell = "t = open('docs/d.txt').read()"
    monkeypatch.chdir(tmp_path / "a")
    run_cash_cell(cash_magics, cell)
    stored = _metadata(cash_magics, backend, "t")["file_dependencies"]
    assert "docs/d.txt" not in stored, "the relative spelling was not folded into its twin's record"
    run_cash_cell(cash_magics, "t = None")
    run_cash_cell(cash_magics, cell)
    assert cash_magics.cash_status("dict")["last_cell"]["statements"][0]["status"].name == "RESTORED"
    assert cash_magics.shell.user_ns["t"] == "from a"

    monkeypatch.chdir(tmp_path / "b")
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["t"] == "from b"
