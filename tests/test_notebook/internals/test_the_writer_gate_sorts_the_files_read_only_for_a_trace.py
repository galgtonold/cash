"""The writer gate sorts the files read above only when a trace is written.

Each writer the upstream check considers is recorded for the decision
trace (``CASH_TRACE_FILE``) with the first of the files the cells above
read, sorted. The sort ran whether a trace was written or not: below a loop
over 50,000 documents, 100,000 paths sorted for each writer before every
cell, and the record dropped.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cash.notebook.upstream import file_writers
from cash.notebook.upstream.file_writers import FileWriterScheduler

N = 500
WRITER = "with open('out.csv', 'a') as f:\n    f.write('x')"


def _gate() -> FileWriterScheduler:
    state = SimpleNamespace(
        executed_write_stmt_codes=set(),
        executed_write_input_lineages={},
        variable_lineage={},
        executed_cell_codes={},
    )
    return FileWriterScheduler(SimpleNamespace(user_ns={}), state, None)


def _considered(gate, tmp_path) -> list[int]:
    reads = {str(tmp_path / f"doc{i}.txt") for i in range(N)} | {"out.csv"}
    trace = [SimpleNamespace(stmt_code=WRITER, outputs=set(), inputs=set(), input_hashes={})]
    return gate.find_stale_file_writer_indices(trace, relevant_read_paths=reads, writer_positions=(0,))


@pytest.fixture
def sorts(monkeypatch):
    """The sizes of what the gate sorts."""
    sizes: list[int] = []

    def counting(items, *args, **kwargs):
        items = list(items)
        sizes.append(len(items))
        return sorted(items, *args, **kwargs)

    monkeypatch.setattr(file_writers, "sorted", counting, raising=False)
    return sizes


def test_no_trace_no_sort(tmp_path, monkeypatch, sorts):
    monkeypatch.delenv("CASH_TRACE_FILE", raising=False)
    monkeypatch.chdir(tmp_path)

    assert _considered(_gate(), tmp_path) == []  # an append is never re-fired

    assert not [size for size in sorts if size > N]


def test_a_trace_still_records_the_files_read(tmp_path, monkeypatch, sorts):
    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setenv("CASH_TRACE_FILE", str(trace_file))
    monkeypatch.chdir(tmp_path)

    _considered(_gate(), tmp_path)

    considered = [line for line in trace_file.read_text(encoding="utf-8").splitlines() if "writer_considered" in line]
    assert len(considered) == 1
    assert '"read_paths": [' in considered[0]
