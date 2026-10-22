"""A statement that wrote more files than a provenance holds is not probed
for every one of them.

``tar.extractall()`` writes every member of the archive. The write
provenance a later reader uses to skip the writer holds at most
``_MAX_PROVENANCE_FILES`` paths, but each written path was probed for
existence first, and only then counted: 100,000 probes per statement, three
statements deep (the ``if``, the ``with``, the call), to learn there were too
many. The count stops once it is past the limit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cash.notebook.statement import records
from cash.notebook.statement.records import StatementRecords
from cash.notebook.tracking_state import TrackingState


class _Backend:
    def __init__(self) -> None:
        self.stored: dict[str, dict] = {}

    def set_metadata_only(self, key, record) -> None:
        self.stored[key] = record


def _records(backend):
    shell = SimpleNamespace(user_ns={})
    return StatementRecords(shell, TrackingState(), SimpleNamespace(backend=backend), None)


@pytest.fixture
def probes(monkeypatch):
    seen = []
    own_stat = records.own_stat

    def counted(path):
        seen.append(path)
        return own_stat(path)

    monkeypatch.setattr(records, "own_stat", counted)
    return seen


def _written(tmp_path, count):
    paths = []
    for i in range(count):
        path = tmp_path / f"{i}.txt"
        path.write_text("x", encoding="utf-8")
        paths.append(str(path))
    return frozenset(paths)


def test_past_the_limit_the_probes_stop_and_nothing_is_recorded(tmp_path, monkeypatch, probes):
    limit = 20
    monkeypatch.setattr(StatementRecords, "_MAX_PROVENANCE_FILES", limit)
    backend = _Backend()
    _records(backend).persist_write_provenance("extract()", set(), None, _written(tmp_path, 500))
    assert len(probes) == limit + 1
    assert backend.stored == {}


def test_up_to_the_limit_every_written_file_is_recorded(tmp_path, monkeypatch, probes):
    monkeypatch.setattr(StatementRecords, "_MAX_PROVENANCE_FILES", 20)
    backend = _Backend()
    written = _written(tmp_path, 20)
    _records(backend).persist_write_provenance("extract()", set(), None, written)
    (record,) = backend.stored.values()
    assert record["paths"] == sorted(written)


def test_a_written_file_gone_again_does_not_count_towards_the_limit(tmp_path, monkeypatch, probes):
    monkeypatch.setattr(StatementRecords, "_MAX_PROVENANCE_FILES", 20)
    backend = _Backend()
    kept = _written(tmp_path, 5)
    gone = frozenset(str(tmp_path / f"tmp{i}") for i in range(100))
    _records(backend).persist_write_provenance("extract()", set(), None, kept | gone)
    (record,) = backend.stored.values()
    assert record["paths"] == sorted(kept)
