"""The upstream check asks about every statement above the cell, on every
cell: what it can answer without the backend or the namespace, it does.

In a 400-cell notebook a trivial cell at the end cost 1.6x one at the top.
Two of the questions were a backend lookup per ``def``'s caller for the
files it read when it last ran (most read none), and a namespace lookup per
statement for a user function that writes files, asked even of a statement
that calls nothing.
"""

from __future__ import annotations

import types

from cash.notebook import cache_key
from cash.notebook.upstream import file_writers
from cash.notebook.upstream.file_writers import FileWriterScheduler
from cash.notebook.upstream.read_scope import ReadScope


class _Probe:
    def __init__(self, records: dict | None = None) -> None:
        self.records = records or {}
        self.asked: list[str] = []

    def record(self, key: str):
        self.asked.append(key)
        return self.records.get(key)


def _scope(probe: _Probe) -> ReadScope:
    return ReadScope(types.SimpleNamespace(user_ns={}), types.SimpleNamespace(), probe)


def test_a_statement_with_no_read_record_is_looked_up_once():
    probe = _Probe()
    scope = _scope(probe)
    for _ in range(3):
        assert scope._persisted_reads("x2 = h2(x1)") is None
    assert len(probe.asked) == 1


def test_a_record_this_process_writes_is_seen():
    probe = _Probe()
    scope = _scope(probe)
    assert scope._persisted_reads("df = load(p)") is None
    probe.records[cache_key.read_provenance_key("df = load(p)")] = {"read_provenance": True, "paths": ["a.csv"]}
    cache_key.note_read_provenance_written()
    assert scope._persisted_reads("df = load(p)") == {"a.csv"}


def test_a_recorded_statement_is_read_each_time():
    """Only a missing record is remembered: a record's paths are read anew."""
    code = "df = load(p)"
    probe = _Probe({cache_key.read_provenance_key(code): {"read_provenance": True, "paths": ["a.csv"]}})
    scope = _scope(probe)
    assert scope._persisted_reads(code) == {"a.csv"}
    assert scope._persisted_reads(code) == {"a.csv"}
    assert len(probe.asked) == 2


def test_a_statement_that_calls_nothing_is_no_writer_without_asking_the_namespace(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("asked the namespace about a statement that calls nothing")

    monkeypatch.setattr(file_writers, "statement_calls_user_writer", refuse)
    scheduler = FileWriterScheduler(types.SimpleNamespace(user_ns={}), types.SimpleNamespace(), _Probe())
    assert scheduler._is_file_writer("x2 = x1 + 1", []) is False
    assert scheduler._is_file_writer("d = {'k': x1}\nx3 = d['k']", []) is False
    # A text write is still one, and a call is still asked about.
    assert scheduler._is_file_writer("open('o.txt', 'w').write('x')", []) is True
    monkeypatch.setattr(file_writers, "statement_calls_user_writer", lambda *a, **k: "save")
    assert scheduler._is_file_writer("save(x1)", []) is True
