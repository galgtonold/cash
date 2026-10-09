"""``with open(path) as f: records = [...]`` is cached as any statement is.

The block leaves the closed handle ``f`` bound. It was taken for an open
stream, so every value the block bound got a content digest in its lineage
on every run -- a pickle of all the records -- as a ``# @cash:no-cache``
statement's outputs do. And the RAM tier keeps the closed handle as the
object itself, so on every hit after that each variable of the entry was
deep-copied to see that it could be, then thrown away: 50 us a record,
11.8 s a Run All for 300,000 records after a restart against 2.6 s plain.

A closed handle is no stream (nothing can be read from it); the file the
block read is the statement's dependency. JSON-like data is known to copy
without copying it. A handle still open, and a value that cannot be copied,
are handled as before.
"""

from __future__ import annotations

import copy
import io
import json

from cash.notebook.cache_status import CacheStatus
from cash.notebook.statement import freshness
from cash.notebook.statement import lineage as statement_lineage
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import json, time\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x"


def _events(tmp_path, n=2000):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps({"id": i, "tags": ["a", i]}) + "\n" for i in range(n)), encoding="utf-8")
    return path


def _with(path) -> str:
    return f"with open({str(path)!r}) as f:\n    records = slow([json.loads(line) for line in f])"


def _counting(monkeypatch, module, name):
    calls: list = []
    real = getattr(module, name)

    def spy(value, *args, **kwargs):
        calls.append(value)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


def test_a_closed_handle_does_not_make_the_block_a_stream(cash_magics, statement_processor, tmp_path, monkeypatch):
    path = _events(tmp_path)
    run_cash_cell(cash_magics, SETUP)
    digests = _counting(monkeypatch, statement_lineage, "no_cache_value_digest")

    statement_processor.process_statement(_with(path))

    assert digests == [], "the records were hashed as a stream's outputs"
    assert cash_magics.shell.user_ns["f"].closed


def test_a_hit_does_not_copy_the_records_to_see_that_it_can(cash_magics, statement_processor, tmp_path, monkeypatch):
    path = _events(tmp_path)
    run_cash_cell(cash_magics, SETUP)
    code = _with(path)
    statement_processor.process_statement(code)
    copies = _counting(monkeypatch, freshness.copy, "deepcopy")

    assert statement_processor.process_statement(code)["status"] == CacheStatus.RESTORED

    assert not any(type(c) is list and len(c) == 2000 for c in copies), "the records were deep-copied"
    records = cash_magics.shell.user_ns["records"]
    assert len(records) == 2000 and records[5] == {"id": 5, "tags": ["a", 5]}


def test_the_file_the_block_read_is_still_its_dependency(cash_magics, statement_processor, tmp_path):
    path = _events(tmp_path, 20)
    run_cash_cell(cash_magics, SETUP)
    code = _with(path)
    statement_processor.process_statement(code)
    path.write_text(json.dumps({"id": 99}) + "\n", encoding="utf-8")

    assert statement_processor.process_statement(code)["status"] != CacheStatus.RESTORED
    assert cash_magics.shell.user_ns["records"] == [{"id": 99}]


def test_an_open_handle_is_still_a_stream(cash_magics, statement_processor, tmp_path, monkeypatch):
    path = _events(tmp_path, 20)
    run_cash_cell(cash_magics, SETUP)
    digests = _counting(monkeypatch, statement_lineage, "no_cache_value_digest")

    statement_processor.process_statement(f"fh = open({str(path)!r})")

    assert digests, "an open handle's statement lost its value digest"
    cash_magics.shell.user_ns["fh"].close()


def test_a_handle_that_cannot_say_whether_it_is_closed_is_a_stream():
    from cash.notebook.statement.processor import _open_stream

    class Odd(io.IOBase):
        @property
        def closed(self):
            raise OSError("gone")

    assert _open_stream(Odd())
    assert not _open_stream(_closed(io.StringIO("x")))
    assert _open_stream(io.StringIO("x"))
    assert not _open_stream([1, 2])


def _closed(handle):
    handle.close()
    return handle


def test_json_like_values_are_known_to_copy_and_others_are_copied():
    assert freshness._copies([{"a": [1, (2, "x")]}] * 3)
    assert not freshness._copies([object()])
    loop: list = []
    loop.append(loop)
    assert not freshness._copies(loop)
    assert copy.deepcopy([{"a": [1, (2, "x")]}])  # what it stands for


def test_a_value_that_cannot_be_copied_still_stops_the_hit():
    import threading

    checker = freshness.CacheFreshnessChecker.__new__(freshness.CacheFreshnessChecker)
    checker.last_miss_reason = None
    data = {"variables": {"records": [{"id": 1}], "lock": threading.Lock()}}

    assert checker._invalidate_if_held_by_reference(data) is None
    assert "'lock'" in checker.last_miss_reason


def test_the_probe_after_a_restart_does_not_copy_the_records(tmp_path, monkeypatch):
    """After a restart the entry comes back from disk and the RAM tier keeps
    the closed handle by reference: the entry is marked so, and each hit
    asked every variable to be copied."""
    path = _events(tmp_path, 2000)
    with open(path) as f:
        records = [json.loads(line) for line in f]
    checker = freshness.CacheFreshnessChecker.__new__(freshness.CacheFreshnessChecker)
    checker.last_miss_reason = None
    copies = _counting(monkeypatch, freshness.copy, "deepcopy")
    data = {"variables": {"f": f, "records": records, "n": len(records)}}

    assert checker._invalidate_if_held_by_reference(data) is data
    assert copies == []
