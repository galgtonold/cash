"""A read cash failed to record is never served after the file changed.

Two ways it was:

- ``os.environ`` replaced by a plain mapping (``os.environ = dict(...)``,
  ``mock.patch("os.environ", {...})``) made cash's own ``NUMBA_CACHE_DIR``
  lookup raise inside every ``open`` it watched, and the read was dropped;
- any error in cash's handler of an audited ``open`` was swallowed (it must
  never fail the user's call), and with it the read the handler was to
  record: the call was stored with no dependency on the file.

A failed handler now marks the read as one that is never fresh, and a
notebook statement that made it is not stored; both warn KEY-READ-UNRECORDED.
"""

from __future__ import annotations

import os
import time
import warnings

import pytest

from cash import Cash
from cash.tracking import env_reads, io_watch
from cash.tracking.file_tracker import FileAccessTracker
from cash.tracking.read_classification import is_runtime_cache_path
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def plain_environ(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))


def _reader(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    data = tmp_path / "in.txt"
    data.write_text("v1", encoding="utf-8")

    @cash.cache
    def load():
        with open(data, encoding="utf-8") as fh:
            return fh.read()

    return load, data


def test_the_runtime_cache_check_works_on_a_replaced_environ(plain_environ, monkeypatch):
    os.environ["NUMBA_CACHE_DIR"] = "/data/numba"
    assert is_runtime_cache_path("/data/in.csv") is False
    assert is_runtime_cache_path("/data/numba/x.nbi") is True


def test_a_cached_reader_sees_its_file_change_with_a_replaced_environ(tmp_path, plain_environ):
    load, data = _reader(tmp_path)
    assert load() == "v1"
    time.sleep(0.05)
    data.write_text("v2-longer", encoding="utf-8")
    assert load() == "v2-longer"


def test_cashs_own_lookup_is_not_a_read_of_the_users(monkeypatch):
    monkeypatch.setenv("CASH_TEST_SETTING", "on")
    io_watch.hold()
    try:
        assert type(os.environ).__getitem__ is env_reads._getitem, "the user's reads are watched"
        with env_reads.recording() as reads:
            assert env_reads.environ_get("CASH_TEST_SETTING") == "on"
            assert env_reads.environ_get("CASH_TEST_NOT_SET") is None
        assert not reads
    finally:
        io_watch.release()


def _broken(self, path):
    raise RuntimeError("a bug in cash's read handler")


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


def test_a_handler_that_fails_leaves_the_call_never_served(tmp_path, monkeypatch):
    load, data = _reader(tmp_path)
    monkeypatch.setattr(FileAccessTracker, "track_path", _broken)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert load() == "v1"
    assert _codes(caught).count("KEY-READ-UNRECORDED") == 1
    message = str(next(w.message for w in caught if getattr(w.message, "code", None) == "KEY-READ-UNRECORDED"))
    assert str(data) in message and "RuntimeError" in message
    time.sleep(0.05)
    data.write_text("v2-longer", encoding="utf-8")
    assert load() == "v2-longer"


def test_a_failed_handler_is_noted_on_the_block_and_its_parents(tmp_path, monkeypatch):
    data = tmp_path / "in.txt"
    data.write_text("v1", encoding="utf-8")
    with FileAccessTracker() as outer, FileAccessTracker(propagate_to_parent=True) as inner:
        monkeypatch.setattr(FileAccessTracker, "track_path", _broken)
        data.read_text(encoding="utf-8")
        monkeypatch.undo()
    path = os.path.abspath(str(data))
    for tracker in (inner, outer):
        assert tracker.tracking_failed == [("open", path, "RuntimeError: a bug in cash's read handler")]
        assert path in tracker.get_unresolved_files()


def test_a_statement_whose_read_was_not_recorded_warns_and_runs_every_time(cash_magics, tmp_path, monkeypatch):
    data = tmp_path / "in.txt"
    data.write_text("v1", encoding="utf-8")
    cells = [
        f"import time\nP = {str(data)!r}",
        f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\ntext = open(P, encoding='utf-8').read()",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, cells[0], cells=cells)
    monkeypatch.setattr(FileAccessTracker, "track_path", _broken)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[1], cells=cells)
    assert "KEY-READ-UNRECORDED" in _codes(caught)
    monkeypatch.undo()
    data.write_text("v2-longer", encoding="utf-8")
    run_cash_cell(cash_magics, cells[1], cells=cells)
    assert cash_magics.shell.user_ns["text"] == "v2-longer"
