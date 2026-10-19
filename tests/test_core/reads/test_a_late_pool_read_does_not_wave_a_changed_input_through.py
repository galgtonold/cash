"""A pool thread still reading after the body returned does not wave a changed
input through.

The store step checks that no file the body read moved before it returned
(`FileAccessTracker.inputs_changed_since_read`), walking the tracker's
``read_stats``. A pool thread the body started and did not wait for can still
be reading then, adding to ``read_stats`` mid-walk: the walk raised
"dictionary changed size during iteration", and every caller took the error
for "nothing moved". A file rewritten after the body read it was stored with
its new fingerprint next to a result computed from the old content, and
served on the next call. The walk now goes over copies, and a check that
cannot finish refuses the store.
"""

from __future__ import annotations

import contextvars
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from cash._paths import normalize_path
from cash.tracking import file_tracker
from cash.tracking.file_tracker import FileAccessTracker
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = pytest.mark.core


def _late_read_during_the_check(monkeypatch, read_late):
    """Make the moved-input check's first stat wait for *read_late*, run on
    another thread: a read landing in the middle of the walk."""
    real = file_tracker.regular_file_stat_result
    fired = threading.Event()

    def stat(path):
        if not fired.is_set() and sys._getframe(1).f_code.co_name == "inputs_changed_since_read":
            fired.set()
            read_late()
        return real(path)

    monkeypatch.setattr(file_tracker, "regular_file_stat_result", stat)
    return fired


def test_the_check_finds_the_changed_file_while_a_pool_thread_reads(tmp_path, monkeypatch):
    paths = []
    for i in range(3):
        path = tmp_path / f"f{i}.txt"
        path.write_text("old", encoding="utf-8")
        paths.append(str(path))
    late = tmp_path / "late.txt"
    late.write_text("x", encoding="utf-8")
    go, landed = threading.Event(), threading.Event()

    def read_late():
        go.wait(10)
        late.read_text(encoding="utf-8")
        landed.set()

    tracker = FileAccessTracker(hash_on_read=True)
    with tracker:
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                fh.read()
        # Started in the block, as a pool's worker is: its reads are the block's.
        worker = threading.Thread(target=contextvars.copy_context().run, args=(read_late,))
        worker.start()
    (tmp_path / "f2.txt").write_text("NEW!", encoding="utf-8")

    def wake_the_worker():
        go.set()
        landed.wait(10)

    fired = _late_read_during_the_check(monkeypatch, wake_the_worker)
    moved = tracker.inputs_changed_since_read()
    worker.join(10)
    # Paths are recorded in forward slashes on Windows too.
    assert fired.is_set() and normalize_path(str(late)) in tracker.read_stats, "the late read did not land mid-walk"
    assert moved == [normalize_path(str(tmp_path / "f2.txt"))]


class _Late:
    """A pool the body hands a read to and does not wait for, and a writer
    that rewrites the input once the body has read it."""

    def __init__(self, data, late):
        self.data, self.late = data, late
        self.pool = ThreadPoolExecutor(1)
        self.go, self.landed = threading.Event(), threading.Event()

    def __cash_key__(self):
        return (str(self.data), str(self.late))

    def read_late(self):
        self.go.wait(10)
        self.late.read_text(encoding="utf-8")
        self.landed.set()

    def wake(self):
        self.go.set()
        self.landed.wait(10)


def test_a_cached_call_does_not_store_an_input_that_moved(disk_cash, tmp_path, monkeypatch):
    data, late = tmp_path / "data.txt", tmp_path / "late.txt"
    data.write_text("old", encoding="utf-8")
    late.write_text("x", encoding="utf-8")
    runs = []

    @disk_cash.cache
    def load(late_reads):
        runs.append(1)
        content = late_reads.data.read_text(encoding="utf-8")
        late_reads.pool.submit(contextvars.copy_context().run, late_reads.read_late)
        # Rewritten after the body read it, before the entry is stored.
        late_reads.data.write_text("new", encoding="utf-8")
        return content

    late_reads, again = _Late(data, late), _Late(data, late)
    again.go.set()  # the second call's late read needs no waking
    fired = _late_read_during_the_check(monkeypatch, late_reads.wake)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert load(late_reads) == "old"
            assert fired.is_set(), "the late read did not land mid-check"
            # Not stored: the next call reads the file as it is now.
            assert load(again) == "new"
    finally:
        late_reads.pool.shutdown(wait=True)
        again.pool.shutdown(wait=True)
    assert len(runs) == 2


def test_a_cached_call_is_not_stored_when_the_check_fails(disk_cash, tmp_path, monkeypatch):
    data = tmp_path / "data.txt"
    data.write_text("old", encoding="utf-8")
    runs = []

    @disk_cash.cache
    def load(path):
        runs.append(1)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def broken(self):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(FileAccessTracker, "inputs_changed_since_read", broken)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load(str(data))
        load(str(data))
    assert len(runs) == 2


def test_a_statement_is_not_stored_when_the_check_fails(cash_magics, tmp_path, monkeypatch):
    data = tmp_path / "data.txt"
    data.write_text("old", encoding="utf-8")

    def broken(self):
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(FileAccessTracker, "inputs_changed_since_read", broken)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cell = f"import time\ntext = (open({str(data)!r}).read(), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, cell)
        run_cash_cell(cash_magics, cell)
    read = [s for s in cash_magics._last_cell_metrics["statements"] if s["code"].startswith("text =")]
    assert [str(s["status"].value) for s in read] == ["COMPUTED"], read
