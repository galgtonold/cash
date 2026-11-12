"""Reading the saved notebook while the editor may be writing it.

An editor may still be flushing the ``.ipynb`` when a run begins, and a
read that overlaps the write returns a truncated file, a mix of two
versions or the one being replaced: pre-edit cell sources, which silently
break upstream change detection. ``_read_saved_notebook`` accepts a read
only when the file's signature is the same just before and just after it
and its stamps were old enough that a write would have moved them; it
reads again otherwise, and gives up after a bounded wait.

A settled file must cost no wait at all: a fixed 50 ms pause on every
freshly saved file was most of cash's per-cell cost on tiny cells, as an
editor autosaves before nearly every run.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from cash.notebook import server_discovery as sd

TICK_NS = sd._STAMP_TICK_NS


def _notebook(code: str) -> str:
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}
    return json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


class _Stat:
    """*real* with its change stamps moved: ``os.utime`` can set the mtime
    but not the ctime, which a write always moves to now."""

    def __init__(self, real, shift):
        self._real = real
        self.st_mtime_ns = shift(real.st_mtime_ns)
        self.st_ctime_ns = shift(real.st_ctime_ns)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _stamp(monkeypatch, path, age_ns: int) -> None:
    """Make every stat of *path* show its stamps *age_ns* older than they
    are, on fine stamps (as a filesystem with sub-second stamps leaves):
    a write still moves them, to *age_ns* before it."""
    real_stat = os.stat

    def shift(ns):
        ns -= age_ns
        return ns - 1234 if ns % sd._COARSE_STAMP_NS == 0 else ns

    def stat(p, *a, **k):
        st = real_stat(p, *a, **k)
        return _Stat(st, shift) if os.fspath(p) == os.fspath(path) else st

    monkeypatch.setattr(sd.os, "stat", stat)


def _codes(nb: dict) -> list[str]:
    return [c["source"] for c in nb["cells"]]


@pytest.fixture
def sleeps(monkeypatch):
    """Every pause the reader takes, still taken for real."""
    taken: list[float] = []
    real_sleep = time.sleep

    def sleep(seconds):
        taken.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(sd._time, "sleep", sleep)
    return taken


@pytest.fixture
def nb(tmp_path):
    p = tmp_path / "nb.ipynb"
    p.write_text(_notebook("x = 1"), encoding="utf-8")
    return p


def test_a_freshly_saved_file_that_holds_still_is_read_without_waiting(nb, sleeps, monkeypatch):
    """Saved a moment ago, but more than a timestamp tick ago: no write can
    hide in the read, so it is taken at once."""
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    assert time.time_ns() - os.stat(nb).st_ctime_ns < sd._SAVE_FRESH_WINDOW_S * 1e9  # fresh

    data, st = sd._read_saved_notebook(str(nb))

    assert _codes(data) == ["x = 1"]
    assert st.st_mtime_ns == os.stat(nb).st_mtime_ns
    assert sleeps == [], f"waited on a file nobody was writing: {sleeps}"


def test_the_cell_reader_takes_a_stable_fresh_file_without_sleeping(nb, sleeps, monkeypatch):
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    cells = sd.NotebookCellReaders().saved_file(str(nb))
    assert [c.code for c in cells] == ["x = 1"]
    assert sleeps == []


def test_a_file_written_inside_the_last_tick_is_read_once_the_tick_has_passed(nb, sleeps, monkeypatch):
    """A write in the same tick as the last one could keep every stamp, so
    a file changed less than a tick ago waits out the rest of the tick."""
    _stamp(monkeypatch, nb, 0)
    data, _ = sd._read_saved_notebook(str(nb))
    assert _codes(data) == ["x = 1"]
    assert len(sleeps) == 1 and 0 < sleeps[0] <= TICK_NS / 1e9, sleeps


def test_a_write_landing_during_the_read_is_never_taken(nb, sleeps, monkeypatch):
    """The read returns the old bytes while a save lands under it: the
    signature moved, so the reader reads again and returns the new file."""
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    real_read = sd._read_bytes
    reads: list[bytes] = []

    def read_under_a_save(path):
        data = real_read(path)
        if not reads:
            with open(path, "w", encoding="utf-8") as f:
                f.write(_notebook("x = 2  # edited"))
        reads.append(data)
        return data

    monkeypatch.setattr(sd, "_read_bytes", read_under_a_save)
    data, _ = sd._read_saved_notebook(str(nb))

    assert _codes(data) == ["x = 2  # edited"]
    assert len(reads) == 2
    assert _codes(json.loads(reads[0])) == ["x = 1"], "the first read was not the one overlapped"


def test_a_same_size_save_replacing_the_file_during_the_read_is_never_taken(nb, sleeps, monkeypatch):
    """A save that writes a temporary file and renames it over the
    notebook (same size, the stamp put back): only the file's identity
    tells the two apart, and it does."""
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    before = nb.stat()  # the real stamps, to put back
    real_read = sd._read_bytes
    reads = []

    def read_under_a_replace(path):
        data = real_read(path)
        if not reads:
            tmp = nb.with_suffix(".tmp")
            tmp.write_text(_notebook("x = 9"), encoding="utf-8")  # same size as "x = 1"
            os.utime(tmp, ns=(before.st_atime_ns, before.st_mtime_ns))
            os.replace(tmp, nb)
        reads.append(data)
        return data

    monkeypatch.setattr(sd, "_read_bytes", read_under_a_replace)
    data, _ = sd._read_saved_notebook(str(nb))

    assert _codes(data) == ["x = 9"]
    assert len(reads) == 2


def test_a_truncated_file_is_not_taken_while_it_may_still_be_written(nb, sleeps, monkeypatch):
    """Cut off mid-write and paused: it does not parse, so the reader waits
    and reads again, and gets the finished file."""
    full = _notebook("x = 2")
    nb.write_text(full[: len(full) // 2], encoding="utf-8")
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    real_sleep = time.sleep

    def writer_finishes(seconds):
        sleeps.append(seconds)
        nb.write_text(full, encoding="utf-8")
        real_sleep(seconds)

    monkeypatch.setattr(sd._time, "sleep", writer_finishes)
    data, _ = sd._read_saved_notebook(str(nb))
    assert _codes(data) == ["x = 2"]
    assert sleeps, "a truncated file was retried without waiting for its writer"


def test_a_truncated_file_that_stays_so_is_never_taken(nb, sleeps, monkeypatch):
    full = _notebook("x = 2")
    nb.write_text(full[: len(full) // 2], encoding="utf-8")
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    with pytest.raises(ValueError):
        sd._read_saved_notebook(str(nb))
    # The cell reader shows no cells rather than any half of them.
    assert sd.NotebookCellReaders().saved_file(str(nb)) == ()


def test_a_broken_file_saved_long_ago_is_not_waited_on(nb, sleeps, monkeypatch):
    nb.write_text("{not json", encoding="utf-8")
    _stamp(monkeypatch, nb, 10 * 1_000_000_000)
    with pytest.raises(ValueError):
        sd._read_saved_notebook(str(nb))
    assert sleeps == []


def test_a_file_with_coarse_stamps_is_taken_once_it_holds_still(nb, sleeps, monkeypatch):
    """Whole-second stamps (HFS+, ext3, FAT): the tick is unknown, so a file
    changed within it is read only after it has held still for a poll."""
    real_stat = os.stat

    def stat(p, *a, **k):
        st = real_stat(p, *a, **k)
        if os.fspath(p) != os.fspath(nb):
            return st
        return _Stat(st, lambda ns: ns // 1_000_000_000 * 1_000_000_000)

    monkeypatch.setattr(sd.os, "stat", stat)
    data, _ = sd._read_saved_notebook(str(nb))
    assert _codes(data) == ["x = 1"]
    assert sleeps == [sd._SAVE_POLL_INTERVAL_S]


def test_a_writer_that_never_stops_is_given_up_on(nb, monkeypatch):
    """The cap wins, and the cell reader shows no cells rather than a read
    some write overlapped."""
    monkeypatch.setattr(sd, "_SAVE_MAX_WAIT_S", 0.3)
    real_read = sd._read_bytes
    n = {"reads": 0}

    def read_under_endless_saves(path):
        data = real_read(path)
        n["reads"] += 1
        with open(path, "a", encoding="utf-8") as f:
            f.write(" ")
        return data

    monkeypatch.setattr(sd, "_read_bytes", read_under_endless_saves)
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        sd._read_saved_notebook(str(nb))
    assert time.monotonic() - t0 < 2.0
    assert n["reads"] > 1
    assert sd.NotebookCellReaders().saved_file(str(nb)) == ()


def test_unreadable_path_raises_after_one_retry(tmp_path, sleeps):
    with pytest.raises(OSError):
        sd._read_saved_notebook(str(tmp_path / "does-not-exist.ipynb"))
    assert sleeps == [sd._SAVE_POLL_INTERVAL_S]


def test_saves_from_another_thread_during_reads_are_read_past(nb, monkeypatch):
    """A real writer thread saving while the reader reads, in lockstep: each
    of the first reads is overlapped by a save of a new version, and the
    reader returns only the last version, never one a save overlapped."""
    n_saves = 4
    monkeypatch.setattr(sd, "_SAVE_MAX_WAIT_S", 60.0)
    _stamp(monkeypatch, nb, 3 * TICK_NS)
    your_turn = threading.Semaphore(0)
    saved = threading.Semaphore(0)

    def writer():
        for i in range(n_saves):
            if not your_turn.acquire(timeout=30):
                return
            nb.write_text(_notebook(f"x = {i + 10}" + " " * i), encoding="utf-8")
            saved.release()

    real_read = sd._read_bytes
    reads = []

    def read(path):
        data = real_read(path)
        reads.append(data)
        if len(reads) <= n_saves:
            your_turn.release()  # the save lands before this read is checked
            assert saved.acquire(timeout=30), "the writer thread never saved"
        return data

    monkeypatch.setattr(sd, "_read_bytes", read)
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    data, _ = sd._read_saved_notebook(str(nb))
    t.join(timeout=5)

    assert _codes(data) == [f"x = {n_saves + 9}" + " " * (n_saves - 1)]
    assert len(reads) == n_saves + 1
