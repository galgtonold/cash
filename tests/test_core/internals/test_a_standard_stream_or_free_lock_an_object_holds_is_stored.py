"""An object holding ``sys.stdout``, ``sys.stderr`` or a free lock is stored.

A progress bar holds the stream it writes to (``ProgBar(n,
stream=sys.stdout)``) and its ``write``; an object may hold a lock for its own
use. ``pickle`` refuses both, so the object was never stored and the loop
updating it ran again after every restart. Plain Python, after a restart and a
Run All, has a bar writing to the stream the process has then, and a lock no
one holds. So a stream that is ``sys.stdout`` or ``sys.stderr`` as it is
stored is stored as that name, and a lock no one holds as a new lock of its
kind (`live_state`), in what is stored only: a key still refuses a lock.
Any other file, and a held lock, stay unstorable.

Only what ``pickle`` refuses is stored this way: what could be stored before
is pickled, and keyed, to the same bytes.
"""

from __future__ import annotations

import io
import pickle
import sys
import threading

import pytest

from cash import kept_state, live_state
from cash.notebook._tee import StreamTaken, TeeWriter


class Bar:
    def __init__(self, stream):
        self.stream = stream
        self.out = stream.write
        self.n = 0


class Guarded:
    def __init__(self, lock):
        self.lock = lock
        self.items = [1, 2]


def _round_trip(value):
    return pickle.loads(kept_state.dumps(value))


@pytest.mark.parametrize("name", ["stdout", "stderr"])
def test_a_bar_on_a_standard_stream_restores_onto_the_stream_of_the_time(name, monkeypatch):
    bar = Bar(getattr(sys, name))
    data = kept_state.dumps(bar)
    now = io.StringIO()
    monkeypatch.setattr(sys, name, now)
    restored = pickle.loads(data)
    assert restored.stream is now
    restored.out("x")
    assert now.getvalue() == "x"


def test_a_bar_made_while_a_cell_ran_restores_as_one_made_in_plain_python():
    """While a cell runs, ``sys.stdout.write`` is cash's: the bar stores as a
    bar holding the stream's own ``write``, to the bytes of one that does."""
    plain = kept_state.dumps(Bar(sys.stdout))
    with StreamTaken("stdout", TeeWriter):
        during = kept_state.dumps(Bar(sys.stdout))
    assert during == plain


@pytest.mark.parametrize("kind", [threading.Lock, threading.RLock])
def test_a_free_lock_restores_as_a_new_free_lock_of_its_kind(kind):
    restored = _round_trip(Guarded(kind()))
    assert type(restored.lock) is type(kind())
    assert restored.lock.acquire(blocking=False)
    restored.lock.release()
    assert restored.items == [1, 2]


@pytest.mark.parametrize("kind", [threading.Lock, threading.RLock])
def test_a_free_lock_is_not_keyed(kind):
    """Every free lock would key alike: a key would lose which lock it was,
    so a key still refuses one, as before."""
    with pytest.raises((TypeError, pickle.PicklingError)):
        kept_state.dumps(Guarded(kind()), keyed=True)


@pytest.mark.parametrize("kind", [threading.Lock, threading.RLock])
def test_a_held_lock_stays_unstorable(kind):
    lock = kind()
    lock.acquire()
    try:
        with pytest.raises((TypeError, pickle.PicklingError)):
            kept_state.dumps(Guarded(lock))
    finally:
        lock.release()


def test_any_other_file_stays_unstorable(tmp_path):
    with open(tmp_path / "log.txt", "w", encoding="utf-8") as f:
        with pytest.raises((TypeError, pickle.PicklingError)):
            kept_state.dumps(Bar(f))


def test_a_writer_on_a_file_stays_unstorable(tmp_path):
    with open(tmp_path / "log.txt", "w", encoding="utf-8") as f:
        with pytest.raises((TypeError, pickle.PicklingError)):
            kept_state.dumps(TeeWriter(f))


@pytest.mark.parametrize(
    "value",
    [
        io.StringIO("text"),
        io.BytesIO(b"bytes"),
        {"buf": io.StringIO("a"), "n": [1, 2]},
        Guarded(None),
    ],
    ids=["StringIO", "BytesIO", "dict holding one", "object without a lock"],
)
def test_what_was_storable_is_pickled_as_before(value, monkeypatch):
    """Never asked: what pickle takes is not stored as live state."""
    before = kept_state.dumps(value)

    def refuse(_obj):
        raise AssertionError("asked for a value pickle takes")

    monkeypatch.setattr(kept_state, "live_reduce", refuse)
    assert kept_state.dumps(value) == before


def test_only_the_standard_streams_are_live():
    assert live_state.is_live_stream(sys.stdout)
    assert live_state.is_live_stream(sys.stderr)
    assert not live_state.is_live_stream(io.StringIO())


class _KernelStream(io.TextIOBase):
    """Like ipykernel's ``OutStream``: its reduce succeeds, its state does not pickle."""

    def __init__(self):
        self.session = {"signer": threading.Lock()}
        self.text = []

    def write(self, s):
        self.text.append(s)
        return len(s)


def test_a_kernel_stream_whose_state_does_not_pickle_is_stored_as_the_stream(monkeypatch):
    kernel = _KernelStream()
    monkeypatch.setattr(sys, "stdout", kernel)
    data = kept_state.dumps(Bar(sys.stdout))
    later = _KernelStream()
    monkeypatch.setattr(sys, "stdout", later)
    restored = pickle.loads(data)
    assert restored.stream is later
    restored.out("x")
    assert later.text == ["x"]


def test_a_buffer_bound_as_stdout_is_pickled_as_before(monkeypatch):
    """``redirect_stdout(buf)``: a ``StringIO`` pickle takes, stored as one."""
    buf = io.StringIO("kept")
    before = kept_state.dumps({"buf": buf})
    monkeypatch.setattr(sys, "stdout", buf)
    assert kept_state.dumps({"buf": buf}) == before
