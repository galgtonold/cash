"""A statement that printed nothing does not flush the kernel's streams.

A kernel's stream flushes by asking its IO thread and waiting for the answer.
Every statement flushed stdout and stderr when it ended, printed or not: two
waits a statement, an eighth of what a trivial statement cost. The flush is
what puts a print ahead of what the statement shows next, so it still
happens when the statement wrote something, when a write from outside the
statement is still waiting, and on a stream that does not say.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("IPython")

from cash.notebook._tee import TeeWriter
from cash.notebook.statement.capture import capture_output


class _KernelStream:
    """ipykernel's OutStream, as far as a flush sees it."""

    def __init__(self, pending: bool = False, subprocess_pending: bool = False) -> None:
        self._flush_pending = pending
        self._subprocess_flush_pending = subprocess_pending
        self.flushes = 0
        self.text = ""

    def write(self, text: str) -> int:
        self.text += text
        self._flush_pending = True
        return len(text)

    def flush(self) -> None:
        self.flushes += 1
        self._flush_pending = False


class _PlainStream:
    """A stream that does not say whether a write is waiting."""

    def __init__(self) -> None:
        self.flushes = 0

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        self.flushes += 1


def test_nothing_written_and_nothing_waiting_is_not_flushed():
    stream = _KernelStream()
    TeeWriter(stream).flush_written()
    assert stream.flushes == 0


def test_what_the_statement_wrote_is_flushed():
    stream = _KernelStream()
    writer = TeeWriter(stream)
    writer.write("row 1\n")
    writer.flush_written()
    assert stream.flushes == 1
    assert stream.text == "row 1\n"


@pytest.mark.parametrize("waiting", [{"pending": True}, {"subprocess_pending": True}])
def test_a_write_from_outside_the_statement_still_waiting_is_flushed(waiting):
    stream = _KernelStream(**waiting)
    TeeWriter(stream).flush_written()
    assert stream.flushes == 1


def test_a_stream_that_does_not_say_is_flushed():
    stream = _PlainStream()
    TeeWriter(stream).flush_written()
    assert stream.flushes == 1


def _flushes_of_a_capture(monkeypatch, body) -> tuple[_KernelStream, _KernelStream]:
    out, err = _KernelStream(), _KernelStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    with capture_output(tee=True):
        body()
    return out, err


def test_a_capture_of_a_silent_statement_flushes_neither_stream(monkeypatch):
    out, err = _flushes_of_a_capture(monkeypatch, lambda: None)
    assert (out.flushes, err.flushes) == (0, 0)


def test_a_capture_of_a_statement_that_prints_flushes_the_stream_it_wrote(monkeypatch):
    out, err = _flushes_of_a_capture(monkeypatch, lambda: print("shown"))
    assert out.text == "shown\n"
    assert out.flushes >= 1
    assert err.flushes == 0
