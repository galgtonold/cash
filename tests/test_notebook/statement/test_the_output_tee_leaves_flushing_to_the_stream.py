"""The tee cash puts on ``sys.stdout``/``sys.stderr`` for a statement
writes through to the real stream and records the text, and leaves when to
send it to the stream, as plain Python does: in a kernel, ipykernel's own
timer. A flush of its own, 100 ms into a statement, split what plain
Python sends as one output (a print's text from its newline, two prints in
a row) and sent stderr ahead of stdout written before it.
"""

from __future__ import annotations

import time

from cash.notebook._tee import TeeWriter


class _Stream:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.pending = ""

    def write(self, s: str) -> int:
        self.pending += s
        return len(s)

    def flush(self) -> None:
        if self.pending:
            self.sent.append(self.pending)
        self.pending = ""


def test_writes_are_sent_when_the_stream_sends_them():
    real = _Stream()
    tee = TeeWriter(real)
    print("a", file=tee)
    time.sleep(0.15)
    print("b", file=tee)
    print("c", end="", file=tee)
    assert real.sent == []
    assert real.pending == "a\nb\nc"
    tee.flush()
    assert real.sent == ["a\nb\nc"]
    assert tee.getvalue() == "a\nb\nc"


def test_writelines_is_recorded():
    real = _Stream()
    tee = TeeWriter(real)
    tee.writelines(["a", "b\n"])
    assert real.pending == tee.getvalue() == "ab\n"
