"""``print('x')`` writes ``'x'`` and then ``'\\n'``. The tee cash puts on
the streams flushes the real one at most every 100 ms, and only at the end
of a line: flushed between the two writes (the first print of a statement
100 ms in), the notebook got two outputs for one line where plain Python
gives one.
"""

from __future__ import annotations

from cash.notebook._tee import TeeWriter


class _Stream:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._pending = ""

    def write(self, s: str) -> int:
        self._pending += s
        return len(s)

    def flush(self) -> None:
        if self._pending:
            self.sent.append(self._pending)
        self._pending = ""


def test_a_line_printed_late_is_sent_whole():
    real = _Stream()
    tee = TeeWriter(real)
    tee._last_flush -= 1.0  # the statement ran a second before it printed
    print("'warned\\n'", file=tee)
    tee.flush()
    assert real.sent == ["'warned\\n'\n"]
    assert tee.getvalue() == "'warned\\n'\n"


def test_a_long_run_of_lines_is_still_sent_as_it_goes():
    real = _Stream()
    tee = TeeWriter(real)
    tee._last_flush -= 1.0
    print("first", file=tee)
    assert real.sent == ["first\n"]
