"""`TeeWriter`: record what is written to a stream while passing it through.

A leaf module, so both the statement pipeline (``statement/capture.py``) and a
cached call (``call_unit.py``, which sits beneath the statement package in the
import graph) use the one writer.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterable
from typing import Any

from cash._user_attrs import attribute_or

__all__ = ["StreamTaken", "TeeWriter"]


class TeeWriter:
    """A writer that sends output to both a real stream and a list buffer.

    * Every ``write(s)`` goes to ``self._real`` at once, and the real stream
      sends it when it sends its own: in a kernel, ipykernel's ``OutStream``
      batches what was written and flushes it on its own timer, as it does
      for plain Python. The tee does not flush on its own: flushing on a
      timer of its own split what plain Python sends as one output
      (``print('a'); print('b')`` 100 ms into a statement came as two, and
      a print's text apart from its newline) and showed stderr ahead of
      stdout written before it.
    * Text is kept in a plain Python list (O(1) append) and joined once at
      the end for the metrics/cache record.

    ``writelines`` is recorded too: left to ``__getattr__``, it went straight
    to the real stream and a ``sys.stdout.writelines([...])`` was missing
    from the text cached and replayed on a hit.

    **Known gap**: ``sys.stdout.buffer`` (the binary stream some libraries
    write raw bytes to directly) is forwarded untouched by ``__getattr__`` and
    is NOT recorded. Recording it would need a second, byte-oriented tee wired
    through ``.buffer``; text ``write``/``writelines`` (what ``print`` and
    nearly every library use) are the channels this class covers.
    """

    def __init__(self, real_stream: Any, chunks: list[str] | None = None) -> None:
        self._real = real_stream
        self._chunks: list[str] = chunks if chunks is not None else []

    def write(self, s: str) -> int:
        self._real.write(s)
        self._chunks.append(s)
        return len(s)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._real.flush()

    def getvalue(self) -> str:
        """Return all accumulated text."""
        return "".join(self._chunks)

    # Forward attribute access (encoding, fileno, etc.) to the real stream
    # so libraries that introspect the stream object still work.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _Passthrough:
    """A stream's own ``write`` and ``flush``, for a sink to pass writes on to
    while :class:`StreamTaken` has the stream's ``write`` pointed at it."""

    def __init__(self, stream: Any, write: Callable[[str], Any]) -> None:
        self._stream = stream
        self.write = write

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class StreamTaken:
    """Sends what is written to ``sys.<name>`` to a sink, leaving the stream in place.

    Putting a ``StringIO`` or a :class:`TeeWriter` in ``sys.stdout`` showed
    code a different stream than plain gives it: ``type(sys.stdout)`` was not
    ipykernel's ``OutStream``, ``sys.stdout.name`` raised. The stream stays;
    its ``write`` and ``writelines`` are pointed at the sink, on the instance.
    *make_sink* gets what the sink passes writes on to: the stream's own
    ``write``, or the stream itself when it takes no attributes, in which case
    the sink goes in ``sys.<name>`` as before.
    """

    _METHODS = ("write", "writelines")

    def __init__(self, name: str, make_sink: Callable[[Any], Any]) -> None:
        self.name = name
        self._make_sink = make_sink
        self.sink: Any = None
        self._stream: Any = None
        self._own: dict[str, Any] = {}
        self._swapped = False

    def __enter__(self) -> StreamTaken:
        stream = self._stream = getattr(sys, self.name)
        own = attribute_or(stream, "__dict__", None)
        write = attribute_or(stream, "write", None)
        if own is None or not callable(write):
            return self._swap()
        self._own = {m: own[m] for m in self._METHODS if m in own}
        self.sink = self._make_sink(_Passthrough(stream, write))
        try:
            stream.write = self.sink.write
            stream.writelines = self.sink.writelines
        except (AttributeError, TypeError):
            self._put_back()
            return self._swap()
        return self

    def _swap(self) -> StreamTaken:
        self.sink = self._make_sink(self._stream)
        setattr(sys, self.name, self.sink)
        self._swapped = True
        return self

    def _put_back(self) -> None:
        for m in self._METHODS:
            with contextlib.suppress(AttributeError, TypeError):
                if m in self._own:
                    setattr(self._stream, m, self._own[m])
                elif m in attribute_or(self._stream, "__dict__", {}):
                    delattr(self._stream, m)  # back to the class's method

    def __exit__(self, *exc: Any) -> None:
        if not self._swapped:
            self._put_back()
        elif getattr(sys, self.name) is self.sink:
            setattr(sys, self.name, self._stream)
        # else: the code rebound ``sys.<name>`` itself (``sys.stdout = buf``),
        # and keeps it, as in plain.

    def getvalue(self) -> str:
        return self.sink.getvalue()
