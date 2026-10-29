"""`TeeWriter`: record what is written to a stream while passing it through.

A leaf module, so both the statement pipeline (``statement/capture.py``) and a
cached call (``call_unit.py``, which sits beneath the statement package in the
import graph) use the one writer.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from cash._user_attrs import attribute_or
from cash.live_state import live_reduce

__all__ = ["StreamTaken", "TeeWriter"]


class TeeWriter:
    """A writer that sends output to both a real stream and a list buffer.

    Output is forwarded to the real stream (e.g. Jupyter's IOPub) in batches
    controlled by a time-based flush policy.  This avoids the O(n²) behaviour
    of flushing after every single ``write()`` call — which, in Jupyter, sends
    one ZMQ message per flush, causing extreme slowdown for tight loops with
    many print calls.

    Instead we:
    * Always call ``self._real.write(s)`` so the data enters the kernel's
      output buffer immediately.
    * Only call ``self._real.flush()`` if ≥ ``_FLUSH_INTERVAL_S`` seconds
      have elapsed since the last flush.  Jupyter's own ``OutStream`` uses a
      similar strategy (~200 ms batching).
    * Accumulate text in a plain Python list (O(1) append) and join once at
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

    _FLUSH_INTERVAL_S = 0.1  # seconds – matches ipykernel's default batch interval

    def __init__(self, real_stream: Any, chunks: list[str] | None = None) -> None:
        self._real = real_stream
        self._chunks: list[str] = chunks if chunks is not None else []
        self._last_flush = time.monotonic()

    def write(self, s: str) -> int:
        self._real.write(s)
        self._chunks.append(s)
        now = time.monotonic()
        if now - self._last_flush >= self._FLUSH_INTERVAL_S:
            self._real.flush()
            self._last_flush = now
        return len(s)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._real.flush()
        self._last_flush = time.monotonic()

    def getvalue(self) -> str:
        """Return all accumulated text."""
        return "".join(self._chunks)

    def _cash_wrapped_stream(self) -> Any:
        """The stream this writer passes writes on to."""
        real = self._real
        return real._cash_wrapped_stream() if type(real) is _Passthrough else real

    def __reduce_ex__(self, protocol: Any) -> Any:
        return _stream_reduce(self)

    # Forward attribute access (encoding, fileno, etc.) to the real stream
    # so libraries that introspect the stream object still work.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _stream_reduce(writer: Any) -> tuple:
    """How a writer standing in for a stream is stored: as the stream.

    Code holds what it finds while a cell runs: ``ProgBar(stream=sys.stdout)``
    keeps ``sys.stdout.write``, which is a `TeeWriter`'s then. Plain Python
    holds the stream's own ``write``, so the writer is stored as the stream
    it writes to (`live_state`): ``sys.stdout`` or ``sys.stderr``, or not at
    all. Left to pickle's default, it was stored as an empty writer whose
    ``__getattr__`` recursed as soon as it was read back.
    """
    reduced = live_reduce(writer)
    if reduced is None:
        raise TypeError(
            f"cannot pickle {type(writer).__name__!r} object: it writes to a stream that is not sys.stdout or sys.stderr"
        )
    return reduced


class _Passthrough:
    """A stream's own ``write`` and ``flush``, for a sink to pass writes on to
    while :class:`StreamTaken` has the stream's ``write`` pointed at it."""

    def __init__(self, stream: Any, write: Callable[[str], Any]) -> None:
        self._stream = stream
        self.write = write

    def flush(self) -> None:
        self._stream.flush()

    def _cash_wrapped_stream(self) -> Any:
        """The stream whose ``write`` this passes writes on to."""
        return self._stream

    def __reduce_ex__(self, protocol: Any) -> Any:
        return _stream_reduce(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _StreamMethod:
    """``stream.<name>`` while :class:`StreamTaken` points it at a sink.

    Code keeps what it finds: ``ProgBar(stream=sys.stdout)`` holds
    ``sys.stdout.write``. Stored, that is the stream's own method again
    (``getattr(stream, name)``), as plain Python holds it; a sink's bound
    method stored as the sink, which pickles apart from the stream beside
    it, so the object hashed one way computed and another restored.

    Once its `StreamTaken` has ended, a call goes where a call of the
    stream's method goes now, so the statement capturing then records it,
    as it records a bar made in plain Python; from inside that call, to
    what the stream's method was when this one was put in place (*under*).
    """

    __slots__ = ("_call", "_name", "_stream", "_under", "active")

    def __init__(self, stream: Any, name: str, call: Callable[..., Any], under: Callable[..., Any]) -> None:
        self._stream = stream
        self._name = name
        self._call = call
        self._under = under
        self.active = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.active:
            return self._call(*args, **kwargs)
        if _PASSING.on:
            return self._under(*args, **kwargs)
        _PASSING.on = True
        try:
            return getattr(self._stream, self._name)(*args, **kwargs)
        finally:
            _PASSING.on = False

    def __reduce_ex__(self, protocol: Any) -> Any:
        return getattr, (self._stream, self._name)


class _Passing(threading.local):
    """Set while an ended `_StreamMethod` passes a call on to the stream."""

    on = False


_PASSING = _Passing()


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
        #: The methods this put in place, ended with it.
        self._taken: dict[str, _StreamMethod] = {}
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
            self._taken = {
                "write": _StreamMethod(stream, "write", self.sink.write, write),
                "writelines": _StreamMethod(
                    stream, "writelines", self.sink.writelines, attribute_or(stream, "writelines", self.sink.writelines)
                ),
            }
            stream.write = self._taken["write"]
            stream.writelines = self._taken["writelines"]
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
            if m in self._taken:
                self._taken[m].active = False
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
