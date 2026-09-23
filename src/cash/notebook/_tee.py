"""`TeeWriter`: record what is written to a stream while passing it through.

A leaf module, so both the statement pipeline (``statement/capture.py``) and a
cached call (``call_unit.py``, which sits beneath the statement package in the
import graph) use the one writer.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

__all__ = ["TeeWriter"]


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

    # Forward attribute access (encoding, fileno, etc.) to the real stream
    # so libraries that introspect the stream object still work.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)
