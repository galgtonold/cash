"""The disk tier's eviction priorities, kept where a ranking can afford to read them.

``FileBackend`` ranks eviction candidates from one ``scandir``, because opening
every entry to read its header is ruinous: measured on Windows at 111 us an
entry warm and 5.7 ms cold (antivirus scanning each open), against 2.3 us for
the walk -- 2.2 s to 113 s per ranking at 20k entries. ``scandir`` yields only
size and mtime, and a value-per-byte ranking needs what each entry cost to
compute. So each write and each access flush appends that entry's priority
here, and a ranking reads this one file.

It is ADVISORY. Several processes append to it and any of them may die
mid-line; it can be deleted, or be older than the entries it describes. None
of that is an error: a line that does not parse is skipped, the last line for
an entry wins, and an entry with no line is ranked as if its cost were
unknown. What is lost is ranking precision, never a value.

Format, one record per line::

    <entry filename stem> <priority>
    @L <clock>

The clock line records GreedyDual's inflation value L, so a new process
resumes it instead of restarting at zero (which would rank everything it
writes below everything already on disk).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["RankIndex", "INDEX_FILENAME"]

#: No entry suffix, so every entry glob (listing, sizing, clearing, format
#: migration) passes it by.
INDEX_FILENAME = "_rank.log"

_CLOCK_TAG = "@L"


class RankIndex:
    """Append-only priorities for one cache directory, compacted on demand."""

    #: Records buffered before they are written. An append per write was an
    #: open, write and close on every cache write: measured on Windows at
    #: ~500 us, about doubling the cost of a 1 KB write. Buffering is safe
    #: because the file is advisory -- a process killed with records still
    #: buffered leaves those entries ranked as unknown-cost until read again.
    BATCH = 64

    def __init__(self, cache_dir: str, untracked: Any) -> None:
        self.path = os.path.join(cache_dir, INDEX_FILENAME)
        # cash's own I/O must never become a dependency of the user's code:
        # the caller passes the same `untracked()` its directory walk uses.
        self._untracked = untracked
        self._lock = threading.Lock()
        self._buffer: list[str] = []

    def append(self, records: list[tuple[str, float]], clock: float | None = None) -> None:
        """Buffer priorities, and optionally the clock; written every `BATCH`
        lines and on `flush`."""
        lines = [f"{stem} {priority!r}\n" for stem, priority in records]
        if clock is not None:
            lines.append(f"{_CLOCK_TAG} {clock!r}\n")
        if not lines:
            return
        with self._lock:
            self._buffer.extend(lines)
            full = len(self._buffer) >= self.BATCH
        if full:
            self.flush()

    def flush(self) -> None:
        """Write what is buffered, in one ``write`` so a process's lines stay
        together. Never creates the directory: a directory that is gone was
        cleared, and recreating it here would resurrect a cache without its
        stamp."""
        with self._lock:
            if not self._buffer:
                return
            body, self._buffer = "".join(self._buffer), []
            try:
                with self._untracked(), open(self.path, "a", encoding="ascii") as fh:
                    fh.write(body)
            except OSError:
                logger.debug("Could not append to rank index %s", self.path, exc_info=True)

    def discard(self) -> None:
        """Drop buffered records (the cache they describe was cleared)."""
        with self._lock:
            self._buffer = []

    def load(self) -> tuple[dict[str, float], float, int]:
        """``(priority by stem, clock, line count)``; empty when unreadable.
        Flushes this process's buffer first, so the file is the whole story.
        The priorities are in the order they were last recorded, oldest first."""
        self.flush()
        ranks: dict[str, float] = {}
        clock = 0.0
        count = 0
        try:
            with self._untracked(), open(self.path, encoding="ascii", errors="replace") as fh:
                for line in fh:
                    count += 1
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    try:
                        value = float(parts[1])
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        continue
                    if parts[0] == _CLOCK_TAG:
                        clock = max(clock, value)
                    else:
                        # Moved to the end, so the dict's order is each
                        # entry's LAST record: the order of writes and access
                        # flushes, which a ranking breaks ties by.
                        ranks.pop(parts[0], None)
                        ranks[parts[0]] = value
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not read rank index %s", self.path, exc_info=True)
        return ranks, clock, count

    def compact(self, ranks: dict[str, float], clock: float) -> None:
        """Rewrite the index as exactly *ranks* and *clock*.

        Through a temp file and a rename, so a reader never sees half of it.
        A failed rename (Windows refuses while another process has the file
        open) leaves the old, longer index in place -- still correct, only
        bigger -- and the next ranking tries again.
        """
        tmp = f"{self.path}.{os.getpid()}.tmp"
        body = "".join(f"{stem} {priority!r}\n" for stem, priority in ranks.items())
        body += f"{_CLOCK_TAG} {clock!r}\n"
        try:
            with self._lock, self._untracked():
                with open(tmp, "w", encoding="ascii") as fh:
                    fh.write(body)
                os.replace(tmp, self.path)
        except OSError:
            logger.debug("Could not compact rank index %s", self.path, exc_info=True)
            try:
                os.remove(tmp)
            except OSError:
                pass

    def remove(self) -> None:
        self.discard()
        try:
            with self._lock, self._untracked():
                os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove rank index %s", self.path, exc_info=True)
