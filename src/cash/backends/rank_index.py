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

The file is append-only between compactions, and every process reads all of
it once, so it is kept proportional to the number of entries: whoever finds
it holding many more lines than entries rewrites it as one line per entry
that still exists (`RankIndex.COMPACT_RATIO`), and a process that appended
to it leaves it compact when it closes.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

from .cache_dir import RANK_INDEX_FILENAME
from .entry_format import ENTRY_SUFFIX

logger = logging.getLogger(__name__)

__all__ = ["RankIndex"]

_CLOCK_TAG = "@L"


class RankIndex:
    """Append-only priorities for one cache directory, compacted as it grows."""

    #: Records buffered before they are written. An append per write was an
    #: open, write and close on every cache write: measured on Windows at
    #: ~500 us, about doubling the cost of a 1 KB write. Buffering is safe
    #: because the file is advisory -- a process killed with records still
    #: buffered leaves those entries ranked as unknown-cost until read again.
    BATCH = 64

    #: The file is rewritten once it holds more than this many lines per
    #: entry (plus `COMPACT_SLACK`). Every write and every flushed read
    #: appends a line, so without it the file grew with the number of runs,
    #: not the number of entries, and each process parsed all of it. At 2,
    #: a rewrite of N lines is paid for by N appended ones.
    COMPACT_RATIO = 2.0
    #: The ratio a process leaves behind when it closes (`close`): the next
    #: process then starts from a file of about one line per entry.
    CLOSE_RATIO = 1.1
    #: Lines over the ratio that are let be, so a small index is not
    #: rewritten for a handful of lines.
    COMPACT_SLACK = 64

    def __init__(self, cache_dir: str, untracked: Any) -> None:
        self.cache_dir = cache_dir
        self.path = os.path.join(cache_dir, RANK_INDEX_FILENAME)
        #: Lines in the file and distinct entries among them, as of the last
        #: `load` plus this process's appends since. None until a `load`:
        #: a process that never ranks never reads the file to count it.
        self._lines: int | None = None
        self._stems = 0
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
        stamp. A file grown past `COMPACT_RATIO` is read again and rewritten."""
        if self._append_buffered() and self._outgrown(self.COMPACT_RATIO):
            self.load()

    def close(self) -> None:
        """`flush`, and leave the file near one line per entry (`CLOSE_RATIO`)
        if this process is the one that made it longer."""
        appended = self._append_buffered()
        if (appended or self._lines is not None) and self._outgrown(self.CLOSE_RATIO):
            self.load(ratio=self.CLOSE_RATIO)

    def _append_buffered(self) -> bool:
        """Append the buffer to the file; whether anything was written."""
        with self._lock:
            if not self._buffer:
                return False
            lines, self._buffer = self._buffer, []
            try:
                with self._untracked(), open(self.path, "a", encoding="ascii") as fh:
                    fh.write("".join(lines))
            except OSError:
                logger.debug("Could not append to rank index %s", self.path, exc_info=True)
                return False
            if self._lines is not None:
                self._lines += len(lines)
            return True

    def _outgrown(self, ratio: float) -> bool:
        """By this process's count, does the file hold more than *ratio*
        lines per entry? Unknown before the first `load`."""
        return self._lines is not None and self._lines > ratio * self._stems + self.COMPACT_SLACK

    def discard(self) -> None:
        """Drop buffered records (the cache they describe was cleared)."""
        with self._lock:
            self._buffer = []

    def load(self, compact: bool = True, ratio: float | None = None) -> tuple[dict[str, float], float, int]:
        """``(priority by stem, clock, line count)``; empty when unreadable.
        Flushes this process's buffer first, so the file is the whole story.
        The priorities are in the order they were last recorded, oldest first.

        With *compact*, a file holding more than *ratio* (`COMPACT_RATIO`)
        lines per entry is rewritten as it is read: one line for each entry
        still in the directory. The count returned is the file's before that.
        """
        self._append_buffered()
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
        with self._lock:
            self._lines, self._stems = count, len(ranks)
        if compact and count > (ratio or self.COMPACT_RATIO) * len(ranks) + self.COMPACT_SLACK:
            self.compact(self._still_stored(ranks), clock)
        return ranks, clock, count

    def _still_stored(self, ranks: dict[str, float]) -> dict[str, float]:
        """*ranks* without the entries that are gone (evicted, cleared,
        replaced): their lines would otherwise outlive them for ever. One
        listing of the directory, names only; all of *ranks* when it cannot
        be listed."""
        try:
            with self._untracked(), os.scandir(self.cache_dir) as found:
                stored = {e.name[: -len(ENTRY_SUFFIX)] for e in found if e.name.endswith(ENTRY_SUFFIX)}
        except OSError:
            return ranks
        return {stem: priority for stem, priority in ranks.items() if stem in stored}

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
                self._lines, self._stems = len(ranks) + 1, len(ranks)
        except OSError:
            logger.debug("Could not compact rank index %s", self.path, exc_info=True)
            try:
                os.remove(tmp)
            except OSError:
                pass

    def remove(self) -> None:
        self.discard()
        with self._lock:
            if self._lines is not None:
                self._lines = self._stems = 0
        try:
            with self._lock, self._untracked():
                os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove rank index %s", self.path, exc_info=True)
