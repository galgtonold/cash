"""Which entries the disk cap removed, so a later miss can say so.

A miss on an entry the cap evicted looked like any other: "evicted or
cleared" at best, and nothing at all about the minutes the recompute took.
So each eviction appends one note per entry it removed, and a miss asks
whether its key has one (`FileBackend.eviction_note`).

A note holds the entry's file name, which is a SHA-256 of its key, when it
was evicted, what the value took to compute (``-1`` when this process never
saw its metadata) and its size. Never the key, the arguments or the value.

Like the rank index it is ADVISORY: any process appends, a line that does not
parse is skipped, and a lost file only means a miss says "evicted or cleared"
again. It is read once per process, on the first question, into a dict every
later question looks up; this process's own appends and removals keep that dict
current. Past `MAX_NOTES` it is compacted to the newest ones.

Format, one record per line::

    <entry filename stem> <evicted_at> <compute seconds> <size>
    - <entry filename stem>        (stored again: the note no longer applies)
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, NamedTuple

from .cache_dir import EVICTIONS_FILENAME

logger = logging.getLogger(__name__)

__all__ = ["EvictionLog", "EvictionNote"]

_REMOVED = "-"


class EvictionNote(NamedTuple):
    """What the cap removed: when, what it had cost to compute, how big."""

    evicted_at: float
    #: Compute seconds, or a negative number when unknown.
    seconds: float
    size: int


class EvictionLog:
    """Append-only eviction notes for one cache directory, compacted on demand."""

    #: Notes kept. Measured: 10,000 is about 1 MB on disk, 2 MB in memory and
    #: a 2 ms load; 100,000 was 23 MB and 23 ms, for misses on entries evicted
    #: so long ago that the note no longer says much.
    MAX_NOTES = 10_000

    #: The file size that triggers compaction to the newest `MAX_NOTES`. A
    #: note line is about 95 bytes, so this is a little over `MAX_NOTES`
    #: lines, and a compacted file is well under it -- it cannot re-trigger.
    COMPACT_AT_BYTES = MAX_NOTES * 128

    def __init__(self, cache_dir: str, untracked: Any) -> None:
        self.path = os.path.join(cache_dir, EVICTIONS_FILENAME)
        self._untracked = untracked
        self._lock = threading.Lock()
        #: stem -> note, oldest first; None until the first question.
        self._notes: dict[str, EvictionNote] | None = None

    # -- reading ---------------------------------------------------------

    def _read(self) -> dict[str, EvictionNote]:
        """The notes on disk, oldest first, the newest `MAX_NOTES` of them."""
        notes: dict[str, EvictionNote] = {}
        try:
            with self._untracked(), open(self.path, encoding="ascii", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == _REMOVED:
                        notes.pop(parts[1], None)
                        continue
                    if len(parts) != 4:
                        continue
                    try:
                        note = EvictionNote(float(parts[1]), float(parts[2]), int(parts[3]))
                    except ValueError:
                        continue
                    if not (math.isfinite(note.evicted_at) and math.isfinite(note.seconds)):
                        continue
                    # Moved to the end: the order is each entry's LAST eviction.
                    notes.pop(parts[0], None)
                    notes[parts[0]] = note
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not read eviction log %s", self.path, exc_info=True)
        excess = len(notes) - self.MAX_NOTES
        if excess > 0:
            for stem in list(notes)[:excess]:
                del notes[stem]
        return notes

    def lookup(self, stem: str) -> EvictionNote | None:
        """*stem*'s note, if the cap evicted it. Reads the file on the first call."""
        with self._lock:
            if self._notes is None:
                self._notes = self._read()
            return self._notes.get(stem)

    # -- writing ---------------------------------------------------------

    def _append(self, lines: list[str]) -> None:
        """One write for all of *lines*. Never creates the directory: one that
        is gone was cleared, and a note about a cleared cache means nothing."""
        try:
            with self._untracked(), open(self.path, "a", encoding="ascii") as fh:
                fh.write("".join(lines))
                size = fh.tell()
        except OSError:
            logger.debug("Could not append to eviction log %s", self.path, exc_info=True)
            return
        if size > self.COMPACT_AT_BYTES:
            self._compact()

    def record(self, notes: list[tuple[str, EvictionNote]]) -> None:
        """Note entries the cap has just removed."""
        if not notes:
            return
        lines = [f"{stem} {n.evicted_at:.0f} {n.seconds:.6g} {int(n.size)}\n" for stem, n in notes]
        with self._lock:
            if self._notes is not None:
                for stem, note in notes:
                    self._notes.pop(stem, None)
                    self._notes[stem] = note
            self._append(lines)

    def forget(self, stem: str) -> None:
        """*stem* was stored again, so its note no longer applies.

        Costs nothing until the notes are loaded, which a store's own miss has
        normally done. A note a store leaves behind unloaded only makes a
        later miss of that key say "evicted" where something else removed it.
        """
        with self._lock:
            if self._notes is None or self._notes.pop(stem, None) is None:
                return
            self._append([f"{_REMOVED} {stem}\n"])

    def _compact(self) -> None:
        """Rewrite as the newest `MAX_NOTES` notes, through a temp file and a
        rename. Re-reads the file, so what other processes appended is kept;
        a failed rename leaves the longer, still-correct file."""
        notes = self._read()
        tmp = f"{self.path}.{os.getpid()}.tmp"
        body = "".join(f"{stem} {n.evicted_at:.0f} {n.seconds:.6g} {int(n.size)}\n" for stem, n in notes.items())
        try:
            with self._untracked():
                with open(tmp, "w", encoding="ascii") as fh:
                    fh.write(body)
                os.replace(tmp, self.path)
        except OSError:
            logger.debug("Could not compact eviction log %s", self.path, exc_info=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return
        if self._notes is not None:
            self._notes = notes

    def remove(self) -> None:
        """The directory was emptied: no entry in it was evicted."""
        with self._lock:
            self._notes = None
            try:
                with self._untracked():
                    os.remove(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("Could not remove eviction log %s", self.path, exc_info=True)
