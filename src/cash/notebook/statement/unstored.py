"""Why a statement that ran has no entry on disk, so its next miss can say so.

A statement that ran and was not stored -- refused (it wrote a file), or kept
in memory only (cheaper to run again than to read back) -- misses after a
restart with nothing to compare: no entry, no version, no record of this
session. ``faces = fetch_lfw_people(...)`` ran again after each restart of a
notebook with no reason given. So a statement that took long enough to
matter notes, under its key, why it was not stored; a miss on that key reads
the note (`UnstoredNotes.reason`).

Like the eviction log it is ADVISORY: one line per note, appended by any
process, read once per process; a line that does not parse is skipped, and
a lost file only means a miss says nothing again. Stored on disk later, the
key's note is withdrawn.

Format, one record per line::

    <key>\t<reason>
    slot:<version slot>\t<reason>   (the statement's last run, whatever its key)
    -\t<key>          (stored on disk since: the note no longer applies)
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable

from ...tracking.tracker_context import untracked

logger = logging.getLogger(__name__)

__all__ = ["UnstoredNotes"]

_REMOVED = "-"
#: The prefix of a note under a statement's version slot.
_SLOT = "slot:"
#: Notes kept; past twice this the file is rewritten with the newest.
_MAX_NOTES = 10_000
#: The longest reason kept.
_REASON_CHARS = 240

#: What a statement kept in memory only is told after a restart.
MEMORY_ONLY = (
    "it was kept in memory only when it last ran (not worth the disk it would take), and a restart clears memory"
)


class UnstoredNotes:
    """``cache key -> why it has no entry on disk``, for one cache directory."""

    FILENAME = "unstored_notes.log"

    def __init__(self, cache_dir: str | None) -> None:
        self.path = os.path.join(cache_dir, self.FILENAME) if cache_dir else None
        self._notes: dict[str, str] | None = None
        self._lines = 0
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        notes: dict[str, str] = {}
        lines = 0
        if self.path is None:
            return notes
        try:
            with untracked(), open(self.path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    lines += 1
                    key, sep, reason = line.rstrip("\n").partition("\t")
                    if not sep or not reason:
                        continue
                    if key == _REMOVED:
                        notes.pop(reason, None)
                    else:
                        notes.pop(key, None)
                        notes[key] = reason
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not read %s", self.path, exc_info=True)
        self._lines = lines
        return notes

    def _loaded(self) -> dict[str, str]:
        if self._notes is None:
            self._notes = self._load()
        return self._notes

    def reason(self, key: str | None, slot: str | None = None) -> str | None:
        """The miss reason for *key* when the run before did not store it;
        failing that, for the statement's *slot* (`version_slot`): the last
        run of the same statement, under other inputs, left nothing on disk
        either, so there was no entry to serve whatever the inputs."""
        if self.path is None:
            return None
        with self._lock:
            notes = self._loaded()
            why = notes.get(key) if key else None
            if why is None and slot:
                why = notes.get(_SLOT + slot)
        return f"not stored on disk when it last ran: {why}" if why else None

    def note(self, key: str | None, why: str | None, slot: str | None = None) -> None:
        """Note *why* the run under *key* (of the statement's *slot*) left no
        entry on disk; with *why* None, that there is nothing to say (the
        notes are withdrawn)."""
        names = [n for n in (key, _SLOT + slot if slot else None) if n and "\t" not in n and "\n" not in n]
        if not names or self.path is None:
            return
        if why is not None:
            why = " ".join(why.split())[:_REASON_CHARS]
        with self._lock:
            notes = self._loaded()
            lines = []
            for name in names:
                if why is None:
                    if notes.pop(name, None) is not None:
                        lines.append(f"{_REMOVED}\t{name}\n")
                elif notes.get(name) != why:
                    notes.pop(name, None)
                    notes[name] = why
                    lines.append(f"{name}\t{why}\n")
            if lines:
                self._append(lines)

    def _append(self, lines: Iterable[str]) -> None:
        lines = list(lines)
        try:
            with untracked():
                if self._lines + len(lines) > 2 * _MAX_NOTES:
                    self._compact()
                else:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write("".join(lines))
                    self._lines += len(lines)
        except OSError:
            logger.debug("Could not write %s", self.path, exc_info=True)

    def _compact(self) -> None:
        notes = self._notes or {}
        while len(notes) > _MAX_NOTES:
            del notes[next(iter(notes))]
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("".join(f"{k}\t{v}\n" for k, v in notes.items()))
        os.replace(tmp, self.path)
        self._lines = len(notes)
