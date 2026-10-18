"""Lines for a cache directory's append-only logs, written a batch at a time.

The file-digest table (`digest_table`) and the file versions
(`file_versions`) note a line for each file a statement reads or writes.
Opened for each line, a log cost an ``open`` and a ``close`` per file: under
cash's own open hooks ~120 us each on Linux, and more on Windows, where a
virus scanner looks at every open; E10's loop over 50,000 files spent ~10 s
there. A line now waits for others and is written with them, at once in one
``write``, when `FLUSH_LINES` are waiting, when the oldest has waited
`FLUSH_SECONDS` and another comes, when a cell ends, when the process
starts another (``subprocess``, ``os.system``, ``os.fork``, ...), so a
script a cell runs knows what the cell noted, and when the process exits.

A line that is not written costs a read, never a wrong answer: a digest is
looked up under the exact stat it was taken at, and a version links two
stats only where the bytes were found the same, so a process that lacks the
line hashes the file, or keys a reader on the file's own stat and runs it
again. A kernel killed in the middle of a cell (a restart waits for the
cell to end), or a process that ends by ``os._exit``, loses only that.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time
import weakref

logger = logging.getLogger(__name__)

__all__ = ["FLUSH_LINES", "FLUSH_SECONDS", "LogLines", "flush_all"]

#: Lines waiting before they are written.
FLUSH_LINES = 256
#: Seconds the oldest waiting line waits, at most, for the next to come.
FLUSH_SECONDS = 0.5

_LOGS: weakref.WeakSet[LogLines] = weakref.WeakSet()


class LogLines:
    """The lines waiting for the log at *path*; *lock* guards them (the
    owner's lock, held around `add`, `flush_locked` and `drop`)."""

    def __init__(self, path: str, lock: threading.Lock, label: str) -> None:
        self.path = path
        self.lock = lock
        self._label = label
        self._waiting: list[str] = []
        self._since = 0.0
        _LOGS.add(self)

    def add(self, line: str) -> None:
        """Queue *line* (lock held), and write the queue when it is due."""
        if not self._waiting:
            self._since = time.monotonic()
        self._waiting.append(line)
        if len(self._waiting) >= FLUSH_LINES or time.monotonic() - self._since >= FLUSH_SECONDS:
            self.flush_locked()

    def drop(self) -> None:
        """Forget the waiting lines (lock held): the log was rewritten whole."""
        self._waiting.clear()

    def flush_locked(self) -> None:
        """Append the waiting lines in one write (lock held)."""
        if not self._waiting:
            return
        text, self._waiting = "".join(self._waiting), []
        if not os.path.isdir(os.path.dirname(self.path)):
            return  # a cache dir that is gone: never recreated from here
        try:
            # Append mode, in one write: lines from two processes interleave whole.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            logger.debug("[%s] could not write to %s", self._label, self.path, exc_info=True)

    def flush(self) -> None:
        with self.lock:
            self.flush_locked()


def flush_all() -> None:
    """Write every log's waiting lines: at the end of a cell, and at exit."""
    for log in list(_LOGS):
        try:
            log.flush()
        except Exception:  # a log line never breaks a cell or an exit
            logger.debug("could not write a log's lines", exc_info=True)


#: Audit events of a process starting another, which may read the logs.
#: (``multiprocessing``'s ``spawn`` on POSIX starts its worker with no
#: event: it reads the lines written so far.)
_STARTS = frozenset(
    {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.exec", "os.startfile", "_winapi.CreateProcess"}
)


def _on_audit(event: str, _args: tuple) -> None:
    if event in _STARTS:
        flush_all()


atexit.register(flush_all)
sys.addaudithook(_on_audit)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=flush_all)
