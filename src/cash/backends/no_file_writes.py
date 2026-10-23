"""Rebuilding a stored value may not change a file.

Turning stored bytes back into an object runs the value's own code
(``__setstate__``, a constructor a pickle names). A class of the user's
pickled to reopen its sink or log opened the file with ``"w"`` on every
rebuild: once for the RAM tier's copy of each store, once for every disk
hit, and the file was emptied each time, where plain Python opened it once.
Inside `no_file_writes` such a rebuild fails instead, and the value is
neither stored nor served.

What it refuses is what Python audits: an ``open`` that writes, truncates,
creates or appends, ``os.truncate``, ``os.remove``, ``os.rename``,
``os.rmdir``, ``os.mkdir``, ``os.symlink``, ``os.link``, and a
``sqlite3.connect`` that would create its file. A write through a file
already open (one the value held, a descriptor) is not audited, and not
seen.

What it lets through is a scratch file or folder of the rebuild's own:
made new in the temp folder (an exclusive create, a ``mkdir``) and written,
renamed or removed there. Nothing that was there before changes. Python's
own ``.pyc`` writes for a module the rebuild imports are not made
(``sys.dont_write_bytecode``), so the import itself succeeds.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ..tracking import io_watch

#: The audit events a rebuild may not raise, unless on its own scratch.
EVENTS = frozenset(
    (
        "open",
        "os.truncate",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.mkdir",
        "os.symlink",
        "os.link",
        "sqlite3.connect",
    )
)

_WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_TMPFILE = getattr(os, "O_TMPFILE", 0)


class WouldWriteAFile(pickle.UnpicklingError):
    """Rebuilding a stored value would change a file (`no_file_writes`).

    An `UnpicklingError`, so every backend reads such an entry as a miss
    (`serialization.RESTORE_ERRORS`)."""


def _path(value: object) -> str | None:
    try:
        return os.path.abspath(os.fsdecode(value))  # type: ignore[arg-type]
    except TypeError:
        return None  # a file descriptor


class NoFileWrites:
    """The audit check of one rebuild: a change to a file fails, and is
    remembered, so a rebuild that catches the error still fails."""

    __slots__ = ("made", "refused")

    def __init__(self) -> None:
        self.refused: str | None = None
        #: The scratch files and folders this rebuild made in the temp folder.
        self.made: list[str] = []

    def __call__(self, event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = (*args, None, None, None)[:3]
            if isinstance(path, int):
                return  # a descriptor already open: wrapping it changes nothing
            if isinstance(flags, int):
                if not flags & _WRITES or (_TMPFILE and flags & _TMPFILE == _TMPFILE):
                    return  # a read, or a file with no name (``TemporaryFile``)
                if flags & os.O_CREAT and flags & os.O_EXCL and self._scratch(path, made=True):
                    return
            if self._own(path) or os.path.isdir(path):
                # A folder is not written by opening it: ``NamedTemporaryFile``
                # opens the temp folder with its mode and an opener, whose own
                # open of the file is audited as well.
                return
            self._refuse(f"open({path!r}, {mode!r})")
        elif event == "os.mkdir":
            if _no_dir_fd(args[2]) and self._scratch(args[0], made=True):
                return
            self._refuse(f"os.mkdir({args[0]!r})")
        elif event == "sqlite3.connect":
            database = args[0]
            if database in (":memory:", b":memory:", "") or (_path(database) and os.path.exists(_path(database))):
                return  # an existing file: what it does with it is a write through a handle
            self._refuse(f"sqlite3.connect({database!r})")
        elif event in ("os.rename", "os.link", "os.symlink"):
            src, dst = args[0], args[1]
            if self._own(dst) and (event == "os.symlink" or self._own(src)):
                return
            self._refuse(f"{event}({src!r}, {dst!r})")
        else:  # os.truncate, os.remove, os.rmdir
            path = args[0]
            in_dir = event != "os.truncate" and not _no_dir_fd(args[1])
            if not in_dir and self._own(path):
                return
            if in_dir and self.made and self._relative_name(path):
                # How ``shutil.rmtree`` empties a folder, its files named
                # inside it: a scratch folder being removed.
                return
            self._refuse(f"{event}({path!r})")

    def _scratch(self, value: object, *, made: bool) -> bool:
        """Is *value* a new name right in the temp folder (where ``mkstemp``
        and ``mkdtemp`` make theirs), or in a scratch folder? Recorded as this
        rebuild's own when *made*."""
        path = _path(value)
        if path is None or os.path.lexists(path):
            return False
        parent = os.path.dirname(path)
        if not (_same(parent, tempfile.gettempdir()) or self._own(parent)):
            return False
        if made:
            self.made.append(path)
        return True

    def _own(self, value: object) -> bool:
        path = _path(value)
        return path is not None and any(_under(path, made) for made in self.made)

    @staticmethod
    def _relative_name(value: object) -> bool:
        name = os.fsdecode(value) if isinstance(value, (str, bytes)) else ""
        return bool(name) and os.sep not in name and name not in (".", "..")

    def _refuse(self, what: str) -> None:
        self.refused = what
        raise WouldWriteAFile(f"rebuilding it would {what}")

    def check(self) -> None:
        """Raise if the rebuild asked for a change to a file, caught or not."""
        if self.refused is not None:
            raise WouldWriteAFile(f"rebuilding it would {self.refused}")


def _no_dir_fd(value: object) -> bool:
    return value is None or value == -1  # audited as -1 when not given


def _same(path: str, folder: str) -> bool:
    return path in (folder, os.path.realpath(folder)) or os.path.realpath(path) == os.path.realpath(folder)


def _under(path: str, folder: str) -> bool:
    for root in {folder, os.path.realpath(folder)}:
        if path == root or path.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


_bytecode_lock = threading.Lock()
_bytecode_holds = 0
_bytecode_before = False


@contextmanager
def no_file_writes() -> Iterator[NoFileWrites]:
    """While inside, on this thread, a change to a file fails with
    `WouldWriteAFile`. Call ``check()`` on what it yields after the
    rebuild: the error may have been caught inside it."""
    global _bytecode_holds, _bytecode_before
    check = NoFileWrites()
    with _bytecode_lock:
        if _bytecode_holds == 0:
            _bytecode_before = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
        _bytecode_holds += 1
    try:
        with io_watch.refusing(EVENTS, check):
            yield check
    finally:
        with _bytecode_lock:
            _bytecode_holds -= 1
            if _bytecode_holds == 0:
                sys.dont_write_bytecode = _bytecode_before


def would_write_a_file(exc: BaseException | None) -> bool:
    """Was *exc*, or what caused it, a refused change to a file?"""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, WouldWriteAFile):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False
