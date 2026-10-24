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
own cache of a module the rebuild imports is let through too, while
``importlib`` itself writes it: its ``__pycache__`` folder made, its
``x.pyc.<id>`` created new and renamed onto ``x.pyc`` (or removed when the
write failed). So the import succeeds as it would outside: a process-wide
``sys.dont_write_bytecode`` cost another thread's import its ``.pyc``, and
reset the user's own. The same steps from the value's own code are refused.

A library's first import may write its own caches (matplotlib's font
list): refused inside the rebuild, the library is then imported as the
program would import it, outside the check, and the rebuild is tried again
(`import_what_it_imported`). A module of the user's is not: its import
writes what the program never wrote.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import logging
import os
import pickle
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ..tracking import io_watch

logger = logging.getLogger(__name__)

#: The audit events a rebuild may not raise, unless on its own scratch;
#: and ``import``, never refused, noted (`NoFileWrites.imported`).
EVENTS = frozenset(
    (
        "import",
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

    __slots__ = ("bytecode", "imported", "in_import", "made", "refused", "tempdir")

    def __init__(self, tempdir: str) -> None:
        self.refused: str | None = None
        #: The scratch files and folders this rebuild made in the temp folder.
        self.made: list[str] = []
        #: The modules the rebuild started to import.
        self.imported: list[str] = []
        #: Whether the refused change was asked for while a module was
        #: being imported.
        self.in_import = False
        #: The ``x.pyc.<id>`` files ``importlib`` made new in this rebuild.
        self.bytecode: list[str] = []
        #: The temp folder, resolved before the check is installed: asked
        #: from inside it, ``tempfile.gettempdir()`` waits on the lock its
        #: own first probe of the folder holds, and the process hung.
        self.tempdir = tempdir

    def __call__(self, event: str, args: tuple) -> None:
        if event == "import":
            if args and isinstance(args[0], str):
                self.imported.append(args[0])
            return
        if event == "open":
            path, mode, flags = (*args, None, None, None)[:3]
            if isinstance(path, int):
                return  # a descriptor already open: wrapping it changes nothing
            if isinstance(flags, int):
                if not flags & _WRITES or (_TMPFILE and flags & _TMPFILE == _TMPFILE):
                    return  # a read, or a file with no name (``TemporaryFile``)
                if flags & os.O_CREAT and flags & os.O_EXCL:
                    if self._scratch(path, made=True):
                        return
                    if _bytecode_temp(path) and _by_importlib("_write_atomic"):
                        self.bytecode.append(_path(path))  # type: ignore[arg-type]
                        return
            if self._own(path) or os.path.isdir(path):
                # A folder is not written by opening it: ``NamedTemporaryFile``
                # opens the temp folder with its mode and an opener, whose own
                # open of the file is audited as well.
                return
            self._refuse(f"open({path!r}, {mode!r})")
        elif event == "os.mkdir":
            if _no_dir_fd(args[2]) and (
                self._scratch(args[0], made=True) or (_bytecode_folder(args[0]) and _by_importlib("set_data"))
            ):
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
            if event == "os.rename" and self._own_bytecode(src, dst) and _by_importlib("_write_atomic"):
                return  # ``importlib``'s ``.pyc``, renamed into place
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
            if event == "os.remove" and not in_dir and _path(path) in self.bytecode and _by_importlib("_write_atomic"):
                return  # ``importlib``'s ``.pyc`` that failed to be written
            self._refuse(f"{event}({path!r})")

    def _scratch(self, value: object, *, made: bool) -> bool:
        """Is *value* a new name right in the temp folder (where ``mkstemp``
        and ``mkdtemp`` make theirs), or in a scratch folder? Recorded as this
        rebuild's own when *made*."""
        path = _path(value)
        if path is None or os.path.lexists(path):
            return False
        parent = os.path.dirname(path)
        if not (_same(parent, self.tempdir) or self._own(parent)):
            return False
        if made:
            self.made.append(path)
        return True

    def _own_bytecode(self, src: object, dst: object) -> bool:
        """Is *src* a ``x.pyc.<id>`` this rebuild's ``importlib`` made, and
        *dst* the ``x.pyc`` it was made for?"""
        src_path, dst_path = _path(src), _path(dst)
        return src_path in self.bytecode and dst_path is not None and src_path.rpartition(".")[0] == dst_path

    def _own(self, value: object) -> bool:
        path = _path(value)
        return path is not None and any(_under(path, made) for made in self.made)

    @staticmethod
    def _relative_name(value: object) -> bool:
        name = os.fsdecode(value) if isinstance(value, (str, bytes)) else ""
        return bool(name) and os.sep not in name and name not in (".", "..")

    def _refuse(self, what: str) -> None:
        self.refused = what
        self.in_import = _by_importlib("_find_and_load", depth=None)
        raise WouldWriteAFile(f"rebuilding it would {what}")

    def check(self) -> None:
        """Raise if the rebuild asked for a change to a file, caught or not."""
        if self.refused is not None:
            raise WouldWriteAFile(f"rebuilding it would {self.refused}")


def _bytecode_temp(value: object) -> bool:
    """Is *value* the name ``importlib`` writes a ``.pyc`` under before
    renaming it into place: ``__pycache__/x.pyc.<id>``?"""
    path = _path(value)
    if path is None:
        return False
    folder, name = os.path.split(path)
    stem, _, number = name.rpartition(".")
    return os.path.basename(folder) == "__pycache__" and stem.endswith(".pyc") and number.isdigit()


def _by_importlib(function: str, depth: int | None = 12) -> bool:
    """Is ``importlib``'s own *function* running, within *depth* frames of
    the event (any depth for None)? For the steps that write a ``.pyc``,
    only as part of an import's (`_by_an_import`)."""
    if function in _BYTECODE_STEPS:
        return _by_an_import(function)
    frame = sys._getframe(1)
    while frame is not None and (depth is None or depth > 0):
        code = frame.f_code
        if code.co_filename in _IMPORTLIB_FILES and code.co_name == function:
            return True
        frame = frame.f_back
        if depth is not None:
            depth -= 1
    return False


def _by_an_import(step: str) -> bool:
    """Is the event *step* of an import's own write of a module's ``.pyc``:
    ``importlib``'s frames, unbroken, from *step* through ``_cache_bytecode``
    and ``get_code`` up to the import or reload running them? A frame of
    anything else in between, a module body the import runs calling
    ``get_code`` on another file, or the value's own
    ``SourceFileLoader(...).get_code(...)`` while an outer import runs,
    breaks the chain, and the write is the value's."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_CASH_DIR):
        frame = frame.f_back  # the check's own frames, and the hook calling it
    needed = {step, "_cache_bytecode", "get_code"}
    while frame is not None:
        code = frame.f_code
        if code.co_filename not in _IMPORTLIB_FILES:
            return False
        needed.discard(code.co_name)
        if code.co_name in _IMPORTING:
            return not needed
        frame = frame.f_back
    return False


#: Where cash's own files are: the audit hook and this check run from them.
_CASH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

#: ``importlib``'s functions an import or a reload runs a loader under.
_IMPORTING = frozenset(("_find_and_load", "_load", "_exec"))


#: The steps of an import's write of a module's ``.pyc``: let through only
#: below ``get_code`` and ``_cache_bytecode``.
_BYTECODE_STEPS = frozenset(("_write_atomic", "set_data"))


def _importlib_files() -> frozenset[str]:
    """The file names ``importlib``'s bootstrap code runs under: frozen, or
    from its source where the interpreter was built without it frozen."""
    found = set()
    for name in ("_bootstrap", "_bootstrap_external"):
        found.add(f"<frozen importlib.{name}>")
        module = sys.modules.get(f"importlib.{name}")
        path = getattr(module, "__file__", None)
        if isinstance(path, str):
            found.add(path)
    return frozenset(found)


_IMPORTLIB_FILES = _importlib_files()


def _bytecode_folder(value: object) -> bool:
    path = _path(value)
    return path is not None and os.path.basename(path) == "__pycache__"


def _no_dir_fd(value: object) -> bool:
    return value is None or value == -1  # audited as -1 when not given


def _same(path: str, folder: str) -> bool:
    return path in (folder, os.path.realpath(folder)) or os.path.realpath(path) == os.path.realpath(folder)


def _under(path: str, folder: str) -> bool:
    for root in {folder, os.path.realpath(folder)}:
        if path == root or path.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


_current = threading.local()


@contextmanager
def no_file_writes() -> Iterator[NoFileWrites]:
    """While inside, on this thread, a change to a file fails with
    `WouldWriteAFile`. Call ``check()`` on what it yields after the
    rebuild: the error may have been caught inside it."""
    check = NoFileWrites(tempfile.gettempdir())
    before = getattr(_current, "check", None)
    _current.check = check
    try:
        with io_watch.refusing(EVENTS, check):
            yield check
    finally:
        _current.check = before


def refused_now() -> str | None:
    """What the innermost `no_file_writes` open on this thread refused, if
    anything."""
    check = getattr(_current, "check", None)
    return None if check is None else check.refused


def import_what_it_imported(check: NoFileWrites) -> bool:
    """Import, outside any check, the libraries a rebuild started to import
    when what it refused was asked for during an import; whether any was
    imported.

    A library's own first import may write its caches (matplotlib's font
    list under an empty ``MPLCONFIGDIR``): refused inside the rebuild, the
    import failed and a disk hit became a miss, where plain Python's import
    wrote them once. Imported here as the program would import them, the
    rebuild is tried again; one that still changes a file fails then. A
    module of the user's is not imported: its import-time write is one the
    program, which never imported it, did not make."""
    if check.refused is None or not check.in_import:
        return False
    imported = False
    for name in dict.fromkeys(check.imported):
        if name in sys.modules or not _a_library(name):
            continue
        try:
            importlib.import_module(name)
        except Exception:
            logger.debug("could not import %s for a rebuild", name, exc_info=True)
        imported = imported or name in sys.modules
    return imported


def _a_library(name: str) -> bool:
    """Is the module *name* not the user's, judged by where it is found,
    without importing anything? Found part by part, each in the folders of
    the package above it: a namespace package's portions can sit in a
    library's folder and in the user's (an installed ``acmex.core``, the
    user's ``src/acmex/mine``), so a package is a library's only when all
    of its folders are."""
    from ..install_paths import is_user_path

    spec = None
    path = None
    parts = name.split(".")
    try:
        for at in range(len(parts)):
            part = ".".join(parts[: at + 1])
            module = sys.modules.get(part)
            spec = getattr(module, "__spec__", None) if module is not None else None
            if spec is None:
                spec = importlib.machinery.PathFinder.find_spec(part, path)
            if spec is None:
                return False
            path = list(getattr(module, "__path__", None) or spec.submodule_search_locations or ())
    except Exception:  # noqa: BLE001 - an odd finder or module: not known to be a library
        return False
    if spec.origin in ("built-in", "frozen"):
        return True
    places = [spec.origin] if spec.origin not in (None, "namespace") else list(spec.submodule_search_locations or ())
    return bool(places) and all(isinstance(place, str) and not is_user_path(place) for place in places)


def would_write_a_file(exc: BaseException | None) -> bool:
    """Was *exc*, or what caused it, a refused change to a file?"""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, WouldWriteAFile):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False
