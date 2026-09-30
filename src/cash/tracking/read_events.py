"""The audit events that report reads to the file tracker.

Python raises an audit event for every Python-level ``open`` and directory
listing, which :mod:`cash.tracking.io_watch` delivers here. A read becomes a
dependency of the active tracker, a write an effect for the effect observer,
and a read outside every tracker a credit to the code that made it
(`cash.tracking.read_credit`).
"""

from __future__ import annotations

import glob as glob_module
import os
import sys
from typing import Any

# Imported at module scope, not inside `_on_open`: that runs inside every
# ``open`` in the process, including at interpreter shutdown while sys.modules
# is being torn down. `cash.effect_observer` imports nothing that imports this
# module, so this cannot cycle.
from cash.effect_observer import active_observer as _active_effect_observer
from cash.tracking import io_watch
from cash.tracking.read_classification import is_cash_internal
from cash.tracking.read_credit import note_untracked_read
from cash.tracking.tracker_context import active_tracker

__all__ = ["subscribe_read_events"]


#: Callers whose opens and listings are the interpreter's, not a read of data:
#: the import system (a module's source and bytecode, the ``sys.path``
#: directories it lists) and the source readers behind ``inspect.getsource``
#: and tracebacks, which the ``linecache`` wrappers in `reader_patches` cover
#: instead.
_NOT_A_READ = frozenset(
    {
        "_frozen_importlib",
        "_frozen_importlib_external",
        "importlib._bootstrap",
        "importlib._bootstrap_external",
        "zipimport",
        "tokenize",
        "linecache",
    }
)

#: Packages whose every open and listing is a tool's, never the notebook's.
#: coverage's tracer lists the directory of each source file it first meets
#: (on Windows, to learn the path's case), from inside the trace function, so
#: the statement running at that moment got the directory as an input. Which
#: directories that is depends on coverage's per-process cache: the same
#: import keyed differently in the next kernel, and the two lineage engines
#: disagreed on it.
_TOOL_PACKAGES = frozenset({"coverage"})


#: Callers that read a package's data file through its loader's ``get_data``,
#: which lives in the import system: ``pkgutil.get_data`` and the
#: ``importlib.resources`` / ``pkg_resources`` functions. That read is data
#: (`cash.tracking.read_classification` decides whose), not an import.
_LOADER_DATA_CALLERS = ("pkgutil", "importlib.resources", "importlib_resources", "pkg_resources")


def _not_a_read(frame: Any) -> bool:
    """Whether the call *frame* made is the interpreter's or a tool's, not the user's."""
    name = frame.f_globals.get("__name__") or ""
    if name in _NOT_A_READ:
        if frame.f_code.co_name == "get_data" and frame.f_back is not None:
            caller = frame.f_back.f_globals.get("__name__") or ""
            if any(caller == n or caller.startswith(n + ".") for n in _LOADER_DATA_CALLERS):
                return False
        return True
    return name.partition(".")[0] in _TOOL_PACKAGES


def _audited_caller() -> Any:
    """The frame that made the audited call, seen from an audit consumer."""
    # +1: this helper's own frame.
    return sys._getframe(io_watch.CALLER_DEPTH + 1)


def _is_read_mode(mode: str) -> bool:
    # `w+` and `x+` start from an empty file, so nothing the code reads back
    # existed before it: a write, not an input. Pillow saves every image with
    # "w+b", and would have each `savefig` recorded as a dependency on its own
    # output.
    return "r" in mode or ("+" in mode and "w" not in mode and "x" not in mode)


def _on_open(args: tuple) -> None:
    """The ``open`` audit event: a read is a dependency, a write an effect.

    ``mode`` is the raw mode (``r``, ``w+``, ...) of any Python-level open;
    ``os.open`` reports None there and is not watched, and an integer "path"
    is a descriptor being wrapped, which names no file.
    """
    path, mode, flags = args
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    if not isinstance(mode, str):
        # ``os.open``: not watched as a read, but a file it creates
        # (``tempfile.mkstemp``) is the block's own output.
        if isinstance(flags, int) and _creates(flags):
            tracker = active_tracker.get()
            if tracker is not None:
                tracker.note_created(path)
        return
    caller = _audited_caller()
    if _not_a_read(caller):
        return
    if _is_read_mode(mode):
        tracker = active_tracker.get()
        if tracker is None:
            note_untracked_read(path, caller)
            return
        tracker.track_path(path)
        if "a" not in mode:
            # A file that was not there is an input too, and the docs say so --
            # but only the `os.path.exists` spelling recorded it.
            # `try: open(p) except FileNotFoundError:` kept serving its default
            # after the file appeared. The event comes before the open, so ask the disk.
            try:
                os.stat(path)
            except FileNotFoundError:
                tracker.track_absent(path)
            except (OSError, ValueError):
                pass
    elif any(ch in mode for ch in "wax"):
        # Not a dependency -- a WRITE is an effect, not an input, and folding
        # it into the key would invalidate a function on its own output. It is
        # recorded for the effect observer instead: a cache hit skips this
        # write, and if the analyzer said nothing (the write happened inside a
        # library it does not walk into) that is a silent behaviour change the
        # user should hear about. See cash.effect_observer.
        observer = _active_effect_observer.get()
        if observer is not None and not is_cash_internal(path):
            observer.record_write(path)
        if "a" not in mode:
            # Truncated or new: what the block reads back from it is its own
            # output, not an input (`FileAccessTracker.note_created`).
            tracker = active_tracker.get()
            if tracker is not None:
                tracker.note_created(path)


def _creates(flags: int) -> bool:
    """Do ``os.open`` *flags* make a new file or empty an existing one?"""
    writes = flags & (os.O_WRONLY | os.O_RDWR)
    return bool(writes) and (bool(flags & os.O_TRUNC) or (flags & os.O_CREAT and flags & os.O_EXCL) != 0)


def _on_mkdir(args: tuple) -> None:
    """``os.mkdir`` (``mkdtemp``, ``makedirs``, ``Path.mkdir``): a directory
    the block makes holds only its own output."""
    path = args[0]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    tracker = active_tracker.get()
    if tracker is not None:
        tracker.note_created(path, directory=True)
    observer = _active_effect_observer.get()
    if observer is not None:
        observer.note_created_dir(path)


def _on_listing(args: tuple) -> None:
    """``os.listdir`` / ``os.scandir``: the listed directory is a dependency.

    Listing a directory is how a new file matching a pattern becomes visible:
    a cell that enumerates a directory and reads the matches gets file deps
    only for the files READ on the first run, so a NEW matching file would be
    invisible. Adding or removing an entry bumps the directory's own mtime on
    local filesystems, so the existing freshness check invalidates the
    reader. ``pathlib``'s ``glob`` / ``iterdir`` list through these too.
    """
    tracker = active_tracker.get()
    if tracker is None:
        return
    path = args[0]
    if path is None:
        path = "."
    elif not isinstance(path, (str, bytes, os.PathLike)):
        return  # a descriptor
    if _not_a_read(_audited_caller()):
        return
    tracker.track_path(path)
    # `try: os.listdir(d) except FileNotFoundError:` answers for a directory
    # that is not there yet. The event comes before the listing: ask the disk.
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        tracker.track_absent(path)
    except (OSError, ValueError, TypeError):
        pass


def _on_glob(args: tuple) -> None:
    """``glob.glob`` / ``glob.iglob``: track the directory the pattern enumerates."""
    tracker = active_tracker.get()
    if tracker is None:
        return
    base = _glob_base_dir(args[0])
    if base is not None:
        tracker.track_path(base)


def _glob_base_dir(pattern: Any) -> str | None:
    """Return the longest leading, magic-free directory of a glob *pattern*.

    ``gdir/*.num`` → ``gdir``; ``a/b*/c`` → ``a`` (deepest stable ancestor).
    The directory's mtime is what we track for membership changes.
    """

    try:
        parts = (os.fsdecode(pattern) if isinstance(pattern, bytes) else str(pattern)).replace("\\", "/").split("/")
    except (TypeError, ValueError):
        return None
    base: list[str] = []
    for p in parts[:-1]:  # exclude the filename component
        if glob_module.has_magic(p):
            break
        base.append(p)
    return "/".join(base) or "."


def subscribe_read_events() -> None:
    """Route the audit events of Python-level opens and listings to the tracker."""
    io_watch.subscribe("open", _on_open, outside_scopes=True)
    io_watch.subscribe("os.listdir", _on_listing)
    io_watch.subscribe("os.scandir", _on_listing)
    io_watch.subscribe("glob.glob", _on_glob)
    io_watch.subscribe("os.mkdir", _on_mkdir)
