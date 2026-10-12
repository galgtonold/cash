"""Which files a block of user code wrote, observed as it ran.

A writer's provenance -- the files it produced, and their state then -- lets
the re-execution planner tell after a kernel restart that the writer's effect
is already on disk, instead of re-firing it and rebuilding everything it
needs. Reading the paths from the code covers ``df.to_csv(OUT / 'a.csv')``;
it does not cover ``save_chart(kind)``, whose ``fig.savefig(OUT /
f'{kind}.png')`` sits in a helper, nor a loop over kinds. Those were re-fired
after every restart, and a chart loop re-ran a 263 s sweep to redraw
charts nobody had asked for.

So watch the writes instead, through cash's one audit hook
(:mod:`cash.tracking.io_watch`), inert unless a collector is active in the
current context. Every file opened for writing and every rename/replace
destination (an atomic write) is recorded, whichever library did it -- pandas,
matplotlib and pathlib all open through ``io.open``. Writes a C extension
makes without Python's ``open`` are not seen; their writer keeps no provenance
and is re-fired as before.
"""

from __future__ import annotations

import contextvars
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from cash.install_paths import is_user_code_file
from cash.tracking import io_watch

__all__ = ["WrittenPaths", "kept_writes", "observe_writes"]

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class WrittenPaths(set):
    """The absolute paths a block wrote; :attr:`appended` holds those it
    only ever opened to append to."""

    def __init__(self) -> None:
        super().__init__()
        self.appended: set[str] = set()
        self._replaced: set[str] = set()


#: The collectors of the blocks being observed, innermost last. A statement
#: inside a loop is recorded for both.
_active: contextvars.ContextVar[tuple[WrittenPaths, ...]] = contextvars.ContextVar("cash_write_observers", default=())


def _record(sinks: tuple[WrittenPaths, ...], path: object, appends: bool = False) -> None:
    if path is None or isinstance(path, int):
        return  # a descriptor names no file
    resolved = os.path.abspath(os.fsdecode(os.fspath(path)))
    for sink in sinks:
        sink.add(resolved)
        if not appends:
            sink._replaced.add(resolved)
            sink.appended.discard(resolved)
        elif resolved not in sink._replaced:
            sink.appended.add(resolved)


#: Modules that open a file for the code that calls them (``Path.open``,
#: ``gzip.open``, ``codecs.open``, the ``open`` IPython puts in a kernel's
#: builtins): whose append it is, is their caller's. Cash's own frames too.
_PASS_THROUGH_OPENERS = frozenset(
    {"pathlib", "pathlib._local", "pathlib._abc", "io", "_pyio", "codecs", "gzip", "bz2", "lzma"}
    | {"IPython.core.interactiveshell"}
)


def _passes_through(frame) -> bool:
    name = frame.f_globals.get("__name__") or ""
    return name in _PASS_THROUGH_OPENERS or name == "cash" or name.startswith("cash.")


def _appended_by_user_code() -> bool:
    """Was the append being audited opened by the user's own code (a helper's
    ``open(path, mode)``), not by a library's (a log handler's)?

    Such an append is a write a hit would skip, like a replacing one, and the
    static check cannot see it when the mode is computed (``open(p, mode)``)."""
    try:
        frame = sys._getframe(io_watch.CALLER_DEPTH + 1)
    except ValueError:
        return False
    while frame is not None and _passes_through(frame):
        frame = frame.f_back
    return frame is not None and is_user_code_file(frame.f_code.co_filename)


def _on_open(args: tuple) -> None:
    sinks = _active.get()
    if not sinks:
        return
    path, mode, flags = args
    if isinstance(mode, str):
        written = any(c in mode for c in "wax+")
        # An append the user's own ``open()`` made is a write a hit skips.
        appends = "a" in mode and not _appended_by_user_code()
    else:  # os.open
        written = isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        appends = written and bool(flags & os.O_APPEND) and not flags & os.O_TRUNC
    if written:
        _record(sinks, path, appends)


def _on_move(args: tuple) -> None:
    sinks = _active.get()
    if sinks:
        _record(sinks, args[1])  # the destination of an atomic write


io_watch.subscribe("open", _on_open)
io_watch.subscribe("os.rename", _on_move)
io_watch.subscribe("os.replace", _on_move)


@contextmanager
def observe_writes() -> Iterator[WrittenPaths]:
    """Yield the set of absolute paths the block writes, filled as it runs."""
    written = WrittenPaths()
    token = _active.set(_active.get() + (written,))
    io_watch.hold()
    try:
        yield written
    finally:
        io_watch.release()
        _active.reset(token)


def kept_writes(written: "set[str] | frozenset[str]", storage_root: object = None, appended=frozenset()) -> list[str]:
    """The paths of *written* still there, outside cash's own storage
    (*storage_root*) and not in *appended*, sorted. A file written and
    removed again within the block (a temporary file) left nothing a cache
    hit would skip.

    A file only appended to by a library is left to the static writer check,
    which reads ``to_csv(mode='a')`` in the user's code: an append seen in a
    library's code is a log handler's, as often as not. One the user's own
    code opened (``open(p, mode)`` in a helper) is never in *appended*."""
    if not written:
        return []
    root = None
    if isinstance(storage_root, (str, os.PathLike)):
        root = os.path.normcase(os.path.abspath(os.fspath(storage_root))) + os.sep
    return sorted(
        p
        for p in written
        if p not in appended
        and not (root and os.path.normcase(p).startswith(root))
        and os.path.lexists(p)
        # A folder opened to write is ``O_TMPFILE``: a file with no name.
        and not os.path.isdir(p)
    )
