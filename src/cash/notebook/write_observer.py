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
from collections.abc import Iterator
from contextlib import contextmanager

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


def _on_open(args: tuple) -> None:
    sinks = _active.get()
    if not sinks:
        return
    path, mode, flags = args
    if isinstance(mode, str):
        written = any(c in mode for c in "wax+")
        appends = "a" in mode
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

    A file only appended to is left to the static writer check, which reads
    ``open(p, 'a')`` and ``to_csv(mode='a')`` in the user's code: an append
    seen in a library's code is a log handler's, as often as not."""
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
