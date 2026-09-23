"""Which files a block of user code wrote, observed as it ran.

A writer's provenance -- the files it produced, and their state then -- lets
the re-execution planner tell after a kernel restart that the writer's effect
is already on disk, instead of re-firing it and rebuilding everything it
needs. Reading the paths from the code covers ``df.to_csv(OUT / 'a.csv')``;
it does not cover ``save_chart(kind)``, whose ``fig.savefig(OUT /
f'{kind}.png')`` sits in a helper, nor a loop over kinds. Those were re-fired
after every restart, and r23s3's chart loop re-ran a 263 s sweep to redraw
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

__all__ = ["observe_writes"]

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

#: The collectors of the blocks being observed, innermost last. A statement
#: inside a loop is recorded for both.
_active: contextvars.ContextVar[tuple[set[str], ...]] = contextvars.ContextVar("cash_write_observers", default=())


def _record(sinks: tuple[set[str], ...], path: object) -> None:
    if path is None or isinstance(path, int):
        return  # a descriptor names no file
    resolved = os.path.abspath(os.fsdecode(os.fspath(path)))
    for sink in sinks:
        sink.add(resolved)


def _on_open(args: tuple) -> None:
    sinks = _active.get()
    if not sinks:
        return
    path, mode, flags = args
    if isinstance(mode, str):
        written = any(c in mode for c in "wax+")
    else:  # os.open
        written = isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
    if written:
        _record(sinks, path)


def _on_move(args: tuple) -> None:
    sinks = _active.get()
    if sinks:
        _record(sinks, args[1])  # the destination of an atomic write


io_watch.subscribe("open", _on_open)
io_watch.subscribe("os.rename", _on_move)
io_watch.subscribe("os.replace", _on_move)


@contextmanager
def observe_writes() -> Iterator[set[str]]:
    """Yield the set of absolute paths the block writes, filled as it runs."""
    written: set[str] = set()
    token = _active.set(_active.get() + (written,))
    io_watch.hold()
    try:
        yield written
    finally:
        io_watch.release()
        _active.reset(token)
