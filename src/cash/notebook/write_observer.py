"""Which files a block of user code wrote, observed as it ran.

A writer's provenance -- the files it produced, and their state then -- lets
the re-execution planner tell after a kernel restart that the writer's effect
is already on disk, instead of re-firing it and rebuilding everything it
needs. Reading the paths from the code covers ``df.to_csv(OUT / 'a.csv')``;
it does not cover ``save_chart(kind)``, whose ``fig.savefig(OUT /
f'{kind}.png')`` sits in a helper, nor a loop over kinds. Those were re-fired
after every restart, and r23s3's chart loop re-ran a 263 s sweep to redraw
charts nobody had asked for.

So watch the writes instead: one ``sys.addaudithook`` for the process, inert
unless a collector is active in the current context. Every file opened for
writing and every rename/replace destination (an atomic write) is recorded,
whichever library did it -- pandas, matplotlib and pathlib all open through
``io.open``. Writes a C extension makes without Python's ``open`` are not
seen; their writer keeps no provenance and is re-fired as before.
"""

from __future__ import annotations

import contextvars
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["observe_writes"]

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

#: The collectors of the blocks being observed, innermost last. A statement
#: inside a loop is recorded for both.
_active: contextvars.ContextVar[tuple[set[str], ...]] = contextvars.ContextVar("cash_write_observers", default=())
_installed = False


def _written_path(event: str, args: tuple) -> object | None:
    if event == "open":
        path, mode, flags = args
        if isinstance(mode, str):
            return path if any(c in mode for c in "wax+") else None
        return path if isinstance(flags, int) and flags & _WRITE_FLAGS else None
    if event in ("os.rename", "os.replace"):
        return args[1]
    return None


def _hook(event: str, args: tuple) -> None:
    # Called for every audited event in the process: return at once unless a
    # block is being observed, and never raise -- an exception here would fail
    # the user's own open().
    try:
        sinks = _active.get()
        if not sinks:
            return
        path = _written_path(event, args)
        if path is None or isinstance(path, int):
            return
        path = os.fsdecode(os.fspath(path))
        resolved = os.path.abspath(path)
        for sink in sinks:
            sink.add(resolved)
    except Exception:  # noqa: BLE001 - see above
        return


@contextmanager
def observe_writes() -> Iterator[set[str]]:
    """Yield the set of absolute paths the block writes, filled as it runs."""
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
    written: set[str] = set()
    token = _active.set(_active.get() + (written,))
    try:
        yield written
    finally:
        _active.reset(token)
