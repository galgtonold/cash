"""A thread a cell started that is still running when the cell ends
(``NOTEBOOK-THREAD-ALIVE``).

What such a thread changes after the cell finished is outside cash's view:
the statements that read those values were keyed, and stored, on what they
held when the cell ended. Cash cannot follow the thread; the user can join
it before the cell ends.

Cheap: the idents of the threads alive before the cell, compared with the
threads alive after it. Idents, not the ``Thread`` objects: a set of those
held for the whole cell would put every live thread, and all its target and
arguments hold, in reach of the walks that look for the functions a value
holds. A
new thread counts only when the code it runs is the user's (its target, or
the ``run`` of a ``Thread`` subclass, by its code's file): the workers of a
pool, a library's monitor thread and cash's own writers run library code and
are left out, whatever the library. Once per cell source.
"""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import dataclass, field

from cash.diagnostics import log_diagnostic, warn_diagnostic
from cash.exceptions import CashWarning
from cash.install_paths import is_user_code_file
from cash.notebook.file_round_trips import quote
from cash.source_norm import exact_source_digest

logger = logging.getLogger(__name__)

__all__ = ["CODE", "BackgroundThreads", "user_code_of"]

CODE = "NOTEBOOK-THREAD-ALIVE"

#: How many threads the message names.
_NAMED = 3


def _code_of(fn: object) -> object | None:
    """The code object *fn* runs: through a ``partial`` and a bound method."""
    for _ in range(8):
        if isinstance(fn, functools.partial):
            fn = fn.func
            continue
        inner = getattr(fn, "__func__", None)
        if inner is not None:
            fn = inner
            continue
        break
    return getattr(fn, "__code__", None)


def user_code_of(thread: threading.Thread) -> object | None:
    """The function of the user's *thread* runs, or None when it runs
    library code (or nothing cash can name)."""
    fn = getattr(thread, "_target", None)
    if fn is None and isinstance(thread, threading.Timer):
        fn = thread.function
    if fn is None:
        run = getattr(type(thread), "run", None)
        if run is threading.Thread.run:
            return None
        fn = run
    code = _code_of(fn)
    if code is None or not is_user_code_file(getattr(code, "co_filename", None)):
        return None
    return fn


def _label(fn: object) -> str:
    while isinstance(fn, functools.partial):
        fn = fn.func
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


@dataclass
class BackgroundThreads:
    """The threads alive as a cell starts, and the cells already warned of."""

    _before: frozenset[int | None] = frozenset()
    _warned: set[str] = field(default_factory=set)

    def cell_starts(self) -> None:
        self._before = frozenset(t.ident for t in threading.enumerate())

    def cell_ends(self, raw_cell: str, cell_number: int | None) -> None:
        """Warn when a thread of the user's that this cell started is alive."""
        before, self._before = self._before, frozenset()
        started = [t for t in threading.enumerate() if t.ident not in before and t.is_alive()]
        if not started:
            return
        alive = [(t, fn) for t in started if (fn := user_code_of(t)) is not None]
        if not alive:
            return
        digest = exact_source_digest(raw_cell)
        if digest in self._warned:
            return
        self._warned.add(digest)
        cell = f"cell {cell_number}" if cell_number is not None else f"the cell {quote(raw_cell)}"
        names = ", ".join(f"'{t.name}' (running {_label(fn)})" for t, fn in alive[:_NAMED])
        if len(alive) > _NAMED:
            names += f" and {len(alive) - _NAMED} more"
        what = (
            f"{cell} started thread {names}, which still runs after the cell finished, so what it "
            "changes from now on is outside cash's view. The statements that read those values "
            "were keyed and stored on what they held when the cell ended."
        )
        fix = (
            "join the thread before the cell ends (t.join()), or have it return its result "
            "(concurrent.futures: future.result()) in the cell that uses it."
        )
        log_diagnostic(logger, CODE, what, fix)
        location = ("<cash>", cell_number) if cell_number is not None else None
        warn_diagnostic(CashWarning, CODE, what, fix, location=location)
