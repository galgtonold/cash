"""Garbage collections run in a context of their own.

The finalizers a collection runs are no work of the code it interrupted:
they run wherever an allocation made the collector start. In that code's
context they ran under its file tracker, so a ``__del__`` of something long
gone that submitted to a thread pool, started a thread or wrote a file
counted as the cached call's own: work still running as the call returned
(`FileAccessTracker.work_outlived_block`) kept its result out of the cache
-- on a loaded machine, where even a short task had not finished yet -- and
a write refused the call for the session. Whether a collection fell inside
a call depended on what ran before it, so the same cheap call was stored
alone and ran again in a full test run.

On Python 3.10 and 3.11 the collector can start inside any allocation, including the ones a
``ContextVar.set`` or ``reset`` makes while it builds the context's new
mapping. A finalizer the collection runs (a ``__del__``, a generator closed
as garbage) that sets or resets a context variable then replaces the mapping
the interrupted call is half-way through: when that call finishes, it writes
back the mapping it started from, and the finalizer's change is lost; the
mapping it started from may even have been freed under it. cash's blocks set
and reset their variables all the time (the file tracker, the effect
observer, the write observer), and a lost change surfaced as a
``LookupError`` from a later reset whose value had vanished. Python 3.12
runs the collector only between bytecodes, never inside such a call.

Here each collection enters a fresh, empty context, and leaves it when it is
done: what the finalizers set and reset lands there, and the context of the
code they interrupted is left as it was, and no tracker or observer of
that code sees what they do. A finalizer resetting a token made
outside the collection gets the ValueError of a token from another context
(`reset_in_any_context` takes that), instead of changing the interrupted
context at a moment nothing chose.

With the C extension (3.12 on) the callback is ``gc_own_context`` in C, which
a collection can call at any depth; without it, a Python function that gives
itself frames past the recursion limit for its own calls.
"""

from __future__ import annotations

import contextvars
import gc
import platform
import sys
from typing import Any

__all__ = ["INSTALLED"]

#: Frames past the recursion limit a collection's enter and exit may use.
_HEADROOM = 50


def _native_callback() -> Any:
    """The C extension's callback (``cgc.c``), or None without it."""
    from ._native import cwatch

    return getattr(cwatch, "gc_own_context", None)


def _install() -> bool:
    if platform.python_implementation() != "CPython":
        return False
    native = _native_callback()
    if any(getattr(callback, "_cash_own_context", False) or callback is native for callback in gc.callbacks):
        return True  # a second import of this module: one context per collection
    if native is not None:
        # Called from C, it takes no Python frame: the Python callback below
        # cannot even be called a frame short of the recursion limit, which on
        # 3.14 a collection deep in a recursion reported as an exception
        # ignored in a GC callback, and ran in the interrupted code's context.
        gc.callbacks.append(native)
        return True
    import ctypes

    # Prototypes of our own, so the shared ``ctypes.pythonapi`` functions keep
    # whatever argtypes other code gave them.
    prototype = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)
    enter = prototype(("PyContext_Enter", ctypes.pythonapi))
    leave = prototype(("PyContext_Exit", ctypes.pythonapi))
    entered: list[contextvars.Context] = []
    # The program's own limit while one is raised for a collection: a
    # collection that cannot put it back leaves it for the next one. Read
    # back from the raised value instead, the limit climbed 50 frames a
    # collection under a recursion that collects at every allocation, past
    # the C stack (a crash on Windows and macOS, 3.10). Kept by plain
    # assignment: deeper than the limit even a call to a list method raises.
    program_limit: int | None = None

    def own_context(phase: str, info: dict[str, Any]) -> None:
        # A collection can start at any depth, also a frame short of the
        # recursion limit (a pickler deep in a 20,000-object chain): ctypes
        # checks its argument's type in Python, which then raised there, and
        # the collection ran in the interrupted code's context. The calls
        # get a few frames more than the limit for themselves.
        nonlocal program_limit
        try:
            limit = sys.getrecursionlimit()
            if program_limit is not None and limit == program_limit + _HEADROOM:
                limit = program_limit  # still raised for an earlier collection
            program_limit = limit
            sys.setrecursionlimit(limit + _HEADROOM)
            try:
                if phase == "start":
                    context = contextvars.Context()
                    enter(context)
                    entered.append(context)
                elif entered:
                    leave(entered.pop())
            finally:
                sys.setrecursionlimit(limit)  # deeper than *limit* now: raises, and the next collection puts it back
                program_limit = None
        except (RecursionError, ctypes.ArgumentError):
            # Too deep even with the frames above: the collection runs in the
            # interrupted code's context, as without cash.
            pass

    own_context._cash_own_context = True  # type: ignore[attr-defined]
    gc.callbacks.append(own_context)
    return True


#: True when collections run in a context of their own (CPython).
INSTALLED = _install()
