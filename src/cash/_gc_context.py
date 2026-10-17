"""Garbage collections run in a context of their own on Python 3.10 and 3.11.

There the collector can start inside any allocation, including the ones a
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
code they interrupted is left as it was. A finalizer resetting a token made
outside the collection gets the ValueError of a token from another context
(`reset_in_any_context` takes that), instead of changing the interrupted
context at a moment nothing chose.
"""

from __future__ import annotations

import contextlib
import contextvars
import gc
import platform
import sys
from typing import Any

__all__ = ["INSTALLED"]

#: Frames past the recursion limit a collection's enter and exit may use.
_HEADROOM = 50


def _install() -> bool:
    if sys.version_info >= (3, 12) or platform.python_implementation() != "CPython":
        return False
    if any(getattr(callback, "_cash_own_context", False) for callback in gc.callbacks):
        return True  # a second import of this module: one context per collection
    import ctypes

    # Prototypes of our own, so the shared ``ctypes.pythonapi`` functions keep
    # whatever argtypes other code gave them.
    prototype = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)
    enter = prototype(("PyContext_Enter", ctypes.pythonapi))
    leave = prototype(("PyContext_Exit", ctypes.pythonapi))
    entered: list[contextvars.Context] = []

    def own_context(phase: str, info: dict[str, Any]) -> None:
        # A collection can start at any depth, also a frame short of the
        # recursion limit (a pickler deep in a 20,000-object chain): ctypes
        # checks its argument's type in Python, which then raised there, and
        # the collection ran in the interrupted code's context. The calls
        # get a few frames more than the limit for themselves.
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(limit + _HEADROOM)
        try:
            if phase == "start":
                context = contextvars.Context()
                enter(context)
                entered.append(context)
            elif entered:
                leave(entered.pop())
        finally:
            with contextlib.suppress(RecursionError):  # deeper than the old limit now: left raised
                sys.setrecursionlimit(limit)

    own_context._cash_own_context = True  # type: ignore[attr-defined]
    gc.callbacks.append(own_context)
    return True


#: True when collections run in a context of their own (CPython 3.10, 3.11).
INSTALLED = _install()
