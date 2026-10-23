"""Objects an output holds that stand for the process, not for data.

A progress bar holds the stream it writes to: ``pyprind.ProgBar(n,
stream=sys.stdout)``. ``sys.stdout`` cannot be pickled, so neither could the
bar, and a loop that called ``pbar.update()`` ran again after every restart.
What plain Python gives after a restart and a Run All is a bar writing to the
stream the process has then. So a stream that *is* ``sys.stdout`` or
``sys.stderr`` (or ``sys.__stdout__`` / ``sys.__stderr__``) when it is
stored is stored as that name, and a restore binds whatever the name holds
now (`current_stream`). Any other file object stays unstorable.

A lock an object holds for its own use is stored as a fresh unlocked lock of
the same kind, when it is not held as it is stored; a held one stays
unstorable.

Only what ``pickle`` refuses is stored this way (`live_reduce`), so nothing
that could be stored before is stored, or keyed, differently.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

__all__ = ["current_stream", "is_live_stream", "live_reduce"]

#: The ``sys`` names a stored stream may stand for, in the order they are
#: tried: an object holding ``sys.stdout`` restores to the stdout of the
#: process it is restored in.
_STREAM_NAMES = ("stdout", "stderr", "__stdout__", "__stderr__")

_LOCK = type(threading.Lock())
_RLOCK = type(threading.RLock())


def current_stream(name: str) -> Any:
    """The stream ``sys.<name>`` holds now: what a stored one restores to."""
    return getattr(sys, name)


def _stream_name(obj: Any) -> str | None:
    """The ``sys`` name *obj* is bound to now, or None. A writer cash puts in
    front of a stream while a cell runs (``notebook/_tee.py``) stands for the
    stream it writes to."""
    wrapped = getattr(type(obj), "_cash_wrapped_stream", None)
    if wrapped is not None:
        obj = wrapped(obj)
    for name in _STREAM_NAMES:
        if getattr(sys, name, None) is obj:
            return name
    return None


def is_live_stream(obj: Any) -> bool:
    """Whether *obj* is one of the process's standard output streams: a
    restore hands back the stream of the process, as it hands back a module."""
    return _stream_name(obj) is not None


def _unheld(lock: Any) -> bool:
    """Whether *lock* is free: neither ``threading.Lock`` nor, before 3.14,
    ``threading.RLock`` says so but by its repr."""
    return repr(lock).startswith("<unlocked")


def live_reduce(obj: Any) -> tuple | None:
    """A reduce for *obj*, which ``pickle`` refused, when it is a standard
    stream or a free lock (see the module docstring); else None."""
    name = _stream_name(obj)
    if name is not None:
        return current_stream, (name,)
    kind = type(obj)
    if kind is _LOCK and _unheld(obj):
        return threading.Lock, ()
    if kind is _RLOCK and _unheld(obj):
        return threading.RLock, ()
    return None
