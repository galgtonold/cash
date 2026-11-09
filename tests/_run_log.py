"""A list of what a function under test was called with, filled from outside it.

A test counts a callee's runs by having it note its argument: ``note(v)``
in the body, ``note = calls.note`` in the closure, ``calls == [5]`` after.
The note has to be something a cache hit may skip, since skipping the run
is what a hit is, and that leaves the callee keyed as before. A list's own
``append`` held in the closure is not: it changes the list, which a hit
must not skip, so a callee holding one runs every time. Nor is a logger's
method, which a hit may skip but which is no plain data, so a call of its
holder is not keyed on content.

``note`` raises an audit event (``sys.audit``, a builtin over a string): it
changes no object the callee or the notebook can read, and a hook this
module installs once appends the argument to the list the event names. A
test whose callee must hold nothing but plain data and code, to be keyed on
content, raises it itself: ``sys.audit(event, v)`` with ``event =
calls.event``, a string.
"""

from __future__ import annotations

import functools
import itertools
import sys
import weakref

_PREFIX = "tests.run_log."
_IDS = itertools.count()
_LOGS: weakref.WeakValueDictionary[str, RunLog] = weakref.WeakValueDictionary()


def _hook(event: str, args: tuple) -> None:
    if event.startswith(_PREFIX):
        log = _LOGS.get(event)
        if log is not None:
            log.append(args[0])


sys.addaudithook(_hook)


class RunLog(list):
    """What ``note`` was called with, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.event = f"{_PREFIX}{next(_IDS)}"
        _LOGS[self.event] = self
        self.note = functools.partial(sys.audit, self.event)
