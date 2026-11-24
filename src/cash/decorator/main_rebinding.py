"""Which namespaces a watched world provably reads nothing of: a plain
rebinding there leaves a memo standing.

A script or notebook rebinds its variables all the time, and each one whose
name scanned code names (an attribute name counts: ``ids`` for
``self.ids``) moves the C extension's world epoch. A memo that holds only
while that epoch stands still -- a class fold (`class_memo`), a state
segment (`state_memo`) -- then works again on every call of a loop at the
top of a script, hits included, though nothing in its world moved.

The extension remembers the latest moves of each epoch and the namespace
each one was a plain rebinding in (neither value a callable or a module):
``world_moved_only_in`` for the world epoch, ``core_moved_only_in`` for the
core epoch the state memo compares. It notes which namespaces a scan
watches (`watching`). A memo holds across moves that were all plain
rebindings in ``__main__`` when its world does not read ``__main__``
(`outside_namespaces`); any other change still breaks it as before.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable, Iterable
from typing import Any

from .. import _native
from .._type_memo import class_module
from .._user_attrs import plain_getattr
from . import hit_memo

__all__ = ["outside_namespaces", "watching"]

_cw = _native.cwatch

_LOCK = threading.RLock()
#: The sets of the `watching` calls open on the thread holding `_LOCK`,
#: outermost first: a scan inside another one notes into both.
_OPEN: list[set[int]] = []


def watching(into: set[int], scan: Callable[[], Any]) -> Any:
    """Run *scan* (a `hit_memo` scan) and add to *into* the id of every
    namespace it watches (`cwatch.world_watch_log`): the module globals of
    the code it met, the modules whose attributes that code names, each
    dict it read. The modules every scan watches are watched first, so only
    the scan's own are noted. Another thread's watch meanwhile adds its own,
    and a scan run inside this one adds to both, which only makes the sets
    larger: fewer namespaces count as outside."""
    log = getattr(_cw, "world_watch_log", None)
    if log is None:
        into.add(id(getattr(sys.modules.get("__main__"), "__dict__", None)))
        return scan()
    with _LOCK:
        hit_memo.watch_modules()
        if _OPEN:
            noted = log(False)
            for outer in _OPEN:
                outer.update(noted)
        _OPEN.append(into)
        log(True)
        try:
            return scan()
        finally:
            noted = log(False)
            _OPEN.pop()
            into.update(noted)
            for outer in _OPEN:
                outer.update(noted)
            if _OPEN:
                log(True)


def outside_namespaces(
    reports: Iterable[Any], cells: tuple, tags: tuple, watched: set[int], defined_in: Iterable[str] = ()
) -> frozenset[int]:
    """The namespaces a plain rebinding in which leaves a watched world
    standing: the script's or notebook's (``__main__``), unless the world
    reads it.

    The world reads ``__main__`` when the scan watched its namespace
    (*watched*: a module of the world naming ``__main__``'s variables is
    among them), a module of *defined_in* (the class or function the memo
    is for) is ``__main__``, a class or function the scan met (*cells*,
    *tags*) was defined there, a function met runs in it or names it
    (``import __main__``, ``sys.modules["__main__"]``), or an analysis of
    *reports* resolved a helper through it. A change there that is not a
    plain rebinding (a function, a class or a module bound, a name of a
    dict no key reads) still moves the world as before; so does any change
    anywhere else."""
    main = getattr(sys.modules.get("__main__"), "__dict__", None)
    if type(main) is not dict or id(main) in watched:
        return frozenset()
    if "__main__" in defined_in:
        return frozenset()
    for report in reports:
        paths = [m for m, _chain in report.helper_resolution_paths.values()]
        paths.extend(m for m, _chain, _ref in report.helper_bindings)
        if "__main__" in paths:
            return frozenset()
    for x in (*cells, *tags):
        if isinstance(x, type):
            if class_module(x) == "__main__":
                return frozenset()
        elif type(x) is types.FunctionType:
            if x.__globals__ is main or plain_getattr(x, "__module__", None) == "__main__" or _names_main(x.__code__):
                return frozenset()
    return frozenset((id(main),))


def _names_main(code: types.CodeType) -> bool:
    """Does *code*, or code nested in it, name ``__main__``?"""
    if "__main__" in code.co_names or "__main__" in code.co_consts:
        return True
    return any(type(c) is types.CodeType and _names_main(c) for c in code.co_consts)
