"""What a thread a statement started and left running can change.

A statement that starts a thread of the user's (``threading.Thread``, a
``Timer``, ``executor.submit``) and returns while it runs was stored as it
stood when it returned: a file the thread read, or a variable it changed,
after that was in no key. Worse, a later statement reading that variable was
keyed on its lineage, which the thread's write did not move, so a later run
served what the variable held the first time (``RESULT['v']`` filled by a
thread from a file since edited). The decorator refuses the same case
(STORE-WORK-RUNNING); here:

* the statement that left the work running is not stored (its file tracker
  counts the work, `FileAccessTracker.work_outlived_block`);
* the variables of the notebook the work's function can reach (the globals
  its code names, the objects its closure holds) are noted, and a statement
  reading one is not stored either, until a statement binds the name again;
* while the work runs (a thread, or a process the statement started and did
  not wait for), no statement is served from the cache: a ``time.sleep(1.0)``
  left to wait for a timer's write was skipped after its first run, and the
  read below it ran before the write (`still_running`).

The cell still warns NOTEBOOK-THREAD-ALIVE (`background_threads`).
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or, instance_of

__all__ = ["ThreadReach", "reachable"]

#: Values that are code or a namespace, not state a thread changes.
_NOT_STATE = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)


def _codes(code: types.CodeType) -> Iterable[types.CodeType]:
    stack = [code]
    while stack:
        c = stack.pop()
        yield c
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))


def _function(fn: Any) -> Any:
    for _ in range(8):
        inner = attribute_or(fn, "__func__", None) or attribute_or(fn, "func", None)
        if inner is None:
            break
        fn = inner
    return fn


def reachable(functions: Iterable[Any], user_ns: Mapping[str, Any]) -> dict[str, str]:
    """The variables of *user_ns* each of *functions*, handed to a thread,
    can change, to the function's name: the globals its code names, the
    variables bound to what its closure holds, and the object a method
    bound to one is of (``D.__setitem__``)."""
    found: dict[str, str] = {}
    by_id: dict[int, list[str]] | None = None

    def holding(value: Any) -> list[str]:
        nonlocal by_id
        if by_id is None:
            by_id = {}
            for name, held in user_ns.items():
                if not name.startswith("_") and not isinstance(held, _NOT_STATE):
                    by_id.setdefault(id(held), []).append(name)
        return by_id.get(id(value), [])

    for fn in functions:
        owner = attribute_or(fn, "__self__", None)
        fn = _function(fn)
        label = attribute_or(fn, "__qualname__", None) or repr(fn)
        if owner is not None and not instance_of(owner, types.ModuleType):
            for name in holding(owner):
                found.setdefault(name, label)
        code = attribute_or(fn, "__code__", None)
        if code is None:
            continue
        if attribute_or(fn, "__globals__", None) is user_ns:
            for c in _codes(code):
                for name in c.co_names:
                    if name in user_ns and not instance_of(user_ns[name], _NOT_STATE):
                        found.setdefault(name, label)
        for cell in attribute_or(fn, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            for name in holding(value):
                found.setdefault(name, label)
    return found


class ThreadReach:
    """The notebook's variables work left running in a thread can change."""

    def __init__(self) -> None:
        #: name -> what reaches it, for the message.
        self.names: dict[str, str] = {}
        #: Work statements left running, while it runs: whether it still
        #: does, and what it is, for the message.
        self._running: list[tuple[Callable[[], bool], str]] = []

    def left_running(self, runs: Callable[[], bool], label: str) -> None:
        """A statement returned with work still running: *runs* says whether
        it still does, *label* names it."""
        self._running.append((runs, label))

    def still_running(self) -> str | None:
        """What work a statement left running has not finished yet, or None."""
        if self._running:
            self._running = [(runs, label) for runs, label in self._running if runs()]
        return self._running[0][1] if self._running else None

    def note(self, functions: Iterable[Any], user_ns: Mapping[str, Any]) -> None:
        """Note what each of *functions* can reach in *user_ns*
        (`reachable`)."""
        for name, label in reachable(functions, user_ns).items():
            self.names.setdefault(name, label)

    def reached(self, names: Iterable[str]) -> list[tuple[str, str]]:
        """Those of *names* running work can change: ``(name, its function)``."""
        if not self.names:
            return []
        return sorted((name, self.names[name]) for name in names if name in self.names)

    def rebound(self, names: Iterable[str]) -> None:
        """*names* were bound again by a statement that left no work running."""
        if self.names:
            for name in names:
                self.names.pop(name, None)
