"""A table of functions read on every call, and what about it can change.

A cached function that reads a dispatch table or a step list (``HANDLERS =
{"a": a, ...}``) keys the table's value, the code of every function in it
and what that code reads, on every call. Built afresh each time, a hit on a
table of 1000 functions cost 100 ms or more for a body that takes a
microsecond. Most of that work depends only on the table's structure and on
the functions' code objects, so a `CodeTable` snapshot of those -- every
object in the table by identity, each function's code, defaults and
closure, and every global its code names -- stands guard for it: while the
snapshot holds, what was built from it is built again the same.

Only a plain table qualifies: builtin containers of numbers, strings and
functions with no closure, no attributes of their own and no import in their
body. Anything else is built on every call, as before.
"""

from __future__ import annotations

import dis
import types
from typing import Any

from ..install_paths import is_user_code_module
from ..value_types import IMMUTABLE_PRIMS
from .closure_fold import iter_code_scopes
from .user_code import is_cash_wrapper, is_user_code_object

__all__ = ["CodeTable", "MISSING"]

#: How many objects a table may hold to be snapshotted at all.
_MAX_ITEMS = 200_000

_PRIMS = frozenset(IMMUTABLE_PRIMS)
#: A value a function's code may name and still be checked by identity:
#: one that cannot change without becoming another object.
_SETTLED_TYPES = (types.ModuleType, types.BuiltinFunctionType, *IMMUTABLE_PRIMS)

#: A name its globals do not hold (a builtin, or not defined yet).
MISSING = object()


def _items(value: Any) -> tuple[list[Any], list[types.FunctionType]] | None:
    """Every object in *value*, in a fixed order, and the functions among
    them; None when it holds anything but builtin containers, immutable
    primitives and plain functions."""
    items: list[Any] = []
    functions: list[types.FunctionType] = []
    stack = [value]
    while stack:
        x = stack.pop()
        items.append(x)
        t = type(x)
        if t is dict:
            stack.extend(x.keys())
            stack.extend(x.values())
        elif t is list or t is tuple:
            stack.extend(x)
        elif t in _PRIMS:
            pass
        elif t is types.FunctionType and not is_cash_wrapper(x):
            functions.append(x)
        else:
            return None
        if len(items) > _MAX_ITEMS:
            return None
    return items, functions


def _function_state(fn: types.FunctionType) -> tuple | None:
    """What of *fn* the snapshot checks by identity, or None when *fn* does
    not qualify: a closure, attributes of its own, or a default that can
    change in place."""
    if fn.__closure__ is not None or fn.__dict__:
        return None
    for defaults in (fn.__defaults__ or (), (fn.__kwdefaults__ or {}).values()):
        if any(type(d) not in _PRIMS for d in defaults):
            return None
    return (fn, fn.__code__, fn.__defaults__, fn.__kwdefaults__)


def _imports(code: types.CodeType) -> bool:
    return any(ins.opname in ("IMPORT_NAME", "IMPORT_FROM") for ins in dis.get_instructions(code))


class CodeTable:
    """A snapshot of a plain table of functions (`CodeTable.of`), and of the
    globals its functions name when *names* (`CodeTable.with_names`)."""

    __slots__ = ("_functions", "_items", "_names", "value")

    def __init__(self, value: Any, items: list[Any], functions: list[tuple]) -> None:
        self.value = value
        self._items = items
        self._functions = functions
        self._names: list[tuple[dict, str, Any]] = []

    @classmethod
    def of(cls, value: Any) -> CodeTable | None:
        """A snapshot of *value*, or None when it is no plain table of
        functions."""
        found = _items(value)
        if found is None or not found[1]:
            return None
        items, functions = found
        states: dict[int, tuple] = {}
        for fn in functions:
            if id(fn) in states:
                continue
            state = _function_state(fn)
            if state is None:
                return None
            states[id(fn)] = state
        return cls(value, items, list(states.values()))

    @property
    def functions(self) -> list[types.FunctionType]:
        return [state[0] for state in self._functions]

    def with_names(self) -> bool:
        """Add to the snapshot every global the functions' code names, and
        each attribute it names of a user module among them. False -- the
        snapshot cannot stand for what the code reads -- when one is
        anything but a module, a builtin or an immutable primitive (a user
        function or class, a mutable container: what it holds can change
        without a new object), or the code imports or reads ``__doc__``."""
        seen: set[tuple[int, str]] = set()
        for fn in self.functions:
            g = fn.__globals__
            names: set[str] = set()
            for scope in iter_code_scopes(fn.__code__):
                if _imports(scope):
                    return False
                names.update(scope.co_names or ())
            if "__doc__" in names:
                return False
            for name in sorted(names):
                value = g.get(name, MISSING)
                if not self._settled(g, name, value, seen):
                    return False
                if isinstance(value, types.ModuleType) and is_user_code_module(value):
                    # A library module's attributes are not folded, and the
                    # code they hold is not followed: only a user module's.
                    attrs = vars(value)
                    for attr in sorted(names):
                        if not self._settled(attrs, attr, attrs.get(attr, MISSING), seen):
                            return False
        return True

    def _settled(self, space: dict, name: str, value: Any, seen: set) -> bool:
        if value is not MISSING and not isinstance(value, _SETTLED_TYPES):
            if not (isinstance(value, (types.FunctionType, type)) and not is_user_code_object(value)):
                return False
        if (id(space), name) not in seen:
            seen.add((id(space), name))
            self._names.append((space, name, value))
        return True

    def holds(self, value: Any) -> bool:
        """Is *value* still the table this snapshot was taken of, unchanged?"""
        if value is not self.value:
            return False
        found = _items(value)
        if found is None:
            return False
        items = found[0]
        if len(items) != len(self._items) or any(a is not b for a, b in zip(items, self._items)):
            return False
        for fn, code, defaults, kwdefaults in self._functions:
            if (
                fn.__code__ is not code
                or fn.__defaults__ is not defaults
                or fn.__kwdefaults__ is not kwdefaults
                or fn.__dict__
            ):
                return False
        return all(space.get(name, MISSING) is value for space, name, value in self._names)
