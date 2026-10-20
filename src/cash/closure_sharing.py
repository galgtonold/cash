"""Functions in a value whose closure holds another part of it.

``return arr, lambda: arr``: a function from a module cannot be pickled, so
a stored value keeps it by reference while the rest is copied, and the
copy's ``arr`` is no longer the one the function returns. A write through a
view of one stops reaching the other, which no plain run does. Such a value
is not stored: in a notebook the statement runs again, and ``@cash.cache``
says STORE-PARTS-SHARED. A value too big to look through in the time the
check has (`closure_check_budget`) is not stored either
(STORE-PARTS-UNCHECKED).
"""

from __future__ import annotations

import gc
import pickle
import types
from collections.abc import Mapping
from typing import Any, NamedTuple

from cash._clock import CpuLimit
from cash._user_attrs import own_dict
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["ClosureCheck", "closure_check_budget", "closures_sharing"]

_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)
_CONTAINERS = frozenset({tuple, list, set, frozenset})
#: The builtin containers whose subclasses' items `_Walker` reads.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset)
#: Not looked inside: pickled by reference, as a function is.
_OPAQUE = (type, types.ModuleType, types.FunctionType)
#: The check may take this many times what the statement or call took,
#: but not less than `CLOSURE_CHECK_FLOOR_S`: past it the value is not
#: stored, which a check that costs more than the work it saves earns.
CLOSURE_CHECK_FACTOR = 2.0
CLOSURE_CHECK_FLOOR_S = 1.0
#: Items looked at between two reads of the clock.
_CLOCK_EVERY = 4096


class ClosureCheck(NamedTuple):
    """What `closures_sharing` found: the *names* holding a function whose
    closure reaches another part of the value, and whether the walk
    *finished* within its time. Unfinished, nothing is known, and the
    value is not stored."""

    names: list[str]
    finished: bool = True


def closure_check_budget(cost: float) -> float:
    """Seconds `closures_sharing` may take on the outputs of work that took
    *cost* seconds."""
    return max(CLOSURE_CHECK_FLOOR_S, CLOSURE_CHECK_FACTOR * cost)


def _builtin_of(kind: type) -> type:
    """The builtin container *kind* subclasses."""
    return next(base for base in kind.__mro__ if base in _BUILTIN_CONTAINERS)


def _parts(item: Any) -> Any:
    """What `closures_sharing` looks inside *item* for: the items of a
    builtin container and the attributes of a user object -- the parts
    pickle writes with the value."""
    kind = type(item)
    if kind in _CONTAINERS:
        return item
    if kind is dict:
        return item.values()
    if issubclass(kind, _OPAQUE):
        return ()
    attributes = own_dict(item)
    own = attributes.values() if attributes is not None else ()
    if issubclass(kind, _BUILTIN_CONTAINERS):
        return [*own, *_items(item, kind)]
    return own


def _items(item: Any, kind: type) -> list:
    """The items of *item*, of a subclass *kind* of a builtin container (a
    named tuple, an ``OrderedDict``), by the builtin's own reading, not a
    method the class may define."""
    if issubclass(kind, dict):
        return list(dict.values(item))
    return list(_builtin_of(kind).__iter__(item))


def closures_sharing(captured_vars: Mapping[str, Any], seconds: float) -> ClosureCheck:
    """The names among *captured_vars* (a statement's outputs, or one
    ``{"result": value}``) that are, or hold in a container or an object's
    attributes, a function whose closure reaches an object found elsewhere
    in the outputs (``return arr, lambda: box['a']`` with ``box = {'a':
    arr}``). Kept by reference, such a function would not share it with
    the restored copies.

    Walked whole, within *seconds* of this thread's CPU time (`CpuLimit`):
    a cap on objects let a big output use it up and pass the function
    behind it unseen, so past the time the answer is unfinished and the
    caller stores nothing."""
    limit = CpuLimit(seconds)
    found = _functions_held(captured_vars, limit)
    if found is None:
        return ClosureCheck([], finished=False)
    functions, places = found
    if not functions:
        return ClosureCheck([])
    names: set[str] = set()
    for name, fn in functions:
        if name in names:
            continue
        reached: set[int] = set()
        if _reaches(fn, places(), reached, limit):
            names.add(name)
        elif limit.passed():
            return ClosureCheck(sorted(names), finished=False)
    return ClosureCheck(sorted(names))


class _OutOfTime(Exception):
    """The walk passed its `CpuLimit`."""


def _nothing(*_: Any) -> None:
    return None


class _Sink:
    """Where `_Walker` writes: nowhere."""

    @staticmethod
    def write(data: Any) -> int:
        return len(data)


class _Walker(pickle.Pickler):
    """Pickle's own walk, in C, over the builtin containers of a value,
    told by `reducer_override` what to do at every other object: a
    function is noted and not entered, a class or module is not entered,
    and an object is entered at its parts (`_parts`) -- no code of the
    object's runs, as its ``__reduce__`` would. Pickle's memo is the set of
    objects met, kept once each."""

    def __init__(self, limit: CpuLimit) -> None:
        super().__init__(_Sink(), protocol=5, buffer_callback=_nothing)
        self.limit = limit
        self.name = ""
        self.functions: list[tuple[str, types.FunctionType]] = []
        self.met = 0

    def reducer_override(self, obj: Any) -> Any:
        self.met += 1
        if self.met % _CLOCK_EVERY == 0 and self.limit.passed():
            raise _OutOfTime
        if obj is _nothing:
            return NotImplemented
        kind = type(obj)
        if kind is types.FunctionType:
            if obj.__closure__:
                self.functions.append((self.name, obj))
            return _nothing, ()
        if issubclass(kind, _OPAQUE):
            return _nothing, ()
        # The state is pickled after *obj* is in the memo, so an object that
        # holds itself is met once.
        if issubclass(kind, _BUILTIN_CONTAINERS):
            return _nothing, (), (own_dict(obj), _items(obj, kind))
        attributes = own_dict(obj)
        return (_nothing, (), attributes) if attributes else (_nothing, ())


def _functions_held(captured_vars: Mapping[str, Any], limit: CpuLimit) -> tuple[list, Any] | None:
    """The functions with a closure that *captured_vars* hold, each with
    its name, and how to get the ids of every object met; None past
    *limit*. A value pickle's own walk cannot finish (one nested deeper
    than its recursion) is walked by `_walk`, more slowly."""
    walker = _Walker(limit)
    try:
        for name, value in captured_vars.items():
            walker.name = name
            walker.dump(value)
    except _OutOfTime:
        return None
    except Exception:  # noqa: BLE001 - RecursionError, or any other stop of pickle's walk: walked in Python
        places: set[int] = set()
        functions: list = []
        for name, value in captured_vars.items():
            if not _walk([value], places, limit, name, functions):
                return None
        return functions, lambda: places
    memo = walker.memo
    return walker.functions, lambda: set(memo.copy())


def _fresh(parts: Any, seen: set[int]) -> list:
    """The items of *parts* that are not atoms and not in *seen*, now
    added to it; comprehensions, so a big container of numbers or plain
    records costs little per item."""
    leaves = _LEAVES
    fresh = [part for part in parts if type(part) not in leaves and id(part) not in seen]
    seen.update(map(id, fresh))
    return fresh


def _walk(roots: list, places: set[int], limit: CpuLimit, name: str, functions: list) -> bool:
    """Add to *places* what *roots* hold, and to *functions* the functions
    with a closure among them; False when *limit* ran out first. A
    container the garbage collector does not track holds no function and
    nothing that holds one, so its items are counted and not opened."""
    is_tracked = gc.is_tracked
    stack = _fresh(roots, places)
    done = 0
    while stack:
        item = stack.pop()
        done += 1
        if done % _CLOCK_EVERY == 0 and limit.passed():
            return False
        if type(item) is types.FunctionType:
            if item.__closure__:
                functions.append((name, item))
            continue
        fresh = _fresh(_parts(item), places)
        stack.extend([part for part in fresh if is_tracked(part)])
    return True


def _reaches(fn: types.FunctionType, places: set[int], reached: set[int], limit: CpuLimit) -> bool:
    """Does what *fn*'s closure holds reach an object in *places*? Another
    function met on the way is not followed: its own check is its own."""
    stack = []
    for cell in fn.__closure__ or ():
        try:
            stack.append(cell.cell_contents)
        except ValueError:  # an empty cell
            continue
    leaves = _LEAVES
    done = 0
    while stack:
        held = stack.pop()
        if type(held) in leaves or type(held) is types.FunctionType or id(held) in reached:
            continue
        if id(held) in places:
            return True
        reached.add(id(held))
        done += 1
        if done % _CLOCK_EVERY == 0 and limit.passed():
            return False
        stack.extend(_parts(held))
    return False
