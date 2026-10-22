"""Functions in a value whose closure holds another part of it.

``return arr, lambda: arr``: a function from a module cannot be pickled, so
a stored value keeps it by reference while the rest is copied, and the
copy's ``arr`` is no longer the one the function returns. A write through a
view of one stops reaching the other, which no plain run does. Such a value
is not stored: in a notebook the statement runs again, and ``@cash.cache``
says STORE-PARTS-SHARED. A value too big to look through in the time the
check has (`closure_check_budget`) is not stored either
(STORE-PARTS-UNCHECKED).

No code of a user's class runs: what its ``__getstate__`` or
``__reduce__`` would hand pickle and its own attributes do not hold is not
seen.
"""

from __future__ import annotations

import pickle
import sys
import types
from collections.abc import Iterable, Mapping
from itertools import chain, islice
from typing import Any, NamedTuple

from cash._clock import CpuLimit
from cash._plain_data import TREE_NODES, in_a_look, tree_facts, tree_levels
from cash._user_attrs import own_dict
from cash.install_paths import is_user_code_module
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["ClosureCheck", "closure_check_budget", "closures_sharing"]

#: Hold nothing: not looked inside, and no place of the value's either.
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES) | {pickle.PickleBuffer, memoryview}
#: The builtin containers, read in bulk a level at a time.
_SEQUENCES = frozenset({tuple, list, set, frozenset})
#: The builtin containers whose subclasses' items `_object_parts` reads.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset)
#: Not looked inside: pickled by reference, as a function is.
_OPAQUE = (type, types.ModuleType, types.FunctionType)
#: The check may take this many times what the statement or call took,
#: but not less than `CLOSURE_CHECK_FLOOR_S`: past it the value is not
#: stored, which a check that costs more than the work it saves earns.
CLOSURE_CHECK_FACTOR = 2.0
CLOSURE_CHECK_FLOOR_S = 1.0
#: Objects read one by one between two reads of the clock.
_CLOCK_EVERY = 4096
#: Items of a level taken in at once between two reads of the clock.
_CHUNK = 1 << 20


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


def closures_sharing(captured_vars: Mapping[str, Any], seconds: float) -> ClosureCheck:
    """The names among *captured_vars* (a statement's outputs, or one
    ``{"result": value}``) that are, or hold anywhere, a function whose
    closure reaches an object found elsewhere in the outputs (``return
    arr, lambda: box['a']`` with ``box = {'a': arr}``). Kept by reference,
    such a function would not share it with the restored copies.

    Walked whole, a level at a time, within *seconds* of this thread's CPU
    time (`CpuLimit`): a cap on objects let a big output use it up and pass
    the function behind it unseen, so past the time the answer is
    unfinished and the caller stores nothing. JSON-like data (exact dicts,
    lists and tuples over plain values, `tree_facts`) holds no function and
    is not walked unless a function elsewhere may reach into it."""
    limit = CpuLimit(seconds)
    plain: set[str] = set()
    for name, value in captured_vars.items():
        found = _plain(value, limit)
        if found is None:
            return ClosureCheck([], finished=False)
        if found:
            plain.add(name)
    places: dict[int, Any] = {}
    functions: list[tuple[str, types.FunctionType]] = []
    deferred: list[tuple[str, Any]] = []  # (name, what a deferred array hands pickle)
    for name, value in captured_vars.items():
        if name not in plain and not _walk(name, value, places, functions, limit, deferred):
            return ClosureCheck([], finished=False)
    if not functions:
        return ClosureCheck([])
    for name, value in [*((name, captured_vars[name]) for name in plain), *deferred]:
        if not _walk(name, value() if callable(value) else value, places, functions, limit):
            return ClosureCheck([], finished=False)
    names: set[str] = set()
    for name, fn in functions:
        if name in names:
            continue
        found = _reaches(fn, places, limit)
        if found is None:
            return ClosureCheck(sorted(names), finished=False)
        if found:
            names.add(name)
    return ClosureCheck(sorted(names))


def _plain(value: Any, limit: CpuLimit) -> bool | None:
    """Is *value* JSON-like data? None when *limit* ran out first. In a
    `one_look` the answer is the share check's, read once for both;
    outside one, the levels are read with the clock checked at each."""
    if type(value) not in TREE_NODES:
        return False
    try:
        if in_a_look():
            return tree_facts(value) is not None
        for _level in tree_levels(value):
            if limit.passed():
                return None
    except Exception:  # noqa: BLE001 - not plain (an object, an unhashable key): walked
        return False
    return True


def _walk(
    name: str, value: Any, places: dict[int, Any], functions: list, limit: CpuLimit, deferred: list | None = None
) -> bool:
    """Add to *places* the id of every object *value* holds, and to
    *functions* each function with a closure among them, with *name*;
    False when *limit* ran out first. A level's builtin containers are read
    in bulk; an atom is neither a place nor looked into.

    With *deferred*, an object array holding JSON-like data only (a frame's
    column of dicts, `_plain_cells`) holds no function and is added to it
    unread, with *name*, as a call giving its parts: its cells are places
    only if a function is found.
    Fifty views of a frame of 100,000 dicts were each read cell by cell,
    with no function anywhere."""
    level: list | None = _fresh([[value]], places, limit)
    if level is None:
        return False
    while level:
        held: list[Iterable[Any]] = []
        kinds = set(map(type, level))
        for kind in kinds:
            group = level if len(kinds) == 1 else [item for item in level if type(item) is kind]
            if kind in _SEQUENCES:
                held.append(chain.from_iterable(group))
            elif kind is dict:
                held.append(chain.from_iterable(group))  # the keys
                held.append(chain.from_iterable(map(dict.values, group)))
            elif kind is types.FunctionType:
                functions.extend((name, fn) for fn in group if fn.__closure__)
            elif not issubclass(kind, _OPAQUE):
                for done, item in enumerate(group, 1):
                    if done % _CLOCK_EVERY == 0 and limit.passed():
                        return False
                    if deferred is not None and _object_array(kind, item):
                        plain = _plain_cells(item, limit)
                        if plain is None:
                            return False
                        if plain:  # the array is a place already: its parts are walked
                            deferred.append((name, lambda item=item, kind=kind: _object_parts(item, kind)))
                            continue
                    held.append(_object_parts(item, kind))
        if limit.passed():
            return False
        level = _fresh(held, places, limit)
        if level is None:
            return False
    return True


def _object_array(kind: type, item: Any) -> bool:
    """Is *item*, of *kind*, a numpy array of objects?"""
    return kind.__name__ == "ndarray" and kind.__module__ == "numpy" and item.dtype.hasobject


def _plain_cells(array: Any, limit: CpuLimit) -> bool | None:
    """Does the object *array* hold JSON-like data only (`tree_levels`)?
    None when *limit* ran out first."""
    try:
        for _level in tree_levels(array.tolist()):
            if limit.passed():
                return None
    except Exception:  # noqa: BLE001 - not plain (an object, an unhashable key): walked
        return False
    return True


def _fresh(held: list[Iterable[Any]], places: dict[int, Any], limit: CpuLimit) -> list | None:
    """What *held* yields that is not an atom and not in *places*, each
    once, now added to it; None when *limit* ran out first. Taken in
    chunks, each a few passes in C: a level of ten million strings is
    typed and dropped, never remembered one by one.

    *places* holds each object by its id while the look lasts: the tuples
    and lists a reduce hands over are made for the look, and one dropped
    could leave its id to a later one, read as seen and not looked into."""
    leaves = _LEAVES
    items = chain.from_iterable(held)
    fresh: list = []
    while chunk := list(islice(items, _CHUNK)):
        if not leaves.issuperset(map(type, chunk)):
            new = {id(item): item for item in chunk if type(item) not in leaves}
            for key in places.keys() & new.keys():
                del new[key]
            places.update(new)
            fresh.extend(new.values())
        if limit.passed():
            return None
    return fresh


_SLOTS: dict[type, tuple] = {}
_USERS: dict[type, bool] = {}
_REDUCES: dict[type, bool] = {}
_MODULE_OF = type.__dict__["__module__"].__get__


def _slot_readers(kind: type) -> tuple:
    """The member descriptor of each ``__slots__`` entry along *kind*'s
    MRO: how pickle reads a slotted object, read without running its code."""
    found = _SLOTS.get(kind)
    if found is None:
        readers = []
        for klass in kind.__mro__:
            own = klass.__dict__
            if "__slots__" in own:
                readers.extend(m for m in own.values() if type(m) is types.MemberDescriptorType)
        found = _SLOTS[kind] = tuple(readers)
    return found


def _users(kind: type) -> bool:
    """Is *kind* defined in the user's code? Its own methods are never
    asked, where a library's (``__reduce_ex__``) are."""
    found = _USERS.get(kind)
    if found is None:
        module = _MODULE_OF(kind)  # type's own reading: no metaclass of the user's runs
        found = _USERS[kind] = is_user_code_module(sys.modules.get(module)) if module else False
    return found


def _object_parts(item: Any, kind: type) -> list:
    """What pickle writes with *item*, an object of *kind* that is no
    builtin container: its attributes, its slots, the items of a builtin
    container it subclasses, and, for a library's object with no attributes
    of its own (a numpy object array, a ``functools.partial``) or a reduce
    of its own (a pandas block), what its own ``__reduce_ex__`` hands
    pickle. A pandas block keeps its values out of its attributes, and once
    its dtype is read it has one (``_cache``): read by its attributes alone,
    a function in a frame's object column went unseen after ``df.dtypes``.
    No code of a user's class runs."""
    if kind is types.MethodType:  # read by the type's own members, as pickle reads it
        return [item.__func__, item.__self__]
    parts: list = []
    attributes = own_dict(item)
    if attributes:
        parts.extend(attributes.values())
    for reader in _slot_readers(kind):
        try:
            parts.append(reader.__get__(item, kind))
        except AttributeError:  # an empty slot
            continue
    if issubclass(kind, _BUILTIN_CONTAINERS):
        parts.extend(_items(item, kind))
    elif not _users(kind) and (not attributes or _reduces(kind)):
        parts.extend(_reduced(item, kind))
    return parts


def _reduces(kind: type) -> bool:
    """Does *kind*, a library's class, define how pickle reduces it
    (``__reduce_ex__`` or ``__reduce__`` other than ``object``'s)?"""
    found = _REDUCES.get(kind)
    if found is None:
        found = _REDUCES[kind] = (
            kind.__reduce_ex__ is not object.__reduce_ex__ or kind.__reduce__ is not object.__reduce__
        )
    return found


def _items(item: Any, kind: type) -> list:
    """The items of *item*, of a subclass *kind* of a builtin container (a
    named tuple, an ``OrderedDict``), by the builtin's own reading, not a
    method the class may define."""
    if issubclass(kind, dict):
        return [*dict.keys(item), *dict.values(item)]
    base = next(base for base in kind.__mro__ if base in _BUILTIN_CONTAINERS)
    return list(base.__iter__(item))


def _reduced(item: Any, kind: type) -> list:
    """What a library's *item* hands pickle (``__reduce_ex__``): the
    arguments, the state and the items of the value it reduces to. A numpy
    array holds its items only with object dtype (otherwise its bytes, which
    a copy may read through a view, so its base); reduced, an array that is
    not contiguous would be copied."""
    if kind.__name__ == "ndarray" and kind.__module__ == "numpy":
        if not item.dtype.hasobject:
            return [] if item.base is None else [item.base]
    try:
        reduced = item.__reduce_ex__(5)
    except Exception:  # noqa: BLE001 - not picklable: stored by reference or not at all
        return []
    if not isinstance(reduced, tuple):
        return []
    parts: list = [*reduced[1:2], *reduced[2:3]]
    for extra in reduced[3:5]:
        if extra is not None:
            try:
                parts.extend(extra)
            except Exception:  # noqa: BLE001 - the library's own iterator
                continue
    return parts


def _reaches(fn: types.FunctionType, places: dict[int, Any], limit: CpuLimit) -> bool | None:
    """Does what *fn*'s closure holds reach an object in *places*? None
    when *limit* ran out first. Another function met on the way is not
    followed: its own check is its own."""
    held = []
    for cell in fn.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell
            continue
    reached: dict[int, Any] = {}
    level = _fresh([held], reached, limit)
    while level:
        if any(id(item) in places for item in level if not issubclass(type(item), _OPAQUE)):
            return True
        parts: list = []
        for done, item in enumerate(level, 1):
            if done % _CLOCK_EVERY == 0 and limit.passed():
                return None
            kind = type(item)
            if kind in _SEQUENCES:
                parts.append(item)
            elif kind is dict:
                parts.append(item)
                parts.append(item.values())
            elif not issubclass(kind, _OPAQUE):
                parts.append(_object_parts(item, kind))
        level = _fresh(parts, reached, limit)
    return None if level is None else False
