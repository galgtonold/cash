"""The notebook functions a value holds, at any depth, so a statement that
reaches one through a container keys the globals it reads at the call.

``fns = [make]`` above ``r = fns[0](3)``: the statement names ``fns``, a
list, and a call key walks only into what an input names as a function
(``cache_key.called_function_dependencies``). ``make`` reads ``cfg`` when it
runs, so an edited ``cfg`` must re-key ``r``; without a notebook file to
simulate there was nothing else to move it, and the old result was served.
The same holds for a dict of builders, a ``functools.partial``, a lambda in
a dict, a function kept as another's default or in its closure, a class
attribute, and an object of a notebook class whose methods read globals.

A walk that meets more objects than `WALK_LIMIT` stops and says so
(`Held.complete`): what it did not see may hold a function, so the
statement runs uncached, with a reason (``statement.processor``), rather
than be keyed without it.
"""

from __future__ import annotations

import collections
import functools
import itertools
import operator
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

from .._memo import CODE_OBJECTS, LruMemo
from ..install_paths import is_user_code_module
from ..value_types import LEAF_TYPES

__all__ = ["WALK_LIMIT", "Held", "held_notebook_code"]

#: Objects one walk visits before it gives up. A model, a figure, a frame
#: and their containers hold far fewer that are not plain values (a list of
#: numbers is gathered at C speed and stops the walk).
WALK_LIMIT = 2_000_000

_LEAVES = frozenset(LEAF_TYPES)
_SEQUENCES = (list, tuple, set, frozenset)
_PLAIN = frozenset((dict, list, tuple))
_MAIN_NAMES = frozenset(("__main__", "builtins"))
#: ``Py_TPFLAGS_HEAPTYPE``: set on every class a ``class`` statement makes.
_HEAP_TYPE = 1 << 9


class Held(NamedTuple):
    """What `held_notebook_code` found."""

    #: The code of each notebook function held, a notebook class's methods
    #: included, in the order first met.
    codes: tuple[types.CodeType, ...]
    #: False when the walk stopped at `WALK_LIMIT` before it saw everything.
    complete: bool = True


_NOTHING = Held(())

#: ``(id(value), lineage)`` -> the functions and notebook classes a walk
#: found, and whether it saw everything. The lineage moves whenever a cell
#: rebinds or changes the value, so a container walked once is not walked
#: again for every statement reading it. A class's methods are read again
#: each time: ``M.build = other`` changes what an object of ``M`` runs.
_MEMO: LruMemo[tuple[int, str], tuple[tuple[Any, ...], tuple[type, ...], bool]] = LruMemo(CODE_OBJECTS)


def held_notebook_code(value: Any, namespace: Mapping[str, Any], lineage: str | None = None) -> Held:
    """The code of the notebook functions *value* holds, and of the methods
    of the notebook classes it holds or is an object of.

    A notebook function is one whose globals are *namespace*. Followed: the
    items (and non-string keys) of a list, tuple, set or dict; a function's
    defaults, closure and attributes (``__wrapped__``); a bound method's
    function and object; a partial's function and arguments; a static or
    class method; a property's accessors; a notebook class's attributes
    (along its MRO); and the ``__dict__`` and slots of any other object, a
    library's included -- a ``FunctionTransformer(make)`` calls ``make``.
    Not entered: modules, library classes, and what holds no attributes (a
    numpy array).

    *lineage* is the value's lineage, when it has one: the walk is then
    remembered for it.
    """
    if value is None or type(value) in _LEAVES or isinstance(value, types.ModuleType):
        return _NOTHING
    if type(value) is types.FunctionType and _plain_function(value):
        return Held((value.__code__,)) if _is_notebook(value, namespace) else _NOTHING
    memo_key = (id(value), lineage) if lineage else None
    found = _MEMO.get(memo_key) if memo_key is not None else None
    if found is None:
        found = _walk(value, namespace)
        if memo_key is not None:
            _MEMO[memo_key] = found
    functions, classes, complete = found
    codes: dict[int, types.CodeType] = {}
    for fn in functions:
        code = getattr(fn, "__code__", None)
        if isinstance(code, types.CodeType):
            codes.setdefault(id(code), code)
    for cls in classes:
        for code in _class_codes(cls, namespace):
            codes.setdefault(id(code), code)
    return Held(tuple(codes.values()), complete)


def _plain_function(fn: types.FunctionType) -> bool:
    """A function that holds nothing more to walk: no defaults, no closure,
    no attributes. Most notebook functions; answered without a walk."""
    return not fn.__defaults__ and not fn.__kwdefaults__ and not fn.__closure__ and not fn.__dict__


def _globals_ids(namespace: Mapping[str, Any]) -> frozenset[int]:
    """The ids of the dicts a notebook function's ``__globals__`` can be:
    *namespace*, or each map of a ``ChainMap`` over it."""
    maps = getattr(namespace, "maps", None)
    if isinstance(maps, list):
        return frozenset(id(m) for m in maps)
    return frozenset((id(namespace),))


def _is_notebook(fn: Any, namespace: Mapping[str, Any]) -> bool:
    return id(getattr(fn, "__globals__", None)) in _globals_ids(namespace)


def _walk(value: Any, namespace: Mapping[str, Any]) -> tuple[tuple[Any, ...], tuple[type, ...], bool]:
    """Breadth first from *value*: ``(functions, classes, complete)``."""
    globals_ids = _globals_ids(namespace)
    functions: dict[int, Any] = {}
    classes: dict[int, type] = {}
    seen: set[int] = {id(value)}
    level: list[Any] = [value]
    visited = 0
    while level:
        visited += len(level)
        if visited > WALK_LIMIT:
            return tuple(functions.values()), tuple(classes.values()), False
        children: list[Any] = []
        # One step per TYPE in the level, not per object: the items of a
        # list of 100,000 records are gathered at C speed.
        for kind, group in _by_type(level):
            _visit_group(kind, group, globals_ids, functions, classes, children)
        if not children:
            break
        kinds_of = list(map(type, children))
        # An object that leads only to its class (`_inert`: an openpyxl
        # cell's style array, 400,000 of them) is not visited: its class is.
        inert = {kind for kind in set(kinds_of) if _inert(kind)}
        skipped = _LEAVES | inert if inert else _LEAVES
        others = list(itertools.compress(children, map(operator.not_, map(skipped.__contains__, kinds_of))))
        others.extend(inert)
        others = _without_plain_records(others)
        found = dict(zip(map(id, others), others))
        for key in seen & found.keys():
            del found[key]
        seen.update(found)
        level = list(found.values())
    return tuple(functions.values()), tuple(classes.values()), True


def _without_plain_records(objects: list[Any]) -> list[Any]:
    """*objects* less the plain dicts, lists and tuples that hold only plain
    values, when every one of a type does: a million records read as rows
    are dropped in one pass at C speed, before the walk dedups them by id
    (most of what a walk over them cost)."""
    if not set(map(type, objects)) & _PLAIN:
        return objects
    kept: list[Any] = []
    for kind, group in _by_type(objects):
        if kind is dict:
            items = itertools.chain(
                itertools.chain.from_iterable(group), itertools.chain.from_iterable(map(dict.values, group))
            )
        elif kind in _PLAIN:
            items = itertools.chain.from_iterable(group)
        else:
            kept.extend(group)
            continue
        if not set(map(type, items)) <= _LEAVES:
            kept.extend(group)
    return kept


def _visit_group(
    kind: type,
    group: list[Any],
    globals_ids: frozenset[int],
    functions: dict[int, Any],
    classes: dict[int, type],
    children: list[Any],
) -> None:
    """Note the notebook functions and classes in *group*, objects of type
    *kind*, and put what they lead to in *children*."""
    if issubclass(kind, types.ModuleType):
        return
    if issubclass(kind, dict):
        keys = list(itertools.chain.from_iterable(group))
        key_kinds = set(map(type, keys))
        # Keys that are tuples of plain values (an openpyxl sheet's cells by
        # ``(row, column)``) lead nowhere: one pass says so, as for a list.
        if not key_kinds <= _LEAVES and not (
            key_kinds <= _LEAVES | {tuple}
            and set(map(type, itertools.chain.from_iterable(_of_type(keys, tuple)))) <= _LEAVES
        ):
            children.extend(keys)
        if not set(map(type, itertools.chain.from_iterable(map(dict.values, group)))) <= _LEAVES:
            children.extend(itertools.chain.from_iterable(map(dict.values, group)))
        if kind is dict:
            return
    elif issubclass(kind, _SEQUENCES):
        # A list of numbers or strings leads nowhere: one pass at C speed
        # says so, and adds none of its items.
        if not set(map(type, itertools.chain.from_iterable(group))) <= _LEAVES:
            children.extend(itertools.chain.from_iterable(group))
        if kind in _SEQUENCES:
            return
    elif kind is types.FunctionType:
        for fn in group:
            if id(fn.__globals__) in globals_ids:
                functions.setdefault(id(fn), fn)
            children.extend(fn.__defaults__ or ())
            children.extend((fn.__kwdefaults__ or {}).values())
            children.extend(_cell_contents(fn.__closure__))
            children.extend(fn.__dict__.values())
        return
    elif kind is types.MethodType:
        for method in group:
            children.extend((method.__func__, method.__self__))
        return
    elif issubclass(kind, functools.partial):
        for part in group:
            children.append(part.func)
            children.extend(part.args)
            children.extend(part.keywords.values())
        if kind is functools.partial:
            return
    elif kind in (staticmethod, classmethod):
        children.extend(member.__func__ for member in group)
        return
    elif kind is property:
        for prop in group:
            children.extend(f for f in (prop.fget, prop.fset, prop.fdel) if f is not None)
        return
    elif issubclass(kind, type):
        for cls in group:
            if _notebook_class(cls, globals_ids):
                classes.setdefault(id(cls), cls)
                for klass in _own_mro(cls, globals_ids):
                    children.extend(vars(klass).values())
        return
    # Any other object: its class (a notebook class's methods run on it),
    # its attributes and its slots.
    children.append(kind)
    if _may_have_dict(kind):
        for obj in group:
            try:
                attrs = object.__getattribute__(obj, "__dict__")
            except (AttributeError, TypeError):
                attrs = None
            if isinstance(attrs, dict):
                children.extend(attrs.values())
    for name in _slot_names(kind):
        getter = _slot_getter(kind, name)
        if getter is not None:
            try:
                # What ``object.__getattribute__`` reads, at C speed: the
                # slot's own descriptor (400,000 openpyxl cells, 8 slots each).
                column = list(map(getter, group))
            except (AttributeError, TypeError):
                pass  # a slot not set on some object: one at a time
            else:
                _add_column(column, children)
                continue
        for obj in group:
            try:
                children.append(object.__getattribute__(obj, name))
            except (AttributeError, TypeError):
                pass


def _add_column(column: list[Any], children: list[Any]) -> None:
    """Put what one slot of a group of objects holds in *children*, at C
    speed: nothing for a column of plain values (each cell's value), the
    class alone for a column of `_inert` objects (each cell's style array),
    each object once (the sheet that 400,000 cells name as their parent)."""
    kinds = set(map(type, column))
    if kinds <= _LEAVES:
        return
    if all(map(_inert, kinds - _LEAVES)):
        children.extend(kinds - _LEAVES)
        return
    children.extend(dict(zip(map(id, column), column)).values())


def _of_type(objects: list[Any], kind: type) -> Iterator[Any]:
    """The objects of exact type *kind* in *objects*, at C speed."""
    return itertools.compress(objects, map(operator.is_, map(type, objects), itertools.repeat(kind)))


#: A type with more than 1/`_MAJOR` of a level's objects is split out of it
#: in a pass at C speed of its own; the rest are sorted one at a time.
_MAJOR = 16


def _by_type(objects: list[Any]) -> list[tuple[type, list[Any]]]:
    """*objects* grouped by exact type, each group in its order in *objects*.
    The types most of them have are split out at C speed (a level of
    400,000 openpyxl cells and as many index tuples was split by a
    comprehension per type), the few others one at a time."""
    counts = collections.Counter(map(type, objects))
    if len(counts) == 1:
        return [(next(iter(counts)), objects)]
    major = [kind for kind, n in counts.items() if n * _MAJOR > len(objects)]
    types_of = list(map(type, objects))
    groups = {
        kind: list(itertools.compress(objects, map(operator.is_, types_of, itertools.repeat(kind)))) for kind in major
    }
    if len(major) < len(counts):
        majors = frozenset(major)
        for obj in itertools.compress(objects, map(operator.not_, map(majors.__contains__, types_of))):
            groups.setdefault(type(obj), []).append(obj)
    return list(groups.items())


#: What `_visit_group` walks into other than an object's attributes.
_WALKED_KINDS = (
    types.ModuleType,
    dict,
    *_SEQUENCES,
    types.FunctionType,
    types.MethodType,
    functools.partial,
    staticmethod,
    classmethod,
    property,
    type,
)


@functools.lru_cache(maxsize=1024)
def _inert(kind: type) -> bool:
    """An object of *kind* leads `_visit_group` to its class alone: no
    instance dict, no slots, and not a kind it walks into otherwise."""
    if kind in _LEAVES or issubclass(kind, _WALKED_KINDS):
        return False
    return not _may_have_dict(kind) and not _slot_names(kind)


@functools.lru_cache(maxsize=1024)
def _may_have_dict(kind: type) -> bool:
    """Can ``object.__getattribute__(obj, "__dict__")`` find anything on an
    object of *kind*: it has an instance dict, or a class along its MRO
    defines ``__dict__``. An openpyxl cell has slots only, and asking each of
    400,000 for its ``__dict__`` raised as many AttributeErrors."""
    if getattr(kind, "__dictoffset__", 1):
        return True
    try:
        return any("__dict__" in vars(klass) for klass in kind.__mro__)
    except TypeError:
        return True


@functools.lru_cache(maxsize=1024)
def _slot_getter(kind: type, name: str) -> Callable[[Any], Any] | None:
    """The ``__get__`` of the slot descriptor ``object.__getattribute__``
    finds for *name* on an object of *kind* (the first class along the MRO
    that defines *name*), or None when that is not a plain slot."""
    try:
        for klass in kind.__mro__:
            found = vars(klass).get(name, _MISSING)
            if found is not _MISSING:
                return found.__get__ if type(found) is types.MemberDescriptorType else None
    except TypeError:
        return None
    return None


_MISSING = object()


def _cell_contents(cells: tuple[Any, ...] | None) -> list[Any]:
    found = []
    for cell in cells or ():
        try:
            found.append(cell.cell_contents)
        except ValueError:  # an empty cell
            pass
    return found


@functools.lru_cache(maxsize=1024)
def _slot_names(kind: type) -> tuple[str, ...]:
    """The slots *kind* and its bases declare, but ``__dict__`` and ``__weakref__``."""
    names: list[str] = []
    for klass in kind.__mro__:
        slots = vars(klass).get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        try:
            names.extend(
                _mangled(klass, s) for s in slots if isinstance(s, str) and s not in ("__dict__", "__weakref__")
            )
        except TypeError:
            continue
    return tuple(dict.fromkeys(names))


def _mangled(klass: type, name: str) -> str:
    """The attribute a slot *name* of *klass* is stored under: a private
    ``__x`` is ``_Klass__x``, as for any private name in a class body (read
    by its own name, it was not found and its contents were not walked)."""
    owner = klass.__name__.lstrip("_")
    if name.startswith("__") and not name.endswith("__") and owner:
        return f"_{owner}{name}"
    return name


def _member_functions(member: Any) -> list[Any]:
    """The functions a class attribute runs: a method, a static or class
    method, a property's accessors."""
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        return [f for f in (member.fget, member.fset, member.fdel) if isinstance(f, types.FunctionType)]
    return [member] if isinstance(member, types.FunctionType) else []


def _defined_in(klass: type, globals_ids: frozenset[int]) -> bool:
    """Whether a cell defined *klass*: its module is the notebook's
    (``__main__``), or one of its own methods has the notebook's globals."""
    try:
        members = list(vars(klass).values())
    except TypeError:
        return False
    if any(id(fn.__globals__) in globals_ids for member in members for fn in _member_functions(member)):
        return True
    module = sys.modules.get(getattr(klass, "__module__", None) or "")
    return module is not None and id(getattr(module, "__dict__", None)) in globals_ids


def _own_mro(cls: type, globals_ids: frozenset[int]) -> list[type]:
    """The classes in *cls*'s MRO a cell defined."""
    return [klass for klass in cls.__mro__ if klass is not object and _defined_in(klass, globals_ids)]


def _notebook_class(cls: type, globals_ids: frozenset[int]) -> bool:
    """Whether a cell defined *cls* or a class it derives from. A class of a
    library is told at once by its module: its 500 members are not read."""
    if not cls.__flags__ & _HEAP_TYPE:
        return False  # a type built into the interpreter or an extension
    name = getattr(cls, "__module__", None) or ""
    module = sys.modules.get(name)
    # ``builtins`` too: a class a cell ``exec``s with no ``__name__`` in
    # its namespace says it is from there.
    if name not in _MAIN_NAMES and module is not None and id(getattr(module, "__dict__", None)) not in globals_ids:
        if not is_user_code_module(module):
            return False
    return bool(_own_mro(cls, globals_ids))


def _class_codes(cls: type, namespace: Mapping[str, Any]) -> list[types.CodeType]:
    """The code of the methods the notebook's classes in *cls*'s MRO hold now."""
    globals_ids = _globals_ids(namespace)
    codes = []
    for klass in _own_mro(cls, globals_ids):
        for member in list(vars(klass).values()):
            for fn in _member_functions(member):
                if id(fn.__globals__) in globals_ids:
                    codes.append(fn.__code__)
    return codes
