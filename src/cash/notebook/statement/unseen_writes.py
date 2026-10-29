"""Code of the user's that ran in a statement and changes something its key
and outputs leave out.

The key counts the functions a statement reaches along the paths cash can
read from its text: a name it calls, a method, a special method of an object
it names, an item it reads. Others run unseen: ``Vec.__add__`` on an object
in an object-dtype column (``s + 1``), a closure's cell, a classmethod, an
item two steps down (``d[k][0] + 1``). When such a function appends to a
global list (``calls.append(1)``), a served entry drops the append.

On Python 3.12+ every function of the user's that ran is recorded
(``trace_recorder.Unit.codes``). After a run, `unseen_writers` reads each
one's own text for what it changes outside its objects: a global it changes
or rebinds (``calls.append(1)``), a variable of a closure it changes. A
method of a library object it calls (``porter.stem(w)``) is taken to leave
the object alone, as for a function the statement calls by name.
What it calls by name needs no reading: that ran too, and is read in turn.
A change the statement's outputs already hold is kept by a served entry.
Any other means the run is not stored (`UNSEEN_WRITES_REASON`), and a hit on
an entry whose recorded trace holds such a function is not served.
"""

from __future__ import annotations

import ast
import dis
import functools
import inspect
import json
import os
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..._user_attrs import attribute_or, has_attribute, instance_of, own_dict
from ...analysis.callee_effects import free_vars_mutated_in_function, scope_locals
from ...analysis.mutations import MUTATING_METHODS
from ...exceptions import SOURCE_RETRIEVAL_ERRORS
from ...install_paths import is_user_code_file
from ...source_reading import getsource

__all__ = [
    "UNSEEN_WRITES_REASON",
    "COULD_RUN_REASON",
    "COULD_RUN_TOO_BIG_REASON",
    "library_made",
    "live_writers",
    "called_methods",
    "own_writes",
    "unseen_writers",
    "writers_without_a_trace",
]

#: What the statement says when it is not stored for one.
UNSEEN_WRITES_REASON = (
    "{fn}() ran in this statement and changes {targets}, which the statement's own text does not reach; "
    "a stored result would not change {it}"
)
#: The same without a record of what ran (`writers_without_a_trace`): the
#: method may run, nothing says it did.
COULD_RUN_REASON = (
    "{fn}() could run in this statement (no record of what runs is kept here: Python before 3.12, "
    "CASH_TRACE=0, or a debugger holding sys.monitoring) and changes {targets}, which the statement's own "
    "text does not reach; a stored result would not change {it}"
)
#: And when a value the statement reads was too big to look through: every
#: class of the user's alive with such a method counts.
COULD_RUN_TOO_BIG_REASON = (
    "{fn}() could run in this statement: a value it reads is too big to look through, and no record of what "
    "runs is kept here (Python before 3.12, CASH_TRACE=0, or a debugger holding sys.monitoring); it changes "
    "{targets}, which the statement's own text does not reach, and a stored result would not change {it}"
)
#: What `writers_without_a_trace` gives for a special method set on a class
#: after it was made, and the reason said for it: what it does is not the
#: point, an edit to it cannot be seen.
SET_LATER = "<set on the class later>"
SET_LATER_REASON = (
    "{fn}() may run in this statement and was set on its class after the class was made, so cash cannot "
    "see an edit to it without a record of what ran; the statement runs every time"
)

#: What stands for the methods when the objects could not be looked
#: through at all (an error in cash's own walk), and the reason said.
UNREADABLE = "<unreadable>"
UNREADABLE_REASON = (
    "cash could not look through what this statement reads for a special method of yours that could run "
    "in it, and no record of what runs is kept here (Python before 3.12, CASH_TRACE=0, or a debugger holding "
    "sys.monitoring); the statement runs every time"
)

#: ``CASH_UNSEEN_LOG=<path>``: each function found, one JSON line, to count them.
_LOG_ENV = "CASH_UNSEEN_LOG"

_UNKNOWN = "?"


def unseen_writers(
    codes: Mapping[types.CodeType, tuple[dict, str]],
    statement_codes: Iterable[types.CodeType],
    outputs: Iterable[str],
    namespace: Mapping[str, Any],
    statement: str = "",
) -> list[tuple[str, tuple[str, ...]]]:
    """``(function, what it changes)`` for each function of *codes* (code ->
    its globals and module, as the trace records them) that changes
    something outside its objects the statement's *outputs* do not hold.
    The statement's own code (*statement_codes*, with the functions and
    comprehensions inside it) is the statement's analysis's, not looked at."""
    own: set[types.CodeType] = set()
    pending = list(statement_codes)
    while pending:
        code = pending.pop()
        if code in own:
            continue
        own.add(code)
        pending.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    kept = set(outputs)
    # A generator expression, a lambda or a function made inside one that
    # ran is part of its text, read with it (``sum(i * i for i in r)``).
    inside = {c for code in codes for c in code.co_consts if isinstance(c, types.CodeType)}
    found: list[tuple[str, tuple[str, ...]]] = []
    for code, (globals_, _module) in list(codes.items()):
        if code in own or code in inside or code.co_qualname == "<module>" or library_made(code, globals_):
            continue
        writes = own_writes(code) | set(_method_writes(code, globals_))
        if not writes:
            continue
        targets: list[str] = []
        for name in sorted(writes):
            if name == _UNKNOWN:
                targets.append("what its unreadable text changes")
            elif name in code.co_freevars:
                targets.append(f"{name!r} of the function it was made in")
            elif globals_ is namespace and name in kept:
                continue
            elif name in globals_:
                targets.append(repr(name) if globals_ is namespace else f"{globals_.get('__name__')}.{name}")
        if targets:
            found.append((_held_as(code, (namespace, globals_)), tuple(targets)))
    if found:
        _log(found, statement)
    return found


def _held_as(code: types.CodeType, spaces: Iterable[Mapping[str, Any]]) -> str:
    """The name to say *code*'s function by: the class attribute holding it
    when it was set on a class after the class was made (``Vec.__add__ =
    _add``: ``Vec.__add__``, the name a reader knows), else its own."""
    for ns in spaces:
        try:
            classes = [v for v in list(ns.values()) if issubclass(type(v), type)]
        except Exception:  # noqa: BLE001 - a namespace that cannot be listed
            continue
        for cls in classes:
            for attr, member in list(_CLASS_DICT(cls).items()):
                if issubclass(type(member), (staticmethod, classmethod)):
                    member = member.__func__
                if type(member) is types.FunctionType and member.__code__ is code:
                    return f"{_CLASS_QUALNAME(cls)}.{attr}"
    return code.co_qualname


def live_writers(
    codes: Mapping[types.CodeType, tuple[dict, str]],
    found: Iterable[tuple[str, tuple[str, ...]]],
    notebook: Mapping[str, Any],
    notebook_name: str,
) -> list[Callable[[], bool]]:
    """For each function `unseen_writers` *found* among *codes*, whether it
    is still there: alive, and named by its module and qualified name
    still. Redefined (its name now gives other code) or gone, it is not.
    One that cannot be found at all stands as there for good."""
    import gc
    import weakref

    names = {fn for fn, _ in found}
    out: list[Callable[[], bool]] = []
    for code, (_globals, module) in list(codes.items()):
        qualname = code.co_qualname
        if qualname not in names and _held_as(code, (notebook, _globals)) not in names:
            continue
        function = _named_function(module, qualname, notebook, notebook_name)
        if function is not None and function.__code__ is code:
            out.append(
                functools.partial(_still_named, weakref.ref(function), module, qualname, notebook, notebook_name)
            )
            continue
        # A lambda in a dict, a function made in another: what holds its
        # code (looked for once, for a statement refused).
        function = next(
            (f for f in gc.get_referrers(code) if type(f) is types.FunctionType and f.__code__ is code), None
        )
        out.append(_always if function is None else functools.partial(_alive, weakref.ref(function)))
    return out or [_always]


def _named_function(module: str, qualname: str, notebook: Mapping[str, Any], notebook_name: str) -> Any:
    """The function *qualname* of *module* names now (a method, a property's
    or a ``cached_property``'s function), or None."""
    from ...tracking.trace_items import candidates, resolve

    try:
        _code, obj, _owner = resolve(module, qualname, notebook, notebook_name)
        held = [obj, attribute_or(obj, "func", None)] if instance_of(obj, functools.cached_property) else [obj]
        return next((f for o in held for f in candidates(o) if f.__code__.co_qualname == qualname), None)
    except Exception:  # noqa: BLE001 - not found by its name
        return None


def _always() -> bool:
    return True


def _alive(ref: Any) -> bool:
    return ref() is not None


def _still_named(ref: Any, module: str, qualname: str, notebook: Mapping[str, Any], notebook_name: str) -> bool:
    """Whether the function *ref* holds is alive and still the one
    *qualname* of *module* names (gone from there, it cannot be told)."""
    function = ref()
    if function is None:
        return False
    now = _named_function(module, qualname, notebook, notebook_name)
    return now is None or now.__code__ is function.__code__


#: How many values `writers_without_a_trace` opens for one statement (past
#: it, or a container too big to open, a class of the user's with a writing
#: special method and an object alive counts), and how many steps down it
#: goes.
_OPEN_AT = 20_000
_DEPTH = 6
#: How many items of containers and arrays of objects it reads in all, at C
#: speed (a million in 30 ms). An array of numbers is not read.
_READ_AT = 10_000_000
#: How many of those items it looks into one by one (0.5 s for a million):
#: those that are not plain values.
_SCAN_AT = 2_000_000


def writers_without_a_trace(
    statement: str,
    namespace: Mapping[str, Any],
    outputs: Iterable[str] = (),
    too_big: set[str] | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """What `unseen_writers` would find, read from the objects instead of a
    trace (none on Python 3.10 and 3.11, with ``CASH_TRACE=0``, or while a
    debugger holds `sys.monitoring`): a special method of a class of the
    user's whose objects *statement* reaches through what it reads (an item
    of a container, ``d[k][0]``; an attribute; an object-dtype array or
    column; a function's closure or the globals it names) that changes
    something outside its objects. Any of them may run unseen. *too_big*,
    when given, gets the ones looked at only because a value was too big
    to look through (every class alive with one)."""
    from ...analysis import handed_callables as hc

    try:
        code = compile(statement, "<cash-statement>", "exec")
    except (SyntaxError, ValueError):
        return []
    names: set[str] = set()
    pending_codes = [code]
    while pending_codes:
        current = pending_codes.pop()
        # What it reads, not what it only binds (``d = {'a': [Vec()]}``).
        names.update(i.argval for i in dis.get_instructions(current) if i.opname in _LOADS)
        pending_codes.extend(c for c in current.co_consts if isinstance(c, types.CodeType))
    todo: list[tuple[Any, int]] = []
    for name in names:
        value = namespace.get(name, _MISSING)
        if value is not _MISSING and not issubclass(type(value), types.ModuleType):
            todo.append((value, 0))
    seen: set[int] = set()
    classes: dict[type, None] = {}
    # Functions of the user's it reaches: the statement's analysis reads
    # one from its text, which ``inner.__code__ = <a lambda compiled from a
    # string>`` leaves it none of; the trace reads the code that ran.
    functions: dict[Any, None] = {}
    over = False
    steps = 0
    work = [0, 0]  # items read at C speed, items looked into one by one
    while todo:
        value, depth = todo.pop()
        if value is _OVER:
            over = True
            continue
        if depth > _DEPTH:
            # Deeper than an operator reaches from what the statement
            # names in one expression (``d[k][0].items[1] + 1``).
            continue
        # Every value taken counts, seen before or not: a grid of the same
        # few ints is many steps and few new objects.
        steps += 1
        if steps > _OPEN_AT:
            over = True
            break
        if id(value) in seen:
            continue
        seen.add(id(value))
        kind = type(value)
        # The items a container or an array hands over are read too: a
        # grid of numbers is a few thousand values and millions of items.
        work[0] += _item_count(value, kind, hc)
        if work[0] > _READ_AT:
            over = True
            break
        if kind not in classes and _users_special(kind, hc):
            classes[kind] = None
        if kind is types.FunctionType:
            functions[value] = None
        todo.extend((item, depth + 1) for item in _held(value, kind, hc, work))
        if work[0] > _READ_AT or work[1] > _SCAN_AT:
            over = True
            break
    reached = set(classes)
    if over and hc.special_classes_exist(namespace):
        classes.update(dict.fromkeys(_with_objects_alive(_live_writing_classes(namespace, hc))))
    found: list[tuple[str, tuple[str, ...]]] = [
        (fn.__qualname__, ("what its unreadable text changes",)) for fn in functions if _unreadable_users(fn)
    ]
    for cls in classes:
        before = len(found)
        for attr, fn in _users_special(cls, hc):
            fn_code = attribute_or(fn, "__code__", None)
            if fn_code is None or library_made(fn_code, fn.__globals__) or hc.dataclass_made(fn, cls):
                continue
            if not _named_where_held(fn, cls):
                # Set on the class after it was made (``Vec.__add__ =
                # lambda self, o: 10``): no key reads it, so an edit to it
                # would go unseen.
                found.append((f"{_CLASS_QUALNAME(cls)}.{attr}", (SET_LATER,)))
                continue
            writes = own_writes(fn_code) | set(_method_writes(fn_code, fn.__globals__))
            if fn.__globals__ is namespace:
                writes -= set(outputs)
            if writes:
                what = ("what its unreadable text changes" if w == _UNKNOWN else repr(w) for w in sorted(writes))
                found.append((fn.__qualname__, tuple(what)))
        if too_big is not None and cls not in reached:
            too_big.update(fn for fn, _ in found[before:])
    return found


def _unreadable_users(fn: Any) -> bool:
    """A function of the user's whose code has no text to read: what it
    changes is unknown."""
    code = fn.__code__
    return (
        # What Python 3.14 compiles for a class's or function's annotations
        # (``P.__annotate__``) has no text of its own and builds a dict.
        code.co_name != "__annotate__"
        and is_user_code_file(code.co_filename)
        and not library_made(code, fn.__globals__)
        and _UNKNOWN in own_writes(code)
    )


_MISSING = object()

_LOADS = frozenset({"LOAD_NAME", "LOAD_GLOBAL", "LOAD_FROM_DICT_OR_GLOBALS"})


#: Values no special method of the user's runs on: not taken from a container.
_PLAIN = frozenset({int, float, complex, str, bytes, bool, type(None)})


def _not_plain(items: Iterable[Any]) -> list[Any]:
    if _PLAIN.issuperset(map(type, items)):
        return []
    return [item for item in items if type(item) not in _PLAIN]


def _held(value: Any, kind: type, hc: Any, work: list[int]) -> Iterable[Any]:
    """What `writers_without_a_trace` looks at inside *value*. Read by type
    and Python's own slots: no property, ``__getattr__`` or method of the
    user's runs (``dtype``, ``columns``, ``to_numpy``), and plain values
    are not taken. What it reads is added to *work*: the items of the rows
    it reads through (`_records`), and the items it looks into."""
    try:
        if kind in hc.OPENED:
            held = value.data if issubclass(kind, hc.HELD_IN_DATA) else value
            if issubclass(type(held), dict):
                items = [*dict.values(held), *dict.keys(held)]
            else:
                items = held
            if _PLAIN.issuperset(map(type, items)):
                return []  # read at C speed, a million in about 30 ms
            return _records(_not_plain(items), hc, work)
        if kind is types.FunctionType:
            if not is_user_code_file(value.__code__.co_filename) or library_made(value.__code__, value.__globals__):
                return []
            out = [cell.cell_contents for cell in value.__closure__ or () if _filled(cell)]
            g = value.__globals__
            out.extend(g[n] for n in value.__code__.co_names if n in g and not isinstance(g[n], types.ModuleType))
            return out
        if issubclass(kind, (types.MethodType, classmethod, staticmethod)):
            return [value.__func__, attribute_or(value, "__self__", None)]
        if issubclass(kind, functools.partial):
            return [value.func, *value.args, *value.keywords.values()]
        array = _array_kind(kind)
        if array is not None:
            return _array_items(value, array, hc, work)
        if issubclass(kind, type):
            if hc.is_users_class(value) or _holds_users_code(value):
                held = _not_plain(type.__dict__["__dict__"].__get__(value).values())
                # And its bases of the user's: ``class D(C): pass`` holds
                # nothing of the ``C.m`` that ``D().m(3)`` runs.
                bases = type.__dict__["__bases__"].__get__(value)
                held.extend(base for base in bases if hc.is_users_class(base) or _holds_users_code(base))
                return held
            return []
        if hc.users_instance_type(kind) or kind is types.SimpleNamespace:
            attributes = own_dict(value)
            held = _not_plain(attributes.values()) if attributes else []
            if kind is not types.SimpleNamespace:
                # Its class too: ``obj.m(3)`` runs the function its class
                # holds, which may have been given other code.
                held.append(kind)
            return held
    except Exception:  # noqa: BLE001 - a value that cannot be looked into
        return []
    return []


def _records(items: list[Any], hc: Any, work: list[int]) -> list[Any]:
    """*items* (counted into *work*) less the rows that hold nothing to
    look at, so 25,000 rows are a few values, not past `_OPEN_AT`: a list,
    tuple, dict or set of plain values (a JSON record, ``(i, name)``), and
    after the first of its class an object of the user's whose attributes
    are plain values or such containers (a dataclass with a ``list``
    field). Past `ITEM_TYPES_AT` of them, or past what one statement
    reads, ``[_OVER]``."""
    work[1] += len(items)
    if work[1] > _SCAN_AT:
        return [_OVER]
    firsts: dict[type, Any] = {}
    checked: set[int] = set()
    out = []
    for item in items:
        if id(item) in checked:
            continue  # the same row again: ``[row] * 1000``
        checked.add(id(item))
        kind = type(item)
        if kind in _FLAT and _flat(item, kind, _FLAT_DEPTH, work):
            continue
        if kind in firsts:
            if _flat_record(item, work):
                continue
        elif hc.users_instance_type(kind) and _flat_record(item, work):
            firsts[kind] = item
        if work[0] > _READ_AT:
            return [_OVER]
        out.append(item)
        if len(out) > hc.ITEM_TYPES_AT:
            return [_OVER]
    return out


#: Containers whose own methods are Python's: one holding only plain values
#: (or such containers, `_FLAT_DEPTH` down) runs nothing of the user's when
#: an operator meets it or its items. A ``defaultdict`` is not one: its
#: ``default_factory`` may be the user's.
_FLAT = frozenset({list, tuple, dict, set, frozenset})
_FLAT_DEPTH = 2


def _flat(value: Any, kind: type, depth: int, work: list[int]) -> bool:
    """Whether *value*, of the `_FLAT` type *kind*, holds only plain values
    or such containers, *depth* down; its items are counted into *work*."""
    work[0] += len(value)
    if work[0] > _READ_AT:
        return False
    items = [*dict.values(value), *dict.keys(value)] if kind is dict else value
    if _PLAIN.issuperset(map(type, items)):
        return True
    if depth <= 1:
        return False
    for item in items:
        inner = type(item)
        if inner not in _PLAIN and not (inner in _FLAT and _flat(item, inner, depth - 1, work)):
            return False
    return True


def _flat_record(item: Any, work: list[int]) -> bool:
    """Whether the attributes of *item* are plain values or containers of
    them (`_flat`)."""
    attributes = own_dict(item)
    if attributes is None:
        return False
    for value in attributes.values():
        kind = type(value)
        if kind not in _PLAIN and not (kind in _FLAT and _flat(value, kind, _FLAT_DEPTH, work)):
            return False
    return True


def _with_objects_alive(classes: list[type]) -> list[type]:
    """Of *classes*, those an object of (or of a subclass of) may be alive:
    a special method runs on an object, so a class with none alive cannot
    have one inside a value too big to look through (a timer's
    ``__exit__`` in a helper module never used). Every object of a class
    made by a ``class`` statement is tracked by the garbage collector;
    read once, at C speed (about 0.06 s for two million objects). Asked
    only when a value was too big and such a class exists."""
    import gc

    from ..held_functions import objects_frozen_since_import

    if not classes or objects_frozen_since_import():
        return classes  # objects ``gc.freeze`` moved out of the collector's lists
    objects = gc.get_objects()
    kinds = dict(zip(map(id, map(type, objects)), map(type, objects)))
    del objects
    alive: set[int] = set()
    for kind in kinds.values():
        alive.update(map(id, _CLASS_MRO(kind)))
    return [c for c in classes if id(c) in alive or not _CLASS_FLAGS(c) & _HAVE_GC]


_HAVE_GC = 1 << 14


def _array_kind(kind: type) -> str | None:
    """``"ndarray"``, ``"series"`` (a Series or an Index), ``"frame"`` or
    ``"extension"`` for *kind*, a subclass of the user's included (``class
    S(pd.Series)``, a masked array); None for anything else. Asked only of
    numpy and pandas as they are loaded."""
    np = sys.modules.get("numpy")
    if np is not None and issubclass(kind, np.ndarray):
        return "ndarray"
    pd = sys.modules.get("pandas")
    if pd is None or not has_attribute(pd, "Series"):
        return None
    if issubclass(kind, pd.DataFrame):
        return "frame"
    if issubclass(kind, (pd.Series, pd.Index)):
        return "series"
    if issubclass(kind, pd.api.extensions.ExtensionArray):
        return "extension"
    return None


def _item_count(value: Any, kind: type, hc: Any) -> int:
    """How many items *value* hands `writers_without_a_trace`, counted
    before any is read: a container's, or a numpy array's of objects. One
    of numbers holds no object of the user's and is not read; a Series,
    Index or frame hands over its array of objects as a value of its own,
    counted then. Nothing of the user's runs (a subclass's ``__len__``)."""
    if kind in hc.OPENED:
        held = value.data if issubclass(kind, hc.HELD_IN_DATA) else value
        return len(held)
    if _array_kind(kind) == "ndarray":
        np = sys.modules["numpy"]
        if np.ndarray.dtype.__get__(value) == np.dtype(object):
            return int(np.ndarray.size.__get__(value))
    return 0


def _array_items(value: Any, array: str, hc: Any, work: list[int]) -> list[Any]:
    """The items of a numpy array, a pandas Series, Index or extension
    array holding objects (``arr + 1`` runs each item's ``__add__``), or an
    exact DataFrame's columns holding them; nothing for one of plain values
    (numbers, dates, strings of either kind). Read through numpy's and
    pandas's own descriptors: a property of a subclass of the user's does
    not run."""
    np = sys.modules["numpy"]
    if array == "ndarray":
        if np.ndarray.dtype.__get__(value) != np.dtype(object):
            return []
        # A base-class view: no method of a subclass's runs.
        items = np.asarray(value).ravel()
        if _PLAIN.issuperset(map(type, items)):
            return []  # strings of a pandas 2 column
        return _records(_not_plain(items), hc, work)
    pd = sys.modules["pandas"]
    if array == "frame":
        if type(value) is not pd.DataFrame:
            return []  # a subclass's columns come through its own constructors
        # The column arrays themselves: a column taken as a Series
        # (``value[c]``) is a view copy-on-write counts, so the frame's
        # kept digest no longer vouched for it, and every pass of a grid
        # loop over the frame read its text columns again.
        arrays = vars(pd.DataFrame).get("_iter_column_arrays")
        if arrays is None:
            return [value[c] for c, d in zip(value.columns, value.dtypes) if d == np.dtype(object)]
        return [a for a in arrays(value) if type(a) is np.ndarray and a.dtype == np.dtype(object)]
    if array == "series":
        base = pd.Series if issubclass(type(value), pd.Series) else pd.Index
        held = _object_array(vars(base)["array"].__get__(value, base), pd, np)
    else:
        held = _object_array(value, pd, np)
    return [] if held is None else [held]


def _object_array(held: Any, pd: Any, np: Any) -> Any:
    """The numpy array of objects the pandas array *held* keeps its items
    in (``NumpyExtensionArray``), or a ``Categorical``'s categories in; None
    for one of plain values (numbers, dates, strings of either kind)."""
    if issubclass(type(held), pd.Categorical):
        categories = vars(pd.Categorical)["categories"].__get__(held, pd.Categorical)
        held = vars(pd.Index)["array"].__get__(categories, pd.Index)
    if type(held).__name__ != "NumpyExtensionArray" or not type(held).__module__.startswith("pandas"):
        return None
    found = held._ndarray
    return found if type(found) is np.ndarray and found.dtype == np.dtype(object) else None


#: Stands for what a container too big to open holds.
_OVER = object()


_HEAPTYPE = 1 << 9

#: Read through ``type``'s own descriptors: a metaclass's
#: ``__getattribute__`` is not asked.
_CLASS_DICT = type.__dict__["__dict__"].__get__
_CLASS_MRO = type.__dict__["__mro__"].__get__
_CLASS_FLAGS = type.__dict__["__flags__"].__get__
_CLASS_QUALNAME = type.__dict__["__qualname__"].__get__


def _users_special(cls: type, hc: Any) -> list[tuple[str, Any]]:
    """The special methods and property functions of *cls*'s MRO written in
    code of the user's, whatever module the class names, each with the name
    the class holds it under: a function set on a class later makes it one
    (``Vec.__add__ = lambda self, o: 10``)."""
    if issubclass(cls, (type, types.ModuleType, types.FunctionType, types.MethodType)):
        return []
    out: list[Any] = []
    for klass in _CLASS_MRO(cls):
        if not _CLASS_FLAGS(klass) & _HEAPTYPE:
            continue
        for name, member in list(_CLASS_DICT(klass).items()):
            kind = type(member)
            if issubclass(kind, property):
                fns = [member.fget, member.fset, member.fdel]
            elif name.startswith("__") and name.endswith("__") and name not in hc.NOT_RUN_BY_A_STATEMENT:
                fns = [member.__func__ if issubclass(kind, (staticmethod, classmethod)) else member]
            else:
                continue
            out.extend(
                (name, fn)
                for fn in fns
                if type(fn) is types.FunctionType and is_user_code_file(fn.__code__.co_filename)
            )
    return out


def _holds_users_code(cls: type) -> bool:
    """A class made in code of the user's: one holding a function written
    there, whatever module it names."""
    if not _CLASS_FLAGS(cls) & _HEAPTYPE:
        return False
    for member in list(_CLASS_DICT(cls).values()):
        if issubclass(type(member), (staticmethod, classmethod)):
            member = member.__func__
        if type(member) is types.FunctionType and is_user_code_file(member.__code__.co_filename):
            return True
    return False


def _named_where_held(fn: Any, cls: type) -> bool:
    """Whether *fn* is held by a class of *cls*'s MRO under its own
    qualified name: written in the class's body, not set on it later."""
    qualname = attribute_or(fn, "__qualname__", "")
    for klass in _CLASS_MRO(cls):
        for attr, member in list(_CLASS_DICT(klass).items()):
            if issubclass(type(member), (staticmethod, classmethod)):
                member = member.__func__
            if member is fn:
                return qualname == f"{_CLASS_QUALNAME(klass)}.{attr}"
    return True


def _filled(cell: Any) -> bool:
    try:
        cell.cell_contents
    except ValueError:
        return False
    return True


def _live_writing_classes(namespace: Mapping[str, Any], hc: Any) -> list[type]:
    from cash.notebook.held_functions import users_module_classes

    try:
        bound = [v for v in list(namespace.values()) if isinstance(v, type) and hc.is_users_class(v)]
    except Exception:  # noqa: BLE001 - changed while read
        bound = []
    return [c for c in bound + users_module_classes() + hc.live_users_classes() if hc.reaching_special_functions(c)]


#: The methods ``@dataclass`` writes from text (``dataclasses._create_fn``).
_DATACLASS_MADE = frozenset(
    {
        "__init__",
        "__repr__",
        "__eq__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__setattr__",
        "__delattr__",
        "__getstate__",
        "__setstate__",
    }
)


def library_made(code: types.CodeType, globals_: Mapping[str, Any]) -> bool:
    """Is *code* a method a library wrote for a class of the user's, from
    text of its own: what ``@dataclass`` (``__create_fn__``), attrs
    (``<attrs generated ...>``) or ``namedtuple`` (its ``__new__``) make.
    They set the object's own fields and change nothing outside it; what
    they call of the user's (a ``default_factory``, ``__post_init__``) ran
    as code of its own and is read in turn. Their text has no file, so read
    as any other function's they would count as changing what their
    unreadable text changes: every statement making such an object of a
    class in a module of the user's was refused."""
    filename = code.co_filename
    if filename.startswith("<attrs generated"):
        return True
    if filename != "<string>":
        return False
    if attribute_or(code, "co_qualname", "").startswith("__create_fn__.<locals>."):
        return True
    if sys.version_info < (3, 11) and code.co_firstlineno == 2 and code.co_name in _DATACLASS_MADE:
        # No ``co_qualname`` before 3.11: ``__create_fn__`` puts the method
        # it makes on its second line, under one of these names.
        return True
    name = globals_.get("__name__") if isinstance(globals_, Mapping) else None
    return code.co_name == "<lambda>" and isinstance(name, str) and name.startswith("namedtuple_")


#: How many functions `unreadable_changers` follows from one statement.
_CALLEES_FOLLOWED = 200


def unreadable_changers(names: Iterable[str], namespace: Mapping[str, Any]) -> list[str]:
    """The functions of the user's, among *names* and the functions they call
    by name, whose text cannot be read (``exec`` of a ``def``, a lambda
    compiled from a string) and whose bytecode does not show that they change
    nothing (`own_writes` gives ``"?"``). The statement's own analysis reads
    callees by their text and took such a one for a builtin, so a hit was
    served while traces were off and what it changes was left unchanged."""
    found: list[str] = []
    seen: set[int] = set()
    pending = [(name, namespace) for name in sorted(names)]
    while pending and len(seen) < _CALLEES_FOLLOWED:
        name, scope = pending.pop()
        fn = _method_named(name, scope) if "." in name else scope.get(name) if isinstance(scope, Mapping) else None
        code = attribute_or(fn, "__code__", None) if type(fn) is types.FunctionType else None
        if code is None or id(code) in seen:
            continue
        seen.add(id(code))
        if not is_user_code_file(code.co_filename) and code.co_filename != "<string>":
            continue
        if _function_node(code) is None:
            if _UNKNOWN in own_writes(code):
                found.append(name)
            continue
        pending.extend((callee, fn.__globals__) for callee in code.co_names)
    return found


def _method_named(path: str, scope: Mapping[str, Any]) -> Any:
    """The function ``obj.method`` names (``k.go`` for a ``k`` of a class an
    ``exec`` defined), read as stored: from the object's class or the class
    itself, a static or class method unwrapped. None for anything else."""
    base, _, attr = path.partition(".")
    value = scope.get(base, _MISSING) if isinstance(scope, Mapping) else _MISSING
    if value is _MISSING or instance_of(value, types.ModuleType):
        return None
    try:
        member = inspect.getattr_static(value, attr)
    except AttributeError:
        return None
    if instance_of(member, (staticmethod, classmethod)):
        member = member.__func__
    if type(member) is not types.FunctionType or library_made(member.__code__, member.__globals__):
        return None
    return member


def called_methods(tree: ast.AST | None) -> frozenset[str]:
    """``name.attr`` for each call ``name.attr(...)`` in *tree*: the methods
    a statement calls on what it names, for `unreadable_changers`."""
    if tree is None:
        return frozenset()
    return frozenset(
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
    )


@functools.lru_cache(maxsize=4096)
def own_writes(code: types.CodeType) -> frozenset[str]:
    """The free names the function of *code* changes by its own text: a
    global or closure variable it changes in place (``calls.append(1)``,
    ``d[k] = v``) or rebinds, as the statement's own analysis reads a
    callee. ``"?"`` when its text cannot be read."""
    node = _function_node(code)
    if node is None:
        # Its text cannot be read (a lambda compiled from a string): its
        # bytecode says what it changes where that is plain.
        writes = _bytecode_writes(code)
        return frozenset((_UNKNOWN,)) if writes is None else writes
    try:
        changed = set(free_vars_mutated_in_function(node))
    except RecursionError:
        return frozenset((_UNKNOWN,))
    changed |= _rebound(node)
    changed |= _handed_writes(node) - scope_locals(node)
    return frozenset(changed)


#: Builtins that neither change what they are given nor reach anything else:
#: the only calls code read off its bytecode may make. Any other name is a
#: global or a closure variable, which may hold a stream that ``list`` or a
#: ``for`` draws from.
_PURE_BUILTINS = frozenset(
    {
        "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable", "chr", "complex", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "id", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "oct", "ord", "pow", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    }
)  # fmt: skip
_STORES = frozenset({"STORE_GLOBAL", "DELETE_GLOBAL", "STORE_DEREF", "DELETE_DEREF"})
#: An object changed in place: on which object, the bytecode alone does not
#: say plainly.
_IN_PLACE = frozenset({"STORE_ATTR", "DELETE_ATTR", "STORE_SUBSCR", "DELETE_SUBSCR", "STORE_SLICE", "DELETE_SLICE"})
_NAME_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF", "LOAD_CLASSDEREF", "LOAD_FROM_DICT_OR_DEREF"})
#: What calls or draws: a call, or an iteration (``for``, ``list(it)``'s
#: inlined comprehension, ``yield from``).
_CALLS_OR_DRAWS = ("CALL", "GET_ITER", "FOR_ITER", "GET_AITER", "SEND", "YIELD_FROM")


def _bytecode_writes(code: types.CodeType) -> frozenset[str] | None:
    """The free names *code* (and the code made inside it) rebinds, read off
    its bytecode; None when it may change anything else. What the bytecode
    shows is taken as plain only where it calls nothing but the builtins of
    `_PURE_BUILTINS` on values that are not globals or closure variables:
    a method, a function held in a name, an attribute set or an item set
    may change an object, which the text tells apart and the bytecode does
    not."""
    out: set[str] = set()
    names: set[str] = set()
    calls = attributes = False
    pending = [code]
    seen: set[types.CodeType] = set()
    while pending:
        c = pending.pop()
        if c in seen:
            continue
        seen.add(c)
        instructions = list(dis.get_instructions(c))
        for i, ins in enumerate(instructions):
            op = ins.opname
            if op == "PUSH_NULL" and not _made_or_builtin(_pushed_beside_null(instructions, i)):
                # A value called (``f(1)`` on an argument, ``getattr(...)(x)``).
                return None
            if op in _STORES:
                if op.endswith("_DEREF") and ins.argval not in c.co_freevars:
                    continue
                out.add(ins.argval)
            elif op in _IN_PLACE or op.startswith("IMPORT_"):
                return None
            elif op in _NAME_LOADS:
                names.add(ins.argval)
            elif op in ("LOAD_ATTR", "LOAD_METHOD", "LOAD_SUPER_ATTR"):
                attributes = True
            elif op.startswith(_CALLS_OR_DRAWS):
                calls = True
        pending.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    if calls and (attributes or not names <= _PURE_BUILTINS or sys.version_info < (3, 12)):
        return None
    return frozenset(out)


def _pushed_beside_null(instructions: list[dis.Instruction], i: int) -> dis.Instruction | None:
    """The instruction that pushes what the ``PUSH_NULL`` at *i* is pushed
    beside to be called: after it on 3.12, before it from 3.13 on."""
    j = i + 1 if sys.version_info < (3, 13) else i - 1
    return instructions[j] if 0 <= j < len(instructions) else None


def _made_or_builtin(ins: dis.Instruction | None) -> bool:
    """A function made right there (its code is read too) or a builtin of
    `_PURE_BUILTINS` loaded by name."""
    if ins is None:
        return False
    if ins.opname in ("MAKE_FUNCTION", "SET_FUNCTION_ATTRIBUTE"):
        return True
    return ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and ins.argval in _PURE_BUILTINS


#: Functions that change their first argument: ``next(GEN)`` moves it on.
_FIRST_ARG_CHANGERS = frozenset({"next", "heappush", "heappop", "heapify", "heapreplace", "heappushpop", "shuffle"})
_FIRST_ARG_MODULES = frozenset({"heapq", "random"})


def _handed_writes(node: ast.AST) -> set[str]:
    """Free names *node* changes in ways the shared reader leaves to a
    statement's own analysis: a global handed to a function that changes
    it (``next(GEN)``, ``heapq.heappush(H, 1)``, ``random.shuffle(L)``), or
    whose mutating method is handed on (``map(calls.append, rows)``)."""
    out: set[str] = set()
    called: set[int] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        called.add(id(sub.func))
        func = sub.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in _FIRST_ARG_MODULES
            else None
        )
        if name in _FIRST_ARG_CHANGERS and sub.args and isinstance(sub.args[0], ast.Name):
            out.add(sub.args[0].id)
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and id(sub) not in called
            and isinstance(sub.ctx, ast.Load)
            and sub.attr in MUTATING_METHODS
            and isinstance(sub.value, ast.Name)
        ):
            out.add(sub.value.id)
    return out


def _method_writes(code: types.CodeType, globals_: Mapping[str, Any]) -> list[str]:
    """Globals *code* changes through a method of theirs that changes its
    object (``STATE.bump()``, ``bump`` setting ``self.n``): a method of a
    class written in code of the user's."""
    node = _function_node(code)
    if node is None:
        return []
    own = scope_locals(node)
    out: list[str] = []
    for sub in ast.walk(node):
        if not (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id not in own
        ):
            continue
        owner = globals_.get(sub.func.value.id, _MISSING)
        if owner is _MISSING or isinstance(owner, (type, types.ModuleType)):
            continue
        method = attribute_or(type(owner), sub.func.attr, None)
        if isinstance(method, types.FunctionType) and is_user_code_file(method.__code__.co_filename):
            if _writes_its_object(method.__code__):
                out.append(sub.func.value.id)
    return out


@functools.lru_cache(maxsize=4096)
def _writes_its_object(code: types.CodeType) -> bool:
    """Whether the method of *code* changes its first parameter's object:
    stores to ``self.x`` or ``self[k]``, ``self.x += 1``, a mutating method
    on ``self`` or ``self.x``."""
    node = _function_node(code)
    if node is None or not node.args.args:
        return node is None
    me = node.args.args[0].arg

    def rooted(target: ast.AST) -> bool:
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return isinstance(target, ast.Name) and target.id == me

    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = sub.targets if isinstance(sub, (ast.Assign, ast.Delete)) else [sub.target]
            if any(isinstance(t, (ast.Attribute, ast.Subscript)) and rooted(t) for t in targets):
                return True
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in MUTATING_METHODS
            and rooted(sub.func.value)
        ):
            return True
    return False


@functools.lru_cache(maxsize=4096)
def _function_node(code: types.CodeType) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | None:
    """The definition of *code*'s function, from its file, or None."""
    comprehension = _COMPREHENSIONS.get(code.co_name)
    if comprehension is not None:
        return _comprehension_node(code, comprehension)
    try:
        source = getsource(code)
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    import textwrap

    text = textwrap.dedent(source)
    lines = source.splitlines() or [""]
    shift = len(lines[0]) - len((text.splitlines() or [""])[0])
    first_line_moved = 0
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # A lambda in the middle of a line: the line alone may not parse.
        try:
            tree = ast.parse("(" + text.strip().rstrip(",").rstrip("\\") + ")")
        except SyntaxError:
            # A lambda in a literal over several lines (``'key': lambda r:
            # r[1],``, as black writes a dict): read from the whole text.
            return _lambda_in_its_text(code) if code.co_name == "<lambda>" else None
        first_line_moved = 1 - (len(text) - len(text.lstrip()))
    if code.co_name != "<lambda>":
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == code.co_name:
                return node
        return None
    lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
    if len(lambdas) < 2:
        return lambdas[0] if lambdas else None
    if sys.version_info < (3, 11):
        return None  # no positions to tell them apart by
    # Several on its lines (``{'g': lambda o: 1, 'f': lambda o: calls.append(1)}``):
    # the one whose body starts where *code*'s first instruction of its own
    # does; none found, its text is not known.
    starts = [
        (line, col)
        for line, _, col, end in code.co_positions()
        if line is not None and col is not None and not (col == 0 and end == 0)
    ]
    if not starts:
        # A body that is a constant (``lambda o: 1``) keeps no position of
        # its own on 3.12: it writes nothing, whichever of them it is.
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        for node in lambdas:
            args = node.args
            own = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            if isinstance(node.body, ast.Constant) and tuple(own) == names:
                return node
        return None
    want = min(starts)
    for node in lambdas:
        line = code.co_firstlineno + node.body.lineno - 1
        col = node.body.col_offset + shift - (first_line_moved if node.body.lineno == 1 else 0)
        if (line, col) == want:
            return node
    return None


def _lambda_in_its_text(code: types.CodeType) -> ast.Lambda | None:
    """The lambda of *code* read from the whole of its file or cell: the
    one starting on its first line, or, of several there, the one whose
    body starts where *code*'s does. None when it cannot be told."""
    import linecache

    try:
        tree = ast.parse("".join(linecache.getlines(code.co_filename)))
    except (SyntaxError, ValueError):
        return None
    lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno]
    if len(lambdas) < 2:
        return lambdas[0] if lambdas else None
    if sys.version_info < (3, 11):
        return None
    starts = [
        (line, col)
        for line, _, col, end in code.co_positions()
        if line is not None and col is not None and not (col == 0 and end == 0)
    ]
    want = min(starts) if starts else None
    return next((node for node in lambdas if (node.body.lineno, node.body.col_offset) == want), None)


_COMPREHENSIONS = {
    "<genexpr>": ast.GeneratorExp,
    "<listcomp>": ast.ListComp,
    "<setcomp>": ast.SetComp,
    "<dictcomp>": ast.DictComp,
}

_NO_ARGUMENTS = ast.arguments(
    posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
)


def _comprehension_node(code: types.CodeType, kind: type) -> ast.Lambda | None:
    """A generator expression (or, before 3.12, a comprehension) that ran
    apart from the function it was made in, as a function whose body it is:
    the one of its kind on its first line of its file (all of them, when
    the line has several). None if the file no longer has one there."""
    import linecache

    text = "".join(linecache.getlines(code.co_filename))
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    found = [node for node in ast.walk(tree) if isinstance(node, kind) and node.lineno == code.co_firstlineno]
    if not found:
        return None
    # Several on the line: what any of them changes.
    body = found[0] if len(found) == 1 else ast.Tuple(elts=found, ctx=ast.Load())
    return ast.Lambda(args=_NO_ARGUMENTS, body=body)


def _own_nodes(node: ast.AST) -> Iterable[ast.AST]:
    """The nodes of *node*'s body, not of functions or classes defined in it."""
    body = node.body if isinstance(node.body, list) else [node.body]  # type: ignore[attr-defined]
    pending = list(body)
    while pending:
        sub = pending.pop()
        yield sub
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(sub))


def _rebound(node: ast.AST) -> set[str]:
    """Names declared ``global`` or ``nonlocal`` and bound in *node*."""
    declared: set[str] = set()
    bound: set[str] = set()
    for sub in _own_nodes(node):
        if isinstance(sub, (ast.Global, ast.Nonlocal)):
            declared.update(sub.names)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
    return declared & bound


def _log(found: list[tuple[str, tuple[str, ...]]], statement: str) -> None:
    path = os.environ.get(_LOG_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:  # @cash:assume-safe
            for fn, targets in found:
                row = {"fn": fn, "targets": list(targets), "stmt": statement[:200], "pid": os.getpid()}
                fh.write(json.dumps(row) + "\n")
    except OSError:
        pass
