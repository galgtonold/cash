"""Does a variable's object have a holder other than the variable itself?

A cache hit restores a statement's outputs by binding each name to a
deserialised COPY. That is the same as running the statement only when nothing
else holds the object the name ends up bound to, or any object inside it:

* ``models = {'m': m}`` -- the dict holds ``m``'s model. Restored, it holds a
  copy, and ``m.fit()`` in a later cell is invisible through ``models['m']``;
* ``fitted = m.fit(5)`` returns ``m`` itself, which a copy is not;
* ``d['a'] = f(d)`` after ``d = dfs[0]``, or in ``for d in dfs:`` -- the
  statement changes the frame ``dfs`` holds. Restored, it rebinds ``d`` to a
  changed copy and ``dfs[0]`` keeps its old contents.

Statement shapes cannot tell these apart from the same statements on a fresh
object, so the question is asked of the live objects, by reference count: an
object whose count is higher than the references the outputs themselves
account for has another holder (another variable, a container, a library's
registry). That answer can only err towards "shared", which costs a re-run,
never a wrong value.

The walk goes through what restoring copies wholesale and what identity
matters in: the builtin containers, and the attributes of objects of the
notebook's own classes and of the classes of the user's own modules, of ``SimpleNamespace`` / dataclass instances and of
estimators (a ``Pipeline`` holds the step objects it was built from), and
the object a bound method is bound to and the cells a closure keeps
(``hooks = {'log': tracker.log}``). Everything else is one leaf, whose own
count is checked. Values with no identity worth keeping (numbers, strings,
classes, functions bound to nothing, enum members, numpy scalars and dtypes,
...) are skipped.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import datetime
import decimal
import enum
import fractions
import gc
import inspect
import pathlib
import re
import sys
import types
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from time import perf_counter as _perf_counter
from typing import Any

from cash import _plain_data
from cash.tracking.function_tracker import is_local_module

__all__ = [
    "GaveUp",
    "Leaves",
    "anchors_held",
    "attrs_copied_apart",
    "holds_part_of",
    "output_history",
    "referring_names",
    "refers_back",
    "share_group",
    "shared_names",
    "view_base_names",
]

#: Values whose identity no program relies on: equal ones are interchangeable.
VALUE_TYPES: tuple[type, ...] = (
    int,
    float,
    complex,
    str,
    bytes,
    bool,
    type(None),
    range,
    slice,
    type(Ellipsis),
    type(NotImplemented),
    datetime.date,
    datetime.time,
    datetime.timedelta,
    datetime.tzinfo,
    decimal.Decimal,
    fractions.Fraction,
    pathlib.PurePath,
    uuid.UUID,
    enum.Enum,
    # Pickled by reference, so a restore hands back the very same object.
    type,
    types.ModuleType,
    types.CodeType,
)

#: Functions and methods: values too, unless bound to an object or closing
#: over one (`_bound_objects`).
_CALLABLE_TYPES: tuple[type, ...] = (types.FunctionType, types.BuiltinFunctionType, types.MethodType)

#: The commonest value types, by exact type: one set lookup tells a string or a
#: datetime from a container where `isinstance` against `VALUE_TYPES` walks the
#: whole tuple (and its abstract classes), the cost of a walk over millions of
#: leaves. A subclass or any other type still goes through `isinstance`.
EXACT_VALUE_TYPES: frozenset[type] = frozenset(
    {
        int,
        float,
        complex,
        str,
        bytes,
        bool,
        type(None),
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
    }
)

#: The leaves of a tree `_walk` asks about as one object (`_plain_data.held_only_by_parents`).
TREE_LEAVES: tuple[type, ...] = tuple(EXACT_VALUE_TYPES)

#: The builtin containers by exact type: never a value type, so no `isinstance`.
EXACT_CONTAINER_TYPES: frozenset[type] = frozenset({list, dict, set, tuple, frozenset})

#: The builtin containers a restore copies along with their contents.
_CONTAINERS = (list, dict, set, tuple, frozenset)

#: Containers whose own identity does not matter -- only their contents'.
_IMMUTABLE_CONTAINERS = (tuple, frozenset)


def library_value_types() -> tuple[type, ...]:
    """The value types of numpy and pandas, when they are already imported.

    A ufunc (``np.log`` in a ``FunctionTransformer``) is pickled by its name,
    so a restore hands back the very same object; the numpy modules that
    import it by name are not holders a restore must keep."""
    found: list[type] = []
    np = sys.modules.get("numpy")
    if np is not None:
        found += [np.generic, np.dtype, np.ufunc]
    pd = sys.modules.get("pandas")
    if pd is not None:
        found += [pd.Timestamp, pd.Timedelta, pd.Period, pd.Interval, type(pd.NA), type(pd.NaT)]
    return tuple(found)


def _bound_objects(value: Any) -> list[Any] | None:
    """What a function or method carries along when it is restored, or None
    when it carries nothing.

    A bound method (``tracker.log``, ``results.append``) pickles its
    ``__self__`` by value, and a deep copy of a builtin one keeps the very
    object of the run that stored it; a closure (``make(store)``) keeps the
    cells of that run. Either way the restored hook would write into an
    object that is not the one the notebook's names are bound to.

    A closure a library's module binds by its own name (sklearn's
    ``load_iris``, wrapped by its parameter check) carries nothing: pickle
    and a deep copy hand back that very function (`_found_by_name`). Counted
    as carrying its cells, each import of one searched every object the
    garbage collector tracks for their holders, and found the module: 2 s
    on E11's ``from sklearn.datasets import load_iris``.
    """
    if isinstance(value, types.FunctionType):
        return list(value.__closure__) if value.__closure__ and not _found_by_name(value) else None
    owner = getattr(value, "__self__", None)
    if owner is None or isinstance(owner, (types.ModuleType, type)):
        return None
    return [owner]


def _found_by_name(fn: types.FunctionType) -> bool:
    """*fn* is found again by its module and qualified name, as pickle looks
    it up, in a module other than the notebook's own (``__main__``, whose
    functions a store keeps by value)."""
    module = getattr(fn, "__module__", None) or ""
    if module == "__main__":
        return False
    target: Any = sys.modules.get(module)
    for part in (getattr(fn, "__qualname__", "") or "").split("."):
        if target is None or part.startswith("<"):
            return False
        try:
            target = inspect.getattr_static(target, part, None)
        except Exception:  # noqa: BLE001 - an object that cannot be asked holds no such name
            return False
    return target is fn


def is_value(value: Any, value_types: tuple[type, ...]) -> bool:
    """Whether *value* has no identity a restore must keep (`VALUE_TYPES`,
    a function or method that carries nothing).

    An object whose attributes all raise, ``__class__`` too (numpy's
    ``BagObj``, ``np.load(...).f``), makes an ``isinstance`` against an ABC
    raise: it is read by its type alone."""
    try:
        if isinstance(value, value_types):
            return True
        return isinstance(value, _CALLABLE_TYPES) and _bound_objects(value) is None
    except Exception:  # noqa: BLE001 - see above
        return issubclass(type(value), value_types)


_ESTIMATOR_CLASSES: dict[type, bool] = {}


def _is_estimator_class(cls: type) -> bool:
    """Whether *cls* has scikit-learn's estimator interface. A pipeline or an
    ensemble holds the estimators it was built from (``Pipeline([('s',
    scaler), ...])``), and a restore copies them along with it."""
    known = _ESTIMATOR_CLASSES.get(cls)
    if known is None:
        try:
            known = callable(getattr(cls, "get_params", None)) and callable(getattr(cls, "fit", None))
        except Exception:  # noqa: BLE001 - a class that cannot be asked is a leaf
            known = False
        _ESTIMATOR_CLASSES[cls] = known
    return known


_LOCAL_MODULE_CLASSES: dict[type, bool] = {}


def _is_local_module_class(cls: type) -> bool:
    """Whether *cls* was defined in one of the user's own modules (a
    ``mylib.py`` next to the notebook, their package): its objects are the
    user's as much as a class defined in a cell, and ``training_rows(ds)``
    returning ``ds.rows`` hands back a list the argument holds."""
    known = _LOCAL_MODULE_CLASSES.get(cls)
    if known is None:
        try:
            module = sys.modules.get(cls.__module__)
            known = isinstance(module, types.ModuleType) and is_local_module(module)
        except Exception:  # noqa: BLE001 - a class that cannot be asked is a leaf
            known = False
        _LOCAL_MODULE_CLASSES[cls] = known
    return known


def attributes_of(value: Any) -> dict[str, Any] | None:
    """The attributes a restore copies with *value* and the caller can reach,
    for an object of the notebook's own classes or of a class from the
    user's own modules, a ``SimpleNamespace``, a dataclass or an estimator;
    ``None`` for anything else (a leaf)."""
    cls = type(value)
    if not (
        cls.__module__ == "__main__"
        or cls is types.SimpleNamespace
        or dataclasses.is_dataclass(cls)
        or _is_estimator_class(cls)
        or _is_local_module_class(cls)
    ):
        return None
    try:
        own = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        own = None
    if own is not None and type(own) is not dict:
        return None
    slots = _slot_members(cls)
    if not slots:
        return own
    # ``__slots__`` values live outside ``__dict__``: an object of such a
    # class holds them as surely as its attributes (``Slotted(rows).rows``).
    found = {}
    for name, member in slots:
        try:
            found[name] = member.__get__(value, cls)
        except AttributeError:
            continue  # a slot never set holds nothing
    if own:
        found.update(own)
    return found


_SLOT_MEMBERS: dict[type, tuple[tuple[str, Any], ...]] = {}


def _slot_members(cls: type) -> tuple[tuple[str, Any], ...]:
    """``(name, member descriptor)`` for each ``__slots__`` entry along
    *cls*'s MRO, read from each class's own ``__dict__`` (a private slot
    under its mangled name)."""
    known = _SLOT_MEMBERS.get(cls)
    if known is None:
        found: list[tuple[str, Any]] = []
        for klass in cls.__mro__:
            for name, member in vars(klass).items():
                if isinstance(member, types.MemberDescriptorType) and "__slots__" in vars(klass):
                    found.append((name, member))
        known = _SLOT_MEMBERS[cls] = tuple(found)
    return known


def _values_only(value: Any, exact: frozenset[type]) -> bool:
    """Is *value* an exact list, tuple, set, frozenset or dict whose items
    (a dict's keys and values) are all of the *exact* value types? What
    `_walk`'s first step finds for it, without copying the items out."""
    kind = type(value)
    if kind is dict:
        return exact.issuperset(map(type, value)) and exact.issuperset(map(type, value.values()))
    if kind is list or kind is tuple or kind is set or kind is frozenset:
        return exact.issuperset(map(type, value))
    return False


def children_of(value: Any) -> Iterable[Any] | None:
    """What *value* holds that a restore copies with it, or None for a leaf."""
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, _CONTAINERS):
        return list(value)
    if isinstance(value, _CALLABLE_TYPES):
        return _bound_objects(value)
    if isinstance(value, types.CellType):
        try:
            return [value.cell_contents]
        except ValueError:
            return None
    attrs = attributes_of(value)
    if attrs is not None:
        return list(attrs.values())
    return _library_children(value)


#: The standard library's wrapping containers, each holding its contents in
#: its own ``__dict__`` (``UserDict.data``, ``ChainMap.maps``).
_WRAPPING_CONTAINERS = (collections.UserDict, collections.UserList, collections.ChainMap)


def _library_children(value: Any) -> list[Any] | None:
    """What a container of the standard library or of numpy or pandas holds
    that a restore copies with it, or None for a leaf: a ``deque``'s items,
    a ``UserDict``'s ``data``, a ``ChainMap``'s ``maps``, an object array's
    items (one that owns them: a view's items are its base's), and a
    pandas object's ``attrs``."""
    kind = _LIBRARY_KINDS.get(type(value))
    if kind is None:
        kind = _library_kind(type(value))
    if kind is _LEAF:
        return None
    if kind is _DEQUE:
        return list(value)
    if kind is _WRAPPING:
        try:
            own = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            return None
        return list(own.values()) if type(own) is dict else None
    if kind is _ARRAY:
        if value.dtype.kind == "O" and value.base is None:
            return list(value.flat)
        return None
    attrs = _pandas_attrs(value)
    return [attrs] if attrs else None


_LEAF, _DEQUE, _WRAPPING, _ARRAY, _PANDAS = "leaf", "deque", "wrapping", "array", "pandas"

#: `_library_kind` by type: a walk asks for every leaf it meets.
_LIBRARY_KINDS: dict[type, str] = {}


def _library_kind(cls: type) -> str:
    """Which of `_library_children`'s containers *cls* is, if any."""
    if issubclass(cls, collections.deque):
        kind = _DEQUE
    elif issubclass(cls, _WRAPPING_CONTAINERS):
        kind = _WRAPPING
    else:
        np = sys.modules.get("numpy")
        pd = sys.modules.get("pandas")
        if np is not None and issubclass(cls, np.ndarray):
            kind = _ARRAY
        elif pd is not None and issubclass(cls, (pd.DataFrame, pd.Series)):
            kind = _PANDAS
        else:
            kind = _LEAF
    if len(_LIBRARY_KINDS) > 4096:
        _LIBRARY_KINDS.clear()
    _LIBRARY_KINDS[cls] = kind
    return kind


def _pandas_attrs(value: Any) -> dict | None:
    """A pandas object's own ``attrs`` dict, without making one (pandas
    copies it to every object derived from this one, so it is this one's)."""
    try:
        attrs = object.__getattribute__(value, "_attrs")
    except AttributeError:
        return None
    return attrs if type(attrs) is dict else None


def attrs_copied_apart(roots: Iterable[Any]) -> bool:
    """Whether a deep copy of *roots* would part an object a pandas object's
    ``attrs`` holds from another place in *roots* that holds it too.

    pandas deep-copies ``attrs`` itself, without the copy's memo, so a
    restored frame's ``attrs`` holds copies of their own: ``df.attrs['rows']
    = rows`` restored with ``rows`` leaves two lists where the run left one.
    JSON-like roots hold no pandas object: they are walked only when a
    pandas object with ``attrs`` was found, for what they share with it.
    """
    pd = sys.modules.get("pandas")
    if pd is None:
        return False
    value_types = VALUE_TYPES + library_value_types()
    frames: list[dict] = []
    seen: set[int] = set()
    roots = list(roots)
    trees = [_plain_data.is_tree(root, TREE_LEAVES) for root in roots]
    stack = [root for root, tree in zip(roots, trees) if not tree]
    while stack:
        obj = stack.pop()
        if is_value(obj, value_types) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            attrs = _pandas_attrs(obj)
            if attrs:
                frames.append(attrs)
            continue
        children = children_of(obj)
        if not children or EXACT_VALUE_TYPES.issuperset(map(type, children)):
            continue
        stack.extend(children)
    if not frames:
        return False
    inside: dict[int, int] = {}
    for index, attrs in enumerate(frames):
        for key in _identities(attrs, value_types):
            if inside.setdefault(key, index) != index:
                return True  # in the attrs of two pandas objects
    if any(key in seen for key in inside):
        return True
    return any(key in inside for root, tree in zip(roots, trees) if tree for key in _identities(root, value_types))


def _identities(value: Any, value_types: tuple[type, ...]) -> Iterable[int]:
    """The ids of the objects *value* is and holds, as `_walk` goes through
    them, that have an identity of their own: not values, nor tuples and
    frozensets (their contents are yielded)."""
    seen: set[int] = set()
    stack = [value]
    while stack:
        obj = stack.pop()
        if is_value(obj, value_types) or id(obj) in seen:
            continue
        seen.add(id(obj))
        if not isinstance(obj, _IMMUTABLE_CONTAINERS):
            yield id(obj)
            yield from view_bases(obj)
        stack.extend(children_of(obj) or ())


def view_bases(value: Any) -> list[int]:
    """The ids of the arrays a numpy view *value* sits on (its ``.base``
    chain), empty for anything else. ``arr[2:5]`` is a new object, but
    writing through it writes into ``arr``."""
    np = sys.modules.get("numpy")
    if np is None or not isinstance(value, np.ndarray):
        return []
    found: list[int] = []
    base = value.base
    while base is not None and id(base) not in found:
        found.append(id(base))
        base = getattr(base, "base", None)
    return found


def view_base_names(
    values: Iterable[Any], user_ns: Mapping[str, Any], skip_name: Callable[[str], bool] = lambda name: False
) -> set[str]:
    """The variables of *user_ns* bound to an array that a numpy view among
    *values*, or held in them, sits on (`view_bases`).

    The share check counts references, and an array is a leaf to it: ``head
    = big[:5]`` holds ``big`` through ``head.base``, which no count of
    ``head`` shows, while a later ``big *= 3`` changes what ``head`` reads.
    Walked as `_identities` walks, under the walk budget in progress
    (`check_walk_budget`); JSON-like data holds no array and is not walked.
    """
    np = sys.modules.get("numpy")
    if np is None:
        return set()
    value_types = VALUE_TYPES + library_value_types()
    bases: set[int] = set()
    views: list[Any] = []
    seen: set[int] = set()
    stack = [value for value in values if not _plain_data.is_tree(value)]
    pops = 0
    while stack:
        pops += 1
        if not pops & 1023:
            check_walk_budget()
        obj = stack.pop()
        if is_value(obj, value_types) or id(obj) in seen:
            continue
        seen.add(id(obj))
        found = view_bases(obj)
        if found:
            bases.update(found)
            views.append(obj)
        stack.extend(children_of(obj) or ())
    del stack, seen
    if not bases:
        return set()
    return {name for name, value in list(user_ns.items()) if on_bases(value, bases, views, np) and not skip_name(name)}


def on_bases(value: Any, bases: set[int], views: list[Any], np: Any) -> bool:
    """Whether the array *value* is one of *bases* (ids), or a view of one
    that overlaps one of *views*: ``big = np.arange(6.0).reshape(3, 2)`` is
    itself a view, so ``big[0].base`` is the buffer under both, not ``big``."""
    if id(value) in bases:
        return True
    if not isinstance(value, np.ndarray) or value.base is None:
        return False
    if not any(found in bases for found in view_bases(value)):
        return False
    return any(np.may_share_memory(value, view) for view in views)


#: How many objects `refers_back` looks through below a value.
_BACK_REF_OBJECTS = 5_000

#: What `refers_back` does not walk into: shared by everything, never a
#: part of one value.
_NOT_PARTS = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.CodeType, types.FrameType)


def refers_back(value: Any) -> bool:
    """Whether an object *value* holds -- any reference the garbage
    collector sees, a few levels down -- holds *value* itself, as each
    sheet of an openpyxl workbook holds its workbook.

    For a refusal's wording only (`_shared_reason`): `excess_refs` counts
    such references as holders, since a leaf is not walked. Looks at
    `_BACK_REF_OBJECTS` objects at most, breadth first; False past that."""
    if value is None or is_value(value, VALUE_TYPES):
        return False
    seen = {id(value)}
    level = [value]
    looked = 0
    while level and looked < _BACK_REF_OBJECTS:
        below = []
        for obj in level:
            for ref in gc.get_referents(obj):
                if ref is value:
                    if obj is not value:
                        return True
                    continue
                if id(ref) in seen or not gc.is_tracked(ref) or isinstance(ref, _NOT_PARTS):
                    continue
                seen.add(id(ref))
                below.append(ref)
                looked += 1
                if looked >= _BACK_REF_OBJECTS:
                    break
            if looked >= _BACK_REF_OBJECTS:
                break
        level = below
    del level
    return False


def holds_a_held_object(value: Any) -> bool:
    """Whether an object *value* holds (not *value* itself) has a holder
    outside *value*, by reference count, or *value* holds a numpy view (whose
    base anything may hold). When neither, nothing in *value* below its root
    can be part of another object, which spares `holds_part_of` a walk over
    large sources."""
    value_types = VALUE_TYPES + library_value_types()
    nodes, inbound, checked, _owner, _breaks = _walk({"": value}, [], value_types)
    root = id(value)
    if excess_refs(nodes, inbound, [key for key in checked if key != root]):
        return True
    return any(view_bases(obj) for obj in nodes.values())


def holds_part_of(value: Any, sources: Iterable[Any]) -> bool:
    """Whether *value* is or holds an object one of *sources* is or holds.

    A function's result restored as a copy is only the same as calling the
    function when it holds nothing the caller has too: ``bundle(model, df)``
    returning ``{'model': model, ...}``, or ``pick(cfg, 'a')`` returning
    ``cfg['a']``, hand back an object a later ``model.fit()`` or
    ``c['n'] = 5`` must reach through both names. Walked as `_walk` walks:
    the builtin containers and the attributes of the notebook's own objects.
    A numpy view holds the arrays it sits on (`view_bases`), on both sides:
    ``window(a, 2)`` returning ``a[2:5]`` holds part of ``a``.
    """
    value_types = VALUE_TYPES + library_value_types()
    own = set(_identities(value, value_types))
    if not own:
        return False
    return any(key in own for source in sources for key in _identities(source, value_types))


def excess_refs(nodes: dict[int, Any], inbound: dict[int, int], ids: Iterable[int]) -> list[int]:
    """The ids among *ids* whose reference count is above *inbound* plus this
    function's own references (`_OVERHEAD`)."""
    shared = []
    for key in ids:
        if sys.getrefcount(nodes[key]) - inbound[key] > _OVERHEAD:
            shared.append(key)
    return shared


def _calibrate() -> int:
    """How many references `excess_refs` itself adds to an object it checks.

    Measured, not assumed: it depends on the interpreter (3.14 stops counting
    the argument of ``sys.getrefcount``). The probe is held by one list, so
    its count minus that one reference is the overhead.
    """
    holder = [types.SimpleNamespace()]
    key = id(holder[0])
    nodes = {key: holder[0]}
    return sys.getrefcount(nodes[key]) - 1


_OVERHEAD = _calibrate()


def output_history(
    user_ns: Mapping[str, Any], shell: Any = None
) -> tuple[list[Any], list[tuple[Mapping[str, Any], str]]]:
    """The references IPython's output history makes, as ``(containers,
    named)`` for `shared_names`: the ``Out`` dict, and the ``_``, ``__``,
    ``___`` and ``_<n>`` names bound to a value in it.

    Displaying a value stores it there, so without this a statement that
    keeps an object only the history holds (``df = _`` after a cell that
    showed ``load()``) would count the history as a holder and run every
    time. Nothing in a program depends on the history holding the very
    object, so its references are not a reason to re-run. ``Out`` counts only
    when it is IPython's (the same dict as ``_oh``), and a ``_`` the user
    bound to something never displayed is a holder like any other.

    The display hook binds those names twice more in the *shell*: in its own
    ``_``, ``__`` and ``___`` attributes, and in ``user_ns_hidden``, where
    ``push(..., interactive=False)`` records them; and the previous cell's
    ``last_execution_result.result`` holds its value. A cell ending in ``df``
    left ``df`` held there, and the statement updating ``df`` in place in the
    next cell ran every time.
    """
    out = user_ns.get("Out")
    if type(out) is not dict or user_ns.get("_oh") is not out:
        return [], []
    shown = {id(v) for v in out.values() if v is not None}
    named = [n for n in ("_", "__", "___") if user_ns.get(n) is not None and id(user_ns.get(n)) in shown]
    named += [f"_{k}" for k, v in out.items() if v is not None and user_ns.get(f"_{k}") is v]
    held = [(user_ns, n) for n in named]
    hidden = getattr(shell, "user_ns_hidden", None)
    if isinstance(hidden, dict):
        held += [(hidden, n) for n in named if hidden.get(n) is user_ns.get(n)]
    hook_attrs = getattr(getattr(shell, "displayhook", None), "__dict__", None)
    if hook_attrs is not None:
        held += [(hook_attrs, n) for n in ("_", "__", "___") if id(hook_attrs.get(n)) in shown]
    last = getattr(getattr(shell, "last_execution_result", None), "__dict__", None)
    if last is not None and id(last.get("result")) in shown:
        held.append((last, "result"))
    return [out], held


#: How many times `share_group` widens the group before it gives up, and how
#: many containers deep it looks for the variable that holds one. Giving up
#: refuses the statement as before, never drops a holder.
_MAX_ROUNDS = 8
_MAX_DEPTH = 16


def shared_names(
    roots: Mapping[str, Any],
    bindings: Iterable[Mapping[str, Any]],
    cash_held: Iterable[Any] = (),
    named_held: Iterable[tuple[Mapping[str, Any], str]] = (),
    keep_identity: Iterable[str] | None = None,
) -> set[str]:
    """The names in *roots* whose value, or an object inside it, has a holder
    outside *roots*.

    *roots* maps each name to the object it is bound to. *bindings* are the
    mappings known to hold the roots under those names (the user namespace,
    a dict of captured values): each one is an expected reference. So are
    the references made by the containers in *cash_held*, which cash holds
    itself, and by the containers inside them. Anything above that
    -- another variable, a container in an earlier cell's value, a library's
    registry -- makes the name shared. *named_held* are ``(mapping, name)``
    references not to count either (IPython's ``_``, see `output_history`):
    passed by name, because a container listing them would add as many
    references as it discounts.

    The roots are counted as one graph: one root holding another's object
    (``models`` holding ``m``'s model) is an expected reference. With
    *keep_identity*, only what those roots reach is checked; the other roots'
    references still count as expected.
    """
    shared, _found = _check_group(roots, list(bindings), list(cash_held), list(named_held), keep_identity)
    return shared


class WalkBudgetExceeded(Exception):
    """The share check ran past its time budget (:func:`walk_budget`)."""


_DEADLINE: list[float | None] = [None]


@contextlib.contextmanager
def walk_budget(seconds: float) -> Iterator[None]:
    """Let the walks below take at most *seconds* before they raise
    :class:`WalkBudgetExceeded`.

    The share check finds every variable holding an object of a statement's
    outputs, which can mean walking everything a big list holds: 6 s after a
    loop that ran 0.1 s, because its loop variable is an element of the list.
    A statement whose result is refused when the check gives up simply runs
    each time, so the budget costs the cache a store, never a wrong value.
    """
    before = _DEADLINE[0]
    _DEADLINE[0] = _perf_counter() + seconds
    try:
        yield
    finally:
        _DEADLINE[0] = before


def check_walk_budget() -> None:
    """Raise :class:`WalkBudgetExceeded` when the budget of the walk in
    progress is spent; nothing outside :func:`walk_budget`."""
    deadline = _DEADLINE[0]
    if deadline is not None and _perf_counter() > deadline:
        raise WalkBudgetExceeded


#: How many levels of referrers `referring_names` climbs, and how many
#: objects it takes in, before it gives up.
_REFERRER_DEPTH = 8
_REFERRER_OBJECTS = 200_000


def referring_names(
    roots: Iterable[Any],
    user_ns: Mapping[str, Any],
    seconds: float,
    skip_name: Callable[[str], bool] = lambda name: False,
    ignore: Iterable[Any] = (),
) -> tuple[set[str], bool]:
    """``(names, complete)``: the variables of *user_ns* bound to one of
    *roots*, holding one at any depth, or held inside one, found within
    *seconds*.

    The fallback for when `share_group` cannot say (its walk ran out of
    budget, or a holder is not a variable): instead of walking what every
    variable holds, it climbs from *roots* up their referrers
    (`gc.get_referrers`, one heap scan per level) until it reaches the
    namespace, and walks down what *roots* hold. Frames, modules and the
    objects in *ignore* (cash's own references) are not climbed through.
    *complete* is False when the climb or the walk was cut short -- by
    *seconds*, depth or size -- so names may be missing.
    """
    import gc

    deadline = _perf_counter() + seconds
    value_types = VALUE_TYPES + library_value_types()
    roots = [root for root in roots if not is_value(root, value_types)]
    by_id: dict[int, list[str]] = {}
    for name, value in list(user_ns.items()):
        if not skip_name(name) and not is_value(value, value_types):
            by_id.setdefault(id(value), []).append(name)
    names: set[str] = set()
    # Down: what the roots hold.
    try:
        with walk_budget(seconds):
            for root in roots:
                for key in _identities(root, value_types):
                    names.update(by_id.get(key, ()))
                    check_walk_budget()
    except WalkBudgetExceeded:
        return names, False
    # Up: what holds the roots.
    stop = {id(user_ns), id(roots)}
    stop.update(id(obj) for obj in ignore)
    stop.update(id(vars(module)) for module in list(sys.modules.values()) if module is not None)
    seen = {id(root) for root in roots}
    frontier = list(roots)
    stop.add(id(frontier))
    taken = 0
    for _level in range(_REFERRER_DEPTH):
        if not frontier:
            return names, True
        if _perf_counter() > deadline or taken > _REFERRER_OBJECTS:
            return names, False
        # One tuple, passed on as the call's own arguments, so it is known.
        args = tuple(frontier)
        stop.add(id(args))
        referrers = gc.get_referrers(*args)
        stop.add(id(referrers))
        following: list[Any] = []
        stop.add(id(following))
        for obj in referrers:
            key = id(obj)
            if key in seen or key in stop or isinstance(obj, (types.FrameType, types.ModuleType)):
                continue
            seen.add(key)
            names.update(by_id.get(key, ()))
            following.append(obj)
        taken += len(following)
        del referrers, args
        frontier[:] = following
    return names, not frontier


@dataclasses.dataclass
@dataclasses.dataclass
class Leaves:
    """Objects a walk leaves out, by id: an anchor of `share_group`, which a
    restore does not copy but binds to the live object again, is neither a
    node nor walked below. *met* gets the ids the walk reached."""

    ids: set[int] = dataclasses.field(default_factory=set)
    met: set[int] = dataclasses.field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.ids)


@dataclasses.dataclass
class GaveUp:
    """What `share_group` found: the links of a check that succeeded, or what
    it still found when it gave up.

    *names* are the notebook variables found holding an object of the
    outputs, or held in one, beside the holder that is not a variable (a
    function's defaults, a class, a library object such as an
    ``argparse.Namespace`` or a numpy view): a change made in place through
    one of them is a change to what the outputs reach, so they are linked
    though the statement is not stored. *links* pairs them with the names
    whose objects they hold or are held in, ``(owners, holders)``: an
    output, or a variable an earlier round found, so a variable is linked
    only to the outputs it shares an object with. *complete* is False when
    the search was cut short (too many objects, too deep, out of time), so
    names may be missing.
    """

    names: set[str] = dataclasses.field(default_factory=set)
    links: list[tuple[frozenset[str], frozenset[str]]] = dataclasses.field(default_factory=list)
    complete: bool = True
    #: The modules whose namespace was found holding an object of the
    #: outputs, by their ``sys.modules`` name.
    modules: set[str] = dataclasses.field(default_factory=set)

    def add(self, owners: Iterable[str], holders: Iterable[str]) -> None:
        """Note that *holders* share an object with *owners*."""
        held = frozenset(holders)
        if held:
            self.names |= held
            self.links.append((frozenset(owners), held))


def share_group(
    names: Iterable[str],
    values: Mapping[str, Any],
    user_ns: Mapping[str, Any],
    cash_held: Iterable[Any] = (),
    named_held: Iterable[tuple[Mapping[str, Any], str]] = (),
    foreign: Callable[[str], bool] = lambda name: False,
    gave_up: GaveUp | None = None,
    anchorable: Callable[[str], bool] | None = None,
    anchors: dict[str, Any] | None = None,
    anchor_links: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """``(holders, shared)`` for the outputs *names*, bound in *values*.

    When an output's object, or one inside it, has a holder outside the
    outputs (`shared_names`), the holder is looked for: a variable of
    *user_ns*, directly or through containers and attributes a restore
    copies along (`gc.get_referrers`, asked only then). Those variables join
    the group and the count is taken again, until nothing is left over:
    *holders* maps each variable that joined to its value, and *shared* is
    empty. Stored and restored with the outputs as one graph, they keep
    holding the very objects the outputs hold.

    When a holder is anything else -- a library's registry, a closure, a
    module of a library, a suspended frame, an object cash does not copy
    along -- or a variable *foreign* names (IPython's own), *shared* names
    the outputs still shared and *holders* is empty. *gave_up*, when given,
    then gets the variables found holding the outputs' objects all the same
    (`GaveUp`): the search goes past such a holder instead of stopping at
    it, without climbing through it. When the check succeeds, *gave_up*
    gets each round's links all the same: which names were shared, and the
    variables found holding them.

    With *anchorable*, a variable found holding an object of the outputs is
    asked first whether it can be an anchor instead (the statement left its
    value as it was): then its object is not copied with the outputs but
    bound again to the variable's live object on a restore, so whatever else
    holds that object holds the same one the restored outputs do, as in
    plain Python, and is not looked for. *anchors* gets each anchor the
    outputs reach, by name, and no holder that only held an anchor's object
    joins. An anchor that holds an object of the group is not one: the live
    object would hold the old one, not its restored copy. *anchor_links* gets
    the variables found holding an object of the outputs in the rounds that
    took anchors: they share objects with the outputs all the same, so a
    change through one of them moves the outputs on.
    """
    group = {name: values[name] for name in names if values.get(name) is not None}
    bindings = [values, user_ns]
    cash_held = list(cash_held)
    named_held = list(named_held)
    joined: set[str] = set()
    # What each round finds below a root of JSON-like data, for the next
    # round: no code runs between them, so what holds a part of it is the
    # same (`_walk`'s *memo*).
    memo: dict[int, list | None] = {}
    links: list[tuple[set[str], set[str]]] = []
    partial = GaveUp() if gave_up is not None else None
    leaves = Leaves()
    #: The anchor tried for each object, by id; None when it can be none.
    tried: dict[int, str | None] = {}
    rounds = anchor_rounds = 0
    while True:
        leaves.met.clear()
        shared, found = _check_group(group, bindings, cash_held, named_held, None, user_ns, memo, partial, leaves)
        if not shared:
            if anchors is not None:
                anchors.update(_met_anchors(leaves, tried, user_ns))
            if gave_up is not None:
                for owners, holders in links:
                    gave_up.add(owners, holders)
            return {name: group[name] for name in joined}, set()
        if found and anchorable is not None and anchor_rounds < _MAX_ANCHOR_ROUNDS:
            fresh = _anchor_candidates(found, user_ns, tried, anchorable)
            leaves.ids.update(fresh)
            _take_back_holding(group, leaves, fresh, tried, user_ns, memo)
            if fresh & leaves.ids:
                # Counted again with the new anchors left out: a holder that
                # held only their objects is no holder any more.
                anchor_rounds += 1
                if anchor_links is not None:
                    anchor_links |= found
                continue
        if found:
            links.append((shared, found))
        if not found or any(foreign(name) for name in found) or _copies_keep_old_objects(group, leaves):
            break
        joined |= found
        group.update((name, user_ns[name]) for name in found)
        rounds += 1
        if rounds >= _MAX_ROUNDS:
            if partial is not None:
                partial.complete = False
            break
    if gave_up is not None and partial is not None:
        for owners, holders in links:
            gave_up.add(owners, (name for name in holders if not foreign(name)))
        gave_up.add(shared, (name for name in partial.names if not foreign(name)))
        gave_up.modules |= partial.modules
        gave_up.complete = gave_up.complete and partial.complete
    outputs = set(group) - joined
    return {}, (shared & outputs) or outputs


#: More rounds `share_group` may take when it finds anchors: each one takes
#: at least one new anchor or ends.
_MAX_ANCHOR_ROUNDS = 8


def _anchor_candidates(
    found: set[str], user_ns: Mapping[str, Any], tried: dict[int, str | None], anchorable: Callable[[str], bool]
) -> set[int]:
    """The ids of the objects of *found* not tried as anchors yet that
    *anchorable* takes, each noted in *tried* under its name (the first in
    order of two bound to one object)."""
    fresh: set[int] = set()
    for name in sorted(found):
        value = user_ns.get(name)
        key = id(value)
        if value is None or key in tried:
            continue
        ok = anchorable(name)
        tried[key] = name if ok else None
        if ok:
            fresh.add(key)
    return fresh


def _take_back_holding(
    group: Mapping[str, Any],
    leaves: Leaves,
    fresh: set[int],
    tried: dict[int, str | None],
    user_ns: Mapping[str, Any],
    memo: dict[int, list | None] | None = None,
) -> None:
    """Take back from *leaves* the anchors among *fresh* whose object holds
    an object of *group* (`share_group`): bound again on a restore, the live
    anchor would hold the old object, not the restored one."""
    if not fresh:
        return
    value_types = VALUE_TYPES + library_value_types()
    nodes = _group_nodes(group, value_types, leaves, memo)
    whole: dict[int, Any] | None = None
    for key in fresh:
        anchor = user_ns.get(tried[key] or "")
        if id(anchor) != key:
            continue
        inner: dict[int, Any] = {}
        _descend([anchor], "", inner, {}, {}, [], value_types, Leaves(leaves.ids - {key}))
        seen = nodes
        if key in nodes:
            # The anchor is itself a root or a break: what only it holds is
            # left out of *nodes* but in *inner*. Asked of the whole walk.
            if whole is None:
                whole = _group_nodes(group, value_types, leaves, None)
            seen = whole
        if not seen.keys().isdisjoint(inner):
            leaves.ids.discard(key)
            tried[key] = None


def _group_nodes(
    group: Mapping[str, Any], value_types: tuple[type, ...], leaves: Leaves, memo: dict[int, list | None] | None
) -> dict[int, Any]:
    """The objects of *group* an anchor's walk is looked for in
    (`_take_back_holding`): the roots and what they reach, stopping at
    *leaves*. With *memo* (`_walk`'s, read and not added to; None for the
    whole walk), a root of JSON-like data gives only the containers something besides their parent
    holds and what they reach (`_plain_data.held_beyond_parents`).

    The ones left out are held by their parent alone, once: a walk from an
    anchor that is none of them reaches one only through its parent, so it
    reaches the root or a break above it as well, which are kept. The two
    meet the same anchors but for one that is itself a kept container,
    which the caller asks of the whole walk. A break among *leaves* lies on
    no path the whole walk takes: that root is walked whole.
    """
    nodes: dict[int, Any] = {}
    stop = Leaves(set(leaves.ids))
    whole: list[Any] = []
    for root in group.values():
        nodes[id(root)] = root
        if memo is None or type(root) not in (list, dict):
            whole.append(root)
            continue
        key = id(root)
        breaks = memo[key] if key in memo else _plain_data.held_beyond_parents(root, TREE_LEAVES, _MAX_BREAKS)
        kept = None if breaks is None else [obj for obj in breaks if _is_node(obj, value_types)]
        if kept is None or any(id(obj) in stop.ids for obj in kept):
            whole.append(root)
            continue
        fresh = [obj for obj in kept if id(obj) not in nodes]
        nodes.update((id(obj), obj) for obj in fresh)
        _descend(fresh, "", nodes, {}, {}, [], value_types, stop)
    _descend(whole, "", nodes, {}, {}, [], value_types, stop)
    return nodes


def anchors_held(value: Any, anchors: Mapping[str, Any]) -> set[str]:
    """The names of *anchors* (``{name: object}``) whose object *value* is or
    holds, by identity: a walk that stops at each of them."""
    by_id = {id(obj): name for name, obj in anchors.items()}
    if id(value) in by_id:
        return {by_id[id(value)]}
    leaves = Leaves(set(by_id))
    _descend([value], "", {}, {}, {}, [], VALUE_TYPES + library_value_types(), leaves)
    return {by_id[key] for key in leaves.met}


def _met_anchors(leaves: Leaves, tried: dict[int, str | None], user_ns: Mapping[str, Any]) -> dict[str, Any]:
    """The anchors the last walk reached, by name, bound to their objects."""
    met: dict[str, Any] = {}
    for key in leaves.met:
        name = tried.get(key)
        if name is not None and id(user_ns.get(name)) == key:
            met[name] = user_ns[name]
    return met


def _copies_keep_old_objects(group: Mapping[str, Any], leaves: Leaves | None = None) -> bool:
    """Whether *group* holds a closure or a builtin bound method that carries
    an object (``lambda v: store.append(v)``, ``results.append``).

    A deep copy keeps such a function as it is, still bound to the object of
    the run that stored it, while the holder variable is restored as a copy:
    stored together, the two would no longer be the same object."""
    value_types = VALUE_TYPES + library_value_types()
    seen: set[int] = set()
    # JSON-like data holds no function at all: asked a level at a time, where
    # the walk below takes a Python step per object -- most of re-sorting a
    # list of 450,000 parsed pairs a loop variable still held part of.
    stack = [root for root in group.values() if not _plain_data.is_tree(root, TREE_LEAVES)]
    pops = 0
    while stack:
        pops += 1
        if not pops & 1023:
            check_walk_budget()
        obj = stack.pop()
        if is_value(obj, value_types) or id(obj) in seen or (leaves and id(obj) in leaves.ids):
            continue
        seen.add(id(obj))
        if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
            return True
        stack.extend(children_of(obj) or ())
    return False


def _check_group(
    group: Mapping[str, Any],
    bindings: list[Mapping[str, Any]],
    cash_held: list[Any],
    named_held: list[tuple[Mapping[str, Any], str]],
    keep_identity: Iterable[str] | None,
    user_ns: Mapping[str, Any] | None = None,
    memo: dict[int, list | None] | None = None,
    gave_up: GaveUp | None = None,
    leaves: Leaves | None = None,
) -> tuple[set[str], set[str] | None]:
    """``(shared, found)`` for *group*: the names whose objects are held
    beyond what the group, *bindings*, *cash_held* and *named_held* account
    for, and -- with *user_ns* -- the variables of *user_ns* outside the
    group that hold them (`_find_holders`), None when one holder is not such
    a variable (*gave_up* then gets the variables found all the same).
    *memo* is `_walk`'s, kept by the caller across rounds. *leaves* are
    `_walk`'s."""
    value_types = VALUE_TYPES + library_value_types()
    if memo is None:
        memo = {}
    for fast in (True, False):
        nodes, inbound, checked, owner, breaks = _walk(
            group, bindings, value_types, keep_identity, fast=fast, memo=memo, leaves=leaves
        )
        _count_memo(memo, nodes, inbound)
        if not excess_refs(nodes, inbound, checked):
            # Every reference is the group's own already. What cash holds
            # itself only adds to the counts, so it cannot leave one over:
            # not walked. It was walked for every statement, ``x = 0`` too
            # -- the output history holding a displayed list of 400,000
            # tuples, 1.2 s a statement.
            return _inside_library_objects(group, nodes, owner, value_types, user_ns, bindings, cash_held)
        internal = count_held(cash_held, nodes, inbound, value_types, named_held, user_ns)
        for mapping, key in named_held:
            if id(mapping.get(key)) in nodes:
                inbound[id(mapping.get(key))] += 1
        if user_ns is not None:
            _count_index_data(nodes, inbound, user_ns)
        counted = dict(inbound)
        excess = excess_refs(nodes, inbound, checked)
        if excess and _count_leaf_attributes(nodes, inbound):
            excess = excess_refs(nodes, inbound, checked)
        if not excess:
            return _inside_library_objects(group, nodes, owner, value_types, user_ns, bindings, cash_held)
        if _held_within(nodes, counted, excess, value_types):
            return _inside_library_objects(group, nodes, owner, value_types, user_ns, bindings, cash_held)
        shared = {owner[k] for k in excess}
        if user_ns is None:
            return shared, None
        named = {id(mapping) for mapping, _key in named_held if mapping is not user_ns}
        internal |= {id(group), id(bindings), id(cash_held), id(named_held), id(nodes), *nodes, *named}
        internal |= {id(memo), *(id(listed) for listed in memo.values() if listed is not None)}
        internal |= {id(m) for m in bindings if m is not user_ns}
        discounted = {key for mapping, key in named_held if mapping is user_ns}
        if breaks.isdisjoint(excess):
            # Held only in a pandas object's own slots: the search up would
            # stop at that object, so its answer is known without it.
            in_frames = _held_in_pandas_slots(nodes, inbound, excess, user_ns, set(group) | discounted)
            if in_frames is not None:
                if gave_up is not None:
                    gave_up.names |= in_frames
                return shared, None
        partial = GaveUp() if gave_up is not None else None
        found = _find_holders([nodes[k] for k in excess], internal, user_ns, set(group) | discounted, partial)
        if found is not None or breaks.isdisjoint(excess):
            if found is None and gave_up is not None and partial is not None:
                gave_up.names |= partial.names
                gave_up.modules |= partial.modules
                gave_up.complete = gave_up.complete and partial.complete
            return shared, found
        # A container a root read a level at a time holds besides its parent
        # (`_take_breaks`): the search up from it climbs through the
        # containers above it, which the walk left out of *internal*, and a
        # longer climb can give up where the full walk's would not. Asked
        # again of the full walk, with none of these references left.
        del nodes, inbound, checked, owner, breaks, internal, excess, found, partial
    return shared, None


#: Most objects `_held_within` reads inside the walk's leaves, and most
#: nodes a walk may have for it to try.
_WITHIN_MAX_OBJECTS = 20_000
_WITHIN_MAX_NODES = 5_000

#: What `_held_within` never reads into: a holder of one of these is shared.
_NOT_WITHIN = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.CodeType,
    types.FrameType,
)


def _held_within(nodes: dict[int, Any], inbound: dict[int, int], excess: list[int], value_types: tuple) -> bool:
    """Whether each reference the walk left over on *excess* comes from
    inside the walk's leaves, from objects held only from inside in turn.

    A library's object the walk does not look into (`children_of` None)
    can hold its parent: each trace and the layout of a plotly figure point
    back at the figure (``_parent``), so the figure read as held from
    outside, and a search of every object the garbage collector tracks ran
    for each statement making one -- 147 times, 6.6 s, on E23. Here the
    objects inside the leaves are read down (`gc.get_referents`), and each
    reference to an excess node is looked for among them; the objects
    holding it must have no reference from outside either, up to the
    nodes. Anything not found -- an object outside, a C object that does
    not list what it holds, too many objects -- leaves the answer False,
    and the search up runs as before. *inbound* is the walk's own count,
    before `_count_leaf_attributes`.
    """
    if len(nodes) > _WITHIN_MAX_NODES:
        return False
    within: dict[int, Any] = {}
    refs: dict[int, int] = {}
    holders: dict[int, list[int]] = {}
    stack = [obj for obj in nodes.values() if not is_value(obj, value_types) and children_of(obj) is None]
    if not stack:
        return False
    while stack:
        parent = id(stack[-1])
        referents = gc.get_referents(stack.pop())
        for child in referents:
            if is_value(child, value_types) or issubclass(type(child), _NOT_WITHIN):
                continue
            key = id(child)
            refs[key] = refs.get(key, 0) + 1
            holders.setdefault(key, []).append(parent)
            if key in nodes or key in within:
                continue
            if len(within) >= _WITHIN_MAX_OBJECTS:
                return False
            within[key] = child
            stack.append(child)
        child = None  # no reference of this frame's left in the counts
        del referents
    todo = list(excess)
    done: set[int] = set()
    while todo:
        key = todo.pop()
        if key in done:
            continue
        done.add(key)
        held = inbound.get(key, 0) + refs.get(key, 0)
        if key in nodes:
            if sys.getrefcount(nodes[key]) - held > _OVERHEAD:
                return False
        elif sys.getrefcount(within[key]) - held > _OVERHEAD:
            return False
        todo.extend(holder for holder in holders.get(key, ()) if holder in within)
    return True


#: Most objects `_inside_library_objects` reads inside the library objects
#: of a walk before it gives up, which refuses the statement.
_INSIDE_MAX_OBJECTS = 50_000
#: Above how many objects one level of `_variables_holding`'s climb reads
#: the heap once itself: `gc.get_referrers` takes time in proportion to the
#: heap times the objects it is given (33 s for 18,000 on a test worker).
_REFERRERS_BY_CALL = 128

#: IPython's own names, and cash's: never a holder of what a library object
#: holds (`_variables_holding`).
_SHELL_NAME = re.compile(r"_+|_i+\d*|_\d+|_oh|_ih|_dh|In|Out|exit|quit|get_ipython|__cash_\w*")


def _inside_library_objects(
    group: Mapping[str, Any],
    nodes: dict[int, Any],
    owner: dict[int, str],
    value_types: tuple[type, ...],
    user_ns: Mapping[str, Any] | None,
    bindings: list[Mapping[str, Any]],
    cash_held: list[Any],
) -> tuple[set[str], set[str] | None]:
    """`_check_group`'s ``(shared, found)`` for what the library objects
    among *nodes* hold, which the walk does not read (`children_of` None).

    ``torch.optim.SGD(model.parameters())`` holds ``model``'s parameters
    inside its ``param_groups``; read as one closed object, nothing else held
    it, and a hit bound a copy whose parameters were not ``model``'s, so
    training stepped the copy and ``model`` never trained. Here the objects
    inside are read down (`gc.get_referents`), and each one something
    outside them holds too is looked for up its referrers: the variables
    found join the group and are stored with it, as for a holder the walk
    finds. A library's own globals (a plotly validator) are not variables
    and do not count, as for the library object itself. ``(owners, None)``,
    which refuses the statement, when the search could not finish."""
    if user_ns is None:
        return set(), set()
    inside, owners = _held_from_outside(nodes, owner, value_types)
    if inside is None:
        return owners, None
    if not inside:
        return set(), set()
    ignore = {id(nodes), id(group), id(owner), id(bindings), id(cash_held), *nodes}
    ignore.update(id(binding) for binding in bindings if binding is not user_ns)
    ignore.update(id(obj) for obj in cash_held)
    names = _variables_holding(inside, user_ns, set(group), ignore, value_types)
    if names is None:
        return owners, None
    return (owners, names) if names else (set(), set())


def _reads_inside(node: Any, value_types: tuple[type, ...]) -> bool:
    """Whether *node* is a library's own object the walk does not read: not
    a value, nor pandas or numpy data (read by their own rules)."""
    try:
        if is_value(node, value_types) or issubclass(type(node), _NOT_WITHIN) or children_of(node) is not None:
            return False
    except Exception:  # noqa: BLE001 - an object whose attributes raise (see `is_value`)
        return False
    kind = _LIBRARY_KINDS.get(type(node))
    return (kind or _library_kind(type(node))) is _LEAF


def _read_down(obj: Any, value_types: tuple[type, ...]) -> bool | None:
    """None when *obj*, met inside a library object, is not one to look for
    (a value, a type, a function, cash's own record); else whether to read
    what it holds in turn (a library's own object, not pandas or numpy data)."""
    if _is_cash_own(obj) or is_value(obj, value_types) or issubclass(type(obj), _NOT_WITHIN):
        return None
    if _holds_only_values(obj, value_types) or _is_extension_dtype(obj):
        return None
    return (_LIBRARY_KINDS.get(type(obj)) or _library_kind(type(obj))) is _LEAF


def _is_extension_dtype(obj: Any) -> bool:
    """Whether *obj* is a pandas extension dtype (``str``, ``category``,
    ``Int64``): a description of a column's data that pandas requires to be
    hashable and treats as immutable, as numpy's own dtypes are values here.

    ``df["card_scheme"].unique()`` holds the column's ``StringDtype``, which
    every string column and pandas' dtype cache hold too: looked for up its
    referrers, the climb read the heap past its budget and the statement ran
    uncached every time."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return False
    try:
        return isinstance(obj, pd.api.extensions.ExtensionDtype)
    except AttributeError:
        return False


def _holds_only_values(obj: Any, value_types: tuple[type, ...], depth: int = 3) -> bool:
    """Whether *obj* is a tuple or frozenset of values (or of such tuples):
    nothing in it can change, so no identity of it a restore must keep.

    The empty tuple a fitted ``HistGradientBoostingRegressor`` holds is the
    interpreter's one ``()``, held by thousands of objects: looked for up its
    referrers, the climb read the heap seven times over (3 s), never ended,
    and the fit was refused as shared, every run."""
    if type(obj) not in (tuple, frozenset) or depth <= 0:
        return False
    return all(is_value(item, value_types) or _holds_only_values(item, value_types, depth - 1) for item in obj)


def _is_cash_own(obj: Any) -> bool:
    """Whether *obj* is one of cash's own records (its type is cash's)."""
    module = type(obj).__module__
    return isinstance(module, str) and (module == "cash" or module.startswith("cash."))


def _held_from_outside(
    nodes: dict[int, Any], owner: dict[int, str], value_types: tuple[type, ...]
) -> tuple[dict[int, Any] | None, set[str]]:
    """``(inside, owners)``: the objects inside the library objects among
    *nodes* that something besides them holds, by id, and the names of the
    group they are in; *inside* None when there were too many to read.

    A library object one of whose parts holds it back (each trace of a
    plotly figure, each node of a minidom document) is left out: those keep
    the library-object rule (`_held_within`), and their parts share the
    library's own validators and the like with every other such object."""
    roots = {key: owner[key] for key, node in nodes.items() if _reads_inside(node, value_types)}
    if not roots:
        return {}, set()
    stack = [(nodes[key], key) for key in roots]
    within: dict[int, Any] = {}
    refs: dict[int, int] = {}
    root_of: dict[int, int] = {}
    held_back: set[int] = set()
    pops = 0
    while stack:
        parent, root = stack.pop()
        if root in held_back:
            continue
        pops += 1
        if not pops & 255:
            check_walk_budget()
        for child in gc.get_referents(parent):
            read_down = _read_down(child, value_types)
            if read_down is None:
                continue
            key = id(child)
            if key == root:
                held_back.add(root)
                break
            if key in nodes:
                continue
            refs[key] = refs.get(key, 0) + 1
            if key in within:
                continue
            if len(within) >= _INSIDE_MAX_OBJECTS:
                return None, {roots[root]}
            within[key] = child
            root_of[key] = root
            if read_down:
                # pandas and numpy data are read by their own rules, not down.
                stack.append((child, root))
        child = None  # no reference of this frame's left in the counts
    parent = None
    held = [
        key for key in within if root_of[key] not in held_back and sys.getrefcount(within[key]) - refs[key] > _OVERHEAD
    ]
    return {key: within[key] for key in held}, {roots[root_of[key]] for key in held}


def _variables_holding(
    inside: dict[int, Any],
    user_ns: Mapping[str, Any],
    skip: set[str],
    ignore: set[int],
    value_types: tuple[type, ...],
) -> set[str] | None:
    """The variables of *user_ns* not in *skip* that hold an object of
    *inside*, found up `gc.get_referrers`; None when the climb did not end
    within `_REFERRER_DEPTH` levels, or a level reached more than
    `_INSIDE_MAX_OBJECTS` objects, or when an object is held in a container a
    module or a library's class keeps. It stops at a variable, a module, a
    library's class, a frame, cash's own records (`_is_cash_own`: what they hold is the
    statement's bookkeeping) and the objects of *ignore* (cash's own
    references)."""
    by_id: dict[int, list[str]] = {}
    for name, value in list(user_ns.items()):
        if name not in skip and not _SHELL_NAME.fullmatch(name) and not is_value(value, value_types):
            by_id.setdefault(id(value), []).append(name)
    stop = {id(user_ns), id(inside), id(by_id), *ignore}
    module_dicts = {id(vars(module)) for module in list(sys.modules.values()) if module is not None}
    module_dicts.discard(id(user_ns))
    stop.update(module_dicts)
    # Held in a container a library keeps, a library's own object is still
    # the library's (a module-level loss holding sklearn's ``Interval``); the
    # notebook's data or its own object is not, and no copy of it is that one.
    notebook_own = not all(_is_library_own(obj) for obj in inside.values())
    names: set[str] = set()
    for key in inside:
        names.update(by_id.get(key, ()))
    seen = set(inside)
    frontier = [obj for key, obj in inside.items() if key not in by_id]
    stop.add(id(frontier))
    for _level in range(_REFERRER_DEPTH):
        if not frontier:
            return names
        check_walk_budget()
        if len(frontier) > _INSIDE_MAX_OBJECTS:
            return None
        args = tuple(frontier)
        stop.add(id(args))
        referrers = gc.get_referrers(*args) if len(args) <= _REFERRERS_BY_CALL else _referrers_of(args, stop)
        stop.add(id(referrers))
        following: list[Any] = []
        stop.add(id(following))
        for obj in referrers:
            key = id(obj)
            if key in module_dicts and _level and notebook_own:
                # Held in a container a module keeps (``lib.registry.append(df)``):
                # not the module's own global, and no copy is that object.
                return None
            if key in seen or key in stop or issubclass(type(obj), (types.FrameType, types.ModuleType)):
                continue
            seen.add(key)
            if key in by_id:
                names.update(by_id[key])
                continue
            if isinstance(obj, type) and _is_library_class(obj):
                # A library's class, found again by its name as its module
                # is: what it holds itself (sklearn's ``interval_y_pred``) is
                # the library's own, as its globals are. Reached through a
                # container it holds (``SomeLibClass.registry.append(df)``),
                # the object is the notebook's, held where no copy reaches.
                if _level > 1 and notebook_own:
                    return None
                continue
            if not _is_cash_own(obj):
                following.append(obj)
        del referrers, args
        frontier[:] = following
    return names if not frontier else None


def _is_library_own(obj: Any) -> bool:
    """Whether *obj* is a library's own object: of a library's class
    (`_is_library_class`), read as one (not pandas or numpy data). A list,
    a dict or the notebook's own object is not."""
    cls = type(obj)
    if cls.__module__ == "builtins" or not _is_library_class(cls):
        return False
    return (_LIBRARY_KINDS.get(cls) or _library_kind(cls)) is _LEAF


def _is_library_class(cls: type) -> bool:
    """Whether *cls* is a class of a library: not the notebook's (``__main__``)
    nor one of the user's own modules (`_is_local_module_class`).

    sklearn's link classes hold the ``Interval`` a fitted boosting model's
    loss holds too (``interval_y_pred``): climbing past the class, through
    its subclasses, methods and module, read the heap seven times and never
    ended, so the fit was refused as shared."""
    module = getattr(cls, "__module__", None)
    return isinstance(module, str) and module != "__main__" and not _is_local_module_class(cls)


def _referrers_of(objs: tuple[Any, ...], stop: set[int]) -> list[Any]:
    """`gc.get_referrers` of *objs*, in one read of the heap whatever their
    number. The heap's list goes in *stop*."""
    ids = {id(obj) for obj in objs}
    heap = gc.get_objects()
    stop.add(id(heap))
    found = []
    for i, obj in enumerate(heap):
        if not i & 4095:
            check_walk_budget()
        if any(id(child) in ids for child in gc.get_referents(obj)):
            found.append(obj)
    return found


def _count_leaf_attributes(nodes: dict[int, Any], inbound: dict[int, int]) -> bool:
    """Add to *inbound* the references the attributes of each object of the
    walk it does not look into (a library's own object, `children_of` None)
    make to nodes, when nothing outside the walk holds that object; whether
    any were added.

    Such an object is inside the outputs, and a restore copies it with them,
    so what it holds is held within: a LightGBM classifier's
    ``_best_score`` is also its booster's ``best_score``, and the walk,
    which looks into the classifier but not the booster, read that dict as
    held by "another variable or container" and every fit statement re-ran.
    One level: what the attributes hold in turn stays uncounted. Each
    object is read in `_attributes_in`, so that no local here holds it
    while its own count is taken.
    """
    added = False
    containers = EXACT_CONTAINER_TYPES
    for key in list(nodes):
        held = _attributes_in(nodes[key], nodes, containers)
        if held and not excess_refs(nodes, inbound, [key]):
            for child in held:
                inbound[child] += 1
            added = True
    return added


def _attributes_in(value: Any, nodes: dict[int, Any], containers: frozenset[type]) -> list[int]:
    """The ids of the nodes a leaf *value*'s own attributes are bound to."""
    if type(value) in containers or children_of(value) is not None:
        return []
    try:
        own = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return []
    if type(own) is not dict:
        return []
    return [id(attr) for attr in own.values() if id(attr) in nodes]


def _count_memo(memo: dict[int, list | None], nodes: dict[int, Any], inbound: dict[int, int]) -> None:
    """Add to *inbound* the memo's own reference to each node it lists. In
    a function of its own, so that no loop variable is left holding one."""
    for listed in memo.values():
        for key in map(id, listed or ()):
            if key in nodes:
                inbound[key] += 1


def _count_index_data(nodes: dict[int, Any], inbound: dict[int, int], user_ns: Mapping[str, Any]) -> None:
    """Count in *inbound* what the indexes of the frames in *user_ns* hold
    of *nodes*: an index's data and its engine's values.

    An index is immutable: pandas never writes its data in place, and every
    change to a frame's labels (``df.columns = ...``, ``rename``) binds a
    new index. Holding its data is holding a value, which keys and restores
    the same shared or not (K10). Under pandas 2 a fitted model's
    ``feature_names_in_`` is ``X.columns._data`` itself, and every fit on a
    frame was refused as sharing it with ``X``; pandas 3 hands out a copy.
    A frame of the group itself, or an index the walk read, is counted by
    the walk already."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return
    frame_types = (pd.DataFrame, pd.Series)
    seen: set[int] = set()
    try:
        for value in list(user_ns.values()):
            if not isinstance(value, frame_types) or id(value) in nodes or id(value) in seen:
                continue
            seen.add(id(value))
            for axis in vars(value)["_mgr"].axes:
                if id(axis) in nodes or id(axis) in seen:
                    continue
                seen.add(id(axis))
                slots = vars(axis)
                engine = slots.get("_cache", {}).get("_engine")
                for item in (slots.get("_data"), getattr(engine, "values", None)):
                    if item is not None and id(item) in inbound:
                        inbound[id(item)] += 1
    except Exception:  # noqa: BLE001 - another layout: counted as held elsewhere
        return


def _held_in_pandas_slots(
    nodes: dict[int, Any],
    inbound: dict[int, int],
    excess: list[int],
    user_ns: Mapping[str, Any],
    known: set[str],
) -> set[str] | None:
    """The variables bound to the pandas frames whose own slots hold every
    reference left over on *excess*, or None when any is held elsewhere.

    The slot is pandas' item cache (``y = df['t']`` stays in
    ``df._item_cache``; pandas 2 without copy-on-write, where ``y[0] = 99``
    changes ``df`` too); an index's data is counted before
    (`_count_index_data`). `_holders_from_referrers` climbs to the same
    frame and stops there, a library's object, after one search
    of every object the garbage collector tracks per level: 20-60 ms a
    statement. The references are counted, so a reference from anywhere
    else leaves the search to run.
    """
    pd = sys.modules.get("pandas")
    if pd is None:
        return None
    frame_types = (pd.DataFrame, pd.Series)
    wanted = {key: sys.getrefcount(nodes[key]) - inbound[key] - _OVERHEAD for key in excess}
    names: set[str] = set()
    seen: set[int] = set(nodes)
    try:
        for name, value in list(user_ns.items()):
            if name in known or not isinstance(value, frame_types) or id(value) in seen:
                continue
            seen.add(id(value))
            own = vars(value)
            cache = own.get("_item_cache")
            held = False
            if type(cache) is dict:
                for item in cache.values():
                    if id(item) in wanted:
                        wanted[id(item)] -= 1
                        held = True
            if held:
                names.add(name)
    except Exception:  # noqa: BLE001 - another layout: the search up decides
        return None
    if any(left != 0 for left in wanted.values()):
        return None
    return names


def _find_holders(
    targets: list[Any],
    internal: set[int],
    user_ns: Mapping[str, Any],
    known: set[str],
    gave_up: GaveUp | None = None,
) -> set[str] | None:
    """The variables of *user_ns* not in *known* that hold one of *targets*,
    directly or through containers and attributes a restore copies along;
    None when a holder is anything else.

    Looked for from the variables down first (`_holders_from_names`), which
    costs what the namespace's own small containers cost; the search up
    `gc.get_referrers` walks every object the garbage collector tracks, once
    per level of containers -- 0.3 s a statement next to a list of three
    million records. Down is only a first guess: the count taken again with
    the variables it found joined decides, and when that still finds a holder
    the search up runs.
    """
    found = _holders_from_names(targets, internal, user_ns, known)
    if found:
        return found
    return _holders_from_referrers(targets, internal, user_ns, known, gave_up)


#: Most objects `_holders_from_referrers` looks the holders of at once.
#: `gc.get_referrers` compares every reference of every tracked object with
#: each object asked about, so the call costs the heap times their number: 87,573
#: lists (a record per session, all shared with a second variable) did not
#: finish in ten minutes, next to a statement that ran in 0.2 s. Past this
#: the search gives up, which refuses the statement (nothing is stored; it
#: runs as it always did).
_MAX_REFERRER_TARGETS = 64

#: How far `_holders_from_names` looks: containers deep, objects in all, and
#: the largest container it reads. A holder past these is left to the search
#: up `gc.get_referrers`.
_DOWN_DEPTH = 4
_DOWN_NODES = 5_000
_DOWN_WIDEST = 2_000


def _holders_from_names(
    targets: list[Any], internal: set[int], user_ns: Mapping[str, Any], known: set[str]
) -> set[str]:
    """The variables of *user_ns* not in *known* from which one of *targets*
    is reached through the builtin containers and the attributes a restore
    copies along (`attributes_of`), within `_DOWN_DEPTH` levels; empty when
    none is found within the bounds.

    Goes through what the search up goes through, and nothing in *internal*
    (the group's own objects, cash's containers): a variable it names is
    one that search would name too. One it misses holds a reference the
    count still sees, which sends the caller up `gc.get_referrers`."""
    wanted = {id(obj) for obj in targets}
    exact = EXACT_VALUE_TYPES
    roots: dict[int, list[str]] = {}
    for name, value in list(user_ns.items()):
        if name in known or type(value) in exact:
            continue
        key = id(value)
        if key in wanted or key not in internal:
            roots.setdefault(key, []).append(name)
    if not roots:
        return set()
    found: set[str] = set()
    for key in [k for k in roots if k in wanted]:
        found.update(roots.pop(key))
    # Breadth first from every variable at once, remembering who reached
    # whom, then up from the targets to the variables.
    parents: dict[int, list[int]] = {key: [] for key in roots}
    level = [user_ns[names[0]] for names in roots.values()]
    hits: set[int] = set()
    budget = _DOWN_NODES
    for _depth in range(_DOWN_DEPTH):
        below = []
        for obj in level:
            if not isinstance(obj, _CONTAINERS):
                attrs = attributes_of(obj)
                if attrs is None:
                    continue
                children = list(attrs.values())
            elif len(obj) > _DOWN_WIDEST:
                continue
            else:
                children = children_of(obj)
            if not children or exact.issuperset(map(type, children)):
                continue
            parent = id(obj)
            for child in children:
                ckey = id(child)
                if ckey in wanted:
                    hits.add(parent)
                    continue
                if type(child) in exact or ckey in internal:
                    continue
                seen_from = parents.get(ckey)
                if seen_from is not None:
                    seen_from.append(parent)
                    continue
                parents[ckey] = [parent]
                below.append(child)
        budget -= len(below)
        if not below or budget < 0:
            break
        level = below
    # Up from the objects holding a target to the variables bound to them.
    stack = list(hits)
    reached: set[int] = set()
    while stack:
        key = stack.pop()
        if key in reached:
            continue
        reached.add(key)
        if key in roots:
            found.update(roots[key])
        stack.extend(parents.get(key, ()))
    return found


def _holders_from_referrers(
    targets: list[Any],
    internal: set[int],
    user_ns: Mapping[str, Any],
    known: set[str],
    gave_up: GaveUp | None = None,
) -> set[str] | None:
    """`_find_holders` up `gc.get_referrers`, one level of containers per
    call. The ids in *internal* are references already accounted for (the
    group's own objects, the mappings that bind it, cash's own containers);
    this function's own frame and its callers' are too. An object holding a
    reference without telling the garbage collector is not found at all; the
    count taken again afterwards still sees it, and refuses.

    A holder that is not a container nor an object whose attributes a
    restore copies (a function's defaults, a class, a library object, a
    frame) makes the answer None. With *gave_up*, the search does not stop
    there: it notes the holder, does not climb through it, and goes on
    collecting the variables the other holders lead to, which *gave_up*
    gets, with the variables bound to such a holder itself (``args =
    argparse.Namespace(features=features)``). Cut short by the limits
    below, *gave_up* says the names may be incomplete.
    """
    stack = set()
    frame = sys._getframe()
    while frame is not None:
        stack.add(id(frame))
        frame = frame.f_back
    del frame
    seen = set(internal) | stack
    found: set[str] = set()
    # By id only: holding a frame or a library object here would keep it
    # alive, and add to the counts taken again afterwards.
    outside: set[int] = set()
    level = targets
    for _depth in range(_MAX_DEPTH):
        if len(level) > _MAX_REFERRER_TARGETS:
            return _gave_up(gave_up, found, outside, user_ns, known, complete=False)
        if outside and _past_deadline():
            # Past a holder the search used to stop at: within the walk's
            # budget, or the names found so far are all it says.
            return _gave_up(gave_up, found, outside, user_ns, known, complete=False)
        # Passed as one tuple, which the call hands on as it is: it is a
        # referrer too.
        args = tuple(level)
        seen.update((id(level), id(args)))
        level_ids = {id(obj) for obj in level}
        referrers = gc.get_referrers(*args)
        seen.add(id(referrers))
        upper = []
        for holder in referrers:
            if id(holder) in seen:
                continue
            if holder is user_ns:
                found.update(name for name, value in user_ns.items() if id(value) in level_ids and name not in known)
                continue
            if not (isinstance(holder, _CONTAINERS) or attributes_of(holder) is not None) or _is_module_namespace(
                holder
            ):
                # A frame, a closure cell, a function, a module or its
                # namespace, a library's object.
                if gave_up is None:
                    return None
                seen.add(id(holder))
                outside.add(id(holder))
                continue
            seen.add(id(holder))
            upper.append(holder)
        del referrers, args
        if not upper:
            return _gave_up(gave_up, found, outside, user_ns, known) if outside else found
        level = upper
    return _gave_up(gave_up, found, outside, user_ns, known, complete=False)


def _past_deadline() -> bool:
    """Is the budget of the walk in progress spent (`walk_budget`)?"""
    deadline = _DEADLINE[0]
    return deadline is not None and _perf_counter() > deadline


def _is_module_namespace(holder: Any) -> bool:
    """Whether *holder* is the ``__dict__`` of an imported module: what its
    functions read their globals from, never a container of the notebook's."""
    if type(holder) is not dict:
        return False
    name = holder.get("__name__")
    if type(name) is not str:
        return False
    module = sys.modules.get(name)
    return module is not None and getattr(module, "__dict__", None) is holder


def _gave_up(
    gave_up: GaveUp | None,
    found: set[str],
    outside: set[int],
    user_ns: Mapping[str, Any],
    known: set[str],
    complete: bool = True,
) -> None:
    """Hand *gave_up* the variables `_holders_from_referrers` found, and
    those bound to one of the holders it did not climb through (*outside*,
    by id); None, its answer when it gives up."""
    if gave_up is not None:
        value_types = VALUE_TYPES + library_value_types()
        gave_up.names |= found
        if outside:
            # A class or a function holding the object (a class attribute
            # registry, a mutable default) is linked too: what reads it
            # reads the object (``slow(Reg.items)``), though a class or a
            # function is otherwise a value no statement changes in place.
            gave_up.names.update(
                name
                for name, value in list(user_ns.items())
                if id(value) in outside
                and name not in known
                and (isinstance(value, (type, types.FunctionType)) or not is_value(value, value_types))
            )
            # A module's namespace holding it (``HELD.append(rows)`` in a
            # module's function) is named for the caller to weigh: a module
            # is no variable, but a statement may have set its state.
            gave_up.modules.update(
                name
                for name, module in list(sys.modules.items())
                if isinstance(module, types.ModuleType) and id(module.__dict__) in outside
            )
        gave_up.complete = gave_up.complete and complete
    return None


def count_held(
    held: list[Any],
    nodes: dict[int, Any],
    inbound: dict[int, int],
    value_types: tuple[type, ...],
    named_held: Iterable[tuple[Mapping[str, Any], str]] = (),
    bound_in: Mapping[str, Any] | None = None,
) -> set[int]:
    """Add to *inbound* the references that the containers in *held*, which
    cash holds itself, and the containers inside them make to *nodes*; the
    ids of those containers.

    A container inside one of *held* is cash's own only while nothing else
    holds it -- the references *held*, *named_held* and other such
    containers make to it are its whole count. One a notebook variable holds
    too (a list a cell ended with, so ``Out`` holds it: ``frames = [df1,
    df2]\nframes``) is the user's: its references to *nodes* are a holder's,
    and so are those of everything inside it.

    *bound_in*, the notebook's variables when given: a list, dict or tuple
    of JSON-like data a variable is bound to is the user's on its face, and
    is not walked into (`_walk_held`'s *cut*) -- the output history holding
    a displayed list of 400,000 tuples was walked a tuple at a time for
    each statement whose output another variable shares.
    """
    roots = {id(obj) for obj in held}
    named_held = list(named_held)
    cut = _bound_trees(bound_in, named_held) if bound_in is not None else None
    reach, refs, edges = _walk_held(held, nodes, value_types, cut)
    for mapping, key in named_held:
        if id(mapping.get(key)) in reach:
            refs[id(mapping.get(key))] += 1
    users = excess_refs(reach, refs, list(reach))
    if cut:
        met = [key for key in cut if key in reach]
        if met and (
            id(bound_in) in edges
            or not set(met) <= set(users)
            or not all(_plain_data.is_tree(reach[key], TREE_LEAVES) for key in met)
        ):
            # The variables' own mapping is inside what cash holds, so a
            # binding is no proof, or one is no JSON-like data whose parts
            # can be read a level at a time: walked whole, as before.
            del reach, refs, edges, users, met
            return count_held(held, nodes, inbound, value_types, named_held)
        # What is inside a variable's tree is the user's, as the walk below
        # it would have found: a part also reached another way is too.
        others = set(refs).difference(met, users)
        if others:
            under = _parts_under([reach[key] for key in met])
            users.extend(others & under)
            del under
        del met, others
    del reach
    while users:
        key = users.pop()
        if key in refs:
            del refs[key]
            users.extend(child for child in edges.get(key, ()) if child in refs)
    own = roots | set(refs)
    for key in own:
        for child in edges.get(key, ()):
            if child in nodes:
                inbound[child] += 1
    return own


def _bound_trees(bound_in: Mapping[str, Any], named_held: list[tuple[Mapping[str, Any], str]]) -> set[int]:
    """The ids of the lists, dicts and tuples bound to names of
    *bound_in* other than the *named_held* ones (whose binding `count_held`
    counts as cash's own)."""
    skip = {key for mapping, key in named_held if mapping is bound_in}
    return {id(value) for name, value in bound_in.items() if type(value) in _plain_data.TREE_NODES and name not in skip}


def _parts_under(trees: list[Any]) -> set[int]:
    """The ids of everything inside *trees*, JSON-like data, read a level at
    a time."""
    under: set[int] = set()
    for tree in trees:
        for flat, kinds in _plain_data.tree_levels(tree, TREE_LEAVES):
            if not kinds.isdisjoint(_plain_data.TREE_NODES):
                under.update(id(item) for item in flat if type(item) in _CONTAINER_KINDS)
    return under


_CONTAINER_KINDS = frozenset(_plain_data.TREE_NODES)


def _walk_held(
    held: list[Any], nodes: dict[int, Any], value_types: tuple[type, ...], cut: set[int] | None = None
) -> tuple[dict[int, Any], dict[int, int], dict[int, list[int]]]:
    """``(reach, refs, edges)`` for the containers in *held* and inside them:
    *reach* the containers inside them by id (not *held* itself, nor
    *nodes*), *refs* how many references the walked containers make to each
    one, *edges* the ids each walked container refers to (*reach* and
    *nodes* ones, once per reference). Returns before the counts are read,
    so none of its local references are left to inflate them."""
    roots = {id(obj) for obj in held}
    reach: dict[int, Any] = {}
    refs: dict[int, int] = {}
    edges: dict[int, list[int]] = {}
    seen: set[int] = set()
    stack = list(held)
    exact = EXACT_VALUE_TYPES
    containers = EXACT_CONTAINER_TYPES
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        out = edges.setdefault(id(obj), [])
        children = children_of(obj)
        if children is None:
            children = _iterator_parts(obj)
        if not children or exact.issuperset(map(type, children)):
            continue  # nothing but values, asked at C speed (see `_walk`)
        for child in children:
            ctype = type(child)
            if ctype in exact or (ctype not in containers and is_value(child, value_types)):
                continue
            ckey = id(child)
            if ckey in nodes:
                # A walked node: its own references are counted already.
                out.append(ckey)
                continue
            if ctype is tuple or ctype is frozenset:
                for item in child:
                    itype = type(item)
                    if itype not in exact and (itype in containers or not is_value(item, value_types)):
                        break
                else:
                    # Nothing inside to count, and not a node (see `_walk`).
                    continue
            if ckey in roots:
                continue
            out.append(ckey)
            refs[ckey] = refs.get(ckey, 0) + 1
            if ckey not in reach:
                reach[ckey] = child
                if not cut or ckey not in cut:
                    stack.append(child)
    return reach, refs, edges


def _iterator_parts(value: Any) -> list[Any] | None:
    """What an iterator written in C keeps (`gc.get_referents`), or None
    for anything else.

    ``enumerate``, ``zip`` and ``itertools.product`` keep the tuple they
    hand out and fill it again for the next item, so once a loop over one
    ends, that tuple still holds the item the loop took last. Inside what
    cash holds (a loop's iterable), it is cash's: after ``for i, example in
    enumerate(ds):`` it read as a second holder of ``example``, and the
    search of the whole heap for it cost 0.2-0.5 s a run on E11.
    """
    kind = type(value)
    if getattr(kind, "__next__", None) is None or hasattr(value, "__dict__"):
        return None
    return gc.get_referents(value)


#: Most containers below a root of JSON-like data that `_walk` takes on
#: their own, held besides their parent (`_take_breaks`); past it, the
#: root is walked one container at a time.
_MAX_BREAKS = 64


def _walk(
    group: Mapping[str, Any],
    bindings: list[Mapping[str, Any]],
    value_types: tuple[type, ...],
    keep_identity: Iterable[str] | None = None,
    *,
    fast: bool = True,
    memo: dict[int, list | None] | None = None,
    leaves: Leaves | None = None,
) -> tuple[dict[int, Any], dict[int, int], list[int], dict[int, str], set[int]]:
    """``(nodes, inbound, checked, owner, breaks)`` for the objects
    reachable from the roots in *group*, walked as one graph.

    *nodes* holds each object once by id, *inbound* counts the references to
    it the walk accounts for (for a root: *group* itself and each of
    *bindings* that binds it under its name), *checked* are the ids whose
    count must be compared: all but the tuples and frozensets that hold
    nothing mutable, and with *keep_identity* only what those roots reach.
    *owner* names the root each object was first reached from. Returns
    before the counts are read, so none of its local references are left to
    inflate them.

    With *fast*, a root of JSON-like data whose containers are held by their
    parents alone, but for a few (`_take_breaks`), is read a level at a time
    instead of walked: *nodes* leaves out the containers below it that only
    their parent holds, whose counts can never be above *inbound*, and every
    count it does hold is the one the walk with *fast* off makes, so the two
    find the same objects shared. *breaks* are the ids of the containers it
    took on their own.

    *memo*, when given, keeps what a level-at-a-time read found below each
    root by its id, for a later walk over the same objects with no code run
    in between; its lists are references the caller must count.

    *leaves*, when given, are objects the walk does not take: neither counted
    nor walked below, and noted in ``leaves.met`` when reached.
    """
    nodes: dict[int, Any] = {}
    inbound: dict[int, int] = {}
    owner: dict[int, str] = {}
    order: list[int] = []
    identity = set(group) if keep_identity is None else set(keep_identity)
    exact = EXACT_VALUE_TYPES
    taken: set[int] = set()
    # The roots whose identity counts first: what they reach is checked.
    first = [name for name in group if name in identity]
    reached: int | None = None
    for name in [*first, *(name for name in group if name not in identity)]:
        if name not in identity and reached is None:
            reached = len(order)
        root = group[name]
        if is_value(root, value_types):
            continue
        key = id(root)
        inbound[key] = inbound.get(key, 0) + 1 + sum(1 for m in bindings if m.get(name) is root)
        if key in nodes:
            continue
        nodes[key] = root
        owner[key] = name
        order.append(key)
        if _values_only(root, exact):
            # A flat list or dict of values (``[f(i) for i in ...]``): nothing
            # below the root to count, asked at C speed. Before the tree
            # check, which also sizes every item for the facts it keeps.
            continue
        if not fast:
            if _plain_data.held_only_by_parents(root, TREE_LEAVES):
                # Records as a parser returns them: no container below the
                # root has a holder besides its parent, read a level at a time
                # at C speed. Only the root's own count is left to compare;
                # walked one container at a time, a million records took
                # seconds.
                continue
        else:
            if memo is not None and key in memo:
                breaks = memo[key]
            else:
                breaks = _plain_data.held_beyond_parents(root, TREE_LEAVES, _MAX_BREAKS)
                if memo is not None:
                    memo[key] = breaks
            if breaks is not None and _take_breaks(breaks, name, nodes, inbound, owner, order, value_types, leaves):
                taken.update(map(id, breaks))
                continue
            del breaks
        _descend([root], name, nodes, inbound, owner, order, value_types, leaves)
    if reached is None:
        reached = len(order)
    # Children before parents: a tuple counts only when something mutable
    # is inside it, however deep.
    carries: dict[int, bool] = {}
    containers = EXACT_CONTAINER_TYPES
    for ckey in reversed(order):
        value = nodes[ckey]
        if isinstance(value, _IMMUTABLE_CONTAINERS):
            carries[ckey] = any(
                type(c) not in exact
                and (type(c) in containers or not is_value(c, value_types))
                and carries.get(id(c), id(c) in nodes)
                for c in value
            )
    candidates = order[:reached]
    return nodes, inbound, [k for k in candidates if carries.get(k, True)], owner, taken


def _is_node(child: Any, value_types: tuple[type, ...]) -> bool:
    """Whether `_walk` counts *child*, met inside a root: not a value, and
    not a tuple or frozenset holding values alone."""
    exact = EXACT_VALUE_TYPES
    containers = EXACT_CONTAINER_TYPES
    ctype = type(child)
    if ctype in exact or (ctype not in containers and is_value(child, value_types)):
        return False
    if ctype is tuple or ctype is frozenset:
        for item in child:
            itype = type(item)
            if itype not in exact and (itype in containers or not is_value(item, value_types)):
                return True
        # Holds nothing but values: no identity to count, and not a node (the
        # carries loop reads that as "carries nothing"). A walk over millions
        # of such pairs is most of what a list of records costs.
        return False
    return True


def _descend(
    stack: list[Any],
    name: str,
    nodes: dict[int, Any],
    inbound: dict[int, int],
    owner: dict[int, str],
    order: list[int],
    value_types: tuple[type, ...],
    leaves: Leaves | None = None,
) -> None:
    """`_walk` one container at a time below the objects on *stack*, which
    are nodes already: each reference to a node counted once in *inbound*,
    each new node taken under *name*. One of *leaves* is not taken."""
    exact = EXACT_VALUE_TYPES
    containers = EXACT_CONTAINER_TYPES
    pops = 0
    while stack:
        pops += 1
        if not pops & 1023:
            check_walk_budget()
        children = children_of(stack.pop())
        # Nothing but values, by exact type: asked at C speed, where the
        # loop below takes a Python step per item -- 35 ms of storing a
        # list of 200,000 ints.
        if not children or exact.issuperset(map(type, children)):
            continue
        for child in children:
            ctype = type(child)
            if ctype in exact or (ctype not in containers and is_value(child, value_types)):
                continue
            if (ctype is tuple or ctype is frozenset) and not _is_node(child, value_types):
                continue
            ckey = id(child)
            if leaves and ckey in leaves.ids:
                leaves.met.add(ckey)
                continue
            inbound[ckey] = inbound.get(ckey, 0) + 1
            if ckey not in nodes:
                nodes[ckey] = child
                owner[ckey] = name
                order.append(ckey)
                stack.append(child)


def _take_breaks(
    breaks: list[Any],
    name: str,
    nodes: dict[int, Any],
    inbound: dict[int, int],
    owner: dict[int, str],
    order: list[int],
    value_types: tuple[type, ...],
    leaves: Leaves | None = None,
) -> bool:
    """Count the root *name* of JSON-like data from the containers below it
    that something besides their parent holds (*breaks*,
    `_plain_data.held_beyond_parents`), not from every container; False,
    with nothing counted, when that cannot be done.

    Every other container below the root has one reference, its parent's,
    which the walk would count too: it can neither be shared nor be reached
    but through its parent, so leaving it out changes no count the walk
    compares. Each break is counted as the walk counts a child (one
    reference from the root's side, the least it has) and walked below as
    the walk walks it. That is a lower bound only while no break lies below
    another, whose edge the walk below would count a second time: then
    False. ``for r in recs:`` leaves one record ``r`` holds too, and the
    statement's check reads ``recs`` a level at a time instead of a Python
    step per record.
    """
    breaks = [obj for obj in breaks if _is_node(obj, value_types)]
    if leaves and any(id(obj) in leaves.ids for obj in breaks):
        return False
    if not breaks:
        return True
    ids = {id(obj) for obj in breaks}
    for obj in breaks:
        own = id(obj)
        stack = [obj]
        seen = {own}
        while stack:
            children = children_of(stack.pop())
            # Nothing but values, by exact type, asked at C speed: a dict of
            # four parsed columns of two million items each took 2.4 s here.
            if not children or EXACT_VALUE_TYPES.issuperset(map(type, children)):
                continue
            for child in children:
                ckey = id(child)
                if ckey in seen or type(child) in EXACT_VALUE_TYPES:
                    continue
                if ckey in ids:
                    return False
                seen.add(ckey)
                stack.append(child)
    for obj in breaks:
        ckey = id(obj)
        inbound[ckey] = inbound.get(ckey, 0) + 1
        if ckey not in nodes:
            nodes[ckey] = obj
            owner[ckey] = name
            order.append(ckey)
            _descend([obj], name, nodes, inbound, owner, order, value_types, leaves)
    return True
