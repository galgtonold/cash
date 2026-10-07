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
notebook's own classes, of ``SimpleNamespace`` / dataclass instances and of
estimators (a ``Pipeline`` holds the step objects it was built from), and
the object a bound method is bound to and the cells a closure keeps
(``hooks = {'log': tracker.log}``). Everything else is one leaf, whose own
count is checked. Values with no identity worth keeping (numbers, strings,
classes, functions bound to nothing, enum members, numpy scalars and dtypes,
...) are skipped.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import fractions
import gc
import pathlib
import sys
import types
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any

__all__ = ["holds_part_of", "output_history", "share_group", "shared_names"]

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
_EXACT_VALUE_TYPES: frozenset[type] = frozenset(
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

#: The builtin containers by exact type: never a value type, so no `isinstance`.
_EXACT_CONTAINER_TYPES: frozenset[type] = frozenset({list, dict, set, tuple, frozenset})

#: The builtin containers a restore copies along with their contents.
_CONTAINERS = (list, dict, set, tuple, frozenset)

#: Containers whose own identity does not matter -- only their contents'.
_IMMUTABLE_CONTAINERS = (tuple, frozenset)


def library_value_types() -> tuple[type, ...]:
    """The value types of numpy and pandas, when they are already imported."""
    found: list[type] = []
    np = sys.modules.get("numpy")
    if np is not None:
        found += [np.generic, np.dtype]
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
    """
    if isinstance(value, types.FunctionType):
        return list(value.__closure__) if value.__closure__ else None
    owner = getattr(value, "__self__", None)
    if owner is None or isinstance(owner, (types.ModuleType, type)):
        return None
    return [owner]


def is_value(value: Any, value_types: tuple[type, ...]) -> bool:
    """Whether *value* has no identity a restore must keep (`VALUE_TYPES`,
    a function or method that carries nothing)."""
    if isinstance(value, value_types):
        return True
    return isinstance(value, _CALLABLE_TYPES) and _bound_objects(value) is None


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


def attributes_of(value: Any) -> dict[str, Any] | None:
    """The attributes a restore copies with *value* and the caller can reach,
    for an object of the notebook's own classes, a ``SimpleNamespace``, a
    dataclass or an estimator; ``None`` for anything else (a leaf)."""
    cls = type(value)
    if not (
        cls.__module__ == "__main__"
        or cls is types.SimpleNamespace
        or dataclasses.is_dataclass(cls)
        or _is_estimator_class(cls)
    ):
        return None
    try:
        own = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return None
    return own if type(own) is dict else None


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
    return None if attrs is None else list(attrs.values())


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
        stack.extend(children_of(obj) or ())


def holds_part_of(value: Any, sources: Iterable[Any]) -> bool:
    """Whether *value* is or holds an object one of *sources* is or holds.

    A function's result restored as a copy is only the same as calling the
    function when it holds nothing the caller has too: ``bundle(model, df)``
    returning ``{'model': model, ...}``, or ``pick(cfg, 'a')`` returning
    ``cfg['a']``, hand back an object a later ``model.fit()`` or
    ``c['n'] = 5`` must reach through both names. Walked as `_walk` walks:
    the builtin containers and the attributes of the notebook's own objects.
    """
    value_types = VALUE_TYPES + library_value_types()
    own = set(_identities(value, value_types))
    if not own:
        return False
    return any(key in own for source in sources for key in _identities(source, value_types))


def _excess(nodes: dict[int, Any], inbound: dict[int, int], ids: Iterable[int]) -> list[int]:
    """The ids among *ids* whose reference count is above *inbound* plus this
    function's own references (`_OVERHEAD`)."""
    shared = []
    for key in ids:
        if sys.getrefcount(nodes[key]) - inbound[key] > _OVERHEAD:
            shared.append(key)
    return shared


def _calibrate() -> int:
    """How many references `_excess` itself adds to an object it checks.

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


def share_group(
    names: Iterable[str],
    values: Mapping[str, Any],
    user_ns: Mapping[str, Any],
    cash_held: Iterable[Any] = (),
    named_held: Iterable[tuple[Mapping[str, Any], str]] = (),
    foreign: Callable[[str], bool] = lambda name: False,
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
    the outputs still shared and *holders* is empty.
    """
    group = {name: values[name] for name in names if values.get(name) is not None}
    bindings = [values, user_ns]
    cash_held = list(cash_held)
    named_held = list(named_held)
    joined: set[str] = set()
    for _round in range(_MAX_ROUNDS):
        shared, found = _check_group(group, bindings, cash_held, named_held, None, user_ns)
        if not shared:
            return {name: group[name] for name in joined}, set()
        if not found or any(foreign(name) for name in found) or _copies_keep_old_objects(group):
            break
        joined |= found
        group.update((name, user_ns[name]) for name in found)
    outputs = set(group) - joined
    return {}, (shared & outputs) or outputs


def _copies_keep_old_objects(group: Mapping[str, Any]) -> bool:
    """Whether *group* holds a closure or a builtin bound method that carries
    an object (``lambda v: store.append(v)``, ``results.append``).

    A deep copy keeps such a function as it is, still bound to the object of
    the run that stored it, while the holder variable is restored as a copy:
    stored together, the two would no longer be the same object."""
    value_types = VALUE_TYPES + library_value_types()
    seen: set[int] = set()
    stack = list(group.values())
    while stack:
        obj = stack.pop()
        if is_value(obj, value_types) or id(obj) in seen:
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
) -> tuple[set[str], set[str] | None]:
    """``(shared, found)`` for *group*: the names whose objects are held
    beyond what the group, *bindings*, *cash_held* and *named_held* account
    for, and -- with *user_ns* -- the variables of *user_ns* outside the
    group that hold them (`_find_holders`), None when one holder is not such
    a variable."""
    value_types = VALUE_TYPES + library_value_types()
    nodes, inbound, checked, owner = _walk(group, bindings, value_types, keep_identity)
    internal = _count_held(cash_held, nodes, inbound, value_types, named_held)
    named = {id(mapping) for mapping, _key in named_held if mapping is not user_ns}
    for mapping, key in named_held:
        if id(mapping.get(key)) in nodes:
            inbound[id(mapping.get(key))] += 1
    excess = _excess(nodes, inbound, checked)
    if not excess:
        return set(), set()
    shared = {owner[k] for k in excess}
    if user_ns is None:
        return shared, None
    internal |= {id(group), id(bindings), id(cash_held), id(named_held), id(nodes), *nodes, *named}
    internal |= {id(m) for m in bindings if m is not user_ns}
    discounted = {key for mapping, key in named_held if mapping is user_ns}
    return shared, _find_holders([nodes[k] for k in excess], internal, user_ns, set(group) | discounted)


def _find_holders(
    targets: list[Any], internal: set[int], user_ns: Mapping[str, Any], known: set[str]
) -> set[str] | None:
    """The variables of *user_ns* not in *known* that hold one of *targets*,
    directly or through containers and attributes a restore copies along;
    None when a holder is anything else.

    Walks up `gc.get_referrers`, one level of containers per call. The ids in
    *internal* are references already accounted for (the group's own
    objects, the mappings that bind it, cash's own containers); this
    function's own frame and its callers' are too. An object holding a
    reference without telling the garbage collector is not found at all; the
    count taken again afterwards still sees it, and refuses.
    """
    stack = set()
    frame = sys._getframe()
    while frame is not None:
        stack.add(id(frame))
        frame = frame.f_back
    del frame
    seen = set(internal) | stack
    found: set[str] = set()
    level = targets
    for _depth in range(_MAX_DEPTH):
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
            if not (isinstance(holder, _CONTAINERS) or attributes_of(holder) is not None):
                # A frame, a closure cell, a function, a module, a library's object.
                return None
            seen.add(id(holder))
            upper.append(holder)
        del referrers, args
        if not upper:
            return found
        level = upper
    return None


def _count_held(
    held: list[Any],
    nodes: dict[int, Any],
    inbound: dict[int, int],
    value_types: tuple[type, ...],
    named_held: Iterable[tuple[Mapping[str, Any], str]] = (),
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
    """
    roots = {id(obj) for obj in held}
    reach, refs, edges = _walk_held(held, nodes, value_types)
    for mapping, key in named_held:
        if id(mapping.get(key)) in reach:
            refs[id(mapping.get(key))] += 1
    users = _excess(reach, refs, list(reach))
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


def _walk_held(
    held: list[Any], nodes: dict[int, Any], value_types: tuple[type, ...]
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
    exact = _EXACT_VALUE_TYPES
    containers = _EXACT_CONTAINER_TYPES
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        out = edges.setdefault(id(obj), [])
        for child in children_of(obj) or ():
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
                stack.append(child)
    return reach, refs, edges


def _walk(
    group: Mapping[str, Any],
    bindings: list[Mapping[str, Any]],
    value_types: tuple[type, ...],
    keep_identity: Iterable[str] | None = None,
) -> tuple[dict[int, Any], dict[int, int], list[int], dict[int, str]]:
    """``(nodes, inbound, checked, owner)`` for the objects reachable from
    the roots in *group*, walked as one graph.

    *nodes* holds each object once by id, *inbound* counts the references to
    it the walk accounts for (for a root: *group* itself and each of
    *bindings* that binds it under its name), *checked* are the ids whose
    count must be compared: all but the tuples and frozensets that hold
    nothing mutable, and with *keep_identity* only what those roots reach.
    *owner* names the root each object was first reached from. Returns
    before the counts are read, so none of its local references are left to
    inflate them.
    """
    nodes: dict[int, Any] = {}
    inbound: dict[int, int] = {}
    owner: dict[int, str] = {}
    order: list[int] = []
    identity = set(group) if keep_identity is None else set(keep_identity)
    exact = _EXACT_VALUE_TYPES
    containers = _EXACT_CONTAINER_TYPES
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
        stack = [root]
        while stack:
            children = children_of(stack.pop())
            if not children:
                continue
            for child in children:
                ctype = type(child)
                if ctype in exact or (ctype not in containers and is_value(child, value_types)):
                    continue
                if ctype is tuple or ctype is frozenset:
                    for item in child:
                        itype = type(item)
                        if itype not in exact and (itype in containers or not is_value(item, value_types)):
                            break
                    else:
                        # Holds nothing but values: no identity to count, and
                        # not a node (the loop below reads that as "carries
                        # nothing"). A walk over millions of such pairs is
                        # most of what a list of records costs.
                        continue
                ckey = id(child)
                inbound[ckey] = inbound.get(ckey, 0) + 1
                if ckey not in nodes:
                    nodes[ckey] = child
                    owner[ckey] = name
                    order.append(ckey)
                    stack.append(child)
    if reached is None:
        reached = len(order)
    # Children before parents: a tuple counts only when something mutable
    # is inside it, however deep.
    carries: dict[int, bool] = {}
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
    return nodes, inbound, [k for k in candidates if carries.get(k, True)], owner
