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

import contextlib
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
from collections.abc import Callable, Iterable, Iterator, Mapping
from time import perf_counter as _perf_counter
from typing import Any

from cash import _plain_data

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
    # What each round finds below a root of JSON-like data, for the next
    # round: no code runs between them, so what holds a part of it is the
    # same (`_walk`'s *memo*).
    memo: dict[int, list | None] = {}
    for _round in range(_MAX_ROUNDS):
        shared, found = _check_group(group, bindings, cash_held, named_held, None, user_ns, memo)
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
    memo: dict[int, list | None] | None = None,
) -> tuple[set[str], set[str] | None]:
    """``(shared, found)`` for *group*: the names whose objects are held
    beyond what the group, *bindings*, *cash_held* and *named_held* account
    for, and -- with *user_ns* -- the variables of *user_ns* outside the
    group that hold them (`_find_holders`), None when one holder is not such
    a variable. *memo* is `_walk`'s, kept by the caller across rounds."""
    value_types = VALUE_TYPES + library_value_types()
    if memo is None:
        memo = {}
    for fast in (True, False):
        nodes, inbound, checked, owner, breaks = _walk(
            group, bindings, value_types, keep_identity, fast=fast, memo=memo
        )
        _count_memo(memo, nodes, inbound)
        if not excess_refs(nodes, inbound, checked):
            # Every reference is the group's own already. What cash holds
            # itself only adds to the counts, so it cannot leave one over:
            # not walked. It was walked for every statement, ``x = 0`` too
            # -- the output history holding a displayed list of 400,000
            # tuples, 1.2 s a statement.
            return set(), set()
        internal = count_held(cash_held, nodes, inbound, value_types, named_held, user_ns)
        for mapping, key in named_held:
            if id(mapping.get(key)) in nodes:
                inbound[id(mapping.get(key))] += 1
        excess = excess_refs(nodes, inbound, checked)
        if not excess:
            return set(), set()
        shared = {owner[k] for k in excess}
        if user_ns is None:
            return shared, None
        named = {id(mapping) for mapping, _key in named_held if mapping is not user_ns}
        internal |= {id(group), id(bindings), id(cash_held), id(named_held), id(nodes), *nodes, *named}
        internal |= {id(memo), *(id(listed) for listed in memo.values() if listed is not None)}
        internal |= {id(m) for m in bindings if m is not user_ns}
        discounted = {key for mapping, key in named_held if mapping is user_ns}
        found = _find_holders([nodes[k] for k in excess], internal, user_ns, set(group) | discounted)
        if found is not None or breaks.isdisjoint(excess):
            return shared, found
        # A container a root read a level at a time holds besides its parent
        # (`_take_breaks`): the search up from it climbs through the
        # containers above it, which the walk left out of *internal*, and a
        # longer climb can give up where the full walk's would not. Asked
        # again of the full walk, with none of these references left.
        del nodes, inbound, checked, owner, breaks, internal, excess, found
    return shared, None


def _count_memo(memo: dict[int, list | None], nodes: dict[int, Any], inbound: dict[int, int]) -> None:
    """Add to *inbound* the memo's own reference to each node it lists. In
    a function of its own, so that no loop variable is left holding one."""
    for listed in memo.values():
        for key in map(id, listed or ()):
            if key in nodes:
                inbound[key] += 1


def _find_holders(
    targets: list[Any], internal: set[int], user_ns: Mapping[str, Any], known: set[str]
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
    return _holders_from_referrers(targets, internal, user_ns, known)


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
    targets: list[Any], internal: set[int], user_ns: Mapping[str, Any], known: set[str]
) -> set[str] | None:
    """`_find_holders` up `gc.get_referrers`, one level of containers per
    call. The ids in *internal* are references already accounted for (the
    group's own objects, the mappings that bind it, cash's own containers);
    this function's own frame and its callers' are too. An object holding a
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
        if len(level) > _MAX_REFERRER_TARGETS:
            return None
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
            if breaks is not None and _take_breaks(breaks, name, nodes, inbound, owner, order, value_types):
                taken.update(map(id, breaks))
                continue
            del breaks
        _descend([root], name, nodes, inbound, owner, order, value_types)
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
) -> None:
    """`_walk` one container at a time below the objects on *stack*, which
    are nodes already: each reference to a node counted once in *inbound*,
    each new node taken under *name*."""
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
            _descend([obj], name, nodes, inbound, owner, order, value_types)
    return True
