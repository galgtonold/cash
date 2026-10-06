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
notebook's own classes and of ``SimpleNamespace`` / dataclass instances.
Everything else is one leaf, whose own count is checked. Values with no
identity worth keeping (numbers, strings, classes, functions, enum members,
numpy scalars and dtypes, ...) are skipped.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import fractions
import pathlib
import sys
import types
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["output_history", "shared_names"]

#: Values whose identity no program relies on: equal ones are interchangeable.
_VALUE_TYPES: tuple[type, ...] = (
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
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.CodeType,
)

#: The builtin containers a restore copies along with their contents.
_CONTAINERS = (list, dict, set, tuple, frozenset)

#: Containers whose own identity does not matter -- only their contents'.
_IMMUTABLE_CONTAINERS = (tuple, frozenset)


def _library_value_types() -> tuple[type, ...]:
    """The value types of numpy and pandas, when they are already imported."""
    found: list[type] = []
    np = sys.modules.get("numpy")
    if np is not None:
        found += [np.generic, np.dtype]
    pd = sys.modules.get("pandas")
    if pd is not None:
        found += [pd.Timestamp, pd.Timedelta, pd.Period, pd.Interval, type(pd.NA), type(pd.NaT)]
    return tuple(found)


def _attributes_of(value: Any) -> dict[str, Any] | None:
    """The attributes a restore copies with *value* and the caller can reach,
    for an object of the notebook's own classes, a ``SimpleNamespace`` or a
    dataclass; ``None`` for anything else (a leaf)."""
    cls = type(value)
    if not (cls.__module__ == "__main__" or cls is types.SimpleNamespace or dataclasses.is_dataclass(cls)):
        return None
    try:
        own = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return None
    return own if type(own) is dict else None


def _children(value: Any) -> Iterable[Any] | None:
    """What *value* holds that a restore copies with it, or None for a leaf."""
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, _CONTAINERS):
        return list(value)
    attrs = _attributes_of(value)
    return None if attrs is None else list(attrs.values())


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


def shared_names(
    roots: Mapping[str, Any],
    bindings: Iterable[Mapping[str, Any]],
    cash_held: Iterable[Any] = (),
    named_held: Iterable[tuple[Mapping[str, Any], str]] = (),
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
    """
    cash_held = list(cash_held)
    named_held = list(named_held)
    value_types = _VALUE_TYPES + _library_value_types()
    bindings = list(bindings)
    shared: set[str] = set()
    for name in roots:
        if isinstance(roots[name], value_types):
            continue
        # The caller's *roots* holds the root too.
        expected = 1 + sum(1 for m in bindings if m.get(name) is roots[name])
        nodes, inbound, checked = _walk(roots[name], expected, value_types)
        _count_held(cash_held, nodes, inbound, value_types)
        for mapping, key in named_held:
            if id(mapping.get(key)) in nodes:
                inbound[id(mapping.get(key))] += 1
        if _excess(nodes, inbound, checked):
            shared.add(name)
    return shared


def _count_held(held: list[Any], nodes: dict[int, Any], inbound: dict[int, int], value_types: tuple[type, ...]) -> None:
    """Add to *inbound* the references that the containers in *held*, which
    cash holds itself, and the containers inside them make to *nodes*."""
    seen: set[int] = set()
    stack = list(held)
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        for child in _children(obj) or ():
            if isinstance(child, value_types):
                continue
            if id(child) in nodes:
                # A walked node: its own references are counted already.
                inbound[id(child)] += 1
            else:
                stack.append(child)


def _walk(root: Any, expected: int, value_types: tuple[type, ...]) -> tuple[dict[int, Any], dict[int, int], list[int]]:
    """``(nodes, inbound, checked)`` for the objects reachable from *root*.

    *nodes* holds each object once by id, *inbound* counts the references to
    it the walk accounts for (*expected* for the root), and *checked* are the
    ids whose count must be compared: all but the tuples and frozensets that
    hold nothing mutable. Returns before the counts are read, so none of its
    local references are left to inflate them.
    """
    key = id(root)
    nodes: dict[int, Any] = {key: root}
    inbound: dict[int, int] = {key: expected}
    order = [key]
    stack = [root]
    while stack:
        children = _children(stack.pop())
        if not children:
            continue
        for child in children:
            if isinstance(child, value_types):
                continue
            ckey = id(child)
            inbound[ckey] = inbound.get(ckey, 0) + 1
            if ckey not in nodes:
                nodes[ckey] = child
                order.append(ckey)
                stack.append(child)
    # Children before parents: a tuple counts only when something mutable
    # is inside it, however deep.
    carries: dict[int, bool] = {}
    for ckey in reversed(order):
        value = nodes[ckey]
        if isinstance(value, _IMMUTABLE_CONTAINERS):
            carries[ckey] = any(not isinstance(c, value_types) and carries.get(id(c), True) for c in value)
    return nodes, inbound, [k for k in order if carries.get(k, True)]
