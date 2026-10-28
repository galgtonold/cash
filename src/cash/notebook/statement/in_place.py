"""What a loop changed in place, restored into the object it changed.

``for f in files: ...; p.update()`` changes ``p``. A hit that bound ``p`` to
the restored copy put a new object where the old one was: ``weakref.ref(p)``
then gave None, a ``WeakValueDictionary`` lost its entry, and ``{id(p): ...}``
no longer found ``p``. None of those is a reference the share check counts,
and an ``id()`` key is no reference at all. So the copy's state goes into the
live object instead, which keeps its identity: a list's items, a dict's or a
set's, or the ``__dict__`` of an object pickled the default way.

Stored this way only when that restore is what plain Python gives
(`refusal`): an object of another kind, one whose pickling is its own
(``__getstate__`` or ``__reduce__``, which may leave out what the loop
added), or one that reaches itself, whose copy would point at the copy, runs
the loop again, as do other names or containers holding it beside its own
name (`holder_refusal`), and a weak reference the notebook holds to an
object it holds, or an ``id()`` of one kept in a container of the notebook
(`_held_by_the_notebook`): what it holds comes back as copies. A name bound
to the object itself is fine: it is bound to the live object again. An
``id()`` kept in a module, or below a container too big to walk, is not
seen. A hit asks again of the live objects, since a weak reference made
after the store finds a copy too.

An object of a kind no restore goes into, which the loop only assigns into
(``arr[i] = v``), is bound to the stored copy as any output is
(`bound_as_a_copy`), unless the notebook holds it by a weak reference or
its ``id()``.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref
from collections.abc import Iterable, Mapping
from typing import Any

from cash._user_attrs import attribute_or
from cash.live_state import is_live_stream
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["bound_as_a_copy", "holder_refusal", "ready", "refusal", "restorable_kind", "restore_into"]

#: Containers restored by their contents.
_CONTAINERS = (list, dict, set)
#: Most objects one check looks through: past it the loop runs again.
_MAX_SEEN = 200_000
#: A container with more items is not looked inside (`_held_by_the_notebook`).
_MAX_SCANNED = 10_000
#: What pickle's default handling of an object would be overridden by.
_OWN_PICKLING = ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__getnewargs_ex__", "__getnewargs__")
#: Not looked into: what they hold is not the object's state as pickle sees it.
_OPAQUE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.CodeType, types.FrameType)
_HEAP_TYPE = 1 << 9


def _default_object(kind: type) -> bool:
    """An object of a class of Python code, with a ``__dict__``, no slots of
    its own and pickled the default way: its ``__dict__`` is its state."""
    if not kind.__flags__ & _HEAP_TYPE or not kind.__dictoffset__:
        return False
    for klass in kind.__mro__:
        if klass is object:
            continue
        if not klass.__flags__ & _HEAP_TYPE or "__slots__" in vars(klass):
            return False
        if any(name in vars(klass) for name in _OWN_PICKLING):
            return False
    return True


def refusal(obj: Any, namespace: Mapping[str, Any] | None = None) -> str | None:
    """Why *obj* cannot be restored into, or None. What it holds comes back
    as copies, so a weak reference the notebook holds to one, or an
    ``id()`` of one kept in a container of *namespace*, refuses it too
    (`_held_by_the_notebook`)."""
    kind = type(obj)
    if not restorable_kind(obj):
        return f"a {kind.__name__} is not restored into the object it is, and a new one would not be that object"
    parts: set[int] = set()
    weakly: list[Any] = []
    found = _reaches(_state(obj), obj, seen=parts, weakly=weakly)
    if found is None:
        return "it holds too many objects to check that it does not hold itself"
    if found:
        return "it holds itself, and a copy would hold the copy"
    # A weak reference the state holds itself comes back with it (pandas'
    # between the blocks of a frame).
    refs = [ref for item in weakly for ref in weakref.getweakrefs(item) if id(ref) not in parts]
    held = _held_by_the_notebook(parts, refs, namespace or {})
    if held == _WEAK:
        return "a weak reference points at an object it holds, and a restore would hand it a copy"
    if held == _BY_ID:
        return "a container keeps the id() of an object it holds, and a restore would hand it a copy"
    if held == _TOO_MANY:
        return (
            "the notebook holds containers too big to look through for a weak reference to, or the id() "
            "of, an object it holds"
        )
    return None


def restorable_kind(obj: Any) -> bool:
    """Whether *obj* is of a kind a restore can go into: a list, dict or
    set, or an object whose ``__dict__`` is its state."""
    kind = type(obj)
    return kind in _CONTAINERS or _default_object(kind)


def bound_as_a_copy(obj: Any, namespace: Mapping[str, Any]) -> bool:
    """Whether *obj*, which a loop only assigns into, may be bound to the
    stored copy as any output is (``arr[i] = v`` on a NumPy array): it is of
    a kind no restore goes into, not a list, dict or set of a subclass (whose
    own ``__setitem__`` a copy would call), and nothing holds it by identity,
    a weak reference or its ``id()``."""
    if restorable_kind(obj) or isinstance(obj, (list, dict, set)):
        return False
    try:
        refs = weakref.getweakrefs(obj)
    except TypeError:
        refs = []
    if refs:
        # One *obj* holds itself comes back with the copy (pandas keeps one
        # to the frame in its ``_flags``).
        inside: set[int] = set()
        if _reaches(gc.get_referents(obj), _NOTHING, seen=inside) is not None:
            refs = [ref for ref in refs if id(ref) not in inside]
    return _held_by_the_notebook({id(obj)}, refs, namespace) is None


#: A target `_reaches` never finds: to collect what an object holds.
_NOTHING = object()


#: Packages whose objects `_held_by_the_notebook` does not look into: the
#: shell's ``exit`` reaches the shell, and through it cash's own records,
#: which key objects by their ``id()``.
_NOT_THE_NOTEBOOKS = frozenset({"IPython", "ipykernel", "jupyter_client", "traitlets", "zmq", "tornado", "cash"})

#: What `_held_by_the_notebook` finds.
_WEAK = "weak"
_BY_ID = "by id"
_TOO_MANY = "too many"
#: What a container too big to look inside may hold and keep nothing else.
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


def _held_by_the_notebook(parts: set[int], refs: list[Any], namespace: Mapping[str, Any]) -> str | None:
    """`_WEAK` when the notebook holds one of *refs*, the weak references to
    an object of *parts* (by ``id()``): in a variable or a container of
    *namespace* at any depth (dicts, lists, tuples, sets, an object's
    ``__dict__``, so a ``WeakValueDictionary`` too, but not an object of
    IPython's or cash's), or as a ``weakref.finalize``. Those cash keeps
    itself are not the notebook's.
    `_BY_ID` when one of those keeps the ``id()`` of one of *parts* as a
    key, a value or an item: ``reg = {id(p.seen): ...}``, ``{'seen':
    id(p.seen)}``, ``ids = [id(p.seen)]``. Else None.

    A number that only happens to equal an ``id()`` runs the loop again,
    which is always right. What a container of more than `_MAX_SCANNED`
    items holds is checked at C speed but not looked inside (walking each
    item of a list of a million tokens cost every restore 60 ms): one that
    holds anything but numbers, strings and the like gives `_TOO_MANY`, as
    do more objects than `_MAX_SEEN`, and the loop runs again."""
    if any(isinstance(attribute_or(ref, "__callback__"), weakref.finalize) for ref in refs):
        return _WEAK
    ref_ids = {id(ref) for ref in refs}
    budget = _MAX_SEEN
    seen: set[int] = set()
    level = list(namespace.values())
    if parts.intersection(v for v in level if type(v) is int):
        return _BY_ID
    while level:
        if ref_ids and any(id(v) in ref_ids for v in level):
            return _WEAK
        nxt: list[Any] = []
        for value in level:
            kind = type(value)
            if kind in (dict, list, tuple, set, frozenset):
                items = value
            elif (
                isinstance(value, _OPAQUE)
                or type(kind.__module__) is not str
                or (kind.__module__.partition(".")[0] in _NOT_THE_NOTEBOOKS)
            ):
                continue
            else:
                state = attribute_or(value, "__dict__")
                if type(state) is not dict:
                    continue
                items = state
            if id(value) in seen:
                continue
            seen.add(id(value))
            # A dict's keys and values both: ``{id(p.seen): ...}``,
            # ``{'seen': id(p.seen)}``, ``{weakref.ref(p.child): ...}``.
            held = (*items, *items.values()) if isinstance(items, dict) else items
            try:
                if not parts.isdisjoint(held):
                    return _BY_ID
            except TypeError:  # an unhashable item: one at a time
                if any(type(item) is int and item in parts for item in held):
                    return _BY_ID
            if len(held) > _MAX_SCANNED:
                # Not looked inside: only a container of numbers, strings
                # and the like is known to keep nothing else.
                if not _LEAVES.issuperset(map(type, held)):
                    return _TOO_MANY
                continue
            budget -= len(held) + 1
            if budget < 0:
                return _TOO_MANY
            nxt.extend(held)
        level = [v for v in nxt if type(v) not in IMMUTABLE_LEAF_TYPES]
    return None


def holder_refusal(name: str, obj: Any, values: Mapping[str, Any]) -> str | None:
    """Why the other *values* (outputs and holders, by name) stop *obj*,
    stored as *name*, from being restored into: one holds it inside, where
    a restore would put the copy. A value that is *obj* itself is bound to
    the live object again."""
    others = [value for other, value in values.items() if other != name and value is not obj]
    if not others:
        return None
    found = _reaches(others, obj)
    if found is None:
        return f"too many objects hold '{name}' to check them"
    if found:
        return f"another variable or container holds '{name}', and a restore would hand it a copy"
    return None


def _state(obj: Any) -> list[Any]:
    if type(obj) is dict:
        return [*obj.keys(), *obj.values()]
    if type(obj) in (list, set):
        return list(obj)
    return [obj.__dict__]


def _reaches(
    roots: Iterable[Any], target: Any, seen: set[int] | None = None, weakly: list[Any] | None = None
) -> bool | None:
    """Whether *target* is among *roots* or what they hold; None past `_MAX_SEEN`.
    *seen* gets the ``id()`` of each object looked at, and *weakly* each one
    a weak reference points at."""
    module_dicts = {id(vars(m)) for m in list(sys.modules.values()) if isinstance(m, types.ModuleType)}
    if seen is None:
        seen = set()
    level = [item for item in roots if type(item) not in IMMUTABLE_LEAF_TYPES]
    while level:
        nxt: list[Any] = []
        for item in level:
            if item is target:
                return True
            key = id(item)
            if key in seen or key in module_dicts or isinstance(item, _OPAQUE) or is_live_stream(item):
                # A standard stream comes back as itself (`live_state`).
                continue
            seen.add(key)
            if weakly is not None and weakref.getweakrefcount(item):
                weakly.append(item)
            if len(seen) > _MAX_SEEN:
                return None
            nxt.extend(ref for ref in gc.get_referents(item) if type(ref) not in IMMUTABLE_LEAF_TYPES)
        level = nxt
    return False


def ready(restored: Mapping[str, Any], names: Iterable[str], user_ns: Mapping[str, Any]) -> bool:
    """Whether each of *names* is bound to an object of the kind stored, so
    the restore can go into it."""
    return all(name in restored and type(user_ns.get(name)) is type(restored[name]) for name in names)


def restore_into(live: Any, restored: Any) -> None:
    """Give *live* the state of *restored*, an object of its kind (`ready`)."""
    kind = type(live)
    if kind is list:
        live[:] = restored
    elif kind is dict or kind is set:
        live.clear()
        live.update(restored)
    else:
        state = live.__dict__
        state.clear()
        state.update(restored.__dict__)
