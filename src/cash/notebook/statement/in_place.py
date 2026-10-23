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
name (`holder_refusal`). A name bound to the object itself is fine: it is
bound to the live object again.
"""

from __future__ import annotations

import gc
import sys
import types
from collections.abc import Iterable, Mapping
from typing import Any

from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["holder_refusal", "ready", "refusal", "restore_into"]

#: Containers restored by their contents.
_CONTAINERS = (list, dict, set)
#: Most objects one check looks through: past it the loop runs again.
_MAX_SEEN = 200_000
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


def refusal(obj: Any) -> str | None:
    """Why *obj* cannot be restored into, or None."""
    kind = type(obj)
    if kind not in _CONTAINERS and not _default_object(kind):
        return f"a {kind.__name__} is not restored into the object it is, and a new one would not be that object"
    found = _reaches(_state(obj), obj)
    if found is None:
        return "it holds too many objects to check that it does not hold itself"
    if found:
        return "it holds itself, and a copy would hold the copy"
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


def _reaches(roots: Iterable[Any], target: Any) -> bool | None:
    """Whether *target* is among *roots* or what they hold; None past `_MAX_SEEN`."""
    module_dicts = {id(vars(m)) for m in list(sys.modules.values()) if isinstance(m, types.ModuleType)}
    seen: set[int] = set()
    level = [item for item in roots if type(item) not in IMMUTABLE_LEAF_TYPES]
    while level:
        nxt: list[Any] = []
        for item in level:
            if item is target:
                return True
            key = id(item)
            if key in seen or key in module_dicts or isinstance(item, _OPAQUE):
                continue
            seen.add(key)
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
