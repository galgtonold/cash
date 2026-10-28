"""A value that finds another object by its identity, and so cannot be
restored on its own.

``wr = weakref.ref(p.child)`` and ``reg = {id(p.seen): 'seen'}`` are bound to
what they point at by identity, not by holding it. Restored from the cache,
such a value points at the object as it was when it was stored, while the
statement that binds ``p`` runs (or is restored) and binds a new one: the
weak reference finds nothing, or the old object, and the ``id()`` key finds
nothing, where plain Python finds ``p``'s part. Copied with the cache's RAM
tier, a weak reference even keeps pointing at the old object itself. So a
statement whose value holds either runs again, as a plain run does.

What is looked at:

- a weak reference (``weakref.ref``, a ``WeakValueDictionary``'s or a
  ``WeakSet``'s, a proxy) held anywhere in the value, whose object is alive
  and not inside the value itself (a restore rebuilds those together); a
  class, a module or a function comes back as itself and does not count;
- an ``int`` in the value equal to the ``id()`` of an object the value holds
  or the notebook's variables hold, at any depth.

The value is walked through lists, tuples, dicts, sets and the ``__dict__``
of objects of the user's own classes, not into a library's objects (a
pandas frame keeps weak references between its own parts). A container of
more than `_MAX_SCANNED` items is not looked inside, and the walk stops at
`_MAX_SEEN` objects: a registry of ids is small, and walking a big value
item by item cost each store more than the check is worth.
"""

from __future__ import annotations

import sys
import types
import weakref
from typing import Any

from cash._user_attrs import attribute_or
from cash.install_paths import is_user_code_module

__all__ = ["IdentityHeld", "identity_held_reason"]

#: A container with more items is not looked inside.
_MAX_SCANNED = 10_000
#: Most objects one walk looks at.
_MAX_SEEN = 200_000
#: Restored as themselves, by reference: a weak reference to one stays right.
_BY_REFERENCE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType)
_REFERENCES = (weakref.ReferenceType, weakref.ProxyType, weakref.CallableProxyType)
_CONTAINERS = (list, tuple, dict, set, frozenset)
#: Hold their weak references in ``.data``.
_WEAK_CONTAINERS = (weakref.WeakValueDictionary, weakref.WeakKeyDictionary, weakref.WeakSet)
#: The smallest ``id()`` looked for: an object's address, aligned to 8.
_LOWEST_ADDRESS = 1 << 16


class IdentityHeld(str):
    """The refusal `identity_held_reason` gives: the reason, with the
    variable it names and how its value finds the object (*kind*: ``"a weak
    reference"`` or ``"an id()"``), for the warning (``NOTEBOOK-IDENTITY-HELD``)."""

    name: str
    kind: str

    def __new__(cls, reason: str, name: str, kind: str) -> IdentityHeld:
        made = super().__new__(cls, reason)
        made.name, made.kind = name, kind
        return made


def identity_held_reason(name: str, value: Any, user_ns: dict[str, Any]) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored apart from what it
    points at by identity (see the module docstring); None when it can."""
    user_kinds: dict[type, bool] = {}
    seen: set[int] = set()
    refs: list[Any] = []
    numbers: set[int] = set()
    _walk([value], seen, user_kinds, refs, numbers)
    for ref in refs:
        if not issubclass(type(ref), weakref.ReferenceType):
            if _alive(ref):
                return IdentityHeld(
                    f"'{name}' holds a weak proxy; restored, it would point at another object, so the statement "
                    "re-runs every time",
                    name,
                    "a weak reference",
                )
            continue
        target = ref()
        if target is not None and id(target) not in seen and not issubclass(type(target), _BY_REFERENCE):
            return IdentityHeld(
                f"'{name}' holds a weak reference to an object it does not hold; restored, it would "
                "point at another object, so the statement re-runs every time",
                name,
                "a weak reference",
            )
    candidates = {n for n in numbers if n >= _LOWEST_ADDRESS and not n % 8}
    if not candidates:
        return None
    held = set(seen)
    from .output_refusals import is_history_name

    roots = [v for k, v in user_ns.items() if k != name and not is_history_name(k)]
    if not _walk(roots, held, user_kinds, [], set()) or not candidates.isdisjoint(held):
        return IdentityHeld(
            f"'{name}' holds the id() of an object the notebook holds; restored, the object would be "
            "another one, so the statement re-runs every time",
            name,
            "an id()",
        )
    return None


def _alive(proxy: Any) -> bool:
    """Whether the object a weak proxy points at is alive."""
    try:
        proxy.__class__  # raises once the object is gone
    except ReferenceError:
        return False
    except Exception:  # noqa: BLE001 - the object's own lookup raised: it is there
        return True
    return True


def _walk(roots: list[Any], seen: set[int], user_kinds: dict[type, bool], refs: list[Any], numbers: set[int]) -> bool:
    """Look through *roots* (see the module docstring), adding the ``id()``
    of each object to *seen*, each weak reference to *refs* and each
    ``int`` to *numbers*. False when it stopped at `_MAX_SEEN`."""
    level = roots
    while level:
        nxt: list[Any] = []
        for item in level:
            kind = type(item)
            if kind is int:
                numbers.add(item)
                continue
            if id(item) in seen:
                continue
            if issubclass(kind, _REFERENCES):
                seen.add(id(item))
                refs.append(item)
                continue
            if kind in _CONTAINERS or issubclass(kind, (list, dict, set)):
                items: Any = list(item.keys()) + list(item.values()) if issubclass(kind, dict) else item
            elif issubclass(kind, _WEAK_CONTAINERS):
                data = attribute_or(item, "data")
                items = (
                    list(data.keys()) + list(data.values())
                    if type(data) is dict
                    else list(data)
                    if type(data) is set
                    else []
                )
            elif _users(kind, user_kinds):
                state = attribute_or(item, "__dict__")
                if type(state) is not dict:
                    continue
                items = list(state.values())
            else:
                seen.add(id(item))
                continue
            seen.add(id(item))
            if len(seen) > _MAX_SEEN:
                return False
            if len(items) > _MAX_SCANNED:
                continue
            nxt.extend(items)
        level = nxt
    return True


def _users(kind: type, user_kinds: dict[type, bool]) -> bool:
    """Whether *kind* is a class of the user's code."""
    known = user_kinds.get(kind)
    if known is None:
        module = sys.modules.get(attribute_or(kind, "__module__") or "")
        known = user_kinds[kind] = module is not None and is_user_code_module(module)
    return known
