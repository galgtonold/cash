"""Shared objects held inside a stored value, found again on a hit.

A cached value comes back as a copy, and a copy of ``MISSING = object()``
is a different object: ``x is MISSING`` held on the call that computed the
value and failed on every hit. A value that IS such an object is looked up
by name (`cash.decorator.store.ResultStore.restore_identity`); this module
does the same for one held anywhere inside the value: in a list, tuple or
dict, or in an attribute of an object of the user's own class.

`find` runs when the value is stored. It walks the value a level at a time
and lists, by first appearance, every distinct object of the kinds the
caller looks for: for each, the name it is known by (``["global",
"MISSING"]``), or none when it is another object of that kind. A copy keeps
the order and the sharing (a pickle round trip writes a repeated object
once), so `put_back` walks the copy the same way, meets the copies in the
same order, and puts each named one's current object in its place.

A level at a time, not an item at a time: a million records holding the
sentinel are a few passes at C speed per level, where a walk with a Python
step per item took longer than the copy itself. Bounded: a walk gives up
past `MAX_NODES` containers or `MAX_FOUND` such objects, and a copy that
does not line up (another count, another kind) is left as it is.
"""

from __future__ import annotations

import sys
import types
import weakref
from bisect import bisect_right
from collections.abc import Callable, Mapping
from itertools import accumulate, chain, compress, islice
from typing import Any

__all__ = ["find", "put_back"]

#: Containers and objects a walk looks into before it gives up.
MAX_NODES = 5_000_000
#: Objects of the kinds looked for that one value may hold.
MAX_FOUND = 256
#: Levels a walk goes down before it gives up.
MAX_LEVELS = 64

_ATOMS = frozenset({str, int, float, bool, type(None), bytes, complex, bytearray})


class _GiveUp(Exception):
    pass


def type_name(t: type) -> str:
    """How an entry names a kind: ``module.qualname``."""
    return f"{t.__module__}.{t.__qualname__}"


def find(value: Any, named: Mapping[int, list]) -> list | None:
    """``[[kind, name, type], ...]`` for *value*: one per distinct object it
    holds of a type in *named* (``id(object) -> [kind, name]``), in walk
    order; ``kind`` and ``name`` are ``""`` for an object not in *named*.

    None when *value* holds none of the *named* objects, or the walk gave up.
    """
    if not named:
        return None
    try:
        if id(value) in named:
            held = [value]
        else:
            held = _found(_levels(value, frozenset()), lambda flat: _where(flat, map(named.__contains__, map(id, flat))))
        if not held:
            return None
        kinds = frozenset(map(type, held))
        objects = _found(_levels(value, kinds), _of_kinds(kinds), value)
    except (_GiveUp, RecursionError):
        return None
    return [[*(named.get(id(obj)) or ("", "")), type_name(type(obj))] for obj in objects]


def put_back(value: Any, entries: list, resolve: Callable[[str, str], Any]) -> Any:
    """*value* with each object `find` named replaced by what *resolve*
    (``kind, name -> object``, raising ``LookupError`` when gone) gives now.

    *value* is a copy private to the caller: its lists, dicts and objects
    are changed in place, its tuples rebuilt. Unchanged when anything does
    not line up.
    """
    try:
        current = []
        kinds: set[type] = set()
        for kind, name, recorded in entries:
            if not kind:
                current.append(None)
                continue
            obj = resolve(kind, name)
            if type_name(type(obj)) != recorded:
                return value
            current.append(obj)
            kinds.add(type(obj))
        if {type_name(k) for k in kinds} != {recorded for _k, _n, recorded in entries}:
            return value  # a kind only unnamed objects have: cannot be told apart
        frozen_kinds = frozenset(kinds)
        levels = list(_levels(value, frozen_kinds))
        copies = _found(levels, _of_kinds(frozen_kinds), value)
        if len(copies) != len(entries):
            return value
        swap = {}
        for copy, obj, (_kind, _name, recorded) in zip(copies, current, entries):
            if type_name(type(copy)) != recorded:
                return value
            if obj is not None and copy is not obj:
                swap[id(copy)] = obj
        if not swap:
            return value
        return _swap(value, levels, swap)
    except (_GiveUp, LookupError, RecursionError, TypeError, ValueError, AttributeError):
        return value


def _of_kinds(kinds: frozenset) -> Callable[[list], list]:
    return lambda flat: _where(flat, map(kinds.__contains__, map(type, flat)))


def _where(items: list, mask: Any) -> list:
    """The *items* whose *mask* entry is true, at C speed."""
    return list(compress(items, mask))


def _found(levels: Any, pick: Callable[[list], list], root: Any = None) -> list:
    """The distinct objects *pick* takes from *root* and each level's
    items, in order."""
    found: dict[int, Any] = {}
    if root is not None:
        for obj in pick([root]):
            found[id(obj)] = obj
    for _nodes, flat in levels:
        for obj in pick(flat):
            found.setdefault(id(obj), obj)
        if len(found) > MAX_FOUND:
            raise _GiveUp
    return list(found.values())


def _levels(value: Any, kinds: frozenset):
    """``(nodes, items)`` per level of *value*: the containers and objects
    the walk looks into at that level, and what they hold, in order (a
    list's or tuple's items, a dict's values, an object's attributes).

    Each container is looked into once, where the walk first meets it; an
    object of *kinds* is never looked into."""
    seen: set[int] = {id(value)}
    nodes = [value] if type(value) not in kinds and _is_node_type(type(value)) else []
    budget = MAX_NODES
    for _ in range(MAX_LEVELS):
        if not nodes:
            return
        budget -= len(nodes)
        if budget < 0:
            raise _GiveUp
        flat = _items(nodes)
        yield nodes, flat
        inner = frozenset(t for t in set(map(type, flat)) if t not in kinds and _is_node_type(t))
        if not inner:
            return
        nodes = _where(flat, map(inner.__contains__, map(type, flat)))
        ids = list(map(id, nodes))
        if len(set(ids)) == len(ids) and seen.isdisjoint(ids):
            seen.update(ids)  # each met once: no Python step per node
        else:
            nodes = [x for x in nodes if id(x) not in seen and not seen.add(id(x))]
    raise _GiveUp


def _items(nodes: list) -> list:
    """What *nodes* hold, in order: at C speed when they are all lists, all
    tuples or all dicts."""
    kinds = set(map(type, nodes))
    if len(kinds) == 1:
        (only,) = kinds
        if only is list or only is tuple:
            return list(chain.from_iterable(nodes))
        if only is dict:
            return list(chain.from_iterable(map(dict.values, nodes)))
    return list(chain.from_iterable(map(_children, nodes)))


def _children(node: Any) -> Any:
    """What the walk looks into below a node, in order."""
    if isinstance(node, (list, tuple)):
        return node
    if isinstance(node, dict):
        return node.values()
    attrs = getattr(node, "__dict__", None)
    return attrs.values() if type(attrs) is dict else ()


def _is_node_type(t: type) -> bool:
    """Does the walk look into objects of type *t*?"""
    if t is list or t is tuple or t is dict:
        return True
    if t in _ATOMS:
        return False
    try:
        return _NODE_TYPE[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return False
    if issubclass(t, list):
        found = False  # a list subclass may rebuild itself its own way
    elif issubclass(t, dict):
        found = True
    elif issubclass(t, tuple):
        found = hasattr(t, "_fields") and hasattr(t, "_make")
    elif issubclass(t, (type, types.ModuleType, types.FunctionType, types.MethodType)):
        found = False
    else:
        # A class of the user's: a library's object keeps its state its own way.
        from .install_paths import is_user_code_module

        found = is_user_code_module(sys.modules.get(t.__module__))
    _NODE_TYPE[t] = found
    return found


_NODE_TYPE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _swap(value: Any, levels: list, swap: dict[int, Any]) -> Any:
    """*value* with each item whose ``id`` is in *swap* replaced, deepest
    level first: a tuple holding one is rebuilt, and the rebuilt tuple
    replaces it a level up."""
    for nodes, flat in reversed(levels):
        positions = _where(range(len(flat)), map(swap.__contains__, map(id, flat)))
        if not positions:
            continue
        builtin = set(map(type, nodes)) <= {list, tuple, dict}
        ends = list(accumulate(map(len if builtin else _length, nodes)))
        rebuilt: dict[int, list] = {}
        for i in positions:
            n = bisect_right(ends, i)
            node = nodes[n]
            j = i - (ends[n - 1] if n else 0)
            new = swap[id(flat[i])]
            if isinstance(node, list):
                node[j] = new
            elif isinstance(node, tuple):
                rebuilt.setdefault(n, list(node))[j] = new
            else:
                target = node if isinstance(node, dict) else node.__dict__
                target[next(islice(target, j, None))] = new
        for n, items in rebuilt.items():
            node = nodes[n]
            swap[id(node)] = tuple(items) if type(node) is tuple else type(node)._make(items)
    return swap.get(id(value), value)


def _length(node: Any) -> int:
    if isinstance(node, (list, tuple, dict)):
        return len(node)
    attrs = getattr(node, "__dict__", None)
    return len(attrs) if type(attrs) is dict else 0
