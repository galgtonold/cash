"""The ``_cash_lineage_hash`` tag: held beside an instance, never on it.

The notebook's lineage store tags a value it records, and the decorator and
the loop handler trust that tag as the value's identity. A tag set on a CLASS
is inherited by every instance, so reading it with ``getattr`` made every
instance of the class the same value: ``from pathlib import Path`` in a cached
cell tagged ``Path``, and every path argument to a ``@cash.cache`` function
then keyed alike -- a call on one file was served another file's result
(it first showed up as a "flaky" test in the unit suite). Classes, modules and
functions are therefore never tagged. Both paths keep their tags in the side
table below, so the user's object is never changed; `own_tag` still reads an
instance's own ``__dict__`` after it, for a value that carries such an
attribute itself.
"""

from __future__ import annotations

import inspect
import threading
import types
import weakref
from typing import Any

__all__ = ["clear_tags", "own_tag", "set_tags", "taggable"]

#: Every tag, held BESIDE the value rather than on it:
#: ``id(value) -> (weakref to value, {name: tag})``. Written onto the object,
#: ``_cash_lineage_*`` showed up in the user's own data -- ``vars(ns)``,
#: ``SimpleNamespace.__eq__``, a ``__dict__``-based ``__eq__`` or ``repr``,
#: ``json.dumps(vars(obj))``, the pickle of the result -- and a function that
#: returns its argument tagged the caller's object. Keyed by identity, never
#: by ``==`` or ``hash``, and dropped by the weakref's callback when the value
#: is collected, so a reused ``id`` never inherits a dead object's tags.
_SIDE: dict[int, tuple[weakref.ref, dict[str, Any]]] = {}
_SIDE_LOCK = threading.Lock()


def taggable(value: Any) -> bool:
    """Whether *value* may carry a lineage tag: not a class, module or function."""
    return not (isinstance(value, (type, types.ModuleType)) or inspect.isroutine(value))


def _forget(key: int, ref: weakref.ref) -> None:
    with _SIDE_LOCK:
        entry = _SIDE.get(key)
        if entry is not None and entry[0] is ref:
            del _SIDE[key]


def set_tags(value: Any, **tags: Any) -> bool:
    """Tag *value* without touching it; False if it cannot be tagged.

    What a later `own_tag` returns first. Only objects that could carry an
    attribute of their own (they have a ``__dict__``) and take a weak
    reference are tagged: a list, dict, tuple, number or string never was.
    """
    if not taggable(value):
        return False
    try:
        object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return False
    key = id(value)
    with _SIDE_LOCK:
        entry = _SIDE.get(key)
        if entry is not None and entry[0]() is value:
            entry[1].update(tags)
            return True
        try:
            ref = weakref.ref(value, lambda r, k=key: _forget(k, r))
        except TypeError:
            return False
        _SIDE[key] = (ref, dict(tags))
    return True


def clear_tags(value: Any) -> None:
    """Drop *value*'s side-table tags, so its own attributes speak again."""
    with _SIDE_LOCK:
        entry = _SIDE.get(id(value))
        if entry is not None and entry[0]() is value:
            del _SIDE[id(value)]


def own_tag(value: Any, name: str = "_cash_lineage_hash") -> Any:
    """*value*'s own *name* tag, never one inherited from its class.

    The side table (`set_tags`) first, then the instance's own attribute.
    """
    if isinstance(value, type):
        return None
    if _SIDE:
        entry = _SIDE.get(id(value))
        if entry is not None and entry[0]() is value and name in entry[1]:
            return entry[1][name]
    try:
        own = object.__getattribute__(value, "__dict__")
    except (AttributeError, TypeError):
        return None
    try:
        return own.get(name)
    except (AttributeError, TypeError):
        return None
