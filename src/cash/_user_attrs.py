"""Reading an attribute of a notebook's object without its errors becoming cash's."""

from __future__ import annotations

import collections
import types
from typing import Any

from ._type_memo import mro_of, TypeMemo, type_key

_address = object.__hash__


def attribute_or(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr(obj, name, default)``, with any error the lookup raises taken
    as the attribute missing.

    Python's own default covers AttributeError only. A class whose
    ``__getattr__`` raises KeyError for a name it does not hold (a record
    reading its fields from a dict) failed cash's own question about it --
    ``fit``, ``__code__`` -- and the cell with it, where plain Python never
    asks. The question is cash's, so the answer is: not there.

    ``__code__`` is there only as a code object: a config class whose
    ``__getattr__`` answers every name (``self._d.get(name, 0)``) hands back
    ``0``, which cash's walk of what a function runs read as code and failed
    the cell with.
    """
    kind = type(obj)
    own = _OWN_LOOKUP.get(type_key(kind))
    if own is None:
        own = _own_lookup(kind)
    try:
        found = object.__getattribute__(obj, name) if own else getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return default
    if name == "__code__" and type(found) is not types.CodeType:
        return default
    return found


#: Per type, `_own_lookup`'s answer, held weakly: a class made per call
#: (``df.itertuples()``) is not kept alive.
_OWN_LOOKUP: TypeMemo[bool] = TypeMemo()


def _own_lookup(kind: type) -> bool:
    """Does *kind* look its attributes up with a ``__getattribute__`` of
    its own, written in Python? cash's questions about such an object
    (``fit``, ``__code__``, the ``__class__`` an ``isinstance`` asks for
    when the type does not match) ran it, where plain Python never asks:
    they are answered by Python's generic lookup and the real type."""
    own = _OWN_LOOKUP.get(type_key(kind))
    if own is None:
        own = _OWN_LOOKUP.put(kind, type(kind.__getattribute__) is types.FunctionType)
    return own


_MISSING = object()


def has_attribute(obj: Any, name: str) -> bool:
    """``hasattr(obj, name)``, with any error the lookup raises taken as the
    attribute missing (`attribute_or`)."""
    return attribute_or(obj, name, _MISSING) is not _MISSING


def instance_of(obj: Any, kinds: type | tuple[type, ...]) -> bool:
    """``isinstance(obj, kinds)``, with any error the object's own lookups
    raise answered by its real type.

    When the type does not match, ``isinstance`` asks the object for its
    ``__class__`` (a proxy's way to pass for what it wraps), and so does an
    ABC's check. An object whose ``__getattribute__`` raises for it failed
    cash's own question and the cell with it, where plain Python never asks.
    An object with a ``__getattribute__`` of its own in Python is not asked
    at all (`_own_lookup`): its real type answers, so a proxy written that
    way passes for itself.
    """
    kind = type(obj)
    own = _OWN_LOOKUP.get(type_key(kind))
    if own is None:
        own = _own_lookup(kind)
    if own:
        return issubclass(kind, kinds)
    try:
        return isinstance(obj, kinds)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return issubclass(kind, kinds)


def is_routine(obj: Any) -> bool:
    """``inspect.isroutine(obj)``, with any error the object's own lookups
    raise (`instance_of`) taken as not a routine: a function, method or
    builtin is not an object whose lookups raise."""
    import inspect

    if _own_lookup(type(obj)):
        return False
    try:
        return inspect.isroutine(obj)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return False


#: What Python's own ``__dict__`` slot is, on a class or a module.
_SLOT_READERS = (types.GetSetDescriptorType, types.MemberDescriptorType)
#: Per type, whether the ``__dict__`` its MRO finds first is the slot
#: Python gives an instance (`own_dict`), held weakly.
_DICT_READERS: TypeMemo[bool] = TypeMemo()


def own_dict(obj: Any) -> dict | None:
    """The attribute dict Python keeps for *obj*, read without running code
    of the object's, or None when it has none.

    ``obj.__dict__`` runs a ``__dict__`` the class defines itself (a
    property computing it), and a lazy tree's makes a new child at each
    read: a walk over what an object holds read one more each time, and
    never ended, where plain Python never asks. The slot Python gives an
    instance is read instead; a class whose ``__dict__`` is its own, not
    that slot, has none that cash reads.
    """
    kind = type(obj)
    slot = _DICT_READERS.get(type_key(kind))
    if slot is None:
        slot = False
        for klass in kind.__mro__:
            found = klass.__dict__.get("__dict__")
            if found is not None:
                slot = type(found) in _SLOT_READERS
                break
        _DICT_READERS.put(kind, slot)
    if not slot:
        return None
    try:
        # The slot found first along the MRO is a data descriptor, so this
        # reads through it: not kept, as it refers to its class.
        attributes = object.__getattribute__(obj, "__dict__")
    except Exception:  # noqa: BLE001 - the object's own slot, not cash's error
        return None
    return attributes if type(attributes) is dict else None


#: The builtin containers a class of the user's can derive from.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset, collections.deque)


def builtin_base(value: Any) -> type | None:
    """The builtin container *value*'s real type is or derives from, or None.

    Decided by the real type, not by ``isinstance``: a proxy
    (``wrapt.ObjectProxy({...})``, ``MagicMock(spec=dict)``, a class whose
    ``__class__`` says ``dict``) passes ``isinstance`` for what it claims to
    be but holds no storage of that type. cash does not open it as one; it
    is read as what it really is."""
    kind = type(value)
    if kind in _BUILTIN_CONTAINERS:
        return kind
    if not issubclass(kind, _BUILTIN_CONTAINERS):
        return None
    found = _DERIVED_BASE.get(type_key(kind))
    if found is not None and found[0] == _address(mro_of(kind)):
        return found[1]
    return _derived_base(kind)


#: Per subclass, ``(its MRO's address, the builtin container it derives
#: from)``: read along its MRO once, not at every object a walk meets.
_DERIVED_BASE: TypeMemo[tuple[int, type]] = TypeMemo()


def _derived_base(kind: type) -> type:
    mro = mro_of(kind)
    base = next(c for c in mro if c in _BUILTIN_CONTAINERS)
    _DERIVED_BASE.put(kind, (_address(mro), base))
    return base


def builtin_items(value: Any) -> list[Any] | None:
    """What the builtin container *value* is or derives from holds: a dict's
    keys then values, the items of a list, tuple, set, frozenset or deque;
    None when its real type derives from none of them (`builtin_base`).

    Read through the builtin type's own methods, not the class's: a dict
    subclass whose ``values()`` raises, or a list subclass whose
    ``__iter__`` does, failed cash's look into it, and the cell with it,
    where plain Python never asked."""
    base = type(value)
    if base not in _BUILTIN_CONTAINERS:
        base = builtin_base(value)
        if base is None:
            return None
    if base is dict:
        return [*dict.keys(value), *dict.values(value)]
    return list(base.__iter__(value))
