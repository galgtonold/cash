"""Reading an attribute of a notebook's object without its errors becoming cash's."""

from __future__ import annotations

import types
from typing import Any

from ._memo import ByClass


def attribute_or(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr(obj, name, default)``, with any error the lookup raises taken
    as the attribute missing.

    Python's own default covers AttributeError only. A class whose
    ``__getattr__`` raises KeyError for a name it does not hold (a record
    reading its fields from a dict) failed cash's own question about it --
    ``fit``, ``__code__`` -- and the cell with it, where plain Python never
    asks. The question is cash's, so the answer is: not there.
    """
    kind = type(obj)
    own = _OWN_KNOWN.get(id(kind))
    if own is None:
        own = _own_lookup(kind)
    try:
        if own:
            return object.__getattribute__(obj, name)
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return default


#: Per type, `_own_lookup`'s answer, gone with the type. Dropped whole
#: past `_OWN_LOOKUP_MAX` types as well.
_OWN_LOOKUP = ByClass()
_OWN_KNOWN = _OWN_LOOKUP.data
_OWN_LOOKUP_MAX = 4096


def _own_lookup(kind: type) -> bool:
    """Does *kind* look its attributes up with a ``__getattribute__`` of
    its own, written in Python? cash's questions about such an object
    (``fit``, ``__code__``, the ``__class__`` an ``isinstance`` asks for
    when the type does not match) ran it, where plain Python never asks:
    they are answered by Python's generic lookup and the real type."""
    own = _OWN_KNOWN.get(id(kind))
    if own is None:
        if len(_OWN_LOOKUP) >= _OWN_LOOKUP_MAX:
            _OWN_LOOKUP.clear()
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
    own = _OWN_KNOWN.get(id(kind))
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
#: Per type, whether an instance's own attribute dict is the slot Python
#: gives it (`own_dict`). Not the slot's descriptor: it refers to its class,
#: which would never go.
_DICT_READERS = ByClass()
_DICT_SLOTS = _DICT_READERS.data


def own_dict(obj: Any, subclasses: bool = False) -> dict | None:
    """The attribute dict Python keeps for *obj*, read without running code
    of the object's, or None when it has none.

    ``obj.__dict__`` runs a ``__dict__`` the class defines itself (a
    property computing it), and a lazy tree's makes a new child at each
    read: a walk over what an object holds read one more each time, and
    never ended, where plain Python never asks. The slot Python gives an
    instance is read instead; a class whose ``__dict__`` is its own, not
    that slot, has none that cash reads.

    A ``dict`` subclass set as the slot (``self.__dict__ = Attrs(...)``)
    is given only with *subclasses*: read it with ``dict``'s own methods
    (``dict.values(own)``), so none of the subclass's runs.
    """
    kind = type(obj)
    slot = _DICT_SLOTS.get(id(kind))
    if slot is None:
        slot = False
        # Through ``type``'s own descriptors: a metaclass's hook is not asked.
        for klass in type.__dict__["__mro__"].__get__(kind):
            found = type.__dict__["__dict__"].__get__(klass).get("__dict__")
            if found is not None:
                slot = type(found) in _SLOT_READERS
                break
        _DICT_READERS.put(kind, slot)
    if not slot:
        return None
    try:
        # The type's first ``__dict__`` is that slot, a data descriptor:
        # the generic lookup calls it and nothing of the object's.
        attributes = object.__getattribute__(obj, "__dict__")
    except Exception:  # noqa: BLE001 - the object's own slot, not cash's error
        return None
    if type(attributes) is dict or (subclasses and isinstance(attributes, dict)):
        return attributes
    return None
