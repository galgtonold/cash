"""Reading an attribute of a notebook's object without its errors becoming cash's."""

from __future__ import annotations

import types
from typing import Any


def attribute_or(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr(obj, name, default)``, with any error the lookup raises taken
    as the attribute missing.

    Python's own default covers AttributeError only. A class whose
    ``__getattr__`` raises KeyError for a name it does not hold (a record
    reading its fields from a dict) failed cash's own question about it --
    ``fit``, ``__code__`` -- and the cell with it, where plain Python never
    asks. The question is cash's, so the answer is: not there.
    """
    try:
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return default


_MISSING = object()


def has_attribute(obj: Any, name: str) -> bool:
    """``hasattr(obj, name)``, with any error the lookup raises taken as the
    attribute missing (`attribute_or`)."""
    return attribute_or(obj, name, _MISSING) is not _MISSING


#: What Python's own ``__dict__`` slot is, on a class or a module.
_SLOT_READERS = (types.GetSetDescriptorType, types.MemberDescriptorType)
#: Per type, how to read an instance's own attribute dict (`own_dict`).
_DICT_READERS: dict[type, Any] = {}


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
    reader = _DICT_READERS.get(kind)
    if reader is None:
        reader = False
        for klass in kind.__mro__:
            found = klass.__dict__.get("__dict__")
            if found is not None:
                reader = found if type(found) in _SLOT_READERS else False
                break
        _DICT_READERS[kind] = reader
    if reader is False:
        return None
    try:
        attributes = reader.__get__(obj, kind)
    except Exception:  # noqa: BLE001 - the object's own slot, not cash's error
        return None
    return attributes if type(attributes) is dict else None
