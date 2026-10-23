"""Reading an attribute of a notebook's object without its errors becoming cash's."""

from __future__ import annotations

import abc
import sys
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
    kind = type(obj)
    own = _OWN_LOOKUP.get(kind)
    if own is None:
        own = own_lookup(kind)
    try:
        if own:
            return object.__getattribute__(obj, name)
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return default


#: Per type, `own_lookup`'s answer. Dropped whole past `_OWN_LOOKUP_MAX`
#: types: a class made per call (``df.itertuples()``) is not kept alive.
_OWN_LOOKUP: dict[type, bool] = {}
_OWN_LOOKUP_MAX = 4096


def own_lookup(kind: type) -> bool:
    """Does *kind* look its attributes up with a ``__getattribute__`` of
    its own, written in Python? cash's questions about such an object
    (``fit``, ``__code__``, the ``__class__`` an ``isinstance`` asks for
    when the type does not match) ran it, where plain Python never asks:
    they are answered by Python's generic lookup and the real type."""
    own = _OWN_LOOKUP.get(kind)
    if own is None:
        if len(_OWN_LOOKUP) >= _OWN_LOOKUP_MAX:
            _OWN_LOOKUP.clear()
        own = _OWN_LOOKUP[kind] = type(kind.__getattribute__) is types.FunctionType
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
    at all (`own_lookup`): its real type answers, so a proxy written that
    way passes for itself, and none passes for a function, class, module
    or array by its ``__class__`` alone (`_passable`).
    """
    kind = type(obj)
    passable = _PASSABLE.get(kinds)
    if passable is None:
        passable = _passable(kinds)
    if passable[1]:
        # Kinds that answer ``isinstance`` as ``issubclass`` of ``__class__``.
        if issubclass(kind, kinds):
            return True
        plain = _PLAIN_CLASS.get(kind)
        if plain is None:
            plain = _plain_class(kind)
        if plain:
            # Its ``__class__`` is its type: ``isinstance`` would answer as above.
            return False
    elif own_lookup(kind):
        return issubclass(kind, kinds)
    try:
        # Not by its real type: by the ``__class__`` it answers, but never
        # for a kind whose parts cash reads off the object itself.
        if isinstance(obj, passable[0]):
            return True
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        pass
    if passable[1]:
        return False
    try:
        return issubclass(kind, kinds)
    except TypeError:  # a protocol with data members takes no subclass check
        return False


#: Per type, `_plain_class`'s answer.
_PLAIN_CLASS: dict[type, bool] = {}


def _plain_class(kind: type) -> bool:
    """Do *kind*'s instances answer their real type for ``__class__``, and
    look attributes up as Python does (`own_lookup`)? Then ``isinstance``
    is ``issubclass`` of the type, and the object is never asked."""
    if len(_PLAIN_CLASS) >= _OWN_LOOKUP_MAX:
        _PLAIN_CLASS.clear()
    plain = _PLAIN_CLASS[kind] = own_lookup(kind) or not any(
        "__class__" in vars(klass) for klass in kind.__mro__ if klass is not object
    )
    return plain


def _passable(kinds: type | tuple[type, ...]) -> tuple[tuple[type, ...], bool]:
    """*kinds* without the ones an object passes for only by being one: a
    function, method, code object, cell, frame, module, class or numpy
    array. cash reads their parts straight off the object (a function's
    ``__closure__``, a class's ``__qualname__``, an array's ``.base``), and
    one whose ``__class__`` answers ``FunctionType`` has none of them: the
    read failed the cell. A proxy that passes for one is read as the object
    it is. And whether every kind checks an instance as ``type`` and
    ``ABCMeta`` do, by its ``__class__``'s subclass check."""
    found = _PASSABLE.get(kinds)
    if found is None:
        if len(_PASSABLE) >= _OWN_LOOKUP_MAX:
            _PASSABLE.clear()
        np = sys.modules.get("numpy")
        layouts = (*_LAYOUT_KINDS, np.ndarray) if np is not None else _LAYOUT_KINDS
        every = _flat(kinds)
        found = (
            tuple(k for k in every if not issubclass(k, layouts)),
            all(type(k) in _PLAIN_METACLASSES for k in every),
        )
        _PASSABLE[kinds] = found
    return found


def _flat(kinds: Any) -> tuple[type, ...]:
    """*kinds* as ``isinstance`` reads them: tuples within tuples, and the
    members of a ``X | Y``, as one tuple of classes."""
    if isinstance(kinds, tuple):
        return tuple(kind for part in kinds for kind in _flat(part))
    if isinstance(kinds, types.UnionType):
        return _flat(kinds.__args__)
    return (kinds,)


_PLAIN_METACLASSES = (type, abc.ABCMeta)


#: Kinds whose parts cash reads off the object (`_passable`).
_LAYOUT_KINDS = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.CodeType,
    types.CellType,
    types.FrameType,
    types.ModuleType,
    type,
)
#: Per kinds, `_passable`'s answer.
_PASSABLE: dict[Any, tuple[tuple[type, ...], bool]] = {}


def is_routine(obj: Any) -> bool:
    """``inspect.isroutine(obj)``, with any error the object's own lookups
    raise (`instance_of`) taken as not a routine: a function, method or
    builtin is not an object whose lookups raise."""
    import inspect

    if own_lookup(type(obj)):
        return False
    try:
        return inspect.isroutine(obj)
    except Exception:  # noqa: BLE001 - the object's own error, not one of cash's
        return False


#: What Python's own ``__dict__`` slot is, on a class or a module.
_SLOT_READERS = (types.GetSetDescriptorType, types.MemberDescriptorType)
#: Per type, how to read an instance's own attribute dict (`own_dict`).
_DICT_READERS: dict[type, Any] = {}


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
    reader = _DICT_READERS.get(kind)
    if reader is None:
        reader = False
        # Through ``type``'s own descriptors: a metaclass's hook is not asked.
        for klass in type.__dict__["__mro__"].__get__(kind):
            found = type.__dict__["__dict__"].__get__(klass).get("__dict__")
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
    if type(attributes) is dict or (subclasses and isinstance(attributes, dict)):
        return attributes
    return None
