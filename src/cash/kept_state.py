"""Pickling that keeps the attributes a C base class's reduce drops.

``pickle`` stores an object as its ``__reduce_ex__`` describes it. A class
written in C that defines its own reduce -- ``numpy.ndarray``,
``datetime.date`` -- describes its own data and nothing else, so a Python
subclass's instance attributes are lost: numpy's own ``InfoArray`` example
pickles without its ``info``. Two such arrays with equal data and a different
``info`` keyed alike, and a stored one came back without it.

`dumps` pickles such an instance with its ``__dict__`` added to the state it
restores (`restore_with_dict`), and everything else exactly as ``pickle``
does. It is used wherever cash pickles a value it keys or stores.
"""

from __future__ import annotations

import copyreg
import functools
import io
import operator
import pickle
import sys
import types
import weakref
from typing import Any

__all__ = ["chooses_its_state", "dumps", "left_out_attrs", "restore_with_dict"]


def dumps(
    value: Any,
    protocol: int = pickle.DEFAULT_PROTOCOL,
    *,
    fast: bool = False,
    buffer_callback: Any = None,
    extra: dict | None = None,
    persistent_id: Any = None,
    keyed: bool = False,
) -> bytes:
    """``pickle.dumps(value, protocol)``, keeping the instance attributes a C
    base's reduce drops. *fast* is the pickler's memo-less mode; *extra*
    are reducers by type that come first (a clock test double's);
    *persistent_id* is the pickler's hook of that name. *keyed*
    bytes are for hashing, never loaded: an instance of the user's own class
    also keeps the attributes its own reduce leaves out (`_reduce_for_key`)."""
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=protocol, buffer_callback=buffer_callback)
    pickler.fast = fast
    if persistent_id is not None:
        pickler.persistent_id = persistent_id
    pickler.dispatch_table = _KeepDict(protocol, extra, keyed)
    pickler.dump(value)
    return buf.getvalue()


class _KeepDict(dict):
    """A pickler's ``dispatch_table``: each type's reducer, decided at its
    first instance. The ordinary ``__reduce_ex__`` for most, so the stream
    is the one ``pickle.dumps`` writes; `_reduce_with_dict` for a type whose
    reduce drops its instances' ``__dict__`` (`_drops_dict`)."""

    def __init__(self, protocol: int, extra: dict | None, keyed: bool = False) -> None:
        super().__init__(extra or ())
        self._protocol = protocol
        self._keyed = keyed

    def __missing__(self, t: type) -> Any:
        if issubclass(t, type):
            raise KeyError(t)  # a class: pickled by name, as pickle does
        registered = copyreg.dispatch_table.get(t)
        if self._keyed and (chooses_its_state(t) or (registered is not None and _is_users_class(t))):
            # A user's class whose state its own reduce -- or one registered
            # with ``copyreg.pickle`` -- decides: key it with what it leaves out.
            reducer = functools.partial(_reduce_for_key, protocol=self._protocol, registered=registered)
        elif registered is not None:
            reducer = registered
        else:
            if _drops_dict(t):
                reducer = functools.partial(_reduce_with_dict, protocol=self._protocol)
            else:
                reducer = operator.methodcaller("__reduce_ex__", self._protocol)
        self[t] = reducer
        return reducer


def _drops_dict(t: type) -> bool:
    """Does pickling a *t* instance lose its ``__dict__``?

    When *t*'s reduce is a C type's and that C type has no instance dict of
    its own to know about: the dict is the Python subclass's. Remembered
    per type.
    """
    try:
        return _DROPS_DICT[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_drops_dict(t)
    found = _DROPS_DICT[t] = _decide_drops_dict(t)
    return found


def _decide_drops_dict(t: type) -> bool:
    if not t.__dictoffset__:
        return False
    for klass in t.__mro__:
        own = klass.__dict__
        method = own.get("__reduce_ex__") or own.get("__reduce__")
        if method is None:
            continue
        if klass is object or isinstance(method, _PYTHON_METHODS):
            return False  # object's reduce keeps the dict; a Python one is the class's own say
        return not klass.__dictoffset__
    return False


_PYTHON_METHODS = (types.FunctionType, classmethod, staticmethod)

_DROPS_DICT: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _reduce_with_dict(obj: Any, protocol: int) -> Any:
    """*obj*'s own reduce, with its ``__dict__`` restored after its state."""
    reduced = obj.__reduce_ex__(protocol)
    attrs = getattr(obj, "__dict__", None)
    if not attrs or isinstance(reduced, str):
        return reduced
    rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    state, listitems, dictitems, setter = rest[:4]
    return rebuild, args, (state, setter, dict(attrs)), listitems, dictitems, restore_with_dict


def chooses_its_state(t: type) -> bool:
    """Is *t* a class of the user's whose own Python ``__getstate__``,
    ``__reduce__`` or ``__reduce_ex__`` decides what pickle stores?

    Such a class may leave a setting out of its saved state (a precision, a
    device, a threshold) that its methods still read, so a key of what it
    pickles alone served one setting the other's result. A library's class
    is left to say what its state is. Remembered per type.
    """
    try:
        return _CHOOSES_STATE[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_chooses_state(t)
    found = _CHOOSES_STATE[t] = _decide_chooses_state(t)
    return found


def _decide_chooses_state(t: type) -> bool:
    for klass in t.__mro__:
        if klass is object:
            return False
        own = klass.__dict__
        if any(isinstance(own.get(name), _PYTHON_METHODS) for name in _STATE_METHODS):
            from .install_paths import is_user_code_module

            return is_user_code_module(sys.modules.get(klass.__module__))
    return False


_STATE_METHODS = ("__getstate__", "__reduce__", "__reduce_ex__")

_CHOOSES_STATE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _is_users_class(t: type) -> bool:
    from .install_paths import is_user_code_module

    return is_user_code_module(sys.modules.get(t.__module__))


def _reduce_for_key(obj: Any, protocol: int, registered: Any = None) -> Any:
    """*obj*'s own reduce (or the one *registered* for its type with
    ``copyreg.pickle``), with the instance attributes its state leaves out
    beside it (`chooses_its_state`). Only hashed, never loaded."""
    reduced = obj.__reduce_ex__(protocol) if registered is None else registered(obj)
    if isinstance(reduced, str):
        return reduced
    rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    left_out = left_out_attrs(obj, rest[0])
    if not left_out:
        return reduced
    return rebuild, args, ("__cash_left_out__", rest[0], left_out), *rest[1:4]


def left_out_attrs(obj: Any, state: Any) -> dict:
    """The attributes of *obj* -- in its ``__dict__`` or its ``__slots__``
    -- that *state*, what its reduce saves, does not hold: by name, or
    under their name with another value (a ``__getstate__`` that saves a
    portable default in place of the live setting).

    One that cannot be pickled (a lock or a connection the class leaves out
    for that reason) is given as its type: its content is nothing pickle can
    read.
    """
    attrs = _instance_attrs(obj)
    if not attrs:
        return {}
    saved: dict = {}
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
        if isinstance(state[0], dict):
            saved.update(state[0])
        saved.update(state[1])  # (dict, slots)
    elif isinstance(state, dict):
        saved = state
    left_out = {}
    for name, value in attrs.items():
        if name in saved and _same(saved[name], value):
            continue
        try:
            dumps(value, keyed=True)
        except Exception:  # noqa: BLE001 - whatever pickle refuses
            value = ("__cash_unpicklable__", f"{type(value).__module__}.{type(value).__qualname__}")
        left_out[name] = value
    return left_out


def _instance_attrs(obj: Any) -> dict:
    """*obj*'s ``__dict__`` and the ``__slots__`` it has set."""
    attrs = dict(getattr(obj, "__dict__", None) or ())
    for klass in type(obj).__mro__:
        slots = klass.__dict__.get("__slots__")
        if not slots:
            continue
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{klass.__name__.lstrip('_')}{name}"  # name-mangled slot
            if name in attrs:
                continue
            try:
                attrs[name] = getattr(obj, name)
            except AttributeError:  # a slot never set
                continue
    return attrs


def _same(saved: Any, value: Any) -> bool:
    """Is *saved*, the state's value under an attribute's name, the
    attribute's own *value*? Of one type and equal; when equality gives no
    plain answer (an array), the attribute is keyed beside the state."""
    if saved is value:
        return True
    if type(saved) is not type(value):
        return False
    try:
        return bool(saved == value)
    except Exception:  # noqa: BLE001 - an ambiguous or failing comparison
        return False


def restore_with_dict(obj: Any, packed: tuple) -> None:
    """Set *obj*'s state as pickle would, then its ``__dict__``
    (`_reduce_with_dict`). Named in stored entries: keep its name."""
    state, setter, attrs = packed
    if setter is not None:
        setter(obj, state)
    elif state is not None:
        setstate = getattr(obj, "__setstate__", None)
        if setstate is not None:
            setstate(state)
        else:
            slots = None
            if isinstance(state, tuple) and len(state) == 2:
                state, slots = state
            if state:
                obj.__dict__.update(state)
            for name, value in (slots or {}).items():
                setattr(obj, name, value)
    obj.__dict__.update(attrs)
