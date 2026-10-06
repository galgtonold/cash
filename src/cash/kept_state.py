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
import types
import weakref
from typing import Any

__all__ = ["dumps", "restore_with_dict"]


def dumps(
    value: Any,
    protocol: int = pickle.DEFAULT_PROTOCOL,
    *,
    fast: bool = False,
    buffer_callback: Any = None,
    extra: dict | None = None,
) -> bytes:
    """``pickle.dumps(value, protocol)``, keeping the instance attributes a C
    base's reduce drops. *fast* is the pickler's memo-less mode; *extra*
    are reducers by type that come first (a clock test double's)."""
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=protocol, buffer_callback=buffer_callback)
    pickler.fast = fast
    pickler.dispatch_table = _KeepDict(protocol, extra)
    pickler.dump(value)
    return buf.getvalue()


class _KeepDict(dict):
    """A pickler's ``dispatch_table``: each type's reducer, decided at its
    first instance. The ordinary ``__reduce_ex__`` for most, so the stream
    is the one ``pickle.dumps`` writes; `_reduce_with_dict` for a type whose
    reduce drops its instances' ``__dict__`` (`_drops_dict`)."""

    def __init__(self, protocol: int, extra: dict | None) -> None:
        super().__init__(extra or ())
        self._protocol = protocol

    def __missing__(self, t: type) -> Any:
        if issubclass(t, type):
            raise KeyError(t)  # a class: pickled by name, as pickle does
        reducer = copyreg.dispatch_table.get(t)
        if reducer is None:
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
