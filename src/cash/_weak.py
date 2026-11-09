"""The weak references cash makes, told from the user's.

A variable that finds an object by a weak reference keeps a statement out
of the cache: restored, the object would be a copy the reference does not
find (`identity_holders._weakly_found_by_others`). Cash keeps weak
references to the user's objects too -- its memos drop an entry when the
object goes -- and a plain ``weakref.ref(obj)``, or the one a stdlib
``WeakKeyDictionary`` makes, looked like the user's. Whether a statement was
stored then hung on what cash's memos held at that moment, which an earlier
statement (or test) decided.

Every weak reference cash makes goes through here: `ref` gives one whose
callback is a `_Callback` (a class of this module the user's code never
has), `finalize` is cash's own ``weakref.finalize`` class, and the
containers below are cash's own classes, whose removal callbacks name
them. `is_cash_ref` tells exactly those apart. Not by where a callback's
code is: ``weakref.ref(m, cached_fn)`` or ``weakref.ref(m,
cached_fn.cache_clear)`` hands cash's code to a reference of the user's.
"""

from __future__ import annotations

import types
import weakref
from collections.abc import Callable
from typing import Any

__all__ = [
    "WeakKeyDictionary",
    "WeakSet",
    "WeakValueDictionary",
    "GONE",
    "callback_of",
    "finalize",
    "is_cash_ref",
    "ref",
]


class _Callback:
    """The callback of a reference `ref` made: calls *fn*, if any. Its own
    class, so that the reference is known as cash's whatever *fn* is."""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[Any], object] | None) -> None:
        self.fn = fn

    def __call__(self, reference: Any) -> None:
        fn = self.fn
        if fn is not None:
            fn(reference)


#: The callback of a reference with nothing to run when its object goes.
#: With one, each reference is one of its own, where a plain
#: ``weakref.ref(obj)`` is the one every caller shares.
GONE = _Callback(None)


def ref(obj: Any, callback: Callable[[Any], object] | None = None) -> weakref.ReferenceType:
    """``weakref.ref(obj, callback)``, known as cash's (`is_cash_ref`)."""
    return weakref.ref(obj, GONE if callback is None else _Callback(callback))


class finalize(weakref.finalize):
    """``weakref.finalize``, known as cash's (`is_cash_ref`)."""

    __slots__ = ()


def _copy(self: Any) -> Any:
    """A copy of a container below, of its own class: the standard ones'
    ``copy()`` (and ``|``, which copies) gives a standard one."""
    new = type(self)()
    new.update(self.items())
    return new


class WeakKeyDictionary(weakref.WeakKeyDictionary):
    """``weakref.WeakKeyDictionary``, its references known as cash's."""

    copy = __copy__ = _copy


class WeakValueDictionary(weakref.WeakValueDictionary):
    """``weakref.WeakValueDictionary``, its references known as cash's."""

    copy = __copy__ = _copy


class WeakSet(weakref.WeakSet):
    """``weakref.WeakSet``, its references known as cash's."""


_CONTAINERS = (WeakKeyDictionary, WeakValueDictionary, WeakSet)


#: A reference's callback, read past a subclass's own attributes.
callback_of = weakref.ReferenceType.__callback__.__get__  # type: ignore[attr-defined]


def is_cash_ref(reference: Any) -> bool:
    """Did cash make *reference*: its callback is a `_Callback` (`ref`'s),
    a `finalize` of cash's, or the removal callback of one of the
    containers above (it holds the container as ``selfref``)? A reference
    with no callback is never cash's, nor is a proxy (cash makes none),
    whose attributes are its object's."""
    if not isinstance(reference, weakref.ReferenceType):
        return False
    callback = callback_of(reference)
    kind = type(callback)
    if kind is _Callback or kind is finalize:
        return True
    defaults = getattr(callback, "__defaults__", None) if kind is types.FunctionType else None
    for value in defaults or ():
        if type(value) is weakref.ReferenceType and isinstance(value(), _CONTAINERS):
            return True
    return False
