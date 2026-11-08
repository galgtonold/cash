"""The weak references cash makes, told from the user's.

A variable that finds an object by a weak reference keeps a statement out
of the cache: restored, the object would be a copy the reference does not
find (`identity_holders._weakly_found_by_others`). Cash keeps weak
references to the user's objects too -- its memos drop an entry when the
object goes -- and a plain ``weakref.ref(obj)``, or the one a stdlib
``WeakKeyDictionary`` makes, looked like the user's. Whether a statement was
stored then hung on what cash's memos held at that moment, which an earlier
statement (or test) decided.

Every weak reference cash makes goes through here, or carries a callback
of cash's own code: `ref` gives one with a callback of this module, and the
containers below are cash's own classes, whose removal callbacks name them.
`is_cash_ref` tells exactly those apart.
"""

from __future__ import annotations

import os
import types
import weakref
from typing import Any

__all__ = ["WeakKeyDictionary", "WeakSet", "WeakValueDictionary", "is_cash_ref", "ref"]

#: cash's own source folder: a callback whose code is in it is cash's.
CASH_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def gone(_ref: Any) -> None:
    """The callback of the weak references `ref` makes: none to run. With
    it, each is a reference of its own, where a plain ``weakref.ref(obj)``
    is the one every caller shares."""


def ref(obj: Any) -> weakref.ReferenceType:
    """``weakref.ref(obj)``, known as cash's (`is_cash_ref`)."""
    return weakref.ref(obj, gone)


class WeakKeyDictionary(weakref.WeakKeyDictionary):
    """``weakref.WeakKeyDictionary``, its references known as cash's."""


class WeakValueDictionary(weakref.WeakValueDictionary):
    """``weakref.WeakValueDictionary``, its references known as cash's."""


class WeakSet(weakref.WeakSet):
    """``weakref.WeakSet``, its references known as cash's."""


_CONTAINERS = (WeakKeyDictionary, WeakValueDictionary, WeakSet)


def _cash_code(fn: Any) -> bool:
    if type(fn) is types.MethodType:
        fn = fn.__func__
    code = getattr(fn, "__code__", None) if type(fn) is types.FunctionType else None
    return code is not None and code.co_filename.startswith(CASH_DIR)


def is_cash_ref(reference: Any) -> bool:
    """Did cash make *reference*: its callback is cash's code (`gone`, a
    memo's own), the removal callback of one of the containers above (it
    holds the container as ``selfref``), or a ``weakref.finalize`` of a
    function of cash's? A reference with no callback is never cash's."""
    callback = reference.__callback__
    if callback is None:
        return False
    if type(callback) is weakref.finalize:
        info = weakref.finalize._registry.get(callback)  # type: ignore[attr-defined]
        return info is not None and _cash_code(info.func)
    if _cash_code(callback):
        return True
    defaults = getattr(callback, "__defaults__", None) if type(callback) is types.FunctionType else None
    for value in defaults or ():
        if type(value) is weakref.ReferenceType and isinstance(value(), _CONTAINERS):
            return True
    return False
