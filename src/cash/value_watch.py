"""Reuse a value's digest while watchers prove it unchanged (`digest`).

cash hashes the same values again and again: a notebook variable before and
after every statement that reads it, a big argument on every call of a
cached function. The optional C extension ``cash._cwatch`` watches what a
value's digest is computed from -- its dicts through CPython's dict
watchers, its classes through type watchers, the functions it names through
function watchers, its lists and sets by a snapshot of their item pointers
-- and says when none of it has changed since the digest was taken. That
digest is then handed back instead of hashing the value again.

The digest itself always comes from cash's own Python hashing, so a key is
the same with the extension and without it. Without it (a pure-Python
install, PyPy, Python before 3.12, a free-threaded build, or
``CASH_PURE_PYTHON=1``), `digest` simply hashes: no value is ever "provably
unchanged", which is what cash did before. ``CASH_CWATCH_VERIFY=1`` hashes
again behind every reuse and raises `WatchMismatch` when the two differ.

Which values are watched (`classify` and the C walker): exact dicts, lists,
sets, tuples and frozensets, instances of plain classes, and the classes
and functions they name, over atoms (None, bool, int, float, str, bytes).
A value holding anything else -- an array, a frame, a C object, an instance
of a class that chooses its own pickled state or looks attributes up
itself -- is hashed every time, as before. A value is watched only from
the second time it is hashed, only when it is big enough for hashing to
cost more than watching (``CASH_CWATCH_MIN_BYTES``, 64 KiB), and only while
the watch table fits its budget (``CASH_CWATCH_BUDGET``, 2% of RAM).
"""

from __future__ import annotations

import copyreg
import functools
import os
import sys
import types
from collections.abc import Callable
from time import perf_counter
from typing import Any

from . import _native
from ._native import REASON
from ._native import cwatch as _cw

__all__ = [
    "KIND_PLAIN_KEY",
    "KIND_VALUE",
    "WatchMismatch",
    "classify",
    "digest",
    "enabled",
    "forget",
    "reset",
    "stats",
    "status_line",
]

#: Consumer kinds: each keeps its own digest of a watched value, valid for
#: the value's version it was taken at.
KIND_VALUE = 0  # value_hash.compute_hash
KIND_PLAIN_KEY = 1  # the decorator's key part for plain data (arg_hashing.plain_key_part)


class WatchMismatch(AssertionError):
    """``CASH_CWATCH_VERIFY=1``: a digest the watchers called unchanged is
    not the digest hashing the value again gives."""


VERIFY = os.environ.get("CASH_CWATCH_VERIFY", "") not in ("", "0")

_FIRST = _ARM = _ALREADY = _ARMED = _NEED_TYPE = -1
if _cw is not None:
    _FIRST, _ARM = _cw.FIRST, _cw.ARM
    _ARMED, _ALREADY, _NEED_TYPE = _cw.ARMED, _cw.ALREADY, _cw.NEED_TYPE


def _default_budget() -> int:
    raw = os.environ.get("CASH_CWATCH_BUDGET")
    if raw:
        from .units import parse_size

        return parse_size(raw)
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = 8 << 30
    return max(32 << 20, min(ram // 50, 2 << 30))


def _default_min_bytes() -> int:
    raw = os.environ.get("CASH_CWATCH_MIN_BYTES")
    if raw:
        from .units import parse_size

        return parse_size(raw)
    return 64 << 10


if _cw is not None:
    _cw.configure(
        budget=_default_budget(),
        min_bytes=_default_min_bytes(),
        dispatch_table=copyreg.dispatch_table,
        extension_registry=getattr(copyreg, "_extension_registry", None),
    )


#: Is the fast path on? Read by the hashing entry points before they call in.
WATCHING = _cw is not None


def enabled() -> bool:
    """Is the watcher fast path on in this process?"""
    return _cw is not None


#: Names a class may not define anywhere in its MRO (but `object`) for its
#: instances to be watched: each changes what pickle stores (its state, its
#: reduce, its new-arguments), what an attribute lookup returns
#: (``__getattr__``, ``__getattribute__``), what class pickle sees
#: (``__class__``) or what cash keys instead of the content (``__cash_key__``).
_FORBIDDEN = frozenset(
    {
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__getattr__",
        "__getattribute__",
        "__class__",
        "__cash_key__",
    }
)

_HEAPTYPE = 1 << 9


def classify(t: type) -> tuple[bool, tuple]:
    """Do instances of *t* pickle from their ``__dict__`` and slots alone,
    by cash's pickling (`kept_state.dumps` keyed) as by ``pickle``?
    ``(ok, slot member descriptors)``.

    Only a plain class: the ``type`` metaclass, every class of its MRO but
    `object` defined in Python, none of `_FORBIDDEN` defined, no
    ``functools.cached_property`` (cash leaves views out of the state it
    keys, `kept_state.derived_views`), no reducer registered with
    ``copyreg``, nothing a library content hasher claims, and a name pickle
    can find (not defined inside a function). Its slots must resolve to its
    own slot descriptors (a subclass property of a base slot's name would
    be read instead).
    """
    try:
        return _classify(t)
    except Exception:  # noqa: BLE001 - a class that cannot even be inspected: not watched
        return False, ()


def _classify(t: type) -> tuple[bool, tuple]:
    no = (False, ())
    if type(t) is not type or not (t.__flags__ & _HEAPTYPE) or t.__itemsize__:
        return no
    mro = t.__mro__
    if mro[-1] is not object:
        return no
    for klass in mro[:-1]:
        if type(klass) is not type or not (klass.__flags__ & _HEAPTYPE):
            return no
        own = klass.__dict__
        if not _FORBIDDEN.isdisjoint(own):
            return no
        held_dict = own.get("__dict__")
        if held_dict is not None and type(held_dict) is not types.GetSetDescriptorType:
            return no
        if any(isinstance(member, functools.cached_property) for member in own.values()):
            return no
    if not isinstance(t.__module__, str) or "<locals>" in t.__qualname__:
        return no
    if t in copyreg.dispatch_table:
        return no
    from .content_hashers import builtin_hash_family

    if t.__name__ in ("DataFrame", "Series", "ndarray") or builtin_hash_family(t) is not None:
        return no
    slots: list = []
    for klass in mro[:-1]:
        for name, member in klass.__dict__.items():
            if type(member) is not types.MemberDescriptorType or member.__objclass__ is not klass:
                continue
            for first in mro:
                if name in first.__dict__:
                    if first.__dict__[name] is not member:
                        return no
                    break
            slots.append(member)
    if slots:
        # What pickle reads them by, worked out now: `copyreg._slotnames`
        # writes ``__slotnames__`` into the class on first use, which would
        # move its version tag right after the verdict.
        copyreg._slotnames(t)
    return True, tuple(slots)


def _anchors() -> tuple:
    """Namespaces that may hold a value with no other proof of identity (a
    list, a set, a tuple): the notebook's and ``__main__``'s globals. Such a
    value is watched only while one of them holds it under one name."""
    found = []
    main = sys.modules.get("__main__")
    ns = getattr(main, "__dict__", None)
    if isinstance(ns, dict):
        found.append(ns)
    ipython = sys.modules.get("IPython")
    if ipython is not None:
        try:
            shell = ipython.get_ipython()
        except Exception:  # noqa: BLE001 - no shell
            shell = None
        user_ns = getattr(shell, "user_ns", None)
        if isinstance(user_ns, dict) and all(user_ns is not other for other in found):
            found.append(user_ns)
    return tuple(found)


def _arm(value: Any) -> int:
    cw = _cw
    for _ in range(64):
        status, detail = cw.arm(value, _anchors())
        if status != _NEED_TYPE:
            return status
        ok, slots = classify(detail)
        cw.set_type_verdict(detail, ok, slots)
    return -1


def digest(
    value: Any,
    compute: Callable[[Any], Any],
    kind: int = KIND_VALUE,
    keep: Callable[[Any, Any], bool] | None = None,
) -> Any:
    """``compute(value)``, or the digest it gave before when the watchers
    prove *value* unchanged since then. *keep* says whether a computed
    digest may be reused at all (an identity fallback may not)."""
    cw = _cw
    if cw is None:
        return compute(value)
    found = cw.check(value, kind)
    if type(found) is not int:
        if VERIFY:
            _verify(value, compute, found)
        return found
    if found == _FIRST:
        t0 = perf_counter()
        result = compute(value)
        cw.seen(value, perf_counter() - t0)
        return result
    if found != _ARM:
        return compute(value)
    status = _arm(value)
    if status != _ARMED and status != _ALREADY:
        return compute(value)
    t0 = perf_counter()
    result = compute(value)
    seconds = perf_counter() - t0
    if keep is None or keep(value, result):
        cw.commit(value, kind, result, seconds)
    return result


#: Reuses checked by ``CASH_CWATCH_VERIFY=1`` in this process.
VERIFIED = [0]


def _verify(value: Any, compute: Callable[[Any], Any], reused: Any) -> None:
    fresh = compute(value)
    VERIFIED[0] += 1
    if fresh != reused:
        raise WatchMismatch(
            f"cash._cwatch reused a stale digest for a {type(value).__module__}.{type(value).__qualname__}: "
            f"{reused!r} != {fresh!r}"
        )


def forget(value: Any) -> None:
    """Stop watching *value*."""
    if _cw is not None:
        _cw.forget(value)


def reset() -> None:
    """Drop every watched value and every class verdict."""
    if _cw is not None:
        _cw.reset()
        _cw.configure(
            dispatch_table=copyreg.dispatch_table,
            extension_registry=getattr(copyreg, "_extension_registry", None),
        )


def stats() -> dict:
    """The watch table's counters and size, or ``{"enabled": False, ...}``."""
    if _cw is None:
        return {"enabled": False, "reason": REASON}
    found = dict(_cw.stats())
    found["enabled"] = True
    found["verify"] = VERIFY
    return found


def status_line() -> str:
    """``%cash_status``'s ``fast_path``: on or off and why (`_native`), and
    while on, how big the watch table is and how often it saved a hash."""
    line = _native.status_line()
    if _cw is None:
        return line
    from .units import human_bytes

    s = _cw.stats()
    return (
        f"{line}; watch table {human_bytes(s['table_bytes'])} of {human_bytes(s['budget'])}, "
        f"{s['roots']} value{'s' if s['roots'] != 1 else ''} watched, "
        f"{s['hits']} hash{'es' if s['hits'] != 1 else ''} skipped" + (", verify mode" if VERIFY else "")
    )
