"""Hand a hit the copy handed out before, when nobody can tell (`reuse`).

A hit of a value the memory tier keeps as pickle bytes unpickles a new
copy. When the caller drops each copy unchanged -- a function returning a
big read-only structure (a dictionary trie, a lookup table) called again
and again -- every hit rebuilds the same object graph, and frees the last
one, where a plain ``functools.lru_cache`` hands back the one object.

Here the copy a hit hands out is kept (`retain`), and the next hit of the
same stored entry hands that very object back (`reuse`) when nothing can
tell it from a new copy:

1. the entry is the one the copy was read from (the tier asks with the
   entry's own kept bytes, which a new store replaces);
2. nothing outside the copy holds any part of it: the reference count of
   every object in it is the count of references from inside it, the
   tier's one reference to it aside, and every weak reference to them is
   cash's own (``cash._cwatch.held_alone`` / ``held_still``); a class
   with ``__del__`` is never kept, so no finalizer is delayed;
3. cash's watchers (`value_watch`) prove that nothing in it changed since
   it was handed out: its dicts through dict watchers, its lists, slots
   and instance attributes by their recorded pointers, its classes by
   their version tags.

Whenever one of these cannot be shown -- no C extension, a value the
watchers cannot follow, a class with cached views, a copy someone still
holds or changed -- the hit unpickles a new copy as before. After a few of
those in a row an entry is no longer kept: arming the watchers on a value
costs a walk of it.
"""

from __future__ import annotations

from typing import Any

from . import _weak, value_watch
from ._native import cwatch as _cw

__all__ = ["Handed", "MAX_MISSES", "release", "retain", "reuse"]

#: Hits in a row that could not reuse the copy before an entry's copies are
#: no longer kept.
MAX_MISSES = 3

_HELD_ALONE = getattr(_cw, "held_alone", None)
_HELD_STILL = getattr(_cw, "held_still", None)
_ON = _cw is not None and _HELD_ALONE is not None and _HELD_STILL is not None


class Handed:
    """A copy handed out and kept for the next hit: its one holder (a list
    of one item, the only reference cash keeps), what `held_still` checks
    it by, and the token its watchers keep while it is unchanged."""

    __slots__ = ("count", "holder", "objects", "token", "weakable", "content")

    def __init__(self, holder: list, objects: bytes, weakable: bytes, count: int, token: object, content: int) -> None:
        self.holder = holder
        self.objects = objects
        self.weakable = weakable
        self.count = count
        self.token = token
        #: The value's size as the watchers estimate it, in bytes.
        self.content = content


def _all_cash_refs(refs: list) -> bool:
    return all(_weak.is_cash_ref(ref) for ref in refs)


def retain(holder: list, seconds: float) -> Handed | None:
    """Keep ``holder[0]``, a copy just read and about to be handed out,
    for the next hit: arm the watchers on it and record its objects.
    *holder* must be its only reference. None when it cannot be kept."""
    if not _ON:
        return None
    try:
        status = value_watch.arm_value(holder[0])
        if status != _cw.ARMED and status != _cw.ALREADY:
            return None
        token = object()
        if not _cw.commit(holder[0], value_watch.KIND_HANDED, token, seconds):
            _cw.forget(holder[0])
            return None
        info = _cw.info(holder[0])
        if not info or info.get("views"):
            # A class leaving cached views out of its state: the watchers
            # do not follow what such a view holds.
            _cw.forget(holder[0])
            return None
        found = _HELD_ALONE(holder, 0, True)
        if found is None or not _all_cash_refs(found[0]):
            _cw.forget(holder[0])
            return None
        _refs, objects, weakable, count = found
        return Handed(holder, objects, weakable, count, token, int(info.get("content_bytes") or 0))
    except Exception:  # noqa: BLE001 - not kept: the next hit reads a new copy
        try:
            _cw.forget(holder[0])
        except Exception:  # noqa: BLE001 - a copy the watchers never took needs no forgetting
            pass
        return None


def reuse(handed: Handed) -> Any:
    """The copy *handed* keeps, when nothing can tell it from a new one
    (see the module); else `_NONE`. The caller holds the tier's lock and
    hands the copy out at once."""
    holder = handed.holder
    try:
        # Watchers first: only a value proved unchanged holds the very
        # objects `held_still` reads.
        if _cw.peek(holder[0], value_watch.KIND_HANDED) is not handed.token:
            return NONE
        refs = _HELD_STILL(holder, 0, handed.objects, handed.weakable, handed.count)
        if refs is None or not _all_cash_refs(refs):
            return NONE
    except Exception:  # noqa: BLE001 - not provable: a new copy
        return NONE
    return holder[0]


def release(handed: Handed) -> None:
    """Stop keeping *handed*'s copy."""
    holder = handed.holder
    if holder:
        try:
            _cw.forget(holder[0])
        except Exception:  # noqa: BLE001 - a copy the watchers never took needs no forgetting
            pass
        holder.clear()


#: What `reuse` answers when the copy cannot be handed out again.
NONE = object()
