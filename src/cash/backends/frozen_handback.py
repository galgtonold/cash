"""``frozen=True``: a hit hands back the object a hit handed out before.

A ``frozen=True`` function promises its result is never changed after it
is returned. The RAM tier holds it to that, the way ``functools.lru_cache``
does: the copy a hit hands out is kept (`keep`), and every later hit of
the same stored entry hands that very object back -- whoever else holds it
(a tokenizer keeping each trie it is given by its id). An ordinary hit may
hand back its last copy only when nothing outside holds it
(`handed_copies`); declared frozen, only the "unchanged" part of that proof
is asked for (`check`):

- a numpy array (no object dtype) is handed out read-only, behind a base
  that is read-only too, so that no ``a.flags.writeable = True`` makes it
  writable again; it is unchanged while that still holds and its shape,
  dtype and strides are the same;
- a pandas frame or series (copy-on-write pandas) is kept with a shallow
  copy of its own, so that any write through pandas copies the block it
  writes first: its manager, blocks, block arrays, axes, axis names, flags
  and attributes are then the same objects while nothing changed. Object
  columns must hold only strings and numbers, and ``attrs`` must be empty;
- any other value is watched by the C extension's watchers (`value_watch`):
  dicts, lists, sets, instances of plain classes and their slots, over
  atoms, of at least the watchers' floor (64 KiB).

A value none of these covers is copied on every hit, as before. A kept
object found changed is let go: KEY-FROZEN-MUTATED names its producer
(`warn_changed`), the next hit reads a new copy out of the stored entry,
and the changed object is keyed by its content from then on (`on_changed`).

The same proofs hold a frozen result to its promise when it is passed on
as an argument (`prove`, `verdict`; `decorator.frozen.FrozenResults.audit`):
its watch is kept under a kind of its own (`value_watch.KIND_FROZEN`), so
neither user lets go of the other's (`handed_copies.unwatch`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from .. import _pandas_compat, handed_copies, value_watch

__all__ = [
    "CHANGED",
    "SAME",
    "SMALL",
    "UNANCHORED",
    "UNKNOWN",
    "Kept",
    "check",
    "keep",
    "let_go",
    "mutated_text",
    "on_changed",
    "prove",
    "release",
    "verdict",
]

#: `check`'s verdicts: proved unchanged; seen changed; no longer provable.
SAME = "same"
CHANGED = "changed"
UNKNOWN = "unknown"

#: What `prove` answers for a value the watchers would follow, but not now:
#: under their floor, or not yet held by a name they anchor it to.
SMALL = handed_copies.TOO_SMALL
UNANCHORED = handed_copies.UNANCHORED


class Kept:
    """An object a hit handed out and the next ones hand back: the value,
    what it is proved unchanged by, its size, and the stored entry it was
    read out of (set by the tier)."""

    __slots__ = ("content", "proof", "stored", "value")

    def __init__(self, value: Any, proof: Any, content: int) -> None:
        self.value = value
        self.proof = proof
        self.content = content
        self.stored: Any = None


# -- numpy -------------------------------------------------------------------


class _ArrayProof:
    __slots__ = ("chain", "layout")

    def __init__(self, chain: list, layout: tuple) -> None:
        self.chain = chain
        self.layout = layout


def _keep_array(value: Any) -> Kept | None:
    if value.dtype.hasobject:
        return None  # its cells are objects anyone may change
    chain = [value]
    base = value.base
    while isinstance(base, _np().ndarray):
        chain.append(base)
        base = base.base
    for arr in reversed(chain):
        arr.flags.writeable = False
    handed = value
    if value.base is None:
        # An array owning its memory may be made writable again; a view of
        # a read-only one may not.
        handed = value.view()
        chain.insert(0, handed)
    layout = (handed.shape, handed.dtype, handed.strides)
    return Kept(handed, _ArrayProof(chain, layout), int(value.nbytes))


def _check_array(value: Any, proof: _ArrayProof) -> str:
    if any(arr.flags.writeable for arr in proof.chain):
        return CHANGED
    if (value.shape, value.dtype, value.strides) != proof.layout:
        return CHANGED
    return SAME


# -- pandas ------------------------------------------------------------------


class _FrameProof:
    __slots__ = ("shadow", "snapshot")

    def __init__(self, shadow: Any, snapshot: tuple) -> None:
        self.shadow = shadow
        self.snapshot = snapshot


_ATOM_KINDS = frozenset({"string", "integer", "floating", "boolean", "empty", "mixed-integer-float", "bytes"})


def _copy_on_write(pd: Any) -> bool:
    try:
        if int(pd.__version__.split(".", 1)[0]) >= 3:
            return True
        return pd.options.mode.copy_on_write is True
    except Exception:  # noqa: BLE001 - not provable: copied as before
        return False


def _frame_snapshot(value: Any) -> tuple:
    mgr = _pandas_compat.manager(value)
    blocks = mgr.blocks
    axes = tuple(mgr.axes)
    return (
        mgr,
        blocks,
        tuple(block.values for block in blocks),
        axes,
        tuple(tuple(ax.names) for ax in axes),
        tuple((name, id(item)) for name, item in value.__dict__.items()),
        value.flags.allows_duplicate_labels,
    )


def _same_snapshot(a: tuple, b: tuple) -> bool:
    return (
        a[0] is b[0]
        and a[1] is b[1]
        and len(a[2]) == len(b[2])
        and all(x is y for x, y in zip(a[2], b[2]))
        and len(a[3]) == len(b[3])
        and all(x is y for x, y in zip(a[3], b[3]))
        and a[4] == b[4]
        and a[5] == b[5]
        and a[6] == b[6]
    )


def _keep_frame(value: Any) -> Kept | None:
    pd = sys.modules["pandas"]
    if not _copy_on_write(pd) or value.attrs:
        return None
    infer = pd.api.types.infer_dtype
    nbytes = 0
    for block in _pandas_compat.blocks(value):
        values = block.values
        if getattr(getattr(values, "dtype", None), "hasobject", False):
            # Cells pandas cannot see written: only atoms.
            if infer(values.ravel(), skipna=True) not in _ATOM_KINDS:
                return None
        nbytes += int(getattr(values, "nbytes", 0) or 0)
    # A reference of cash's own to every block: pandas copies a block
    # before any write to it, so a write moves what `_frame_snapshot` reads.
    shadow = value.copy(deep=False)
    return Kept(value, _FrameProof(shadow, _frame_snapshot(value)), nbytes)


def _check_frame(value: Any, proof: _FrameProof) -> str:
    if value.attrs:
        return CHANGED
    return SAME if _same_snapshot(proof.snapshot, _frame_snapshot(value)) else CHANGED


# -- watched object graphs ---------------------------------------------------


class _WatchProof:
    __slots__ = ("kind", "token")

    def __init__(self, token: object, kind: int) -> None:
        self.token = token
        self.kind = kind


def _np() -> Any:
    return sys.modules["numpy"]


def keep(value: Any) -> Kept | None:
    """A `Kept` for *value*, a copy a hit just read out of a frozen entry
    and holds alone, or None when nothing can prove it unchanged later.
    Hand out ``kept.value``: for an array, a read-only view of *value*."""
    kind = type(value)
    module = getattr(kind, "__module__", "") or ""
    try:
        if module.startswith("numpy") and kind is _np().ndarray:
            return _keep_array(value)
        if module.startswith("pandas") and kind.__name__ in ("DataFrame", "Series"):
            return _keep_frame(value)
        found = handed_copies.watch(value)
    except Exception:  # noqa: BLE001 - not provable: copied as before
        return None
    if type(found) is not tuple:
        return None
    token, content = found
    return Kept(value, _WatchProof(token, value_watch.KIND_HANDED), content)


def prove(value: Any) -> object | str | None:
    """What proves *value*, a ``frozen=True`` result about to be passed on
    as an argument, unchanged at each use after (`verdict`), as for a kept
    copy: a copy-on-write pandas frame or series by its blocks, a
    watched value by the watchers (`value_watch.KIND_FROZEN`, its own token
    beside the memory tier's). `SMALL` when the watchers would follow it
    but it is under their floor, `UNANCHORED` when they would once a name
    holds it, None when nothing can prove it. A numpy array is None: it is
    read-only, and keyed by its hash (`FrozenResults.array_hash`)."""
    kind = type(value)
    module = getattr(kind, "__module__", "") or ""
    try:
        if module.startswith("numpy") and kind is _np().ndarray:
            return None
        if module.startswith("pandas") and kind.__name__ in ("DataFrame", "Series"):
            kept = _keep_frame(value)
            return None if kept is None else kept.proof
        found = handed_copies.watch(value, value_watch.KIND_FROZEN)
    except Exception:  # noqa: BLE001 - not provable: audited instead
        return None
    if found is handed_copies.TOO_SMALL:
        return SMALL
    if found is handed_copies.UNANCHORED:
        return UNANCHORED
    if type(found) is not tuple:
        return None
    return _WatchProof(found[0], value_watch.KIND_FROZEN)


def check(kept: Kept) -> str:
    """Is *kept*'s object unchanged since it was kept? `SAME`, `CHANGED`
    (seen changed), or `UNKNOWN` (no longer provable either way)."""
    return verdict(kept.value, kept.proof)


def verdict(value: Any, proof: Any) -> str:
    """Is *value* unchanged since *proof* (`keep`'s, or `prove`'s) was
    taken? `SAME`, `CHANGED` (seen changed), or `UNKNOWN` (no longer
    provable either way)."""
    try:
        if type(proof) is _ArrayProof:
            return _check_array(value, proof)
        if type(proof) is _FrameProof:
            return _check_frame(value, proof)
        return handed_copies.watched(value, proof.token, proof.kind)
    except Exception:  # noqa: BLE001 - not provable: a new copy
        return UNKNOWN


def let_go(value: Any, proof: Any) -> None:
    """Stop proving *value* unchanged by *proof*."""
    if type(proof) is _WatchProof:
        handed_copies.unwatch(value, proof.kind)


def release(kept: Kept) -> None:
    """Stop keeping *kept*'s object."""
    let_go(kept.value, kept.proof)
    kept.value = kept.proof = kept.stored = None


# -- a change seen -----------------------------------------------------------

#: Called with an object a frozen entry handed out and that was seen
#: changed: the decorator stops keying it by its producer
#: (`cash.decorator.frozen`).
_ON_CHANGED: list[Callable[[Any], None]] = []


def on_changed(hook: Callable[[Any], None]) -> None:
    """Call *hook* with each kept object `warn_changed` reports."""
    if hook not in _ON_CHANGED:
        _ON_CHANGED.append(hook)


def mutated_text(kind: str, producer: str, after: str) -> tuple[str, str]:
    """KEY-FROZEN-MUTATED's message and fix, *after* saying what follows."""
    return (
        f"a {kind} returned by {producer}, which is declared "
        f"@cash.cache(frozen=True), has been modified since it was returned. "
        f"Calls that received it before the change may have been served "
        f"results for the unmodified object; {after}",
        f"take frozen=True off {producer} if its result is meant to be "
        f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
    )


def warn_changed(value: Any, producer: str | None) -> None:
    """KEY-FROZEN-MUTATED for *value*, a kept object seen changed."""
    from .._type_memo import class_name
    from ..diagnostics import warn_diagnostic
    from ..exceptions import CashImpurityWarning

    for hook in list(_ON_CHANGED):
        try:
            hook(value)
        except Exception:  # noqa: BLE001 - the warning still goes out
            pass
    name = producer or "a frozen=True function"
    what, fix = mutated_text(
        class_name(type(value)),
        name,
        f"from now on it is keyed by its contents, and the next call of {name} gets a new copy of the stored result.",
    )
    warn_diagnostic(CashImpurityWarning, "KEY-FROZEN-MUTATED", what, fix)
