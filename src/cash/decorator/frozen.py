"""``frozen=True``: results keyed by the call that produced them, audited now
and then for changes."""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import weakref
from collections.abc import Sized
from typing import Any

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning, CashImpurityWarning
from ..object_hashing import builtin_hash
from .arg_hashing import LINEAGE_SRC_DECORATOR, LINEAGE_SRC_FROZEN

#: A frozen object is re-hashed at its 8th use as an argument and every 64th
#: after that (every use under CASH_DEBUG), and compared with the first audit.
FROZEN_AUDIT_FIRST = 8
FROZEN_AUDIT_EVERY = 64


#: Frozen list/tuple/dict results held at once. Past it the oldest goes.
FROZEN_CONTAINERS_MAX = 256


#: How many of a container's elements the cheap audit measures.
FROZEN_SHAPE_SAMPLE = 8


def frozen_shape(obj: Any) -> tuple | None:
    """What *obj* is shaped like, in O(1)-ish work, or ``None``.

    The full audit hashes every byte, so it runs rarely -- the baseline at
    the 8th use and a comparison every 64th after that. That left the
    ordinary shape unprotected: produce a result, change it, pass it again.
    A length, a frame's shape and dtypes, and the lengths of a few elements
    cost nothing to read on EVERY use, and they move for the changes a
    caller actually makes (``model["w"].append(...)``). A change they
    cannot see -- a value overwritten in place, same length -- is still
    caught by the full audit.
    """
    try:
        shape = getattr(obj, "shape", None)
        if shape is not None:
            dtypes = getattr(obj, "dtypes", None)
            dtype = tuple(str(d) for d in dtypes) if dtypes is not None else str(getattr(obj, "dtype", ""))
            return ("shaped", tuple(shape), dtype)
        if isinstance(obj, (str, bytes)):
            return None
        values = list(obj.values())[:FROZEN_SHAPE_SAMPLE] if isinstance(obj, dict) else None
        if values is None and isinstance(obj, (list, tuple)):
            values = list(obj[:FROZEN_SHAPE_SAMPLE])
        inner = tuple(len(v) for v in values or () if isinstance(v, Sized))
        return ("sized", len(obj), inner) if isinstance(obj, Sized) else None
    except Exception:  # noqa: BLE001 - no cheap signal is not a failure
        return None


class FrozenMixin:
    """Results of ``frozen=True`` functions: keyed by producer, audited for changes."""

    def _frozen_arg_names(self, normalized_args: tuple[tuple, dict]) -> list[str]:
        """`explain()`'s list of arguments keyed by a frozen=True producer."""
        args, kwargs = normalized_args
        names = []
        for name, value in [*((f"#{i}", v) for i, v in enumerate(args)), *kwargs.items()]:
            if getattr(value, "_cash_lineage_src", None) == LINEAGE_SRC_FROZEN or (
                self._frozen_arrays and id(value) in self._frozen_arrays
            ):
                producer = (
                    getattr(value, "_cash_lineage_producer", None)
                    or (self._frozen_arrays.get(id(value), [None, None])[1])
                )
                names.append(f"{name} (the result of {producer}, declared frozen)")
        return names

    def _frozen_array_hash(self, arr: Any) -> str | None:
        """The content hash of a frozen function's numpy result, computed once.

        Valid while the array is still that object and still read-only; an
        array made writeable again (``a.flags.writeable = True``) is keyed by
        content from then on.
        """
        entry = self._frozen_arrays.get(id(arr))
        if entry is None:
            return None
        wref, _producer, content_hash = entry
        if wref() is not arr or getattr(arr, "flags", None) is None or arr.flags.writeable:
            self._frozen_arrays.pop(id(arr), None)
            return None
        if content_hash is None:
            content_hash = builtin_hash(arr)
            entry[2] = content_hash
        return content_hash

    def _remember_frozen_container(self, obj: Any, producer: str, lineage: str) -> None:
        """Key a frozen function's list, tuple or dict by its producer's lineage.

        A list of two million parsed rows, passed on to two cached consumers,
        was pickled in full for every call -- warm runs about 9x slower than
        uncached -- and ``frozen=True`` on the parser changed nothing: its fast
        path covered numpy arrays alone, and a list cannot carry a tag. Such a
        result is now keyed like a frozen frame: by the lineage of
        the call that produced it, audited now and then (`_audit_frozen`'s
        schedule).

        It has no weakref either, so the object is held here -- and let go
        again once nothing else holds it, swept on each new entry.
        """
        table = self._frozen_containers
        # What "held by the table alone" reads as, measured the same way: the
        # count differs between Python versions (3.14 borrows references).
        probe = [None, None, None, None, None]
        probe[0] = object()
        alone = sys.getrefcount(probe[0])
        for key, entry in list(table.items()):
            if sys.getrefcount(entry[0]) <= alone:
                table.pop(key, None)
        while len(table) >= FROZEN_CONTAINERS_MAX:
            table.pop(next(iter(table)))
        table[id(obj)] = [obj, producer, f"frozen:{lineage}", 0, None, frozen_shape(obj)]

    def _frozen_container_hash(self, obj: Any) -> str | None:
        """The lineage a frozen list/tuple/dict is keyed by, or None once it
        has been seen to change (KEY-FROZEN-MUTATED, as for a frozen frame)."""
        entry = self._frozen_containers.get(id(obj))
        if entry is None or entry[0] is not obj:
            return None
        entry[3] += 1
        uses = entry[3]
        shape = frozen_shape(obj)
        if len(entry) > 5 and entry[5] is not None and shape != entry[5]:
            self._frozen_containers.pop(id(obj), None)
            self._warn_frozen_mutated(obj, entry[1])
            return None
        due = (
            (self.debug or os.environ.get("CASH_DEBUG"))
            or uses == FROZEN_AUDIT_FIRST
            or (uses > FROZEN_AUDIT_FIRST and uses % FROZEN_AUDIT_EVERY == 0)
        )
        if due:
            try:
                digest = hashlib.sha256(pickle.dumps(obj)).hexdigest()
            except Exception:  # noqa: BLE001 - cannot audit: the declaration stands
                digest = None
            if digest is not None:
                if entry[4] is None:
                    entry[4] = digest
                elif entry[4] != digest:
                    self._frozen_containers.pop(id(obj), None)
                    warn_diagnostic(
                        CashImpurityWarning,
                        "KEY-FROZEN-MUTATED",
                        f"a {type(obj).__name__} returned by {entry[1]}, which is "
                        f"declared @cash.cache(frozen=True), has been modified since "
                        f"it was returned. Calls that received it before the change "
                        f"may have been served results for the unmodified object; "
                        f"from now on it is keyed by its contents.",
                        f"take frozen=True off {entry[1]} if its result is meant to "
                        f"be modified, or modify a copy (`obj = copy.deepcopy(obj)`) "
                        f"instead.",
                    )
                    return None
        return entry[2]

    def _warn_frozen_has_no_effect(self, func_name: str, result: Any) -> None:
        """Say so when ``frozen=True`` cannot apply to what the function returned."""
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "frozen_no_effect",
            f"@cash.cache(frozen=True) on {func_name}: it returned a "
            f"{type(result).__name__}, which cash cannot mark, so frozen=True has "
            f"no effect on it -- a call that receives it still hashes it in full. "
            f"frozen=True applies to a numpy array, a pandas/polars/modin frame, "
            f"a pyarrow table, a list, tuple or dict, and any object that takes "
            f"an attribute.",
            code="KEY-FROZEN-NO-EFFECT",
            fix="return one of those types, or take frozen=True off; "
            "cash.register_hasher gives the type a cheap identity instead.",
        )

    def _warn_frozen_mutated(self, obj: Any, producer: Any = None) -> None:
        """KEY-FROZEN-MUTATED: a result declared frozen is not what it was."""
        producer = producer or getattr(obj, "_cash_lineage_producer", None) or "a frozen=True function"
        try:
            obj._cash_lineage_src = LINEAGE_SRC_DECORATOR
        except (AttributeError, TypeError):
            pass
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {producer}, which is declared "
            f"@cash.cache(frozen=True), has been modified since it was returned. "
            f"Calls that received it before the change may have been served "
            f"results for the unmodified object; from now on it is keyed by its "
            f"contents.",
            f"take frozen=True off {producer} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )

    def _audit_frozen(self, obj: Any) -> bool:
        """Is a frozen=True result still what it was? False once it is not.

        The declaration is trusted, and checked now and then: at the object's
        8th use as an argument and every 64th after that, and at every use
        under CASH_DEBUG. The first check records a baseline -- the content
        hash for a type cash content-hashes, a digest of the pickle otherwise
        -- and each later one compares. On a change, KEY-FROZEN-MUTATED names
        the producer, the object's tag stops being trusted, and it is keyed by
        its content from then on. An object that cannot be pickled cannot be
        audited, and stays trusted.
        """
        key = id(obj)
        entry = self._frozen_uses.get(key)
        if entry is None or entry[0]() is not obj:
            try:
                wref = weakref.ref(obj, lambda _r, k=key, m=self._frozen_uses: m.pop(k, None))
            except TypeError:
                return True
            entry = [wref, 0, None, frozen_shape(obj)]
            if len(self._frozen_uses) >= 4096:
                self._frozen_uses.clear()
            self._frozen_uses[key] = entry
        entry[1] += 1
        uses = entry[1]
        shape = frozen_shape(obj)
        if entry[3] is not None and shape != entry[3]:
            self._frozen_uses.pop(key, None)
            self._warn_frozen_mutated(obj)
            return False
        due = (
            (self.debug or os.environ.get("CASH_DEBUG"))
            or uses == FROZEN_AUDIT_FIRST
            or (uses > FROZEN_AUDIT_FIRST and uses % FROZEN_AUDIT_EVERY == 0)
        )
        if not due:
            return True
        try:
            digest = builtin_hash(obj)
            if digest is None:
                digest = hashlib.sha256(pickle.dumps(obj)).hexdigest()
        except Exception:  # noqa: BLE001 - cannot audit: the declaration stands
            return True
        if entry[2] is None:
            entry[2] = digest
            return True
        if entry[2] == digest:
            return True
        producer = getattr(obj, "_cash_lineage_producer", None) or "a frozen=True function"
        try:
            obj._cash_lineage_src = LINEAGE_SRC_DECORATOR
        except (AttributeError, TypeError):
            pass
        self._frozen_uses.pop(key, None)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {producer}, which is declared "
            f"@cash.cache(frozen=True), has been modified since it was returned. "
            f"Calls that received it before the change may have been served "
            f"results for the unmodified object; from now on it is keyed by its "
            f"contents.",
            f"take frozen=True off {producer} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )
        return False

    def _forget_frozen_container(self, obj: Any) -> None:
        """Stop trusting a frozen result a call was just seen to change."""
        entry = self._frozen_containers.get(id(obj))
        if entry is None or entry[0] is not obj:
            return
        self._frozen_containers.pop(id(obj), None)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {type(obj).__name__} returned by {entry[1]}, which is declared "
            f"@cash.cache(frozen=True), was modified in place by a cached call. "
            f"From now on it is keyed by its contents.",
            f"take frozen=True off {entry[1]} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )
