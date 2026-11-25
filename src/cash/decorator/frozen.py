"""``frozen=True``: results keyed by the call that produced them, proved
unchanged at each use where cash can, and audited now and then where not."""

from __future__ import annotations

import hashlib
import os
import sys
import weakref
from collections.abc import Sized
from typing import TYPE_CHECKING, Any

from .. import _weak, kept_state
from .. import phase_times as _phases
from .._memo import FROZEN_RESULTS, LruMemo
from .._type_memo import class_name
from ..backends import frozen_handback
from ..backends.frozen_handback import mutated_text, on_changed
from ..content_hashers import builtin_hash
from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning, CashImpurityWarning
from ..lineage_tag import own_tag, set_tags
from .arg_hashing import LINEAGE_SRC_DECORATOR, LINEAGE_SRC_FROZEN

if TYPE_CHECKING:
    from ..config.schema import CashConfig
    from .reporting import Notices

#: A frozen object nothing proves unchanged (`frozen_handback.prove`) is
#: re-hashed at its 8th use as an argument and every 64th after that (every
#: use under CASH_DEBUG), and compared with the first audit.
FROZEN_AUDIT_FIRST = 8
FROZEN_AUDIT_EVERY = 64

# What a frozen object passed on is checked by (`_Proof`'s ``kind``).
#: Nothing proves it unchanged: the shape on every use, the content on the
#: audit's schedule.
_SCHEDULE = 0
#: A proof `frozen_handback.verdict` checks in O(1)-ish work on every use.
_PROVED = 1
#: Under the watchers' floor: hashed on every use, which costs about what
#: the bookkeeping of a schedule would (`_full_hash`).
_SMALL = 2
#: Not yet held by a name the watchers anchor it to: tried again at its
#: first use as an argument.
_RETRY = 3


#: What KEY-FROZEN-MUTATED says follows, said by the argument hasher.
_KEYED_BY_CONTENT = "from now on it is keyed by its contents."

#: Every `FrozenResults`, for `_untrust_everywhere`.
_ALL: weakref.WeakSet = _weak.WeakSet()


def _untrust_everywhere(obj: Any) -> None:
    """Key *obj*, an object a frozen entry handed out and the RAM tier saw
    changed (`backends.frozen_handback`), by its content from now on."""
    for results in list(_ALL):
        results.untrust(obj)


on_changed(_untrust_everywhere)


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
            dtype = _dtype_names(dtypes) if dtypes is not None else str(getattr(obj, "dtype", ""))
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


def _dtype_names(dtypes: Any) -> tuple[str, ...]:
    """``str`` of each column's dtype, each distinct dtype object named once.

    A 3,000-column frame holds one or two dtype objects, and ``str`` of a
    numpy dtype is slow: naming every column took ~10 ms on each use of the
    frame as an argument, a second of a grid of 32 backtests over three such
    frames.
    """
    names: dict[int, str] = {}
    out = []
    for d in dtypes:
        name = names.get(id(d))
        if name is None:
            name = names[id(d)] = str(d)
        out.append(name)
    return tuple(out)


def _full_hash(obj: Any, builtin: bool = True) -> str | None:
    """*obj*'s content digest, for the audit: its `builtin_hash` (when
    *builtin*), else a digest of its pickle; None when it cannot be read.
    Timed as ``key.frozen_audit`` (`phase_times`)."""
    phase = _phases.sub_begin(_phases.KEY_FROZEN_AUDIT) if _phases.ON else None
    try:
        digest = builtin_hash(obj) if builtin else None
        if digest is None:
            digest = hashlib.sha256(kept_state.dumps(obj)).hexdigest()
        return digest
    except Exception:  # noqa: BLE001 - cannot audit: the declaration stands
        return None
    finally:
        if phase is not None:
            _phases.sub_end(phase)


def _start_proof(obj: Any, builtin: bool) -> tuple[int, Any, str | None]:
    """``(kind, proof, baseline)`` for *obj*, a frozen result handed out or
    first passed on: `_PROVED` and what proves it unchanged
    (`frozen_handback.prove`); `_SMALL` and its digest now; `_RETRY`; or
    `_SCHEDULE`."""
    proof = frozen_handback.prove(obj)
    if proof is None:
        return _SCHEDULE, None, None
    if proof is frozen_handback.SMALL:
        return _SMALL, None, _full_hash(obj, builtin)
    if proof is frozen_handback.UNANCHORED:
        return _RETRY, None, None
    return _PROVED, proof, None


class FrozenResults:
    """The results of ``frozen=True`` functions: keyed by the call that
    produced them, proved unchanged at each use where cash can (`check`),
    and audited now and then where not."""

    def __init__(self, config: CashConfig, notices: Notices) -> None:
        self._config = config
        self._notices = notices
        #: id(ndarray) -> [weakref, producer, content hash or None]: numpy
        #: results, which cannot carry a tag. Read by the argument hashers.
        self.arrays: dict[int, list] = {}
        #: id(obj) -> [obj, producer, lineage, uses, audit baseline, shape,
        #: proof kind, proof]: a list / tuple / dict result, which carries no
        #: tag and no weakref -- so the object is held here while someone
        #: else holds it too (`remember_container`). Read by the argument
        #: hashers.
        self.containers: dict[int, list] = {}
        # id(obj) -> [weakref, uses, audit baseline or None, shape, proof
        # kind, proof], see `audit`.
        self._uses: LruMemo[int, list] = LruMemo(FROZEN_RESULTS)
        # id(obj) -> (weakref, lineage, content digest): the digests
        # `carry_digest` brought in with a stored or restored result.
        self._carried: LruMemo[int, tuple] = LruMemo(FROZEN_RESULTS)
        _ALL.add(self)

    def untrust(self, obj: Any) -> None:
        """Stop keying *obj*, a frozen result seen changed, by its producer."""
        key = id(obj)
        entry = self.arrays.get(key)
        if entry is not None and entry[0]() is obj:
            self.arrays.pop(key, None)
        entry = self.containers.get(key)
        if entry is not None and entry[0] is obj:
            self._drop_container(key, entry)
        entry = self._uses.get(key)
        if entry is not None and entry[0]() is obj:
            self._drop_use(key, entry, obj)
        entry = self._carried.get(key)
        if entry is not None and entry[0]() is obj:
            self._carried.pop(key, None)
        if own_tag(obj, "_cash_lineage_src") == LINEAGE_SRC_FROZEN:
            set_tags(obj, _cash_lineage_src=LINEAGE_SRC_DECORATOR)

    def arg_names(self, normalized_args: tuple[tuple, dict]) -> list[str]:
        """`explain()`'s list of arguments keyed by a frozen=True producer."""
        args, kwargs = normalized_args
        names = []
        for name, value in [*((f"#{i}", v) for i, v in enumerate(args)), *kwargs.items()]:
            if own_tag(value, "_cash_lineage_src") == LINEAGE_SRC_FROZEN or (self.arrays and id(value) in self.arrays):
                producer = own_tag(value, "_cash_lineage_producer") or (self.arrays.get(id(value), [None, None])[1])
                names.append(f"{name} (the result of {producer}, declared frozen)")
        return names

    def array_hash(self, arr: Any) -> str | None:
        """The content hash of a frozen function's numpy result, computed once.

        Valid while the array is still that object and still read-only; an
        array made writeable again (``a.flags.writeable = True``) is keyed by
        content from then on.
        """
        entry = self.arrays.get(id(arr))
        if entry is None:
            return None
        wref, _producer, content_hash = entry
        if wref() is not arr or getattr(arr, "flags", None) is None or arr.flags.writeable:
            self.arrays.pop(id(arr), None)
            return None
        if content_hash is None:
            content_hash = builtin_hash(arr)
            entry[2] = content_hash
        return content_hash

    def carry_digest(self, obj: Any, digest: str) -> None:
        """Take *digest*, *obj*'s `builtin_hash`, as its content hash.

        A frozen frame passed on was keyed by its content, hashed in full the
        first time each process met it. A process restoring the frame from
        the cache met it for the first time on every hit: a 200 MB frame read
        again in every pool worker, about as long as the restore itself,
        before a consumer could even look its own entry up. The digest is
        taken once, when the result is stored, and kept in the entry, so a
        consumer keys a restored result as fast as the one it was computed
        from -- under the same key, since it is the same content hash.

        It is the audit's baseline too: a change made after the result was
        stored is what the first audit compares against.
        """
        entry = self.arrays.get(id(obj))
        if entry is not None:
            if entry[0]() is obj and entry[2] is None:
                entry[2] = digest
            return
        if own_tag(obj, "_cash_lineage_src") != LINEAGE_SRC_FROZEN:
            return
        lineage = own_tag(obj)
        if lineage is None:
            return
        key = id(obj)
        try:
            wref = _weak.ref(obj, lambda _r, k=key, m=self._carried: m.pop(k, None))
        except TypeError:
            return
        uses = self._uses.get(key)
        if uses is None or uses[0]() is not obj:
            uses = self._new_use(obj)
        if uses is not None and uses[4] == _SCHEDULE and uses[2] is None:
            uses[2] = digest
        self._carried[key] = (wref, lineage, digest)

    def carried_digest(self, obj: Any, lineage: str) -> str | None:
        """The digest `carry_digest` gave *obj*, while it still carries
        *lineage* as its frozen tag."""
        entry = self._carried.get(id(obj))
        if entry is None:
            return None
        wref, carried_lineage, digest = entry
        if wref() is not obj or carried_lineage != lineage or own_tag(obj, "_cash_lineage_src") != LINEAGE_SRC_FROZEN:
            return None
        return digest

    def remember_array(self, arr: Any, producer: str) -> None:
        """Key a frozen function's numpy result by its content, hashed once.

        An array cannot carry a tag; it is made read-only instead, a promise
        numpy enforces: a write raises instead of going stale. Raises what
        numpy raises when the flag cannot be set.
        """
        arr.flags.writeable = False
        entry = self.arrays.get(id(arr))
        if entry is not None and entry[0]() is arr:
            return  # the same array handed back by a hit: its hash stands
        self.arrays[id(arr)] = [
            _weak.ref(arr, lambda _r, k=id(arr), m=self.arrays: m.pop(k, None)),
            producer,
            None,
        ]

    def remember_container(self, obj: Any, producer: str, lineage: str) -> None:
        """Key a frozen function's list, tuple or dict by its producer's lineage.

        A list of two million parsed rows, passed on to two cached consumers,
        was pickled in full for every call -- warm runs about 9x slower than
        uncached -- and ``frozen=True`` on the parser changed nothing: its fast
        path covered numpy arrays alone, and a list cannot carry a tag. Such a
        result is now keyed like a frozen frame: by the lineage of
        the call that produced it, audited now and then (`audit`'s
        schedule).

        It has no weakref either, so the object is held here -- and let go
        again once nothing else holds it, swept on each new entry.
        """
        table = self.containers
        entry = table.get(id(obj))
        if entry is not None and entry[0] is obj and entry[6] != _RETRY:
            # The same object handed back by a hit: what proves it, or its
            # audit's baseline, stands.
            entry[1], entry[2] = producer, f"frozen:{lineage}"
            return
        # What "held by the table alone" reads as, measured the same way: the
        # count differs between Python versions (3.14 borrows references).
        probe = [None, None, None, None, None, None, None, None]
        probe[0] = object()
        alone = sys.getrefcount(probe[0])
        for key, held in list(table.items()):
            if sys.getrefcount(held[0]) <= alone:
                self._drop_container(key, held)
        while len(table) >= FROZEN_CONTAINERS_MAX:
            key = next(iter(table))
            self._drop_container(key, table[key])
        kind, proof, baseline = _start_proof(obj, builtin=False)
        table[id(obj)] = [obj, producer, f"frozen:{lineage}", 0, baseline, frozen_shape(obj), kind, proof]

    def _drop_container(self, key: int, entry: list) -> None:
        """Stop keying *entry*'s container by its producer, and proving it unchanged."""
        if self.containers.get(key) is entry:
            self.containers.pop(key, None)
        if entry[6] == _PROVED:
            frozen_handback.let_go(entry[0], entry[7])
        entry[6], entry[7] = _SCHEDULE, None

    def _drop_use(self, key: int, entry: list, obj: Any) -> None:
        """Forget *entry*, *obj*'s, and stop proving it unchanged."""
        if self._uses.get(key) is entry:
            self._uses.pop(key, None)
        if entry[4] == _PROVED:
            frozen_handback.let_go(obj, entry[5])
        entry[4], entry[5] = _SCHEDULE, None

    def watch(self, obj: Any) -> None:
        """Start proving *obj*, a frozen result just handed out with a tag,
        unchanged at each use as an argument (`audit`), from now on.

        Taken when it is handed out, not at its first use: a change made in
        between would otherwise be the baseline. A result a hit hands back
        again keeps what proves it."""
        key = id(obj)
        entry = self._uses.get(key)
        if entry is not None and entry[0]() is obj and entry[4] != _RETRY:
            return
        self._new_use(obj)

    def _new_use(self, obj: Any) -> list | None:
        """A new `_uses` entry for *obj*, its proof started (`_start_proof`)."""
        key = id(obj)
        try:
            wref = _weak.ref(obj, lambda _r, k=key, m=self._uses: m.pop(k, None))
        except TypeError:
            return None
        kind, proof, baseline = _start_proof(obj, builtin=True)
        entry = [wref, 0, baseline, frozen_shape(obj), kind, proof]
        self._uses[key] = entry
        return entry

    @staticmethod
    def _verdict(obj: Any, kind: int, proof: Any, baseline: str | None, builtin: bool) -> str | None:
        """Is *obj* unchanged by its proof? `frozen_handback.SAME`,
        `CHANGED` or `UNKNOWN` (the proof lost); None when it has none."""
        if kind == _PROVED:
            return frozen_handback.verdict(obj, proof)
        if kind == _SMALL:
            digest = _full_hash(obj, builtin) if baseline is not None else None
            # One that cannot be read: the declaration stands.
            return frozen_handback.CHANGED if digest is not None and digest != baseline else frozen_handback.SAME
        return None

    def container_hash(self, obj: Any) -> str | None:
        """The lineage a frozen list/tuple/dict is keyed by, or None once it
        has been seen to change (KEY-FROZEN-MUTATED, as for a frozen frame)
        or can no longer be proved unchanged (keyed by its content then)."""
        key = id(obj)
        entry = self.containers.get(key)
        if entry is None or entry[0] is not obj:
            return None
        entry[3] += 1
        if entry[6] == _RETRY:
            entry[6], entry[7], baseline = _start_proof(obj, builtin=False)
            if entry[6] == _RETRY:
                entry[6] = _SCHEDULE
            if entry[6] == _SMALL:
                entry[4] = baseline
        verdict = self._verdict(obj, entry[6], entry[7], entry[4], False)
        if verdict == frozen_handback.SAME:
            return entry[2]
        if verdict is not None:
            self._drop_container(key, entry)
            if verdict == frozen_handback.CHANGED:
                self._warn_mutated(obj, entry[1])
            return None
        # Nothing proves it unchanged: its shape on every use, its content
        # on the audit's schedule.
        uses = entry[3]
        shape = frozen_shape(obj)
        if entry[5] is not None and shape != entry[5]:
            self._drop_container(key, entry)
            self._warn_mutated(obj, entry[1])
            return None
        if self._audit_due(uses):
            digest = _full_hash(obj, builtin=False)
            if digest is not None:
                if entry[4] is None:
                    entry[4] = digest
                elif entry[4] != digest:
                    self._drop_container(key, entry)
                    self._warn_mutated(obj, entry[1])
                    return None
        return entry[2]

    def _audit_due(self, uses: int) -> bool:
        """Is the content audit due at an object's *uses*-th use?"""
        return bool(
            (self._config.debug or os.environ.get("CASH_DEBUG"))
            or uses == FROZEN_AUDIT_FIRST
            or (uses > FROZEN_AUDIT_FIRST and uses % FROZEN_AUDIT_EVERY == 0)
        )

    def warn_has_no_effect(self, func_name: str, result: Any) -> None:
        """Say so when ``frozen=True`` cannot apply to what the function returned."""
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "frozen_no_effect",
            f"@cash.cache(frozen=True) on {func_name}: it returned a "
            f"{class_name(type(result))}, which cash cannot mark, so frozen=True has "
            f"no effect on it -- a call that receives it still hashes it in full. "
            f"frozen=True applies to a numpy array, a pandas/polars/modin frame, "
            f"a pyarrow table, a list, tuple or dict, and any object that takes "
            f"an attribute.",
            code="KEY-FROZEN-NO-EFFECT",
            fix="return one of those types, or take frozen=True off; "
            "cash.register_hasher gives the type a cheap identity instead.",
        )

    def _warn_mutated(self, obj: Any, producer: Any = None) -> None:
        """KEY-FROZEN-MUTATED: a result declared frozen is not what it was."""
        producer = producer or own_tag(obj, "_cash_lineage_producer") or "a frozen=True function"
        set_tags(obj, _cash_lineage_src=LINEAGE_SRC_DECORATOR)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            *mutated_text(class_name(type(obj)), producer, _KEYED_BY_CONTENT),
        )

    def audit(self, obj: Any) -> bool:
        """Is a frozen=True result still what it was? False once it is not.

        Proved where cash can, at every use as an argument, from when it was
        handed out (`watch`): a copy-on-write pandas frame by its blocks, a
        dict, list, set or plain instance of 64 KiB or more by the C
        extension's watchers (`frozen_handback.prove`); nothing is hashed.
        A smaller one is hashed at every use and compared with its hash when
        handed out, which costs about what the bookkeeping of a schedule
        does. A proof lost without a change seen (its watch dropped for
        room) lets it go: it is keyed by its content from then on.

        Anything else -- no C extension, a type the watchers cannot follow
        -- is trusted, and checked now and then: its shape at every use, its
        content at its 8th use as an argument and every 64th after that, and
        at every use under CASH_DEBUG. The first check records a baseline --
        the content hash for a type cash content-hashes, a digest of the
        pickle otherwise -- and each later one compares. An object that
        cannot be pickled cannot be audited, and stays trusted.

        On a change, KEY-FROZEN-MUTATED names the producer, the object's tag
        stops being trusted, and it is keyed by its content from then on.
        """
        key = id(obj)
        entry = self._uses.get(key)
        if entry is None or entry[0]() is not obj:
            entry = self._new_use(obj)
            if entry is None:
                return True
        entry[1] += 1
        if entry[4] == _RETRY:
            entry[4], entry[5], baseline = _start_proof(obj, builtin=True)
            if entry[4] == _RETRY:
                entry[4] = _SCHEDULE
            if entry[4] == _SMALL:
                entry[2] = baseline
        verdict = self._verdict(obj, entry[4], entry[5], entry[2], True)
        if verdict == frozen_handback.SAME:
            return True
        if verdict is not None:
            self._drop_use(key, entry, obj)
            if verdict == frozen_handback.CHANGED:
                self._warn_mutated(obj)
            else:
                set_tags(obj, _cash_lineage_src=LINEAGE_SRC_DECORATOR)
            return False
        uses = entry[1]
        shape = frozen_shape(obj)
        if entry[3] is not None and shape != entry[3]:
            self._uses.pop(key, None)
            self._warn_mutated(obj)
            return False
        if not self._audit_due(uses):
            return True
        digest = _full_hash(obj, builtin=True)
        if digest is None:
            return True
        if entry[2] is None:
            entry[2] = digest
            return True
        if entry[2] == digest:
            return True
        self._uses.pop(key, None)
        self._warn_mutated(obj)
        return False

    def forget_container(self, obj: Any) -> None:
        """Stop trusting a frozen result a call was just seen to change."""
        entry = self.containers.get(id(obj))
        if entry is None or entry[0] is not obj:
            return
        self._drop_container(id(obj), entry)
        warn_diagnostic(
            CashImpurityWarning,
            "KEY-FROZEN-MUTATED",
            f"a {class_name(type(obj))} returned by {entry[1]}, which is declared "
            f"@cash.cache(frozen=True), was modified in place by a cached call. "
            f"From now on it is keyed by its contents.",
            f"take frozen=True off {entry[1]} if its result is meant to be "
            f"modified, or modify a copy (`obj = copy.deepcopy(obj)`) instead.",
        )
