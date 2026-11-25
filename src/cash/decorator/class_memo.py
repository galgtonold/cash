"""What a class folds to, remembered in the process while nothing the class
holds or reaches can have moved.

Two folds read the same class again on every call that meets it:

* the code an argument carries (`CodeArgs.carrier_parts`): an instance's
  class, its data, and what its methods and their helpers read. For an
  imbalanced-learn sampler handed to a cached function, with the library
  installed editable (so its code is user code), 8-20 ms a call, hits
  included;
* the digest of a class a call's trace names (`trace_items.class_digest`),
  about a dozen of them, 8-9 ms a miss.

Both are served again while the class's world provably stands still, the
proof the state memo uses (`state_memo`): the class is watched once
(`hit_memo.watch_class`): its attributes along the MRO, the code of its
methods, the globals, modules and classes that code names, and the helpers
the analyses of its methods resolved, transitively. Data no watcher sees is
read whole again on every check (`hit_memo.ContentProbe`). A result is
served while the C extension's world epoch and cash's own generation
(`state_memo.GEN`) are what they were when it was computed, and every cell,
class tag and probe of the watch still holds. Anything else -- a class whose
world holds what can be neither watched nor read whole -- is folded every
time, as before. A class whose world reaches a cached function is watched
for the code arguments carry (`CodeArgs`), and a fold of it remembered only
when it asked for no cached function's state (`Capture.callee_states`),
which is that function's whole world and no part of the watch; the digest
a trace names is not.

A world that keeps moving would be scanned on every call for nothing: a
class whose proof broke before it served is scanned again only after a
doubling number of calls, as the state memo backs off (`_Proof`).

``CASH_CLASS_MEMO=0`` turns it off; ``CASH_CLASS_MEMO_VERIFY=1`` (or the
state memo's or fast path's verify mode) folds again behind every reuse and
raises `ClassMemoMismatch` when anything differs. Without the C extension
nothing is remembered.
"""

from __future__ import annotations

import abc
import os
from collections.abc import Callable
from typing import Any

from .. import _native, _verify, _weak
from .._memo import LruMemo
from .._type_memo import class_module, class_qualname
from . import hit_memo, main_rebinding, state_memo

__all__ = [
    "CALLEE_STATES_BUILT",
    "DIGESTS",
    "ENABLED",
    "MISS",
    "STATS",
    "VERIFY",
    "ClassMemo",
    "ClassMemoMismatch",
    "give_up",
    "proof_for",
    "status",
]


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


#: Is the memo on? Needs the extension's world watch, as the state memo does.
ENABLED = state_memo.ENABLED and _flag("CASH_CLASS_MEMO", "1")
#: Fold again behind every reuse and compare.
VERIFY = ENABLED and (_verify.requested("CASH_CLASS_MEMO_VERIFY") or state_memo.VERIFY)

#: Classes watched at once; a class beyond is scanned again when met again.
_PROOFS = 1024
#: Calls a class whose proof broke unused waits, at most, before a new scan.
_MAX_BACKOFF = 1024

_cw = _native.cwatch

STATS = {"scans": 0, "refused": 0, "served": 0, "recorded": 0, "verified": 0}

#: How many times a cached function's state was asked for
#: (`KeyBuilder.callee_state`): a fold that asked for none has nothing in
#: it that a cached function's own world decides (`Capture.callee_states`).
CALLEE_STATES_BUILT = [0]


class ClassMemoMismatch(_verify.Mismatch):
    """``CASH_CLASS_MEMO_VERIFY=1``: a remembered class fold is not what
    folding it again gives."""


def status() -> dict:
    return {"enabled": ENABLED, "verify": VERIFY, **STATS}


_MISS = object()
#: Scans taken so far: what tells one watch of a class from the next.
_STAMP = [0]


class _Proof:
    """One watch of a class's world (`hit_memo.watch_class`), at the world
    epoch and generation it was taken at."""

    __slots__ = (
        "_cls",
        "backoff",
        "callees_ok",
        "cells",
        "family",
        "gen",
        "outside",
        "reaches_cached",
        "served",
        "skipped",
        "stamp",
        "tags",
        "world",
    )

    def __init__(self, cls: type, callees_ok: bool = False) -> None:
        self._cls = _weak.ref(cls)
        #: May the world reach a cached function, and does it: a result
        #: is then remembered only when its fold asked for no cached
        #: function's state (`Capture.callee_states`).
        self.callees_ok = callees_ok
        self.reaches_cached = False
        self.world = -1
        self.gen = -1
        self.stamp = 0
        self.cells: tuple = ()
        self.tags: tuple = ()
        self.family: Any = None
        #: The namespaces the class's world provably reads nothing of
        #: (`_outside_namespaces`): a plain rebinding there moves the world
        #: epoch, not this watch.
        self.outside: frozenset[int] = frozenset()
        #: Reuses since the last scan, calls skipped since, and how many to skip.
        self.served = 0
        self.skipped = 0
        self.backoff = 0

    @property
    def cls(self) -> type | None:
        return self._cls()

    def holds(self) -> bool:
        if self.world < 0 or self.gen != state_memo.GEN[0]:
            return False
        world = _cw.world_epoch()
        if self.world != world and not (self.outside and _moved_only_in(self.world, self.outside)):
            return False
        # Once a round of checks (`hit_memo.probe_round`): a key build asks
        # again for the watch of each class it meets, about 30 times for an
        # lmfit fit, and only cash's code runs between. What the round found
        # is forgotten when the world epoch moves; a scan gives a new stamp.
        seen = hit_memo.round_checks()
        mark = ("proof", id(self))
        if seen is not None and seen.get(mark) == (self.stamp, self.world, self.gen):
            return True
        cells, tags = _strong(self.cells), _strong(self.tags)
        if cells is None or tags is None or not _cw.world_holds(cells, tags):
            return False
        if self.family is not None and not self.family.holds([None] * 11):
            return False
        # Every move since was a plain rebinding outside the world: it
        # holds at the epoch now, and the next check looks only past it.
        self.world = world
        if seen is not None:
            seen[mark] = (self.stamp, self.world, self.gen)
        return True

    def scan(self, reports: Callable[[], Any]) -> bool:
        """Watch the class now, when due; True when it is watched at the
        current world epoch and generation."""
        if self.skipped < self.backoff:
            self.skipped += 1
            return False
        self.backoff = 0 if self.served or not self.stamp else min(max(self.backoff * 2, 1), _MAX_BACKOFF)
        self.skipped = self.served = 0
        self.world = -1
        _STAMP[0] += 1
        self.stamp = _STAMP[0]
        STATS["scans"] += 1
        gen = state_memo.GEN[0]
        try:
            found = reports()
            world = _cw.world_epoch()
            cls = self.cls
            if cls is None:
                return False
            reached: list = []
            watched: set[int] = set()
            cells, tags, self.family = main_rebinding.watching(
                watched, lambda: hit_memo.watch_class(cls, found, reached if self.callees_ok else None)
            )
            self.reaches_cached = bool(reached)
            self.cells, self.tags = _weakened(cells), _weakened(tags)
            self.outside = _outside_namespaces(cls, found, cells, tags, watched)
        except Exception:  # noqa: BLE001 - fails closed: Ineligible, or a scan that broke: folded every time
            STATS["refused"] += 1
            self.backoff = _MAX_BACKOFF
            return False
        # What moved during the scan is not proven: scanned again next time.
        if _cw.world_epoch() != world or state_memo.GEN[0] != gen:
            return False
        self.world, self.gen = world, gen
        return True


_proofs: LruMemo[tuple, _Proof] = LruMemo(_PROOFS)

#: `cwatch.world_moved_only_in`, or None with an extension without it.
_moved_only_in = getattr(_cw, "world_moved_only_in", None)


def _outside_namespaces(cls: type, reports: Any, cells: tuple, tags: tuple, watched: set[int]) -> frozenset[int]:
    """The namespaces a plain rebinding in which leaves *cls*'s watch
    standing (`main_rebinding.outside_namespaces`): ``__main__``, unless the
    class's world reads it. A notebook rebinds its variables on every cell,
    and each one whose name the scanned code names moved the world epoch, so
    a class handed to every call was folded again on each, hits included."""
    if _moved_only_in is None:
        return frozenset()
    return main_rebinding.outside_namespaces(reports, cells, tags, watched, (class_module(cls),))


class _WeakType:
    """A class a watch names, held weakly: a memo of the process must keep
    no class alive (one defined anew under its name would stay among its
    base's ``__subclasses__``)."""

    __slots__ = ("_ref",)

    def __init__(self, cls: type) -> None:
        self._ref = _weak.ref(cls)

    def __call__(self) -> type | None:
        return self._ref()


def _weakened(flat: tuple) -> tuple:
    return tuple(_WeakType(x) if isinstance(x, type) else x for x in flat)


def _strong(flat: tuple) -> tuple | None:
    """*flat* with each class `_weakened` took back; None when one is gone."""
    out = []
    for x in flat:
        if type(x) is _WeakType:
            x = x()
            if x is None:
                return None
        out.append(x)
    return tuple(out)


def proof_for(cls: type, purpose: str, reports: Callable[[], Any] = tuple, callees_ok: bool = False) -> _Proof | None:
    """The watch of *cls*'s world, holding now, or None when there is none
    to be had. *reports* gives the analyses whose helpers the watch covers
    too; *purpose* tells watches with different reports apart.

    *callees_ok*: a world that reaches a cached function is watched too
    (`_Proof.reaches_cached`); the caller remembers only a fold that asked
    for no cached function's state, and gives up on the class otherwise
    (`give_up`). A cached function's state is its whole world, globals and
    callees included, which this watch does not cover; its wrapper and the
    code it wraps, which a fold reads by name, the watch does."""
    if not ENABLED or type(cls) not in (type, abc.ABCMeta):
        return None
    key = (id(cls), purpose)
    proof = _proofs.get(key)
    if proof is None or proof.cls is not cls:
        proof = _Proof(cls, callees_ok)
        _proofs[key] = proof
    if proof.holds():
        return proof
    return proof if proof.scan(reports) and proof.holds() else None


def give_up(proof: _Proof) -> None:
    """*proof*'s class folds to what cannot be remembered under it (a
    cached function's state): folded every time, scanned again only after
    the longest back-off."""
    STATS["refused"] += 1
    proof.world = -1
    proof.skipped = 0
    proof.backoff = _MAX_BACKOFF


class ClassMemo:
    """Results remembered under the watch of a class (`proof_for`)."""

    def __init__(self, size: int = 4 * _PROOFS) -> None:
        self._entries: LruMemo[tuple, tuple] = LruMemo(size)
        #: (module, qualified name, the rest of the key) -> the key of the
        #: class remembered under that name. What a fold left in the key
        #: build's records can hold its class: a class defined anew under
        #: the same name (a notebook cell run again) lets the old one go.
        self._named: dict[tuple, tuple] = {}

    def get(self, key: tuple, proof: _Proof) -> Any:
        """What `put` remembered for *key* under this very watch of the
        class, or `MISS`."""
        entry = self._entries.get(key)
        if entry is None or entry[0]() is not proof.cls or entry[1] != proof.stamp:
            return _MISS
        proof.served += 1
        STATS["served"] += 1
        return entry[2]

    def put(self, key: tuple, proof: _Proof, value: Any) -> None:
        """Remember *value*, computed under *proof*, when nothing moved
        while it was."""
        cls = proof.cls
        if cls is None or not proof.holds():
            return
        named = (class_module(cls), class_qualname(cls), key[1:])
        before = self._named.get(named)
        if before is not None and before != key:
            self._entries.pop(before)
        if len(self._named) > 4 * self._entries.maxsize:
            self._named.clear()
        self._named[named] = key
        self._entries[key] = (proof._cls, proof.stamp, value)
        STATS["recorded"] += 1


MISS = _MISS

#: `trace_items.class_digest` of a class on its own, by class.
DIGESTS = ClassMemo()


# -- a fold's effects on the key build, captured and replayed ---------------


class _SeenAbove(set):
    """A key build's set of classes folded (`CLASSES_FOLDED`) as a captured
    fold sees it: its own additions here, the build's below, each question
    the build's answered noted, and which it answered "folded"."""

    def __init__(self, below: set) -> None:
        super().__init__()
        self.below = below
        self.asked: set = set()
        self.found: set = set()

    def __contains__(self, key: object) -> bool:
        if set.__contains__(self, key):
            return True
        self.asked.add(key)
        if key in self.below:
            self.found.add(key)
            return True
        return False


class _ReadsAbove(dict):
    """`_SeenAbove` for a key build's record of the globals folds done
    (`READS_FOLDED`): what the build held for each key it answered."""

    def __init__(self, below: dict) -> None:
        super().__init__()
        self.below = below
        self.asked: set = set()
        self.found: dict = {}

    def _below(self, key: Any) -> bool:
        self.asked.add(key)
        if key in self.below:
            self.found.setdefault(key, self.below[key])
            return True
        return False

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return self.below.get(key, default) if self._below(key) else default

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or self._below(key)

    def __getitem__(self, key: Any) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if self._below(key):
            return self.below[key]
        raise KeyError(key)


class Capture:
    """What one fold left in the key build's records, and which of the
    build's own answers it asked for, with the ones it heard "folded"."""

    __slots__ = (
        "classes",
        "classes_asked",
        "callee_states",
        "classes_found",
        "digests",
        "ledger",
        "lookups",
        "owners",
        "reads",
        "reads_asked",
        "reads_found",
        "reports",
        "ttls",
        "watch",
    )


def _records() -> tuple:
    from ..code_swaps import FOLDED_CODE
    from ..dependency_state import STATE_LEDGER
    from .call_state import CAPTURE_WATCH, CARRIER_LOOKUPS, REACHED_TTLS
    from .class_data import CLASSES_FOLDED
    from .global_values import KEY_BUILD_DIGESTS
    from .globals_fold import READS_FOLDED

    return (
        STATE_LEDGER,
        CAPTURE_WATCH,
        FOLDED_CODE,
        REACHED_TTLS,
        KEY_BUILD_DIGESTS,
        CARRIER_LOOKUPS,
        CLASSES_FOLDED,
        READS_FOLDED,
    )


def collecting() -> tuple[bool, ...]:
    """Which of the key build's records this build keeps. A real call keeps
    them all, ``explain()`` no drift-guard watch, the key verify mode builds
    behind a fast hit neither watch nor ledger: a fold captured in one kind
    of build holds only what that kind keeps (`captured`), so it serves only
    a build that keeps the same ones."""
    return tuple(v.get() is not None for v in _records())


def _outside() -> tuple:
    from ..dependency_state import OUTSIDE_READS_FOLDED
    from ..effects import ENVIRONMENT_READS

    return state_memo.OUTSIDE[0], OUTSIDE_READS_FOLDED[0], ENVIRONMENT_READS[0], state_memo.GEN[0]


def captured(compute: Callable[[], Any]) -> tuple[Capture | None, Any]:
    """Run *compute* (a fold into the key build's records) and return what
    it gave, with a `Capture` of what it left in the records, or None when
    it cannot be replayed into another build: it folded something no
    watcher sees (a random stream, a file, the environment). What it found
    the build had folded already (a class another argument's fold went
    through first) is kept with it: the capture serves only a build that
    answers alike (`replayable`). The records end up as running it in
    place leaves them."""
    from ..code_swaps import FoldedCode

    ledger_v, watch_v, code_v, ttls_v, digests_v, lookups_v, classes_v, reads_v = _records()
    below = [v.get() for v in (ledger_v, watch_v, code_v, ttls_v, digests_v, lookups_v, classes_v, reads_v)]
    ledger, watch, code, ttls, digests, lookups, classes, reads = below
    own = (
        None if ledger is None else {},
        None if watch is None else {},
        None if code is None else FoldedCode(),
        None if ttls is None else [],
        None if digests is None else {},
        [],
        None if classes is None else _SeenAbove(classes),
        None if reads is None else _ReadsAbove(reads),
    )
    tokens = [
        (var, var.set(value))
        for var, value in zip((ledger_v, watch_v, code_v, ttls_v, digests_v, lookups_v, classes_v, reads_v), own)
    ]
    before = _outside()
    states_before = CALLEE_STATES_BUILT[0]
    try:
        result = compute()
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
    cap = Capture()
    cap.callee_states = CALLEE_STATES_BUILT[0] - states_before
    cap.ledger, cap.watch = own[0] or {}, own[1] or {}
    cap.owners = dict(own[2].owners) if own[2] is not None else {}
    cap.reports = dict(own[2].reports) if own[2] is not None else {}
    cap.ttls, cap.digests, cap.lookups = tuple(own[3] or ()), own[4] or {}, tuple(own[5])
    own_classes, own_reads = own[6], own[7]
    cap.classes = frozenset(own_classes) if own_classes is not None else frozenset()
    cap.classes_asked = frozenset(own_classes.asked) if own_classes is not None else frozenset()
    cap.classes_found = frozenset(own_classes.found) if own_classes is not None else frozenset()
    cap.reads = dict(own_reads) if own_reads is not None else {}
    cap.reads_asked = frozenset(own_reads.asked) if own_reads is not None else frozenset()
    cap.reads_found = dict(own_reads.found) if own_reads is not None else {}
    replay(cap)
    return (None if _outside() != before else cap), result


def replayable(cap: Capture) -> bool:
    """Would *cap*'s fold, run now, ask the same questions of the key
    build's records and hear the same answers: "not folded yet" where it
    heard that, "folded" where it heard that, by the same fold?"""
    _l, _w, _c, _t, _d, _lk, classes_v, reads_v = _records()
    classes, reads = classes_v.get(), reads_v.get()
    if classes is not None:
        if any((k in classes) != (k in cap.classes_found) for k in cap.classes_asked):
            return False
    elif cap.classes_found:
        return False
    if reads is None:
        return not cap.reads_found
    for k in cap.reads_asked:
        if k in cap.reads_found:
            if k not in reads or not _same_answer(reads[k], cap.reads_found[k]):
                return False
        elif k in reads:
            return False
    return True


def _same_answer(now: Any, then: Any) -> bool:
    """Is a record of the build (`READS_FOLDED`: the function folded and
    the globals it skipped) what it was: the same objects, or equal sets
    and strings?"""
    if now is then:
        return True
    if type(now) is not tuple or type(then) is not tuple or len(now) != len(then):
        return False
    return all(a is b or (type(a) is type(b) and type(a) in (frozenset, str) and a == b) for a, b in zip(now, then))


def replay(cap: Capture) -> None:
    """Leave in the key build's records what *cap*'s fold left in them."""
    ledger_v, watch_v, code_v, ttls_v, digests_v, lookups_v, classes_v, reads_v = _records()
    ledger, watch, code, ttls = ledger_v.get(), watch_v.get(), code_v.get(), ttls_v.get()
    digests, lookups, classes, reads = digests_v.get(), lookups_v.get(), classes_v.get(), reads_v.get()
    if ledger is not None:
        ledger.update(cap.ledger)
    if watch is not None:
        watch.update(cap.watch)
    if code is not None:
        code.owners.update(cap.owners)
        code.reports.update(cap.reports)
    if ttls is not None:
        ttls.extend(cap.ttls)
    if digests is not None:
        for k, v in cap.digests.items():
            digests.setdefault(k, v)
    if lookups is not None:
        lookups.extend(cap.lookups)
    if classes is not None:
        classes.update(cap.classes)
    if reads is not None:
        reads.update(cap.reads)
