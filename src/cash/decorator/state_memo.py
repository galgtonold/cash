"""A cached function's state segment, remembered while nothing it reads moved.

Every call of a ``@cash.cache`` function builds the state segment of its key
(`KeyBuilder._code_state`): the function's code, its helpers', its cached
callees', the globals all of them read, captures, defaults, the environment.
For a library of many small decorated functions that is most of what a call
costs cash -- pvlib's test suite with every function decorated ran 20 times
slower than plain, a third of it in this fold -- and nearly every call folds
exactly what the last one did.

The fold is remembered per function and served again while three things hold:

* **The world**: what the fold reads of the program, watched as the fast hit
  path watches it (`hit_memo.scan`): the function's code, globals, closures,
  defaults, classes, helpers and cached callees, transitively. Every value
  met must be one whose content cannot change unseen (an atom, a tuple of
  them, code, a watched plain dict); anything else (a list, an array, an
  instance) keeps the function on the ordinary path. A change to any of it
  moves the C extension's world epoch, and a renamed function, a rebound
  closure cell or a rebased class is checked on every reuse
  (`cwatch.world_holds`), as is what ABCs registered.
* **Cash's own state**: what the fold reads of cash -- the registrations, the
  analysis of each function and the graph of cached calls, the names a call
  was seen to change (`LearnedMutations`), the random streams a function was
  seen to draw from, the pinned own sources and the settings -- moves the
  generation `GEN` whenever it changes (`moved`). Unlike the fast hit path's
  cash epoch it does not move on a plain miss, which in a library's test
  suite is almost every call.
* **The context of the build**: a capture is served only into a context like
  the one it was taken in (`_mode`): the same containers present (ledger,
  drift-guard watch, folded code, reached TTLs, a key build's digests and
  folds) and all of them still empty, no walk of another cached function's
  state running around it but the function's own, the same explain mode.

A fold whose output does not come from the world alone is never remembered:
one that folded a random stream's epoch (its position moves with every
draw) or a file's content. An environment variable the fold read by name
(``os.environ.get("X")``, ``os.getenv("X")``, ``os.environ["X"]``) is the
exception: the capture keeps what each such read answered
(`effects.ENVIRONMENT_FOLDED`) and is served only while every one of them
answers the same, so a write to another variable -- numba's or a thread
pool's at import or call time -- leaves it standing, and a write to one it
read builds the segment again. The environment read as a whole is no such
read: the key folds the mapping's content as a value, which no watcher
covers, and the segment is not remembered.

What a fold leaves behind besides its digest -- the stages of the state
chain, the ledger that names each part for a miss's explanation, what the
drift guard watches, the code it folded, the run-time lookups by name in
the code a data global carries (`CARRIER_LOOKUPS`), the TTLs and digests of
the cached functions it reached, the classes and globals it folded for the rest of the
key, the cached functions its walk touched -- is captured whole: the fold
runs once into empty containers of its own (`_sandboxed`), and what they hold
afterwards is written into the caller's, on the first call as on every
reuse. A reuse therefore leaves exactly what building again would.

``CASH_STATE_MEMO=0`` turns it off; ``CASH_STATE_MEMO_VERIFY=1`` (or
``CASH_HITPATH_VERIFY=1``) builds the segment again behind every reuse and
raises `StateMemoMismatch` when the digest or anything it leaves behind
differs. Without the C extension there is no proof and nothing is reused.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import _native
from .._active import ACTIVE_CONFIG, EXPLAINING
from ..code_swaps import FOLDED_CODE, FoldedCode
from ..dependency_state import OUTSIDE_READS_FOLDED, STATE_LEDGER
from ..effects import ENVIRONMENT_FOLDED, ENVIRONMENT_READS, ENVIRONMENT_READS_FOLDED, environment_value_now
from . import hit_memo, walk_scope
from .call_state import CAPTURE_WATCH, CARRIER_LOOKUPS, REACHED_TTLS
from .class_data import CLASSES_FOLDED
from .global_values import KEY_BUILD_DIGESTS
from .globals_fold import READS_FOLDED

if TYPE_CHECKING:
    from collections.abc import Callable

    from .cached_function import CachedFunction

__all__ = [
    "ENABLED",
    "GEN",
    "GenerationDict",
    "VERIFY",
    "StateMemoMismatch",
    "folded_outside_world",
    "memo_for",
    "moved",
    "status",
]

_cw = _native.cwatch


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


#: Is the memo on? Needs the extension's world watch.
ENABLED = hit_memo.ENABLED and _cw is not None and hasattr(_cw, "world_holds") and _flag("CASH_STATE_MEMO", "1")
#: Rebuild behind every reuse and compare.
VERIFY = ENABLED and (_flag("CASH_STATE_MEMO_VERIFY", "0") or hit_memo.VERIFY)

_world_epoch = _cw.world_epoch if ENABLED else None
_world_holds = _cw.world_holds if ENABLED else None

#: The generation of cash's own state that a state segment reads. Moved by
#: `moved`, never on a plain call.
GEN = [0]

#: Folds whose output does not come from the world alone (a random stream's
#: epoch, an environment variable, a data source token): counted by
#: `folded_outside_world` and the modules it sums, and a
#: build that counted one is not remembered.
OUTSIDE = [0]

#: Why a remembered segment had to be checked again: the world moved, or
#: cash's own state did (``status()``; for telling what keeps moving).
WHY: Counter = Counter()
#: Reuses, builds remembered, and reuses checked by the verify mode.
STATS = {"reused": 0, "remembered": 0, "verified": 0, "scans": 0, "scans_failed": 0}


def moved() -> None:
    """Something cash keeps that a state segment reads changed (a
    registration, an analysis, a learned mutation or draw, a pin, a
    setting, a clear): nothing remembered before is served again."""
    GEN[0] += 1


class GenerationDict(dict):
    """A dict of cash's own state that a state segment reads (the
    registry's source hashes, functions, data sources, reports): any write
    to it, through the registry or straight into it, is a `moved`. Reads
    are a plain dict's."""

    __slots__ = ()

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        moved()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        moved()

    def __ior__(self, other):
        result = super().__ior__(other)
        moved()
        return result

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        moved()

    def setdefault(self, key, default=None):
        result = super().setdefault(key, default)
        moved()
        return result

    def pop(self, *args):
        result = super().pop(*args)
        moved()
        return result

    def popitem(self):
        result = super().popitem()
        moved()
        return result

    def clear(self) -> None:
        super().clear()
        moved()


def folded_outside_world() -> None:
    """The fold now running folded something no watcher sees."""
    OUTSIDE[0] += 1


def _outside() -> int:
    """Folds no watcher sees, the environment reads by name aside (they
    are checked on every reuse, `_Capture.env`)."""
    return OUTSIDE[0] + OUTSIDE_READS_FOLDED[0]


def _env_holds(env: tuple) -> bool:
    """Does every environment read *env* holds still answer the same?"""
    for entry, value in env:
        if environment_value_now(entry) != value:
            return False
    return True


def _env_replay(env: tuple) -> None:
    """What folding *env* again leaves behind: the counters moved, and the
    reads handed to a build running around this one."""
    if not env:
        return
    ENVIRONMENT_READS[0] += len(env)
    ENVIRONMENT_READS_FOLDED[0] += 1
    outer = ENVIRONMENT_FOLDED.get()
    if outer is not None:
        outer[0] += 1
        for entry, value in env:
            outer[1].setdefault(entry, value)
            if outer[1][entry] != value:
                outer[0] = -(1 << 30)


def status() -> dict:
    """What the memo did in this process."""
    return {"enabled": ENABLED, "verify": VERIFY, "generation": GEN[0], **STATS, **WHY}


class StateMemoMismatch(AssertionError):
    """``CASH_STATE_MEMO_VERIFY=1``: a reused state segment, or what it
    left behind, is not what building it again gives."""


@dataclass(frozen=True)
class _Capture:
    """Everything one fold of the state segment left behind (see the module doc)."""

    state: str
    chain: tuple
    ledger: dict
    watch: dict
    owners: dict
    reports: dict
    ttls: tuple
    digests: dict
    classes: frozenset
    reads: dict
    touched: frozenset
    low: int
    members: frozenset
    lookups: tuple
    #: The environment variables the fold read by name, with what each
    #: answered: served again only while each answers the same.
    env: tuple = ()
    #: Did every environment fold go through `ENVIRONMENT_FOLDED`? (False
    #: when one ran in another thread, or a read answered twice.)
    env_complete: bool = True


def _mode(name: str, note: bool) -> tuple | None:
    """The context a capture taken now stands for, or None when a capture
    cannot: a container the fold writes into already holds something, or
    a walk other than this function's own is running around it."""
    ledger = STATE_LEDGER.get()
    if ledger is not None and (len(ledger) > 1 or (ledger and "@chain" not in ledger)):
        return None
    watch = CAPTURE_WATCH.get()
    folded = FOLDED_CODE.get()
    ttls = REACHED_TTLS.get()
    digests = KEY_BUILD_DIGESTS.get()
    classes = CLASSES_FOLDED.get()
    reads = READS_FOLDED.get()
    lookups = CARRIER_LOOKUPS.get()
    if watch or ttls or digests or classes or reads or lookups:
        return None
    if folded is not None and (folded.owners or folded.reports):
        return None
    from .runtime import CALLEE_STATES

    callees = CALLEE_STATES.get()
    if callees and callees != {name}:
        return None
    frames = walk_scope.stack()
    item = None
    if frames:
        if len(frames) > 1:
            return None
        frame = frames[0]
        if frame.touched or frame.members or frame.low != sys.maxsize:
            return None
        item = frame.item
    building = walk_scope.building_now()
    if building and building != (name,):
        return None
    if not walk_scope.walk_is_idle():
        return None
    config = ACTIVE_CONFIG.get()
    return (
        note,
        ledger is None,
        watch is None,
        folded is None,
        ttls is None,
        digests is None,
        classes is None,
        reads is None,
        lookups is None,
        bool(callees),
        item,
        building,
        EXPLAINING.get(),
        id(config) if config is not None else None,
    )


def _sandboxed(build: Callable[[list], str]) -> _Capture:
    """Run *build* (the fold, into the chain list it is given) on empty
    containers of its own, one for each the caller has, and return what
    they hold afterwards."""
    ledger = STATE_LEDGER.get()
    watch = CAPTURE_WATCH.get()
    folded = FOLDED_CODE.get()
    ttls = REACHED_TTLS.get()
    digests = KEY_BUILD_DIGESTS.get()
    classes = CLASSES_FOLDED.get()
    reads = READS_FOLDED.get()
    new_ledger = None if ledger is None else {}
    new_watch = None if watch is None else {}
    new_folded = None if folded is None else FoldedCode()
    new_ttls = None if ttls is None else []
    new_digests = None if digests is None else {}
    new_classes = None if classes is None else set()
    new_reads = None if reads is None else {}
    new_lookups: list = []
    tokens = (
        (STATE_LEDGER, STATE_LEDGER.set(new_ledger)),
        (CAPTURE_WATCH, CAPTURE_WATCH.set(new_watch)),
        (FOLDED_CODE, FOLDED_CODE.set(new_folded)),
        (REACHED_TTLS, REACHED_TTLS.set(new_ttls)),
        (KEY_BUILD_DIGESTS, KEY_BUILD_DIGESTS.set(new_digests)),
        (CLASSES_FOLDED, CLASSES_FOLDED.set(new_classes)),
        (READS_FOLDED, READS_FOLDED.set(new_reads)),
        (CARRIER_LOOKUPS, CARRIER_LOOKUPS.set(new_lookups)),
    )
    frames = walk_scope.stack()
    chain: list = []
    env_folded: list = [0, {}]
    env_token = ENVIRONMENT_FOLDED.set(env_folded)
    env_before = ENVIRONMENT_READS_FOLDED[0]
    try:
        with walk_scope.apart(frames[0].item if frames else None) as own:
            state = build(chain)
    finally:
        ENVIRONMENT_FOLDED.reset(env_token)
        for var, token in reversed(tokens):
            var.reset(token)
    env = tuple(sorted(env_folded[1].items()))
    env_complete = env_folded[0] == ENVIRONMENT_READS_FOLDED[0] - env_before
    outer = ENVIRONMENT_FOLDED.get()
    if outer is not None and env_folded[0]:
        # A build around this one folded these reads too.
        outer[0] = outer[0] + env_folded[0] if env_complete else -(1 << 30)
        for entry, value in env:
            outer[1].setdefault(entry, value)
            if outer[1][entry] != value:
                outer[0] = -(1 << 30)
    return _Capture(
        state,
        tuple(chain),
        new_ledger or {},
        new_watch or {},
        dict(new_folded.owners) if new_folded is not None else {},
        dict(new_folded.reports) if new_folded is not None else {},
        tuple(new_ttls or ()),
        new_digests or {},
        frozenset(new_classes or ()),
        new_reads or {},
        frozenset(own.touched) if own is not None else frozenset(),
        own.low if own is not None else sys.maxsize,
        frozenset(own.members) if own is not None else frozenset(),
        tuple(new_lookups),
        env,
        env_complete,
    )


def _apply(cap: _Capture, chain: list) -> None:
    """Leave behind in the caller's containers what the fold left in its own."""
    chain.extend(cap.chain)
    ledger = STATE_LEDGER.get()
    if ledger is not None:
        ledger.update(cap.ledger)
    watch = CAPTURE_WATCH.get()
    if watch is not None:
        watch.update(cap.watch)
    folded = FOLDED_CODE.get()
    if folded is not None:
        folded.owners.update(cap.owners)
        folded.reports.update(cap.reports)
    ttls = REACHED_TTLS.get()
    if ttls is not None:
        ttls.extend(cap.ttls)
    digests = KEY_BUILD_DIGESTS.get()
    if digests is not None:
        digests.update(cap.digests)
    classes = CLASSES_FOLDED.get()
    if classes is not None:
        classes.update(cap.classes)
    reads = READS_FOLDED.get()
    if reads is not None:
        reads.update(cap.reads)
    lookups = CARRIER_LOOKUPS.get()
    if lookups is not None:
        lookups.extend(cap.lookups)
    frames = walk_scope.stack()
    if frames:
        frame = frames[0]
        frame.touched |= cap.touched
        frame.low = min(frame.low, cap.low)
        frame.members |= cap.members


def _same(a: Any, b: Any, depth: int = 0) -> bool:
    if a is b:
        return True
    if depth > 40 or type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], depth + 1) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, depth + 1) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - not comparable: not the same
        return False


def _differences(kept: _Capture, built: _Capture) -> list[str]:
    out = [
        name
        for name in (
            "state",
            "chain",
            "ledger",
            "watch",
            "ttls",
            "classes",
            "touched",
            "low",
            "members",
            "lookups",
            "env",
        )
        if not _same(getattr(kept, name), getattr(built, name))
    ]
    for name in ("owners", "reports", "digests", "reads"):
        if getattr(kept, name).keys() != getattr(built, name).keys():
            out.append(name)
    return out


class StateMemo:
    """One cached function's remembered state segments, by context."""

    #: A scan is due this long after the last one, whatever the backoff
    #: (as `hit_memo.HitMemo.RESCAN_AFTER_S`).
    RESCAN_AFTER_S = 0.25

    def __init__(self, spec: CachedFunction) -> None:
        self.spec = spec
        #: The world epoch and generation the scan and the entries are at.
        self.world = -1
        self.gen = -1
        self.cells: tuple = ()
        self.tags: tuple = ()
        self.family: Any = None
        self.entries: dict[tuple, _Capture] = {}
        self.slow = 0
        self.backoff = 1
        self.scanned_at = 0.0
        self.reused_since_scan = 0

    def _holds(self) -> bool:
        if not _world_holds(self.cells, self.tags):
            return False
        family = self.family
        return family is None or family.holds([None] * 11)

    def _scan(self, registry: Any) -> bool:
        """Watch the function's world now, when due; True when it is
        watched at the current world epoch and generation."""
        world, gen = _world_epoch(), GEN[0]
        if self.world == world and self.gen == gen:
            return True
        self.slow += 1
        now = time.monotonic()
        if self.slow < self.backoff and now - self.scanned_at < self.RESCAN_AFTER_S:
            return False
        self.backoff = 1 if self.reused_since_scan else min(self.backoff * 2, 1024)
        self.slow = self.reused_since_scan = 0
        self.scanned_at = now
        self.world = -1
        self.entries = {}
        STATS["scans"] += 1
        try:
            cells, tags, _callees, family = hit_memo.scan(self.spec, registry, environment_ok=True)
        except Exception:  # noqa: BLE001 - Ineligible, or a scan that broke: the ordinary path
            STATS["scans_failed"] += 1
            return False
        self.cells, self.tags, self.family = cells, tags, family
        # What moved during the scan is not proven: the next call scans again.
        if GEN[0] != gen:
            return False
        self.world, self.gen = _world_epoch(), gen
        return True

    def code_state(self, keys: Any, spec: CachedFunction, chain: list, note: bool) -> str:
        """`KeyBuilder._code_state`, served from the memo when it holds."""
        build_now = keys._code_state_now
        mode = _mode(spec.name, note)
        if mode is None:
            WHY["why_context"] += 1
            return build_now(spec, chain, note=note)

        def fold(into: list) -> str:
            return build_now(spec, into, note=note)

        if self.world == _world_epoch() and self.gen == GEN[0]:
            if not self._holds():
                # A cell, a name or a class moved without a dict event:
                # scanned again at the next call.
                self.world = -1
                return build_now(spec, chain, note=note)
            cap = self.entries.get(mode)
            if cap is not None and cap.env and not _env_holds(cap.env):
                # A variable the segment read now answers otherwise: built
                # again below, and remembered with the new answer.
                WHY["why_environment"] += 1
                cap = None
            if cap is not None:
                _apply(cap, chain)
                _env_replay(cap.env)
                STATS["reused"] += 1
                self.reused_since_scan += 1
                if VERIFY:
                    self._verify(cap, fold)
                return cap.state
        else:
            WHY["why_world" if self.world != _world_epoch() else "why_generation"] += 1
            if not self._scan(keys._registry):
                return build_now(spec, chain, note=note)
        world, gen, outside = self.world, self.gen, _outside()
        cap = _sandboxed(fold)
        _apply(cap, chain)
        if (
            _world_epoch() == world
            and GEN[0] == gen
            and _outside() == outside
            and cap.env_complete
            and _env_holds(cap.env)
            and self.world == world
            and len(cap.chain) >= 7
            and cap.chain[5] == cap.chain[4]
            and self._holds()
        ):
            self.entries[mode] = cap
            STATS["remembered"] += 1
        return cap.state

    def _verify(self, kept: _Capture, fold: Callable[[list], str]) -> None:
        world, gen = _world_epoch(), GEN[0]
        try:
            built = _sandboxed(fold)
        except Exception as e:
            raise StateMemoMismatch(f"state of {self.spec.name}: building it again raised {e!r}") from e
        if _world_epoch() != world or GEN[0] != gen:
            return  # the rebuild itself moved something: nothing to compare
        STATS["verified"] += 1
        diff = _differences(kept, built)
        if diff:
            raise StateMemoMismatch(
                f"state of {self.spec.name}: the reused segment differs from a rebuild in {', '.join(diff)} "
                f"({kept.state} reused, {built.state} built)"
            )


def memo_for(spec: CachedFunction) -> StateMemo | None:
    """*spec*'s memo, made on first use; None when the memo is off or the
    function cannot take one (see `hit_memo.plain_spec`)."""
    memo = spec.state_memo
    if memo is not None:
        return memo or None
    if not ENABLED or not hit_memo.plain_spec(spec):
        spec.state_memo = False
        return None
    memo = spec.state_memo = StateMemo(spec)
    return memo
