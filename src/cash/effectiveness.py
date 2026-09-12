"""Notices when ``@cash.cache`` is costing more than it saves.

The notebook path has a cost model that declines to cache work not worth
caching. The decorator deliberately does not: it is an explicit instruction
from the user and second-guessing it is not its job. But that leaves a gap
where cash can make code *slower*, every call, and never say so.

Measured on the case that motivated this -- a 153 MiB DataFrame passed to a
function that sums one column::

    key hash   389.59 ms
    the work    11.31 ms      -> 34x slower, on every call

And it really is every call, for anything cash cannot check cheaply for
changes: outside a notebook a cached result's lineage tag is not trusted (it
is never updated when the object is mutated), so a numpy array, a model or a
pandas frame without copy-on-write is re-hashed from scratch each time. The
verdict names the costliest argument and, when a cached function produced it,
suggests ``frozen=True`` there.

When to speak up
----------------
Two conditions, and both matter:

* **Cumulative waste past a threshold.** "This costs 100 ms twice" is not
  worth a warning; it is the kind of notice people filter away permanently,
  and then the one case that mattered is filtered too. The bar is seconds of
  real, accumulated loss.

* **Overhead exceeds even the BEST case compute.** Compared against the
  *largest* body time observed, not the mean. A function that usually takes
  10 ms but occasionally takes 30 s is worth caching, and flagging it would
  be a confident, wrong diagnosis -- worse than staying quiet.

This module decides; it does not warn. The caller owns the ``warnings.warn``
so the decision stays testable without touching warning filters.

Sibling: ``remote_source.validation_is_expensive`` asks the same question of
freshness checks, and settled on the same 2s absolute bar independently. It
also fires on a second rule this one does not -- "technically net-positive
and still miserable", e.g. 8s of validation to save 60s. That case is
deliberately out of scope here: caching that pays is not misuse, and the ask
was to be conservative about interrupting anyone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Real seconds that must be lost before this is worth interrupting anyone.
# The user pays this cost either way; the only question is whether telling
# them is worth the noise. Two seconds of genuine waste is.
CUMULATIVE_WASTE_SECONDS = 2.0

# Body-time observations kept per function. Bounded because this lives on a
# hot path and because only the maximum is read.
_BODY_SAMPLES = 32

# Calls needed before any verdict. Not about warm-up -- a body time can come
# from an entry restored from a previous session -- but a single observation
# is not a distribution, and the tail is the whole point.
MIN_OBSERVATIONS = 3

# Ceiling on tracked functions, so a long-lived process that decorates
# programmatically cannot grow this without bound. Mirrors the cap on
# ``remote_source._warned_validation_cost``. Past it, new functions are simply
# not tracked: losing a diagnostic is acceptable, leaking is not.
MAX_TRACKED_FUNCTIONS = 1024


@dataclass
class _FunctionLedger:
    overhead_seconds: float = 0.0
    saved_seconds: float = 0.0
    calls: int = 0
    body_samples: deque = field(default_factory=lambda: deque(maxlen=_BODY_SAMPLES))
    warned: bool = False
    #: The overhead split by where it was paid: a hit's is the lookup and the
    #: restore, a miss's the key and the store. Which one dominates decides
    #: what the message blames -- round 20 was told "loading the stored result"
    #: about a function that had never hit.
    hit_overhead: float = 0.0
    miss_overhead: float = 0.0
    hits: int = 0

    def typical_overhead(self) -> float:
        """What one call usually costs: a hit's overhead when there are hits,
        a miss's otherwise. A miss pays its store once per key; averaged in, one
        slow write under load made a hasher that was the whole cost of every
        hit read as "almost none of it is the key"."""
        if self.hits:
            return self.hit_overhead / self.hits
        return self.miss_overhead / max(1, self.calls)


class EffectivenessLedger:
    """Per-function running account of what caching cost versus what it saved.

    ``record`` returns a ``(what, fix)`` pair when the function has crossed
    into "measurably counterproductive", and ``None`` every other time --
    which is almost always. The two halves are separate because the caller
    renders them into a coded diagnostic (``CACHE-NET-LOSS``): the measurement
    is one sentence and the remedy is another, and the doc section carries the
    rest.
    """

    def __init__(self, waste_threshold_seconds: float = CUMULATIVE_WASTE_SECONDS) -> None:
        self._ledgers: dict[str, _FunctionLedger] = {}
        self._threshold = waste_threshold_seconds
        # The last culprit each function reported, for `final_verdicts`.
        self._culprits: dict[str, tuple] = {}

    def record(
        self,
        func_name: str,
        *,
        overhead_seconds: float,
        body_seconds: float | None,
        was_hit: bool,
        culprit: tuple | None = None,
    ) -> tuple[str, str] | None:
        """Account for one call. Returns a ``(what, fix)`` pair, or ``None``.

        ``culprit`` is the costliest argument to hash seen for this function:
        ``(parameter, type name, seconds, producer or None, frame without
        copy-on-write)``, so the message can name it instead of guessing.

        ``body_seconds`` is the function's OWN time, excluding everything cash
        did around it. ``None`` means unknown -- an entry written before this
        was recorded -- and the call is then ignored entirely rather than
        guessed at, because a guess here biases the verdict.
        """
        if body_seconds is None:
            return None

        led = self._ledgers.get(func_name)
        if led is None:
            if len(self._ledgers) >= MAX_TRACKED_FUNCTIONS:
                return None
            led = self._ledgers[func_name] = _FunctionLedger()

        if culprit is not None:
            self._culprits[func_name] = culprit
        led.calls += 1
        led.overhead_seconds += overhead_seconds
        if was_hit:
            led.hit_overhead += overhead_seconds
            led.hits += 1
        else:
            led.miss_overhead += overhead_seconds
        led.body_samples.append(body_seconds)
        if was_hit:
            # A hit is the only time caching actually returns something: the
            # body did not run. A miss saved nothing and still paid overhead.
            led.saved_seconds += body_seconds

        if led.warned or led.calls < MIN_OBSERVATIONS:
            return None

        waste = led.overhead_seconds - led.saved_seconds
        if waste < self._threshold:
            return None

        # The conservative comparison: typical overhead against the LARGEST
        # body time seen. If cash still costs more than the best case it could
        # ever save, the verdict does not depend on which call you look at.
        per_call_overhead = led.overhead_seconds / led.calls
        best_case_saving = max(led.body_samples)
        if per_call_overhead <= best_case_saving:
            return None

        led.warned = True
        return _message(func_name, led, waste, per_call_overhead, best_case_saving, culprit)

    def final_verdicts(self) -> list[tuple[str, str]]:
        """The verdicts a whole run supports, at its end.

        `record` waits for `MIN_OBSERVATIONS` calls of a function, which a
        command-line tool that calls each function once per process never
        reaches: its parser was a net loss of seconds on every run, and nothing
        ever said so (round 19). At the end of the run one call is allowed to
        count -- under the same bar: seconds of real loss, and overhead above
        the largest body time seen.
        """
        out: list[tuple[str, str]] = []
        # Functions losing, each under the bar: ten of them losing 0.4-0.9 s
        # apiece made a run 1.3x slower and nothing said so (round 20).
        small: list[tuple[float, str]] = []
        for func_name, led in self._ledgers.items():
            if led.warned or not led.calls or not led.body_samples:
                continue
            waste = led.overhead_seconds - led.saved_seconds
            per_call_overhead = led.overhead_seconds / led.calls
            best_case_saving = max(led.body_samples)
            if per_call_overhead <= best_case_saving or waste <= 0:
                continue
            if waste < self._threshold:
                small.append((waste, func_name))
                continue
            led.warned = True
            out.append(_message(func_name, led, waste, per_call_overhead,
                                best_case_saving, self._culprits.get(func_name)))
        total = sum(w for w, _ in small)
        if len(small) >= 2 and total >= self._threshold:
            small.sort(reverse=True)
            named = ", ".join(f"{name!r} ({waste:.1f}s)" for waste, name in small[:5])
            more = f" and {len(small) - 5} more" if len(small) > 5 else ""
            out.append((
                f"@cash.cache cost more than it saved across {len(small)} functions "
                f"in this run -- a net loss of about {total:.1f}s together, each "
                f"under the {self._threshold:g}s a single warning waits for: "
                f"{named}{more}. Each one's overhead per call is larger than the "
                f"most its body ever took.",
                "these are cheap functions over large arguments: leave them "
                "uncached, or cache what they are computed from instead -- the "
                "aggregate rather than the rows.",
            ))
        return out

    def reset(self) -> None:
        """Drop all accounting. For tests and ``cash.reset_session()``."""
        self._ledgers.clear()
        self._culprits.clear()


def _message(
    func_name: str,
    led: _FunctionLedger,
    waste: float,
    per_call_overhead: float,
    best_case_saving: float,
    culprit: tuple | None = None,
) -> tuple[str, str]:
    """Say what it cost, and what to do about it.

    Naming a remedy that KEEPS the caching matters more than the number: a
    registered hasher fixes the usual cause (a large argument being content
    hashed in full) without giving up the decorator.

    ``override=True`` is named explicitly because the usual culprit is a
    numpy array or a dataframe, and for exactly those types cash's own
    content hasher runs first. The first version of this message sent a
    design partner to that dead end: it diagnosed the cause correctly, they
    registered a hasher, nothing changed, and nothing said why.

    That silence is gone -- ``Cash.register_hasher`` now REJECTS a plain
    registration for one of those types with ``ValueError`` rather than
    accepting a hasher it would never consult (verified by execution:
    ``register_hasher(np.ndarray, fn)`` raises; the same call with
    ``override=True`` is accepted). Naming the flag here and refusing there
    are two halves of one fix, so do not "simplify" this back to a bare
    ``register_hasher``.
    """
    what = (
        f"@cash.cache on {func_name!r} is costing more than it saves. "
        f"Across {led.calls} calls cash spent {led.overhead_seconds:.2f}s on "
        f"cache keys and lookups to avoid at most {best_case_saving * 1000:.0f}ms "
        f"of work per call -- a net loss of about {waste:.1f}s so far "
        f"({per_call_overhead * 1000:.0f}ms of overhead per call). This usually "
        f"means a large argument is being hashed in full on every call."
    )
    hasher = (
        "register a cheaper hasher for that argument's type "
        "(cash.register_hasher) to keep caching -- for a type cash "
        "fingerprints itself, such as a numpy array or a dataframe, that "
        "registration needs override=True -- or drop the decorator here."
    )
    key_seconds = culprit[2] if culprit is not None else None
    if key_seconds is not None and key_seconds < 0.25 * led.typical_overhead():
        # Not the key: the lookup itself -- reading and rebuilding the stored
        # result. Blaming an argument sent round 19's tester after "'path'
        # (str), about 0ms to hash" for a parser whose hit was a 2M-row restore.
        # And on misses there is nothing to load: the cost is keeping the
        # result -- copying it into memory, writing it (round 20).
        where = ("loading the stored result, which takes longer than running "
                 "the function" if led.hit_overhead > led.miss_overhead else
                 "keeping the result -- copying it into memory and writing it -- "
                 "which takes longer than running the function")
        what = (
            f"@cash.cache on {func_name!r} is costing more than it saves. "
            f"Across {led.calls} calls cash spent {led.overhead_seconds:.2f}s on "
            f"cache keys, lookups and stores to avoid at most "
            f"{best_case_saving * 1000:.0f}ms of work per call -- a net loss of "
            f"about {waste:.1f}s so far. Almost none of it is the key (its "
            f"costliest argument took about {key_seconds * 1000:.0f}ms to hash): "
            f"it is {where}."
        )
        return what, (
            "a result that is slower to load than to compute is not worth "
            "caching: drop the decorator here, or cache something smaller that "
            "the rest is computed from -- the aggregate rather than the rows."
        )
    if culprit is None:
        return what, hasher
    param, type_name, seconds, producer, old_pandas = culprit
    what += (f" The costliest argument is '{param}' ({type_name}), "
             f"about {seconds * 1000:.0f}ms to hash.")
    parts = []
    if producer:
        parts.append(
            f"'{param}' comes from {producer}(): if that result is not modified "
            f"afterwards, declare @cash.cache(frozen=True) on {producer} and it "
            f"is keyed without being hashed")
    if old_pandas:
        parts.append("pandas 3 (copy-on-write) lets cash check a frame for "
                     "changes instead of hashing it on every call")
    parts.append(("otherwise " if parts else "") + hasher)
    return what, "; ".join(parts)
