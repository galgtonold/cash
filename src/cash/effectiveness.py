"""Notices when ``@cash.cache`` is costing more than it saves.

The notebook path has a cost model that declines to cache work not worth
caching. The decorator deliberately does not: it is an explicit instruction
from the user and second-guessing it is not its job. But that leaves a gap
where cash can make code *slower*, every call, and never say so.

Measured on the case that motivated this -- a 153 MiB DataFrame passed to a
function that sums one column::

    key hash   389.59 ms
    the work    11.31 ms      -> 34x slower, on every call

And it really is every call: the arg-hash memo is gated on
``_cash_lineage_hash``, which only notebook-tracked objects carry, so in a
script or library the frame is re-hashed from scratch each time.

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


class EffectivenessLedger:
    """Per-function running account of what caching cost versus what it saved.

    ``record`` returns a message when the function has crossed into
    "measurably counterproductive", and ``None`` every other time -- which is
    almost always.
    """

    def __init__(self, waste_threshold_seconds: float = CUMULATIVE_WASTE_SECONDS) -> None:
        self._ledgers: dict[str, _FunctionLedger] = {}
        self._threshold = waste_threshold_seconds

    def record(
        self,
        func_name: str,
        *,
        overhead_seconds: float,
        body_seconds: float | None,
        was_hit: bool,
    ) -> str | None:
        """Account for one call. Returns a warning message, or ``None``.

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

        led.calls += 1
        led.overhead_seconds += overhead_seconds
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
        return _message(func_name, led, waste, per_call_overhead, best_case_saving)

    def reset(self) -> None:
        """Drop all accounting. For tests and ``cash.reset_session()``."""
        self._ledgers.clear()


def _message(
    func_name: str,
    led: _FunctionLedger,
    waste: float,
    per_call_overhead: float,
    best_case_saving: float,
) -> str:
    """Say what it cost, and what to do about it.

    Naming a remedy that KEEPS the caching matters more than the number: a
    registered hasher fixes the usual cause (a large argument being content
    hashed in full) without giving up the decorator.

    ``override=True`` is named explicitly because the usual culprit is a
    numpy array or a dataframe, and for exactly those types a plain
    ``register_hasher`` is silently never consulted. The first version of
    this message sent a design partner to that dead end: it diagnosed the
    cause correctly, they registered a hasher, nothing changed, and nothing
    said why.
    """
    return (
        f"@cash.cache on {func_name!r} is costing more than it saves. "
        f"Across {led.calls} calls cash spent {led.overhead_seconds:.2f}s on "
        f"cache keys and lookups to avoid at most {best_case_saving * 1000:.0f}ms "
        f"of work per call -- a net loss of about {waste:.1f}s so far "
        f"({per_call_overhead * 1000:.0f}ms of overhead per call). "
        f"This usually means a large argument is being hashed in full on every "
        f"call. Register a cheaper hasher for that type "
        f"(cash.register_hasher) to keep caching -- for a type cash "
        f"fingerprints itself, such as a numpy array or a dataframe, pass "
        f"override=True as well or it will not be consulted -- or drop the "
        f"decorator here."
    )
