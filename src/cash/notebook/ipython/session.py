"""What one ``CashMagics`` keeps for the session: its statistics, provenance
and the compute times its hits are credited against."""

from __future__ import annotations

from typing import Any

from .. import compute_baselines
from ..provenance import ProvenanceTracker

__all__ = ["CashSession", "new_session_stats"]


def new_session_stats() -> dict[str, Any]:
    """A zeroed session-stats dict.

    The single definition of what the stats ARE, so that creating a session and
    resetting one cannot disagree. ``%cash_stats reset`` used to re-list the
    keys by hand, which silently left any later-added counter carrying over the
    reset -- the reset would report success while the next session inherited
    the last one's numbers. Same rule as ``measured_compute``: a
    reset must forget everything the stats claim to summarise.
    """
    return {
        "cells_executed": 0,
        "statements_computed": 0,
        "statements_restored": 0,
        "statements_skipped": 0,
        "total_compute_time": 0.0,
        "total_restored_time": 0.0,
        # GROSS avoided recompute. Every contribution is a ``saved_time``
        # copied off cache metadata — i.e. how long the statement took when
        # it was FIRST computed, on a possibly colder machine. It is an
        # estimate of a counterfactual, never a measurement of this
        # session, and it may overstate without bound.
        "total_time_saved": 0.0,
        # The subset of ``total_time_saved`` whose baseline this session
        # measured itself: the statement was COMPUTED here before it was
        # RESTORED here, so the recompute cost is known under today's
        # conditions rather than assumed from the cache.
        "total_verified_saved": 0.0,
        # The subset whose baseline was measured on this machine in an
        # EARLIER kernel (``compute_baselines``, the least cost ever
        # measured). A Restart & Run All recomputes nothing, so without
        # this the headline net after a restart was "at least -overhead,
        # at best <gross>" -- a range straddling zero in the one reading
        # every user takes.
        "total_measured_saved": 0.0,
        # Cash's OWN added wall-time this session (restore + simulation +
        # hashing + badge machinery), accumulated per cell. Subtracted from
        # the gross ``total_time_saved`` to report an honest NET saving so a
        # session whose overhead outweighs its cache hits reads as a cost,
        # not a phantom win.
        "total_overhead": 0.0,
        # The hit rate over ALL statements is dominated by print/import
        # trivia that cash deliberately never tried to cache, so it made a
        # session where every expensive statement hit read as 14.9% —
        # arithmetically true, practically meaningless. These two
        # count only statements whose compute cost cleared cash's OWN
        # caching floor (``min_execution_time_to_cache_seconds``), i.e. the
        # statements caching was ever on the table for.
        "statements_cacheable_hit": 0,
        "statements_cacheable_miss": 0,
    }


class CashSession:
    """Groups session-level concerns owned by a single CashMagics instance.

    Separating these from execution-level state (backend, shell, tracking
    dictionaries) makes the sub-boundary explicit and each component
    independently addressable.
    """

    __slots__ = ("stats", "provenance", "measured_compute", "measured_decorator_compute", "baselines")

    def __init__(self) -> None:
        self.stats: dict[str, Any] = new_session_stats()
        self.provenance: ProvenanceTracker = ProvenanceTracker()
        # source -> execution_time measured in THIS session. Bounded by the
        # notebook's statement count; pure in-memory floats, no I/O.
        self.measured_compute: dict[str, float] = {}
        # decorator cache_key -> compute time measured in THIS session (a miss).
        # The @cash.cache sibling of measured_compute: it lets a later decorator
        # HIT be credited as VERIFIED under the same rule.
        self.measured_decorator_compute: dict[str, float] = {}
        # The same two measurements, kept on disk beside the cache so the next
        # kernel can still point at one. Bound to a real directory on first
        # use (``CashMagics._baselines``), not here: the backend is not
        # settled while the magics are being constructed.
        self.baselines: Any = compute_baselines.get_store(None)
