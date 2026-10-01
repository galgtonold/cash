"""What one ``CashMagics`` keeps for the session: its statistics, provenance
and the compute times its hits are credited against.

:class:`CashSession` is the one owner of the session statistics: it counts
each cell (:meth:`CashSession.record_cell`), derives what ``%cash_stats``
reports (:meth:`CashSession.summary`) and forgets it all on a reset
(:meth:`CashSession.reset`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import compute_baselines
from ..cache_status import CacheStatus
from ..provenance import ProvenanceTracker

if TYPE_CHECKING:
    from ..statement import ProcessResult

__all__ = ["CashSession", "StatsSummary", "new_session_stats"]


def new_session_stats() -> dict[str, Any]:
    """A zeroed session-stats dict.

    The single definition of what the stats ARE, so that creating a session
    and resetting one cannot disagree: a reset must forget every counter the
    stats summarise, including one added later.
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
        # this the headline net after a restart would be "at least
        # -overhead, at best <gross>": a range straddling zero in the one
        # reading every user takes.
        "total_measured_saved": 0.0,
        # Cash's OWN added wall-time this session (restore + simulation +
        # hashing + badge machinery), accumulated per cell. Subtracted from
        # the gross ``total_time_saved`` to report an honest NET saving so a
        # session whose overhead outweighs its cache hits reads as a cost,
        # not a phantom win.
        "total_overhead": 0.0,
        # The hit rate over ALL statements is dominated by print/import
        # trivia that cash deliberately never tried to cache, so a session
        # where every expensive statement hit would read as a low rate:
        # arithmetically true, practically meaningless. These two count
        # only statements whose compute cost cleared cash's OWN caching
        # floor (``min_execution_time_to_cache_seconds``), i.e. the
        # statements caching was ever on the table for.
        "statements_cacheable_hit": 0,
        "statements_cacheable_miss": 0,
    }


@dataclass(frozen=True)
class StatsSummary:
    """What ``%cash_stats`` reports, derived from the session counters."""

    total_stmts: int
    #: Hits over every statement, the trivial ones included.
    hit_rate: float
    cacheable_hit: int
    cacheable_total: int
    #: Hits over the statements worth caching; None when there were none.
    cacheable_rate: float | None
    gross_saved: float
    measured_saved: float
    overhead: float
    #: The net credited only from verified and measured savings.
    net_saved: float
    #: The net if every restore saved what its entry recorded.
    net_upper: float


class CashSession:
    """The session statistics, provenance and compute baselines of one
    ``CashMagics``.

    *cash_instance* is asked for its backend when the baselines are first
    needed, and *store_floor_s* for the statement store's "too cheap to
    cache" floor each time a cell is counted.
    """

    __slots__ = (
        "stats",
        "provenance",
        "measured_compute",
        "measured_decorator_compute",
        "_baselines",
        "_cash_instance",
        "_store_floor_s",
    )

    def __init__(self, cash_instance: Any, store_floor_s: Callable[[], float]) -> None:
        self._cash_instance = cash_instance
        self._store_floor_s = store_floor_s
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
        # use (``baselines``), not here: the backend is not settled while the
        # magics are being constructed.
        self._baselines: compute_baselines.ComputeBaselineStore = compute_baselines.get_store(None)

    @property
    def baselines(self) -> compute_baselines.ComputeBaselineStore:
        """Measurements from earlier kernels against this cache directory.

        Resolved on use, not in ``__init__``: a notebook's backend is not
        settled when the magics are constructed (``%cash_on`` may still
        replace it), and a store bound to "nowhere to persist" then would
        stay that way for the session, silently reporting no measured
        saving.
        """
        if not self._baselines.persistent:
            resolved = compute_baselines.store_for_backend(getattr(self._cash_instance, "backend", None))
            if resolved is not None and resolved.persistent:
                self._baselines = resolved
        return self._baselines

    # ------------------------------------------------------------------
    # Counting a cell
    # ------------------------------------------------------------------

    def record_cell(self, all_metrics: list[ProcessResult], wall_time: float = 0.0) -> None:
        """Count one cell: its statements, its savings and cash's overhead.

        ``wall_time`` is this cell's full cash-mediated wall time. Cash's own
        overhead for the cell is that wall time minus the user compute that
        would have run anyway (the COMPUTED statements). What remains — cache
        restores, upstream simulation, hashing, badge machinery — is time
        cash *added*, so it is accumulated into ``total_overhead`` and later
        subtracted from the gross ``total_time_saved`` to report an honest
        NET figure. A float subtraction per cell and no I/O beyond
        :meth:`ComputeBaselineStore.flush_soon`'s throttled write: this runs
        on every cell.

        Upstream COMPUTED statements count as user compute, NOT as overhead:
        they are the user's own notebook code, and the state they rebuild is
        state the user would have had to rebuild by hand (the restart pain
        cash exists to absorb). Booking them as cash's overhead would make
        cash understate itself by the size of the user's own ETL on exactly
        the sessions where it helps most.

        The gross saving is credited from ``saved_time``, which is a *stale*
        baseline (see ``total_time_saved``). A restore whose cost was
        measured is additionally credited to ``total_verified_saved`` or
        ``total_measured_saved`` (:meth:`_credit_hit`), which is what
        ``%cash_stats`` reports as the headline NET, so an unverifiable claim
        can never print as a win.
        """
        stats = self.stats
        baselines = self.baselines
        stats["cells_executed"] += 1
        cell_compute_time = 0.0
        # The statement store's own "too cheap to cache" floor, so the
        # cacheable/trivial split matches the decision the cache actually
        # made rather than a second opinion invented here.
        floor = self._store_floor_s()
        for m in all_metrics:
            status = m.get("status")
            if status == CacheStatus.COMPUTED:
                cell_compute_time += self._count_computed(m, floor)
            elif status == CacheStatus.RESTORED:
                self._count_restored(m, floor)
            elif status == CacheStatus.SKIPPED:
                stats["statements_skipped"] += 1
            # A ``@cash.cache`` HIT inside this statement saved real compute
            # that is invisible to the counting above: the value came from the
            # decorator, so the statement itself only did a fast lookup and
            # reads as cheap COMPUTED work. Credited from the same drained call
            # log the badge uses. No double count: a decorator is only invoked
            # when the statement EXECUTES, so a RESTORED statement (whose
            # ``saved_time`` already covers the whole compute) carries no
            # decorator_calls to add.
            self._count_decorator_calls(m.get("decorator_calls"), floor)
        # Overhead = cell wall time minus the user compute that ran this cell.
        # Floor at 0: the wall time always covers the compute it contains, but
        # clamp defensively against clock skew / partial timing.
        stats["total_overhead"] += max(0.0, wall_time - cell_compute_time)
        # A write after a cell that measured something worth keeping, since a
        # Restart & Run All kills the kernel and no exit hook runs; cheap
        # measurements are written every few seconds, not after every cell.
        baselines.flush_soon()

    def _count_computed(self, m: ProcessResult, floor: float) -> float:
        """Count a COMPUTED statement; returns the user compute it took."""
        self.stats["statements_computed"] += 1
        # What the USER's code cost: the statement's wall time less cash's own
        # time inside it -- recording file reads, keying and hashing the
        # arguments of the calls it routes, storing them (``cash_tax``, the
        # same measurement ``CallRouting.price`` uses). Counting that as the
        # user's compute would cancel it out of the overhead.
        exec_time = max(0.0, m.get("execution_time", 0.0) - m.get("cash_tax", 0.0))
        self.stats["total_compute_time"] += exec_time
        code = m.get("code") or None
        self._count_miss(exec_time, floor, self.measured_compute, code, code)
        return exec_time

    def _count_restored(self, m: ProcessResult, floor: float) -> None:
        """Count a RESTORED statement and credit what it saved."""
        self.stats["statements_restored"] += 1
        saved = m.get("saved_time", 0.0)
        self.stats["total_restored_time"] += saved
        code = m.get("code")
        self._credit_hit(saved, floor, self.measured_compute.get(code), code or "")

    def _count_decorator_calls(self, decorator_calls: list[dict[str, Any]] | None, floor: float) -> None:
        """Fold ``@cash.cache`` call metrics into the session totals.

        Without this, a session whose expensive work sits behind the
        decorator would report the inverse of cash's value: a warm pass that
        avoided a long fit via a decorator hit would read as a net *cost*.

        A **miss** records this session's measured compute for that key, so a
        later hit can be credited as VERIFIED rather than merely gross. It is
        NOT added to compute totals: the enclosing statement's
        ``execution_time`` already contains it, and adding it here would
        double-count.

        A **hit** is credited like a RESTORED statement (:meth:`_credit_hit`).
        """
        if not decorator_calls:
            return
        measured = self.measured_decorator_compute
        for call in decorator_calls:
            if call.get("ran_plain"):
                continue  # run without the cache: neither a hit nor a measured miss
            key = call.get("cache_key")
            identity = f"call:{key}" if key is not None else None
            if call.get("cache_hit"):
                self._credit_hit(call.get("time_saved", 0.0) or 0.0, floor, measured.get(key), identity)
            else:
                # A miss's execution_time IS the measured compute for this key.
                self._count_miss(call.get("execution_time", 0.0) or 0.0, floor, measured, key, identity)

    def _count_miss(
        self,
        seconds: float,
        floor: float,
        measured: dict[str, float],
        key: str | None,
        identity: str | None,
    ) -> None:
        """Note a computation this session measured at *seconds*.

        Kept under *key* in *measured* and under *identity* on disk, so a
        later hit here, or in a later kernel, can point at it. A miss on a
        computation worth caching counts towards the cacheable rate.
        """
        if key is not None:
            measured[key] = seconds
        if identity is not None:
            self._baselines.record(identity, seconds)
        if seconds >= floor:
            self.stats["statements_cacheable_miss"] += 1

    def _credit_hit(self, saved: float, floor: float, today: float | None, identity: str | None) -> None:
        """Credit a hit that saved *saved* seconds by the cache's own account.

        *saved* is the cache's stale baseline, so it is trusted for the gross
        figure only, and to answer "was this the kind of computation caching
        was for?": a hit is a fact either way, only the denominator's
        membership rests on the baseline. A saving is VERIFIED only where
        this session measured the same computation itself (*today*), and
        MEASURED where an earlier kernel on this machine did (the baseline
        under *identity*). Either way the smaller figure is credited: if the
        cache's baseline is the smaller it is the one we can defend, and if
        the measurement is smaller the cache's baseline was stale-high and
        must not be credited.
        """
        stats = self.stats
        stats["total_time_saved"] += saved
        if saved >= floor:
            stats["statements_cacheable_hit"] += 1
        if today is not None:
            stats["total_verified_saved"] += min(saved, today)
            return
        before = self._baselines.get(identity) if identity is not None else None
        if before is not None:
            stats["total_measured_saved"] += min(saved, before)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def record_provenance(
        self,
        all_metrics: list[ProcessResult],
        variable_lineage: dict[str, str],
        file_deps: dict[str, Any],
    ) -> None:
        """Record a provenance entry per output variable of each statement.

        *variable_lineage* and *file_deps* are the tracking state's maps, read
        for each output variable's lineage hash and file dependencies.
        """
        for m in all_metrics:
            code = m.get("code", "")
            status = m.get("status", "computed")
            duration_ms = m.get("execution_time", 0.0) * 1000
            # ``rich_outputs`` holds IPython rich-display objects, NOT variable
            # names — never source variable names from it.
            outputs = m.get("restored_vars", []) or m.get("evaluated_vars", [])
            inputs_list = list(m.get("inputs", []))
            # Outputs may contain rich-display dicts; provenance only cares
            # about string variable names.
            var_names = [o for o in (outputs or []) if isinstance(o, str)]

            provenance_status = str(status).lower() if status else "computed"
            for out_var in var_names:
                self.provenance.record(
                    variable=out_var,
                    code=code,
                    inputs=inputs_list,
                    status=provenance_status,
                    duration_ms=duration_ms,
                    lineage_hash=variable_lineage.get(out_var, ""),
                    file_deps=list(file_deps.get(out_var, [])),
                )

    # ------------------------------------------------------------------
    # Reading and resetting
    # ------------------------------------------------------------------

    def summary(self) -> StatsSummary:
        """The rates and nets ``%cash_stats`` reports for the counters."""
        stats = self.stats
        total_stmts = stats["statements_computed"] + stats["statements_restored"] + stats["statements_skipped"]
        hit_rate = (stats["statements_restored"] + stats["statements_skipped"]) / max(total_stmts, 1) * 100

        # The rate over ALL statements answers a question nobody asked: its
        # denominator is dominated by prints, imports and cheap assignments
        # that cash deliberately never tried to cache. Overstating savings is
        # the same failure inverted, so the same rule binds: the number must
        # not imply a conclusion the data does not support, in EITHER
        # direction.
        cacheable_hit = stats.get("statements_cacheable_hit", 0)
        cacheable_miss = stats.get("statements_cacheable_miss", 0)
        cacheable_total = cacheable_hit + cacheable_miss
        cacheable_rate = (cacheable_hit / cacheable_total * 100) if cacheable_total else None

        # Two nets, because two different qualities of evidence.
        #
        # ``gross_saved`` is a counterfactual: each restore is credited with
        # the compute time recorded when the value was FIRST cached. Nothing
        # re-measures that. If the first run was colder — cold page cache,
        # cold imports — the credit is stale-high, and a session that was
        # slower by wall clock would still print a win. That is not fixable
        # by estimating harder: the true recompute cost cannot be known
        # without doing the recompute.
        #
        # So the HEADLINE net is credited only from savings that were
        # measured. The gross figure is still shown, explicitly as an
        # unverified upper bound. This deliberately UNDERSTATES a session
        # that really did save time but never re-measured a baseline: an
        # understatement is a defensible error here, an overstatement is not.
        gross_saved = stats["total_time_saved"]
        verified_saved = stats.get("total_verified_saved", 0.0)
        # Measured on this machine in an earlier kernel, at the least it ever
        # cost. Evidence of the same kind as ``verified``, one run older --
        # and the only kind a Restart & Run All can have.
        measured_saved = stats.get("total_measured_saved", 0.0)
        overhead = stats.get("total_overhead", 0.0)
        return StatsSummary(
            total_stmts=total_stmts,
            hit_rate=hit_rate,
            cacheable_hit=cacheable_hit,
            cacheable_total=cacheable_total,
            cacheable_rate=cacheable_rate,
            gross_saved=gross_saved,
            measured_saved=measured_saved,
            overhead=overhead,
            net_saved=verified_saved + measured_saved - overhead,
            net_upper=gross_saved - overhead,
        )

    def reset(self) -> None:
        """Forget this session's statistics and the baselines behind them."""
        # Rebuilt from the same definition a fresh session uses, so a new
        # counter can never be added to the stats and silently survive a
        # reset.
        self.stats.update(new_session_stats())
        # The baselines are part of the stats, not of the cache: a reset must
        # drop them too (statement and decorator, in memory and on disk), or
        # a later hit would be credited as verified against a measurement the
        # reset claims to have forgotten.
        self.measured_compute.clear()
        self.measured_decorator_compute.clear()
        self.baselines.clear()
