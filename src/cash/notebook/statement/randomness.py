"""What a statement does with randomness: warnings, the badge's RNG pill, and
the seed and draw ledgers a cache key depends on."""

from __future__ import annotations

import ast
import hashlib
import logging
from typing import TYPE_CHECKING

from cash.control_markers import strip_markers
from cash.diagnostics import warn_diagnostic
from cash.tracking.randomness import (
    CashRandomnessWarning,
    RandomnessDetector,
    capture_rng_state,
    check_and_warn_randomness,
    entropy_write_lineage,
    get_drawing_rng_modules,
    get_entropy_reseed_modules,
    get_seeding_rng_modules,
    hidden_lineage_writes,
    hidden_write_lineage,
    publish_seed_epochs,
    rng_modules_changed,
    rng_virtual_var,
    warn_stale_estimator_fit,
    warn_stale_randomness,
    warn_unseeded_estimator_fit,
)

if TYPE_CHECKING:
    from cash.notebook._protocols import ShellProtocol, TrackingState
    from cash.notebook.statement.results import ProcessResult

logger = logging.getLogger(__name__)

_LOG_PROCESSOR = "[PROCESSOR]"

__all__ = ["StatementRandomness"]


class StatementRandomness:
    """The processor's view of randomness, one instance per processor.

    Warns about unseeded draws (on the source, before the lookup, and again
    when a replay is restored), stamps the badge's RNG pill, keeps the seed
    epoch ledger restores and ``@cash.cache`` read, and observes each
    statement's effect on the global RNG streams.
    """

    def __init__(self, shell: ShellProtocol, tracking_state: TrackingState) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.detector = RandomnessDetector()
        #: module -> cache key of the seeding statement in force. Published by
        #: reference so ``@cash.cache`` sees a seed change too: the decorator
        #: consumes the same global stream but had no view of this ledger, and
        #: served a model trained under the previous seed.
        self.seed_epochs: dict[str, str] = {}
        publish_seed_epochs(self.seed_epochs)
        # Source hashes of entropy-reseed statements already warned about, so the
        # "seed(None) does not make everything below it fresh" note fires once.
        self._warned_entropy_reseed: set[str] = set()
        #: What the LAST statement drew (see :meth:`observe_statement`).
        self.observed_draw: set[str] = set()
        #: True when this statement's execution revealed a hidden RNG draw for
        #: the first time, meaning its cache key predates that knowledge.
        self.draw_newly_seen: bool = False
        # The current cell's accumulation, harvested by the cell executor for
        # the replay ledger.
        self._cell_changed: set[str] = set()
        self._cell_pre: dict | None = None
        self._cell_post: dict | None = None

    def begin_statement(self) -> dict:
        """Clear the last statement's observation and snapshot the RNG streams.

        Cleared up front so a statement that RAISES cannot leave the previous
        statement's draw attributed to it. Pass the snapshot to
        :meth:`observe_statement` once the statement has run.
        """
        self.observed_draw = set()
        self.draw_newly_seen = False
        return capture_rng_state()

    def record_seeds(self, code: str, cache_key: str) -> None:
        """Record what a seeding statement starts, once its key is known.

        A seed PRODUCES its module's hidden RNG variable: its lineage (the seed
        statement's own key) goes into the lineage store so downstream draws
        read it as an input. Recorded AFTER the key so a statement that seeds
        and draws at once keys on the state it INHERITS, not the one it opens.
        :attr:`seed_epochs` is kept for the restore's replay guard (the RNG
        STATE axis, separate from lineage).

        An ENTROPY reseed (``seed(None)`` / ``seed()``) asks for a different
        stream on every run, so the values computed after it are not
        replayable; its virtual variable gets a value fresh per execution so
        everything downstream of the reseed recomputes.
        """
        entropy_modules = get_entropy_reseed_modules(code)
        entropy_vars = {rng_virtual_var(m) for m in entropy_modules}
        for var in hidden_lineage_writes(code):
            self.tracking_state.lineage.record(
                var, entropy_write_lineage() if var in entropy_vars else hidden_write_lineage(cache_key)
            )
        for module in get_seeding_rng_modules(code):
            self.seed_epochs[module] = entropy_write_lineage() if module in entropy_modules else cache_key

    def warn_unseeded(self, code: str, allow_random: bool, *, skip_cache: bool = False) -> list:
        """Warn when *code* draws from an unseeded RNG.

        Silent for a ``# @cash:no-cache`` statement (*skip_cache*). The warning
        is about caching a draw, and that statement is never cached: with the
        directive on a line of its own the rewind is off too and every run
        draws again, which is the remedy the warning itself recommends.

        Called on the common path of both ``process_statement`` twins, BEFORE the
        cache lookup, for two reasons:

        * the warning describes the *source*, so it must not depend on whether
          this particular run hit or missed the cache; and
        * ``analyze_code`` doubles as the detector's seed-tracking hook — a
          ``np.random.seed(42)`` statement must mark its module seeded even on a
          cache hit, or the next cell warns spuriously.

        Returns the unseeded calls it found, so the cache-hit path can report the
        replay without re-running the scan (which would double the seed-tracking
        side effect above).

        Never allowed to break execution: this is advisory output, so a detector
        fault must not take the statement down with it.
        """
        code = strip_markers(code)
        try:
            unseeded_calls, _has_seed = check_and_warn_randomness(
                code,
                self.detector,
                suppress_warning=allow_random or skip_cache,
            )
            return list(unseeded_calls)
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            logger.debug("%s Randomness detection failed for statement", _LOG_PROCESSOR)
            return []

    def warn_entropy_reseed(self, code: str) -> None:
        """Warn that ``seed(None)`` cannot make cached values below it fresh.

        An entropy reseed (``np.random.seed(None)`` / bare ``seed()``) asks for a
        different stream every run. But cash caches downstream results, and a
        value whose lineage is source-derived rather than stream-derived -- a
        model from an in-place ``fit()``, and every statement that reads it -- is
        frozen from the run that first computed it. So the user gets a genuinely
        new stream AND a stale cached value describing the previous one, and the
        two silently disagree. No cache key can resolve that: making the value
        fresh-per-run stops it converging (a restore would re-run the producer
        and mint yet another result). The honest fix is to say so.

        Once per reseed statement, and never fatal -- advisory only.
        """
        try:
            stripped = strip_markers(code)
            if not get_entropy_reseed_modules(stripped):
                return
            digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
            if digest in self._warned_entropy_reseed:
                return
            self._warned_entropy_reseed.add(digest)

            warn_diagnostic(
                CashRandomnessWarning,
                "RANDOM-SEED-NONE",
                "seed(None) asks for a different random stream each run, but a "
                "cached value below it is frozen from the run that computed it, "
                "not redrawn -- so what you see may describe a stream that no "
                "longer exists.",
                "put `# @cash:no-cache` on the values that must reflect the "
                "fresh stream, or seed with a fixed integer if you want them "
                "reproducible instead.",
            )
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            logger.debug("%s Entropy-reseed warning failed for statement", _LOG_PROCESSOR)

    def stamp_random_effect(
        self,
        metrics: "ProcessResult",
        code: str,
        unseeded_calls: list,
        unseeded_fits: "list | tuple" = (),
    ) -> None:
        """Record a statement's RNG role on its metric so the badge can show it.

        Purely advisory and never fatal: a draw statement is deliberately still
        cached (a frozen replay), so this only annotates. ``draw`` wins over
        ``seed`` when a statement does both (the draw is the notable effect), and
        an unseeded draw/fit sets ``random_unseeded`` so the badge can flag that
        the cached value is frozen, not freshly drawn.
        """
        try:
            stripped = strip_markers(code)
            draws = bool(get_drawing_rng_modules(stripped)) or bool(unseeded_calls) or bool(unseeded_fits)
            seeds = bool(get_seeding_rng_modules(stripped))
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            return
        if draws:
            metrics["random_effect"] = "draw"
        elif seeds:
            metrics["random_effect"] = "seed"
        else:
            return
        if unseeded_calls or unseeded_fits:
            metrics["random_unseeded"] = True

    def warn_stale(
        self,
        code: str,
        unseeded_calls: list,
        allow_random: bool,
    ) -> None:
        """Announce that a cached unseeded random value was just replayed.

        Called ONLY after a restore has actually succeeded, because that is the
        event being reported: not "this statement contains randomness" (true on
        every run, and already covered by ``warn_unseeded``), but
        "the number you are looking at is a replay of an earlier run".

        That claim can only be made here.  ``warn_unseeded`` runs
        before the cache lookup, where the hit/miss outcome does not exist yet —
        so it can only ever say "may not be reproducible".  On a restore that
        understates it: the value *is* frozen. The prior warning fired on the COLD run,
        when the value is freshly computed and correct, and said nothing on the
        restores, when it is not.  This is the missing half.

        Gated on a successful restore specifically: ``CacheHitServer.serve``
        returns None when restoration fails and the caller falls through to real
        execution, in which case the value is fresh and "replay" would be a lie.

        The dedupe is deliberately kept.  It is keyed on ``(code, message)``, and
        this message is a different claim from the compute-time one, so it lands
        in its own slot: the replay is announced once per statement per session,
        on the first restore.  Removing the dedupe instead would flood a
        re-run — the very thing we avoided so users don't learn to filter
        the whole class away.
        """
        if allow_random or not unseeded_calls:
            return
        try:
            warn_stale_randomness(
                strip_markers(code),
                unseeded_calls,
                self.detector,
                suppress_warning=allow_random,
            )
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            logger.debug("%s Stale-randomness warning failed for statement", _LOG_PROCESSOR)

    def unseeded_estimator_fits(self, est_fit: set[str]) -> list[str]:
        """Return the sorted subset of *est_fit* receivers that are UNSEEDED.

        A bare ``estimator.fit(X, y)`` under ``# @cash:cache-fit`` caches via the
        cache-fit path, but cash's AST
        randomness detector cannot see the randomness inside sklearn's compiled
        ``.fit()``. An estimator built without a ``random_state`` draws fresh
        entropy each fit, so the cached fitted model is a frozen replay -- two
        honest fits would differ. This flags exactly those receivers.

        UNSEEDED iff ``get_params()`` contains ``random_state`` AND it is ``None``.
        An int / ``RandomState`` / ``Generator`` seed -> SEEDED (no warning). No
        ``random_state`` param at all (e.g. ``LinearRegression``) -> deterministic
        (no warning). Any ``get_params`` failure -> no warning: advisory output
        must never crash the statement.
        """
        unseeded: list[str] = []
        for rf in est_fit:
            try:
                est = self.shell.user_ns.get(rf)
                if est is None:
                    continue
                params = est.get_params()
                if "random_state" in params and params["random_state"] is None:
                    unseeded.append(rf)
            except (AttributeError, TypeError, ValueError, KeyError):
                continue
        return sorted(unseeded)

    def warn_unseeded_estimator_fit(
        self,
        code: str,
        est_fit: set[str],
        allow_random: bool,
    ) -> list[str]:
        """Warn that an UNSEEDED estimator ``.fit()`` is cached as a frozen replay.

        The estimator-fit analogue of :meth:`warn_unseeded`, emitted at
        COMPUTE time on the common path (before the cache lookup) so the warning
        describes the *source* independently of this run's hit/miss outcome. Goes
        out on the SAME ``CashRandomnessWarning`` path, so users'
        existing filters catch it.

        Returns the unseeded receivers found, so the cache-hit path can announce
        the replay without re-deriving them. ``# @cash:allow-random`` suppresses
        the warning (but the set is still returned; the restore twin re-checks the
        directive and stays silent too). Never allowed to break execution.
        """
        if not est_fit:
            return []
        unseeded = self.unseeded_estimator_fits(est_fit)
        if not unseeded:
            return []
        try:
            warn_unseeded_estimator_fit(
                strip_markers(code),
                unseeded,
                self.detector,
                suppress_warning=allow_random,
            )
        except (ValueError, AttributeError, RecursionError):
            logger.debug("%s Estimator-fit randomness warning failed", _LOG_PROCESSOR)
        return unseeded

    @staticmethod
    def _statement_calls_fit(tree: ast.Module | None) -> bool:
        """True if the statement calls ``.fit(...)`` / ``.partial_fit(...)``."""
        if tree is None:
            return False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("fit", "partial_fit")
            ):
                return True
        return False

    def _inline_unseeded_fit_outputs(
        self,
        tree: ast.Module | None,
        outputs: set[str],
    ) -> list[str]:
        """Output vars that are UNSEEDED fitted estimators (gap).

        Closes the anonymous/assignment-form fit hole the receiver-name path
        (``MutationClassifier.estimator_fit_receivers``) cannot reach: ``clf =
        RandomForestClassifier(...).fit(X, y)`` caches ``clf`` as an ordinary
        output, but the estimator is constructed inline (no named receiver) and
        the randomness lives inside compiled ``.fit()``, so nothing warns. Detect
        it from the OUTPUT value instead — a statement that calls ``.fit()`` whose
        output duck-types as an sklearn estimator with ``random_state is None``.
        Advisory and never fatal; the value must already exist (post-execute or
        post-restore).
        """
        if not outputs or not self._statement_calls_fit(tree):
            return []
        found: list[str] = []
        for var in outputs:
            try:
                est = self.shell.user_ns.get(var)
                if est is None:
                    continue
                if not (callable(getattr(est, "fit", None)) and callable(getattr(est, "get_params", None))):
                    continue
                params = est.get_params()
                if "random_state" in params and params["random_state"] is None:
                    found.append(var)
            except (AttributeError, TypeError, ValueError, KeyError):
                continue
        return sorted(found)

    def flag_inline_unseeded_fit(
        self,
        metrics: "ProcessResult",
        code: str,
        tree: ast.Module | None,
        outputs: set[str],
        allow_random: bool,
        *,
        is_hit: bool,
    ) -> None:
        """Badge + warn for an inline/assignment-form unseeded fit (gap).

        Stamps the metric so the badge shows the unseeded pill, and routes the
        same warning the named-receiver path uses — the compute-time "detected"
        claim on a miss, the "frozen replay" claim on a hit.
        """
        # Cheap guard on the hot path: no ``fit`` token, no AST walk. A ``.fit`` /
        # ``.partial_fit`` call always spells "fit", so this cannot false-negate.
        if not outputs or "fit" not in code:
            return
        fits = self._inline_unseeded_fit_outputs(tree, outputs)
        if not fits:
            return
        self.stamp_random_effect(metrics, code, [], fits)
        if is_hit:
            self.warn_stale_estimator_fit(code, fits, allow_random)
        else:
            self.warn_unseeded_estimator_fit(code, fits, allow_random)

    def observe_statement(self, pre_rng: dict, code: str | None = None) -> None:
        """Record what this statement did to the global RNG streams.

        THE RNG observer: one before/after diff per statement, serving both
        consumers, at the same granularity the file tracker already works at.

        * ``observed_draw`` -- what this statement DREW, for the badge's
          unseeded pill. A module merely *imported* by the statement is newly
          present in the post-snapshot and ``rng_modules_changed`` reports that
          as changed (the replay side wants to know), but an import is not a
          draw, so it is filtered out here.
        * the per-cell accumulation harvested by :meth:`cell_observation`
          for the replay ledger, which is still keyed by cell. Left UNFILTERED
          so that ledger keeps the semantics it was built on.

        The cell's start position is the FIRST executed statement's pre-state.
        That is deliberately narrower than the cell-wide snapshot this replaced,
        which also spanned cash's own machinery and so recorded a position that
        included cash's work rather than the user's.

        Never allowed to break execution: a capture failure just means this
        statement reports no randomness.
        """
        try:
            post = capture_rng_state()
            changed = rng_modules_changed(pre_rng, post)
            self.observed_draw = {m for m in changed if m in pre_rng}
            if self._cell_pre is None:
                self._cell_pre = pre_rng
            self._cell_changed |= changed
            self._cell_post = post
            self._record_hidden_draws(code, self.observed_draw)
        except (TypeError, AttributeError):  # pragma: no cover - defensive
            self.observed_draw = set()

    def _record_hidden_draws(self, code: str | None, drew: set[str]) -> None:
        """Remember that *code* drew from RNG modules its AST does not mention.

        ``rf4.fit(X, y)`` spells no ``np.random`` yet consumes the global stream.
        Until this ledger existed the observation fed only the badge, so the
        statement never read its module's virtual RNG variable and a re-seed
        above it could not reach it -- its consumers kept hitting across a
        stream that no longer existed.

        Recorded for EVERY statement, including skip-cache ones: a bare
        ``model.fit()`` is not cached itself, but it is exactly the statement
        whose consumers must be invalidated.
        """
        if not code or not drew:
            return
        try:
            visible = get_drawing_rng_modules(strip_markers(code))
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            return
        hidden = set(drew) - set(visible)
        if not hidden:
            return
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        ledger = self.tracking_state.observed_rng_statement_draws
        known = ledger.get(digest, set())
        if hidden - known:
            # First time we have learned this statement draws. Its key was built
            # BEFORE we knew, so it carries no RNG variable. That only leaves a
            # STALE-SEED trap when one of the drawn modules is actually SEEDED:
            # then a later seed change rebuilds the epoch-free key and matches
            # the old value. For a purely UNSEEDED draw there is no epoch to
            # change, so the frozen value is correct and must be kept from the
            # first call -- skipping the write there would redraw and break the
            # freeze-from-first-call contract.
            if hidden & set(self.seed_epochs):
                self.draw_newly_seen = True
        ledger[digest] = known | hidden

    def begin_cell(self) -> None:
        """Open a fresh per-cell RNG accumulation, before the cell's statements run."""
        self._cell_changed = set()
        self._cell_pre = None
        self._cell_post = None

    def cell_observation(self) -> tuple[set[str], dict | None, dict | None]:
        """What this cell's statements changed, and the positions either side.

        Harvested by the cell executor in place of its own before/after diff, so
        a single observer feeds both the badge and the replay ledger.
        """
        return set(self._cell_changed), self._cell_pre, self._cell_post

    def flag_observed_hidden_draw(
        self,
        metrics: "ProcessResult",
        code: str,
        outputs: set[str],
        *,
        skip_cache: bool,
    ) -> None:
        """Stamp the unseeded pill for a draw only the runtime observer saw.

        The AST detector (:meth:`stamp_random_effect`) sees ``np.random.rand()``
        spelled in the cell; object-introspection
        (:meth:`flag_inline_unseeded_fit`) sees a fitted estimator bound as an
        output. Neither sees a draw hidden inside a called function --
        ``x = make_data()`` where ``make_data`` does ``np.random.rand()``. The
        before/after global-RNG diff captured in :meth:`_executing`
        does: the stream advanced with no name and no syntax to give it away.

        Three gates keep this from mislabeling anything the other paths handle:

        * Only when the value is FROZEN. A ``skip_cache`` statement recomputes
          fresh every run, and a statement that binds no output caches nothing --
          either way there is no frozen replay to warn about. That is exactly why
          a bare, uncached ``model.fit()`` (mutates its receiver, binds nothing)
          correctly shows no pill.
        * Only draws the AST could NOT see. A draw the AST spelled out is already
          classified -- seeded or not -- by :meth:`stamp_random_effect` through
          the session seed ledger, so overriding it here would mislabel a draw
          from a module seeded in an *earlier* statement.
        * Only genuinely unseeded modules. A hidden draw from a module already
          seeded this session (the ``seed_epochs`` ledger) is reproducible.
          A seeded estimator fit never touches the global stream at all, so it
          never even reaches here.
        """
        if skip_cache or not outputs:
            return
        changed = self.observed_draw
        if not changed:
            return
        try:
            ast_draws = get_drawing_rng_modules(strip_markers(code))
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            return
        seeded = set(self.seed_epochs)
        hidden_unseeded = changed - ast_draws - seeded
        if not hidden_unseeded:
            return
        metrics["random_effect"] = "draw"
        metrics["random_unseeded"] = True

    def warn_stale_estimator_fit(
        self,
        code: str,
        unseeded_fits: list[str],
        allow_random: bool,
    ) -> None:
        """Announce that a cached UNSEEDED estimator fit was just replayed.

        The restore-time twin of :meth:`warn_unseeded_estimator_fit`, mirroring
        how :meth:`warn_stale` follows :meth:`warn_unseeded`:
        it makes the stronger claim only a successful restore licenses -- the
        fitted model on screen IS a replay, not merely "may differ". Called ONLY
        after ``CacheHitServer.serve`` reports a successful restore.
        """
        if allow_random or not unseeded_fits:
            return
        try:
            warn_stale_estimator_fit(
                strip_markers(code),
                unseeded_fits,
                self.detector,
                suppress_warning=allow_random,
            )
        except (ValueError, AttributeError, RecursionError):
            logger.debug("%s Stale estimator-fit warning failed", _LOG_PROCESSOR)
