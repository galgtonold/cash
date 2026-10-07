"""What a statement changes in place decides its outputs and whether it caches.

A statement that changes an object in place (``lst.append(x)``, a callee
writing a global, a draw on a live Figure) has no assignment target for the
change, so the static analysis of its inputs and outputs misses it. Before
the statement runs, :meth:`MutationRouting.route` adds the receivers it
mutates to its outputs, so their lineage is bumped, and skip-caches it where
the mutation must really happen on every run. After it ran,
:meth:`MutationRouting.observe` does the same for what execution was seen
changing.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...analysis.ast_util import called_names
from ..call_key import changes_its_closure
from ..callee_reach import module_state_names, module_state_writes
from .control_body import is_control_body

if TYPE_CHECKING:
    from ...analysis.mutation_effects import StatementEffects
    from .._protocols import ShellProtocol
    from ..tracking_state import TrackingState
    from .mutations import MutationClassifier
    from .records import StatementRecords
    from .run import StatementRun

__all__ = ["MutationRouting"]


class MutationRouting:
    """Routes a statement's in-place mutations into its outputs and its
    cacheability, through the :class:`MutationClassifier`."""

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        classifier: MutationClassifier,
        records: StatementRecords,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self._classifier = classifier
        self._records = records

    # ------------------------------------------------------------------
    # Before execution
    # ------------------------------------------------------------------

    def route(self, run: StatementRun, effects: StatementEffects) -> None:
        """Add the receivers *run* mutates to its outputs, and skip-cache it
        where the mutation must really happen on every run."""
        mut_pre_route, draw_only, fit_only = self._classify(run)
        self._route_receivers(run, effects, mut_pre_route)
        # Deliberately the SAME treatment the inline spelling of the identical
        # mutation gets in `_route_receivers`: the statement re-executes so the
        # callee's write to a global really happens.
        #
        # The expensive work is NOT lost. Call interception still serves the
        # call inside this statement, keyed on the mutated global's own
        # pre-call state, so what re-executes is the glue around it.
        callee_globals = set(effects.callee_globals)
        if callee_globals:
            _skip(
                run,
                f"Callee mutates: {', '.join(sorted(callee_globals))} "
                "(global lineage bumped; statement re-executes, call still cached)",
            )
        # The same for a callee that changes its closure: `add` appending to
        # the list `make_log` gave it, `counter` bumping a `nonlocal`. The
        # list is no variable of the notebook, so it is not an output and has
        # no lineage; the statement only re-executes. Restored whole, it
        # would skip the call inside, and the list would stay empty. In a loop
        # or branch body too: without a lineage the body statement's key never
        # moves, so nothing else sends it back to run.
        closure_writers = self._callees_changing_their_closure(run.tree)
        if closure_writers:
            _skip(
                run,
                f"Callee changes its closure: {', '.join(sorted(closure_writers))} "
                "(statement re-executes, call still cached)",
            )
        # The same for one that sets state on a local module: `mylib.add(5)`
        # bumping the module's counter, `mylib.K = slow()`. The module is no
        # variable of the notebook either, and a hit restores the statement's
        # outputs, not the module: after a restart the counter stayed at 0.
        # In a loop or branch body too, for the same reason as above.
        modules = module_state_writes(run.code, self.shell.user_ns)
        if modules:
            # The module is an output too, as a list is of `items.append(x)`:
            # its lineage moves with each statement that sets state on it, so
            # once a reload drops that state the upstream check rebuilds it
            # from them (`module_state_names`). The loop or branch owns its
            # body's writes, as with any receiver.
            if not is_control_body(run.code):
                run.outputs = run.outputs | module_state_names(run.code, self.shell.user_ns)
            _skip(
                run,
                f"Sets state on module: {', '.join(sorted(modules))} (statement re-executes, so the module has it)",
            )
        # A draw inside a loop/branch body. Skip the CACHE without touching
        # ``outputs`` -- the statement must re-execute so the artists actually
        # land on the Axes, but bumping its lineage from a per-statement source
        # is precisely what the control-body rule in `_classify` avoids.
        if draw_only:
            _skip(run, f"Draws on: {', '.join(sorted(draw_only))} (live Figure/Axes; statement re-executes)")
        if fit_only:
            _skip(run, f"Fits: {', '.join(sorted(fit_only))} (estimator fitted in place; statement re-executes)")

    def _classify(self, run: StatementRun) -> tuple[set[str], set[str], set[str]]:
        """``(mutated receivers, drawn-on receivers, fitted receivers)`` of
        *run*, and its receivers to observe, assume and record.

        A standalone bare-Expr method call (``lst.append(x)``, ``bus.on(fn)``)
        has no Store target, so AST analysis never surfaces the receiver as an
        output and its lineage stays frozen -> a cached downstream consumer
        would serve a stale value once the mutation is edited. The classifier
        decides which receivers actually mutate (statically known, a prior
        runtime verdict, or assume-mutate); the rest are observed by content
        after execution (:meth:`observe`).

        Control-structure BODY statements arrive here individually with an
        injected marker comment, but the upstream simulation treats the whole
        loop/branch as one unit (its mutations flow through the loop-mutation
        lineage path, not per-body classification). Classifying a body
        statement would bump the receiver with a per-statement source the
        simulation never reproduces -> cross-cell desync. So they are not
        classified, and the control structure owns its body's mutation
        lineage -- with ONE exception: a draw on a live Figure/Axes, or an
        estimator fitted in place, still skip-caches the body statement.
        """
        tree = run.tree
        if is_control_body(run.code):
            draw_only = self._classifier.identity_coupled_call_receivers(tree)
            fit_only = self._classifier.fitted_receivers(tree)
            return set(), draw_only, fit_only
        mut_pre_route, run.mut_observe, run.mut_assumed, run.mut_record = self._classifier.classify(
            tree,
            run.source_hash,
            run.outputs,
        )
        run.est_fit = self._classifier.estimator_fit_receivers(tree, run.outputs) if run.cache_fit else set()
        return mut_pre_route, set(), set()

    def _route_receivers(self, run: StatementRun, effects: StatementEffects, mut_pre_route: set[str]) -> None:
        """Add the mutated receivers and arguments to *run*'s outputs, and
        skip-cache it for a receiver that must be mutated on every run.

        Fitted estimators are the exception, and only under
        ``# @cash:cache-fit`` (``run.est_fit``). A bare ``estimator.fit(X, y)``
        mutates its receiver in place, so it is routed to skip-caching: the
        statement re-executes and is never serialised, which is net-NEUTRAL
        -- a fit that keeps missing cannot cost more than it saves.

        It does NOT make aliases safe. ``backup = clf`` is an ORDINARY
        ASSIGNMENT that cash caches on its own, and restoring it rebinds
        ``backup`` to a pre-fit deserialised copy -- the fit statement has no
        bearing on it either way.

        Caching a bare fit is the OPT-IN path, kept because it is a large win
        when it lands but not the default because its correctness surface
        exceeds what per-statement restore can guarantee:

        * a cache HIT may REBIND the receiver, leaving an alias pointing at
          the pre-fit object. Not fixable per-statement -- on a warm run-all
          the CONSTRUCTOR statement's own hit-restore rebinds the receiver
          before the fit's in-place transfer lands, so the alias graph is
          already broken upstream; and
        * the duck-type gate admits the whole sklearn-compatible universe
          (xgboost/lightgbm/custom), each with its own ``__getstate__``
          contract, and several never restore -- re-serialising every run for
          a net LOSS.

        For reliable ML caching, wrap training in a returning function under
        ``@cash.cache`` instead.

        When opted in, the receiver joins ``outputs`` (so its source-based
        lineage is bumped AND the fitted value is captured/saved) but the
        statement is NOT skip-cached, so the normal lookup runs (hit ->
        in-place restore; miss -> execute + save). A receiver that is BOTH an
        estimator fit AND another genuine skip receiver still skips (the skip
        wins for that receiver). ``est_fit`` also threads to the cache-hit
        path so its restore is IN PLACE. Without the directive ``est_fit`` is
        empty and every site degrades to the skip-cache behaviour.
        """
        est_fit = run.est_fit
        fam = effects.arg_mutations - run.outputs
        skip_pre_route = mut_pre_route - est_fit
        if mut_pre_route or est_fit or fam:
            run.outputs = run.outputs | mut_pre_route | est_fit | fam
        if skip_pre_route:
            _skip(
                run,
                f"In-place mutation on: {', '.join(sorted(skip_pre_route))} "
                "(receiver lineage bumped; statement re-executes)" + self._classifier.cache_fit_hint(skip_pre_route),
            )

    def _callees_changing_their_closure(self, tree: ast.AST | None) -> set[str]:
        """The functions *tree* calls by name, outside loop and branch bodies,
        that change their closure (:func:`changes_its_closure`)."""
        user_ns = self.shell.user_ns
        return {name for name in called_names(tree, "no_control_bodies") if changes_its_closure(user_ns.get(name))}

    # ------------------------------------------------------------------
    # After execution
    # ------------------------------------------------------------------

    def observe(self, run: StatementRun) -> None:
        """Add the receivers execution was seen mutating to *run*'s outputs.

        Broad-precise mutation observation: for a standalone method call whose
        method is not statically known, compare each candidate receiver's
        content after execution against its pre-statement hash. Runs BEFORE
        capture_and_track so a newly-detected mutation is in the outputs (its
        lineage gets bumped) and skip-caches the statement. The verdict is
        recorded for the upstream simulation, which cannot observe execution.
        """
        if not run.mut_record:
            return
        metrics, source_hash = run.metrics, run.source_hash
        newly_mutated = self._classifier.observed_mutations(run.mut_observe, source_hash)
        if newly_mutated:
            # ``run.est_fit`` is non-empty only under ``# @cash:cache-fit``.
            # Those receivers still enter the outputs (source-based lineage
            # bump + fitted value capture) and are still recorded in
            # ``mutation_verdicts`` below (so the upstream simulation bumps
            # downstream lineage on a data edit), but they are NOT
            # skip-cached -- they cache + restore in place. Every
            # other observed mutation -- including a bare fit WITHOUT the
            # directive -- still skip-caches its receiver.
            run.outputs = run.outputs | newly_mutated
            # The metrics hold their own copy of the outputs: without this the
            # badge row would say "Produced -" for ``sc.pp.normalize_total(adata)``
            # on its first run.
            produced = metrics.setdefault("evaluated_vars", [])
            produced.extend(n for n in sorted(newly_mutated) if n not in produced)
            skip_observed = newly_mutated - run.est_fit
            if skip_observed:
                run.skip_cache = True
                metrics.setdefault("uncacheable_reasons", []).append(
                    f"In-place mutation on: {', '.join(sorted(skip_observed))} "
                    "(observed; receiver lineage bumped; statement re-executes)"
                    + self._classifier.cache_fit_hint(skip_observed)
                )
        self.tracking_state.mutation_verdicts[source_hash] = set(run.mut_assumed) | newly_mutated
        self._records.persist_mutation_verdict(source_hash, self.tracking_state.mutation_verdicts[source_hash])


def _skip(run: StatementRun, reason: str) -> None:
    """Skip-cache *run*, giving *reason*."""
    run.skip_cache = True
    run.metrics["uncacheable_reasons"].append(reason)
