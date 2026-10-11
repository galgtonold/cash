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
from ...analysis.mutation_effects import (
    control_structure_mutations,
    is_module_name,
    live_function_source,
    statement_watch_candidates,
)
from ...control_markers import strip_markers
from ...value_types import BUILTIN_NAMES
from ..cache_key import statement_source_hash
from ..call_key import changes_its_closure
from ..callee_reach import CWD, ENVIRON, module_state_names, module_state_writes, process_state_writes, state_holders
from .control_body import is_control_body

if TYPE_CHECKING:
    from ...analysis.mutation_effects import StatementEffects
    from .._protocols import ShellProtocol
    from ..tracking_state import TrackingState
    from .in_loop import InLoopChanges
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
        in_loop: InLoopChanges | None = None,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self._classifier = classifier
        self._records = records
        self._in_loop = in_loop
        #: Loop-body statement texts an earlier kernel recorded no verdict for.
        self._unrecorded: set[str] = set()

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
        run.state_modules = modules
        if modules:
            # The module is an output too, as a list is of `items.append(x)`:
            # its lineage moves with each statement that sets state on it, so
            # once a reload drops that state the upstream check rebuilds it
            # from them (`module_state_names`). The loop or branch owns its
            # body's writes, as with any receiver.
            if not is_control_body(run.code):
                run.state_names = module_state_names(run.code, self.shell.user_ns)
                run.outputs = run.outputs | run.state_names
            _skip(
                run,
                f"Sets state on module: {', '.join(sorted(modules))} (statement re-executes, so the module has it)",
            )
        # The same for one that changes the process's environment or working
        # directory, itself or in a function it calls (`setup()` doing
        # `os.environ["MODE"] = "b"`): a hit would leave them as they are, and
        # a restart's rebuild runs it again (`ReexecutionPlanner`).
        run.process_state = process_state_writes(run.code, self.shell.user_ns)
        if run.process_state:
            what = " and ".join(_PROCESS_WORDS[kind] for kind in sorted(run.process_state))
            _skip(run, f"Changes the {what} (statement re-executes, so the process has it)")
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
        run.verdict_hash = run.source_hash
        if is_control_body(run.code):
            draw_only = self._classifier.identity_coupled_call_receivers(tree)
            fit_only = self._classifier.fitted_receivers(tree)
            # What else it changes in place it changes on every run, as the
            # same line does at cell level: classified like one, its verdict
            # shared by every pass (the text without its marker), but the
            # names stay out of its outputs (`_route_receivers`).
            run.verdict_hash = statement_source_hash(strip_markers(run.code))
            self._recall_verdict(run.verdict_hash)
            user_ns = self.shell.user_ns
            extra = statement_watch_candidates(
                tree,
                user_ns,
                lambda name: is_module_name(name, user_ns),
                lambda name: live_function_source(name, user_ns),
            )
            changed, run.mut_observe, run.mut_assumed, run.mut_record = self._classifier.classify(
                tree, run.verdict_hash, run.outputs, extra
            )
            run.must_run = run.mut_record
            return changed - draw_only - fit_only, draw_only, fit_only
        mut_pre_route, run.mut_observe, run.mut_assumed, run.mut_record = self._classifier.classify(
            tree,
            run.source_hash,
            run.outputs,
        )
        run.est_fit = self._classifier.estimator_fit_receivers(tree, run.outputs) if run.cache_fit else set()
        self._route_unit(run)
        return mut_pre_route, set(), set()

    def _recall_verdict(self, verdict_hash: str) -> None:
        """Take in the verdict an earlier kernel recorded for a statement of
        a loop body, once per text: learnt again after every restart, its
        first pass would run instead of being restored."""
        verdicts = self.tracking_state.mutation_verdicts
        if verdict_hash in verdicts or verdict_hash in self._unrecorded:
            return
        recorded = self._records.recorded_mutation_verdict(verdict_hash)
        if recorded is None:
            self._unrecorded.add(verdict_hash)
        else:
            verdicts[verdict_hash] = recorded

    def _route_unit(self, run: StatementRun) -> None:
        """For a loop or branch run as one unit: skip-cache it when it
        changes in place a name it does not store, as the same calls do at
        cell level (``st.push(v)``, ``push(st, v)``, ``tag(df, i)`` in its
        body); the structure's own lineage update moves those names. A loop
        that hands names to code whose effect is unknown is watched as it
        runs (``loop_watch``), and must run while its verdict is learned."""
        tree = run.tree
        if not _is_structure(tree):
            return
        node = tree.body[0]
        user_ns = self.shell.user_ns
        lineage = self.tracking_state.variable_lineage

        def resolve(name: str) -> str | None:
            return live_function_source(name, user_ns)

        try:
            changed = control_structure_mutations(
                node,
                lambda name: name in BUILTIN_NAMES and name not in lineage,
                lambda name: is_module_name(name, user_ns),
                namespace=user_ns,
                resolve_source=resolve,
            )
        except Exception:  # noqa: BLE001 - what the body changes is unknown: it runs
            _skip(run, "Could not tell what the loop changes in place (statement re-executes)")
            return
        if self._in_loop is not None:
            watch = self._in_loop.open_watch(node, run.code, resolve)
            if watch is not None:
                if watch.verdict is not None:
                    changed |= watch.verdict
                elif watch.learning:
                    run.loop_watch = run.code
                    run.must_run = True
        run.in_place = set(changed) - run.outputs
        if run.in_place:
            _skip(
                run,
                f"In-place mutation on: {', '.join(sorted(run.in_place))} (statement re-executes)",
            )

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
        if is_control_body(run.code):
            # The loop or branch owns what its body changes: the statement
            # runs every time, and the names stay out of its outputs.
            run.in_place = (mut_pre_route | fam) - run.outputs
            if run.in_place:
                _skip(
                    run,
                    f"In-place mutation on: {', '.join(sorted(run.in_place))} (statement re-executes)"
                    + self._classifier.cache_fit_hint(run.in_place),
                )
            return
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

        A statement of a loop or branch body keeps what it changed out of its
        outputs (the structure owns it) and moves it on for the passes after
        it (``InLoopChanges.moved``). A loop run as one unit that was being
        watched learns what it changed (``loop_watch``).
        """
        self._observe_unit(run)
        body = is_control_body(run.code)
        if run.mut_record:
            newly_mutated = self._observe_receivers(run, body)
            verdict_hash = run.verdict_hash or run.source_hash
            self.tracking_state.mutation_verdicts[verdict_hash] = set(run.mut_assumed) | newly_mutated
            self._records.persist_mutation_verdict(verdict_hash, self.tracking_state.mutation_verdicts[verdict_hash])
        if body and run.in_place and self._in_loop is not None:
            self._in_loop.moved(run.in_place, run.cache_key)

    def _observe_receivers(self, run: StatementRun, body: bool) -> set[str]:
        metrics = run.metrics
        newly_mutated = self._classifier.observed_mutations(run.mut_observe, run.verdict_hash or run.source_hash)
        if not newly_mutated:
            return newly_mutated
        if body:
            fresh = newly_mutated - run.outputs - run.in_place
            run.in_place |= newly_mutated - run.outputs
            if fresh:
                run.skip_cache = True
                metrics.setdefault("uncacheable_reasons", []).append(
                    f"In-place mutation on: {', '.join(sorted(fresh))} (observed; statement re-executes)"
                )
            return newly_mutated
        # ``run.est_fit`` is non-empty only under ``# @cash:cache-fit``.
        # Those receivers still enter the outputs (source-based lineage
        # bump + fitted value capture) and are still recorded in
        # ``mutation_verdicts`` (so the upstream simulation bumps
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
        return newly_mutated

    def _observe_unit(self, run: StatementRun) -> None:
        """A loop run as one unit whose watch was learning ran: what it
        changed among the names it was watched for is a change it does not
        store, so it is skip-cached for them too."""
        if run.loop_watch is None or self._in_loop is None:
            return
        seen = self._in_loop.finish_watch(run.loop_watch) - run.outputs - run.in_place
        if seen:
            run.in_place |= seen
            run.skip_cache = True
            run.metrics.setdefault("uncacheable_reasons", []).append(
                f"In-place mutation on: {', '.join(sorted(seen))} (observed; statement re-executes)"
            )

    def note_module_state(self, run: StatementRun) -> None:
        """Take in the local modules *run* was seen setting state on, and
        record those it set state on for the simulation.

        One the text did not say (``mylib.put("K", 5)`` doing ``globals()[name]
        = v``, ``s.apply(5)`` doing ``global K`` in a method) is routed as
        `route` routes one it says: the names holding the module are outputs,
        the statement re-executes, and a reload of the module rebuilds it.
        A loop or branch owns its body's, as with any receiver
        (``update_lineage_after_execution``).
        """
        user_ns = self.shell.user_ns
        observed = run.rebound_modules - run.state_modules
        structure = is_control_body(run.code) or _is_structure(run.tree)
        if observed:
            names = state_holders(observed, run.code, user_ns)
            if structure:
                self.tracking_state.structure_module_writes.update(observed)
            else:
                run.outputs = run.outputs | names
                run.state_names = run.state_names | names
                produced = run.metrics.setdefault("evaluated_vars", [])
                produced.extend(n for n in sorted(names) if n not in produced)
            writers = self.tracking_state.module_state_writers
            for module in observed:
                listed = writers.setdefault(module, [])
                if run.code in listed:
                    listed.remove(run.code)
                listed.append(run.code)
            _skip(
                run,
                f"Seen setting state on module: {', '.join(sorted(observed))} "
                "(statement re-executes, so the module has it)",
            )
        if not structure:
            self._records.note_module_state(run.code, run.state_names, run.state_modules | observed)
        if run.process_state:
            self._records.note_process_state(run.code, run.process_state)


_PROCESS_WORDS = {ENVIRON: "environment", CWD: "working directory"}


def _is_structure(tree: ast.Module | None) -> bool:
    """Whether *tree* is one loop or branch, run as one unit."""
    from ..control_structures.common import is_control_structure

    return tree is not None and len(tree.body) == 1 and is_control_structure(tree.body[0])


def _skip(run: StatementRun, reason: str) -> None:
    """Skip-cache *run*, giving *reason*."""
    run.skip_cache = True
    run.metrics["uncacheable_reasons"].append(reason)
