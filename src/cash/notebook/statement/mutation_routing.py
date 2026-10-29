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
import dis
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..._user_attrs import attribute_or
from ...analysis.ast_util import called_names
from ...analysis.mutation_effects import (
    control_structure_mutations,
    is_module_name,
    live_function_source,
    statement_watch_candidates,
)
from ...control_markers import strip_markers
from ...value_hash import costly_after
from ...value_types import BUILTIN_NAMES
from ..cache_key import statement_source_hash
from ..call_key import changes_its_closure
from ..callee_reach import CWD, ENVIRON, module_state_names, module_state_writes, process_state_writes, state_holders
from ..shared_objects import history_names
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
        # ``Out[2].set_title('x')`` changes the axes ``Out`` holds, not
        # IPython's history: never an output, watched or evicted
        # (`history_names`).
        history = history_names(mut_pre_route | fit_only | run.mut_observe, self.shell.user_ns)
        if history:
            mut_pre_route, fit_only = mut_pre_route - history, fit_only - history
            run.mut_observe = run.mut_observe - history
        self._route_receivers(run, effects, mut_pre_route, fit_only)
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
        if closure_writers and not is_control_body(run.code):
            # The functions sharing a cell with the callee are outputs, as a
            # list is of `items.append(x)`: `rebind(BIG)` setting `nonlocal
            # rows` changes what `draw` reads, so a rebuild of `draw` runs
            # this statement again after the one that made it.
            run.outputs = run.outputs | sharing_a_cell(closure_writers, self.shell.user_ns)
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
            rewatch=True,
        )
        run.est_fit = self._classifier.estimator_fit_receivers(tree, run.outputs)
        if run.est_fit and _is_structure(tree):
            # Stored and restored as outputs: the structure's own lineage
            # update leaves them be (`_finalize_single_unit`), or a hit would
            # move them on from the restored model, whose digest is not the
            # fitted one's (equal dtypes that are one object in a fitted
            # model, copies in a loaded one).
            run.metrics["estimator_fits"] = sorted(run.est_fit)
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
            watch = self._in_loop.open_watch(node, run.code, resolve, run.cache_key)
            if watch is not None:
                if watch.verdict is not None:
                    changed |= watch.verdict
                elif watch.learning:
                    run.must_run = True
                if watch.watching:
                    # Seen running on new inputs: what it changes now is
                    # compared once it ran (`_observe_unit`).
                    run.loop_watch = run.code
        # An estimator fitted in the body (`with warnings.catch_warnings():
        # gs.fit(X, y)`) is stored as the fit at cell level is
        # (`_route_receivers`).
        run.in_place = set(changed) - run.outputs - run.est_fit
        run.unapplied_fills = frozenset(self._unapplied_fills(run, run.in_place))
        run.in_place -= run.unapplied_fills
        if run.unapplied_fills:
            run.metrics["unapplied_fills"] = sorted(run.unapplied_fills)
        if run.in_place:
            _skip(
                run,
                f"In-place mutation on: {', '.join(sorted(run.in_place))} (statement re-executes)",
            )

    def _unapplied_fills(self, run: StatementRun, names: set[str]) -> set[str]:
        """The names among *names*, changed in place by the loop *run*, that
        it may be stored without: each holds the very object a statement
        bound it to that cash did not store (``pbar = ProgBar(...)``: its
        own parts hold it, or it is not worth an entry), the loop only calls
        methods on it and drops
        what they return (``pbar.update()``), nothing else holds it or what
        it holds (`_kept_to_itself`), and nothing after the loop reads it:
        not the rest of its top-level statement, the statements below it in
        the cell, the cells below, or a function of the notebook; none of
        those may read the namespace by name (``eval``, ``globals()``).
        A hit then leaves it as it was, and once the statement finished its
        lineage says so (`TrackingState.unapplied`). Never while the upstream
        check rebuilds, or outside a notebook's cell."""
        state = self.tracking_state
        if not names or not state.unstored or state.running_cell is None:
            return set()
        below, at = state.read_by_later_cells, state.running_statement
        if below is None or at is None or any(reader in below for reader in _NAMESPACE_READERS):
            return set()
        user_ns = self.shell.user_ns
        picked = {
            name
            for name in names
            if name in user_ns
            and state.unstored.get(name) == (id(user_ns[name]), id(type(user_ns[name])))
            and name not in below
            and _only_called_on(run.tree, name)
        }
        if not picked:
            return picked
        body, index = at
        later = _loaded_names(body[index + 1 :])
        if later is None:
            return set()
        picked -= later
        picked = {name for name in picked if _loads(body[index], name) <= _loads(run.tree, name)}
        read = _read_by_functions(user_ns, picked | _NAMESPACE_READERS)
        if read & _NAMESPACE_READERS:
            return set()
        return {
            name
            for name in picked - read
            if _kept_to_itself(user_ns[name], name, user_ns, _methods_called(run.tree, name))
        }

    def _route_receivers(
        self, run: StatementRun, effects: StatementEffects, mut_pre_route: set[str], fit_only: set[str] = frozenset()
    ) -> None:
        """Add the mutated receivers and arguments to *run*'s outputs, and
        skip-cache it for a receiver that must be mutated on every run.

        Fitted estimators are the exception (``run.est_fit``). A bare
        ``estimator.fit(X, y)`` changes its receiver in place; the receiver
        joins ``outputs`` (its lineage moves on and the fitted value is
        stored) and the statement is NOT skip-cached, so a restart followed by
        Run All restores the fit instead of training again. A hit rebinds the
        receiver like any other output: the variables holding it or an object
        inside it (``backup = clf``, the steps of a ``Pipeline``, the
        ``param_grid`` of a ``GridSearchCV``) are found by the share check and
        stored and restored with it, so they hold the restored objects again;
        a holder that is not a variable, or an object it shares with an input
        (``knn._fit_X is X``), refuses the store and the fit re-runs. A
        receiver that is BOTH an estimator fit AND another genuine skip
        receiver still skips.
        """
        est_fit = run.est_fit
        fam = effects.arg_mutations - run.outputs
        if is_control_body(run.code):
            # The loop or branch owns what its body changes: the statement
            # runs every time, and the names stay out of its outputs. An
            # estimator it fits moves on as well, whether the fit was seen
            # (the pass that learnt the verdict) or recalled (every pass
            # after a restart): moved only when seen, the fitted models of
            # `for est in estimators: est.fit(X, y)` left the loop with other
            # lineages after a restart, and what was built on them missed.
            run.in_place = (mut_pre_route | fam | fit_only) - run.outputs
            if run.in_place:
                _skip(
                    run,
                    f"In-place mutation on: {', '.join(sorted(run.in_place))} (statement re-executes)",
                )
            return
        skip_pre_route = mut_pre_route - est_fit
        if mut_pre_route or est_fit or fam:
            run.outputs = run.outputs | mut_pre_route | est_fit | fam
        # A module a `set_*` call is handed names with gets a reason of its
        # own, taken back if the call is seen setting on a name instead
        # (`_settle_provisional`).
        provisional = {
            module: handed
            for module, handed in self._classifier.provisional_modules(run.source_hash).items()
            if module in skip_pre_route
        }
        run.mut_provisional = provisional
        skip_pre_route -= set(provisional)
        if skip_pre_route:
            _skip(
                run,
                f"In-place mutation on: {', '.join(sorted(skip_pre_route))} "
                "(receiver lineage bumped; statement re-executes)" + self._too_costly(skip_pre_route),
            )
        if provisional:
            run.mut_provisional_reason = f"In-place mutation on: {', '.join(sorted(provisional))} (receiver lineage bumped; statement re-executes)"
            _skip(run, run.mut_provisional_reason)

    def _too_costly(self, names: set[str]) -> str:
        """Which of *names* are assumed changed because a check of them ran
        out of its budget (`value_hash.is_costly`), and after how long: the
        lineage they get makes what reads them miss, for that alone."""
        user_ns = self.shell.user_ns
        cut = {name: costly_after(user_ns.get(name)) for name in sorted(names)}
        said = [f"{name} ({seconds:.1f}s)" for name, seconds in cut.items() if seconds is not None]
        if not said:
            return ""
        return f" -- assumed changed, its change check gave up as too costly to read: {', '.join(said)}"

    def _callees_changing_their_closure(self, tree: ast.AST | None) -> set[str]:
        """The functions *tree* calls by name, outside loop and branch bodies,
        that change their closure (:func:`changes_its_closure`)."""
        return callees_changing_their_closure(tree, self.shell.user_ns)

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
        digests = self._classifier.digests
        digests.begin_settled()
        try:
            self._observe_after(run, body)
        finally:
            run.settled_digests = digests.take_settled()

    def _observe_after(self, run: StatementRun, body: bool) -> None:
        if run.mut_record:
            newly_mutated = self._observe_receivers(run, body)
            self._settle_provisional(run, newly_mutated)
            verdict_hash = run.verdict_hash or run.source_hash
            self.tracking_state.mutation_verdicts[verdict_hash] = set(run.mut_assumed) | newly_mutated
            self._records.persist_mutation_verdict(verdict_hash, self.tracking_state.mutation_verdicts[verdict_hash])
        elif run.rewatching:
            self._observe_rewatch(run)
        if body and self._in_loop is not None:
            self._in_loop.note_bindings(run.tree)
            if run.in_place:
                self._in_loop.moved(run.in_place, run.cache_key)

    @staticmethod
    def _settle_provisional(run: StatementRun, newly_mutated: set[str]) -> None:
        """A module a ``set_*`` call was handed names with, when the call was
        seen changing one of them: it set on that name, not on the module
        (``nx.set_node_attributes(G, ...)``), so the module leaves the
        outputs and the reasons. The verdict, which then lists the name and
        not the module, decides the next runs and the simulation alike."""
        settled = {module for module, handed in run.mut_provisional.items() if handed & newly_mutated}
        if not settled:
            return
        run.outputs = run.outputs - settled
        reasons = run.metrics.get("uncacheable_reasons", [])
        if run.mut_provisional_reason in reasons:
            reasons.remove(run.mut_provisional_reason)
        left = sorted(set(run.mut_provisional) - settled)
        if left:
            run.mut_provisional_reason = (
                f"In-place mutation on: {', '.join(left)} (receiver lineage bumped; statement re-executes)"
            )
            reasons.append(run.mut_provisional_reason)
        run.mut_provisional = {m: h for m, h in run.mut_provisional.items() if m not in settled}

    def _observe_rewatch(self, run: StatementRun) -> None:
        """A statement whose verdict is known ran on new inputs, watched for
        what the verdict does not hold (`MutationClassifier.start_rewatch`):
        what it changed now joins the verdict, the outputs and the reasons
        it re-executes, as on the run that learned the verdict."""
        changed = self._classifier.rewatched_changes(run.source_hash) - run.outputs
        if not changed:
            return
        verdicts = self.tracking_state.mutation_verdicts
        verdicts[run.source_hash] = set(verdicts.get(run.source_hash, ())) | changed
        self._records.persist_mutation_verdict(run.source_hash, verdicts[run.source_hash])
        run.outputs = run.outputs | changed
        produced = run.metrics.setdefault("evaluated_vars", [])
        produced.extend(n for n in sorted(changed) if n not in produced)
        run.skip_cache = True
        run.metrics.setdefault("uncacheable_reasons", []).append(
            f"In-place mutation on: {', '.join(sorted(changed))} (observed; receiver lineage bumped; statement re-executes)"
        )

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
        # Estimator-fit receivers (``run.est_fit``) still enter the outputs
        # (lineage bump + fitted value capture) and are still recorded in
        # ``mutation_verdicts`` (so the upstream simulation bumps downstream
        # lineage on a data edit), but they are NOT skip-cached. Every
        # other observed mutation still skip-caches its receiver.
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
            )
        return newly_mutated

    def _observe_unit(self, run: StatementRun) -> None:
        """A loop run as one unit whose watch was learning ran: what it
        changed among the names it was watched for is a change it does not
        store, so it is skip-cached for them too."""
        if run.loop_watch is None or self._in_loop is None:
            return
        seen = self._in_loop.finish_watch(run.loop_watch) - run.outputs - run.in_place - run.unapplied_fills
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


def callees_changing_their_closure(tree: ast.AST | None, user_ns: Mapping[str, Any]) -> set[str]:
    """The functions *tree* calls by name, outside loop and branch bodies,
    that change their closure (:func:`changes_its_closure`), and those of
    *user_ns* a notebook function it calls names in turn: ``def setup():
    rebind([0])``, then ``setup()``, rebinds ``rebind``'s cell."""
    found: set[str] = set()
    seen: set[str] = set()
    todo = list(called_names(tree, "no_control_bodies"))
    while todo and len(seen) < _CALLEES_FOLLOWED:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        fn = user_ns.get(name)
        if changes_its_closure(fn):
            found.add(name)
        if type(fn) is types.FunctionType and fn.__globals__ is user_ns:
            todo.extend(_global_names(fn.__code__))
    return found


#: How many functions `callees_changing_their_closure` looks at.
_CALLEES_FOLLOWED = 256


def _global_names(code: types.CodeType) -> set[str]:
    """The names *code* and the code nested in it look up: its globals among them."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def sharing_a_cell(writers: set[str], user_ns: Mapping[str, Any]) -> set[str]:
    """The functions of *user_ns* that share a closure cell with one of
    *writers*, those included: what a call of one of them changes."""
    cells = {id(cell) for name in writers for cell in attribute_or(user_ns.get(name), "__closure__", None) or ()}
    if not cells:
        return set()
    return {
        name
        for name, value in list(user_ns.items())
        if type(value) is types.FunctionType and any(id(cell) in cells for cell in value.__closure__ or ())
    }


_PROCESS_WORDS = {ENVIRON: "environment", CWD: "working directory"}


def _is_structure(tree: ast.Module | None) -> bool:
    """Whether *tree* is one loop or branch, run as one unit."""
    from ..control_structures.common import is_control_structure

    return tree is not None and len(tree.body) == 1 and is_control_structure(tree.body[0])


#: Calls and names that read the namespace by name, which no text scan follows.
_NAMESPACE_READERS = frozenset({"globals", "locals", "vars", "eval", "exec", "dir", "get_ipython", "__main__"})

#: Most objects `_kept_to_itself` looks at in what the object holds.
_KEPT_MAX = 5000

#: Values whose sharing changes nothing: they cannot be changed in place.
_ATOMS = (str, bytes, int, float, complex, bool, type(None), range)


def _methods_called(tree: ast.AST, name: str) -> set[str]:
    """The methods *tree* calls on *name* (``bar.update()``)."""
    return {
        sub.func.attr
        for sub in ast.walk(tree)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and isinstance(sub.func.value, ast.Name)
        and sub.func.value.id == name
    }


def _kept_to_itself(obj: Any, name: str, user_ns: Mapping[str, Any], methods: set[str]) -> bool:
    """Whether a hit may leave *obj* (bound to *name*) as it was without
    another value changing with it: no other variable, container, default
    or closure holds *obj* or anything it holds (``Bar(sink)`` appending to
    the notebook's ``sink``), nothing it holds is notebook code, and the
    notebook methods the loop calls on it (*methods*) name no notebook
    value, class or namespace reader as a global. Streams it writes to are
    output, which a hit shows again."""
    import gc
    import io

    held: dict[int, Any] = {}
    todo = [obj]
    while todo:
        item = todo.pop()
        if isinstance(item, _ATOMS) or id(item) in held:
            continue
        if isinstance(item, (types.ModuleType, type, io.IOBase)):
            continue
        if isinstance(item, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
            owner = attribute_or(item, "__self__", None)
            if isinstance(item, types.FunctionType) or isinstance(owner, types.ModuleType):
                if attribute_or(item, "__module__", None) == "__main__":
                    return False
                continue
            todo.append(owner)
            continue
        held[id(item)] = item
        if len(held) > _KEPT_MAX:
            return False
        todo.extend(gc.get_referents(item))
    if any(other is not name and id(value) in held for other, value in user_ns.items()):
        return False
    kept = list(held.values())
    for holder in gc.get_referrers(*kept):
        if id(holder) in held or holder is user_ns or holder is kept or isinstance(holder, types.FrameType):
            continue
        return False
    for method in methods:
        func = attribute_or(attribute_or(type(obj), method, None), "__func__", None) or attribute_or(
            type(obj), method, None
        )
        if isinstance(func, types.FunctionType) and not _names_no_notebook_value(func, user_ns, set()):
            return False
    return True


def _names_no_notebook_value(func: types.FunctionType, user_ns: Mapping[str, Any], seen: set[int]) -> bool:
    """Whether notebook function *func*, and the notebook functions it
    names, name as a global only modules, builtins and library code."""
    if attribute_or(func, "__module__", None) != "__main__" or id(func) in seen:
        return True
    seen.add(id(func))

    def globals_of(code: types.CodeType):
        for ins in dis.get_instructions(code):
            if ins.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL", "LOAD_NAME"):
                yield ins.argval
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                yield from globals_of(const)

    for global_name in globals_of(func.__code__):
        if global_name in _NAMESPACE_READERS:
            return False
        if global_name not in user_ns:
            continue
        value = user_ns[global_name]
        if isinstance(value, types.ModuleType):
            continue
        if isinstance(value, types.FunctionType) and _names_no_notebook_value(value, user_ns, seen):
            continue
        if (
            isinstance(value, (type, types.BuiltinFunctionType))
            and attribute_or(value, "__module__", None) != "__main__"
        ):
            continue
        return False
    return True


def _loaded_names(nodes) -> set[str] | None:
    """The names *nodes* read; None when one reads the namespace by name."""
    names: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id in _NAMESPACE_READERS:
                    return None
                names.add(sub.id)
    return names


def _loads(node: ast.AST, name: str) -> int:
    """How often *node* reads *name*."""
    return sum(1 for sub in ast.walk(node) if isinstance(sub, ast.Name) and sub.id == name)


def _only_called_on(tree: ast.AST, name: str) -> bool:
    """Whether *tree* uses *name* only as ``name.method(...)`` statements
    whose result is dropped and whose arguments are literals: what the
    object holds then reaches nothing else the loop changes."""
    allowed: set[int] = set()
    for sub in ast.walk(tree):
        if not (isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call)):
            continue
        func = sub.value.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == name):
            continue
        args = [*sub.value.args, *(k.value for k in sub.value.keywords)]
        if all(_literal(arg) for arg in args):
            allowed.add(id(func.value))
    return all(id(sub) in allowed for sub in ast.walk(tree) if isinstance(sub, ast.Name) and sub.id == name)


def _literal(node: ast.AST) -> bool:
    """A constant, or a string formatted from anything (``f"{i}"``)."""
    if isinstance(node, ast.JoinedStr):
        return True
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _read_by_functions(user_ns: Mapping[str, Any], names: set[str]) -> set[str]:
    """The names among *names* a function or class of the notebook reads as
    a global, which no cell's text shows."""
    read: set[str] = set()

    def visit(code: types.CodeType) -> None:
        read.update(names.intersection(code.co_names))
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                visit(const)

    for value in list(user_ns.values()):
        if isinstance(value, types.FunctionType) and attribute_or(value, "__module__", None) == "__main__":
            visit(value.__code__)
        elif isinstance(value, type) and attribute_or(value, "__module__", None) == "__main__":
            for member in list(vars(value).values()):
                parts = (member.fget, member.fset) if isinstance(member, property) else (member,)
                for part in parts:
                    func = attribute_or(part, "__func__", part)
                    if isinstance(func, types.FunctionType):
                        visit(func.__code__)
    return read


def _skip(run: StatementRun, reason: str) -> None:
    """Skip-cache *run*, giving *reason*."""
    run.skip_cache = True
    run.metrics["uncacheable_reasons"].append(reason)
