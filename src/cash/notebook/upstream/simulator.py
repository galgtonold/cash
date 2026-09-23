"""Notebook simulator: pure-AST + cache-probing replay of upstream cells.

Extracted from ``UpstreamChecker`` so the simulation logic has a clear test
surface independent of the orchestrator. See ``docs/architecture_decisions.md``
ADR-009.

The simulator never executes user code via the IPython kernel. It simulates
statement-by-statement using AST analysis and the cache backend, producing a
plan of statements to re-execute and a list of restored statements. The
orchestrator (``UpstreamChecker``) takes that plan and runs it via the real
``process_statement_callback``.
"""

from __future__ import annotations

import ast
import builtins
import collections
import logging
import os
import sys
import types
from collections.abc import Callable
from typing import Any

from cash.control_markers import strip_markers

from ...analysis.ast_util import parse_cached, resolve_callee
from ...analysis.cacheability import analyze_statement
from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source
from ...analysis.mutation_effects import CellEffects
from ...analysis.mutations import consumed_input_names
from ...analysis.namespace_effects import resolve_literal_path, resolve_path_list, statement_read_paths
from ...tracking.function_tracker import FunctionTracker, is_local_module
from ...value_types import BUILTIN_NAMES
from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from .._trace import is_tracing, trace_event
from ..cache_key import read_provenance_key
from ..consumables import consumable_state, has_diverged, is_consumable_unrestorable
from ._types import CellCheck, ReexecutionPlan, SimulationCache, SimulationResult, apply_collected_mutations
from .mismatch_classifier import MismatchClassifier
from .reexecution_planner import ReexecutionPlanner
from .virtual_lineage import VirtualLineage, loop_derived_vars

__all__ = ["NotebookSimulator"]


def _statement_codes(cell_source: str) -> list[str]:
    """The cell's top-level statements as the runtime keys them (unparsed,
    with an expression's trailing ``;`` kept); the raw text if it does not parse."""
    try:
        clean = clean_cell_source(cell_source)
        tree = parse_cell_source(cell_source)
    except (ValueError, TypeError):
        return [cell_source]
    if tree is None:
        return [cell_source]
    # Local: import cycle upstream.simulator -> ipython.cell_executor -> ... -> upstream.simulator.
    from ..ipython.cell_executor import CellExecutor

    codes = []
    for node in tree.body:
        code = ast.unparse(node)
        if CellExecutor.expr_has_trailing_semicolon(clean, node):
            code += ";"
        codes.append(code)
    return codes


def _bind_literal_paths(stmt: str, bound: dict, namespace) -> None:
    """Record in *bound* a name *stmt* binds to a path or a list of paths.

    ``TF = [Path('other.csv')]`` binds ``TF``; any other binding of a name
    drops it, so a later statement never reads a stale value from here.
    """

    try:
        tree = ast.parse(CodeAnalyzer.strip_magics(stmt))
    except (SyntaxError, ValueError, TypeError):
        return
    node = tree.body[0] if len(tree.body) == 1 else None
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        value = resolve_path_list(node.value, namespace)
        if value is None:
            value = resolve_literal_path(node.value, namespace)
        if value is not None:
            bound[name] = value
            return
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound[n.id] = None


logger = logging.getLogger(__name__)


class NotebookSimulator:
    """Replays upstream cells via AST simulation and cache probing.

    Owned by :class:`UpstreamChecker`, with which it shares the
    ``TrackingState``. :meth:`simulate_upstream` runs the three phases --
    :class:`VirtualLineage`, :class:`MismatchClassifier`,
    :class:`ReexecutionPlanner` -- and applies what they buffered.
    :meth:`simulate_cell` and :meth:`restore_statement` do one cell or one
    statement the same way.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol | None,
        tracking_state: TrackingState,
        compute_hash_fn: Callable[[Any], str] | None = None,
        function_tracker: FunctionTracker | None = None,
    ) -> None:
        self.shell = shell
        self.cash_instance = cash_instance
        self.compute_hash_fn = compute_hash_fn

        # Shared state refs (same dicts as UpstreamChecker / StatementProcessor).
        self.set_tracking_state(tracking_state)
        #: Set by ``reset_caches`` (``%cash_on``): adopt untracked names once.
        self._adopt_untracked_pending = False
        #: The previous simulation's per-cell snapshots, where the next one starts.
        self.cache = SimulationCache()

        # Phase-1 simulator. Shares ``shell``/``cash_instance``/tracking-state
        # references with us so writes are visible on both sides.
        self.virtual_lineage = VirtualLineage(
            shell=shell,
            cash_instance=cash_instance,
            tracking_state=tracking_state,
            compute_hash_fn=compute_hash_fn,
            function_tracker=function_tracker,
            cache=self.cache,
        )

        # Phase-2 classifier. Shares tracking-state references and routes
        # back to ``virtual_lineage`` for cache-probing helpers.
        self.classifier = MismatchClassifier(
            virtual_lineage=self.virtual_lineage,
            tracking_state=tracking_state,
        )

        # Phase-3 planner. Routes into VL + Classifier for helpers that
        # still live on those phases.
        self.planner = ReexecutionPlanner(
            virtual_lineage=self.virtual_lineage,
            classifier=self.classifier,
        )

    def set_tracking_state(self, state: TrackingState) -> None:
        """Re-wire shared state refs (mirrors UpstreamChecker.set_tracking_state)."""
        self.tracking_state = state
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages
        # Propagate to the Phase-1 simulator so its dict refs stay in sync.
        if hasattr(self, "virtual_lineage"):
            self.virtual_lineage.set_tracking_state(state)
        if hasattr(self, "classifier"):
            self.classifier.set_tracking_state(state)

    def reset_caches(self) -> None:
        """Forget the previous simulation.

        Called only by ``%cash_on``, so it also arms the one adoption of
        untracked names -- see ``_adopt_untracked_names``.
        """
        self.virtual_lineage.reset_caches()
        self._adopt_untracked_pending = True

    def _track_modules_bound_before_cash_on(self) -> None:
        """Track the local modules the namespace reached before cash was listening.

        Cash starts tracking a project module when a cell it PROCESSES imports
        it, and the cell that turns cash on is not one of them. So
        `import cash; %cash_on; import helpers as hm` -- the layout the
        quickstart recommends -- left `helpers` untracked: an edit to it
        reloaded nothing, and its source reached no cache key, so a value built
        from it was served pre-edit, even after Restart & Run All (round 28,
        r28s4, exported). Done before pass 1, so this very simulation already
        keys the module's readers on its source.
        """
        ft = self.virtual_lineage.function_tracker
        user_ns = getattr(self.shell, "user_ns", None)
        if ft is None or not user_ns:
            return

        names: set[str] = set()
        for value in list(user_ns.values()):
            if isinstance(value, types.ModuleType):
                names.add(value.__name__)
            else:
                owner = getattr(value, "__module__", None)
                if isinstance(owner, str) and owner != "__main__":
                    names.add(owner)
        for mod_name in names:
            module = sys.modules.get(mod_name)
            if module is None or mod_name in ft.tracked_modules:
                continue
            # cash itself is "local" in a development checkout, and `cash` is
            # bound in every notebook: without this it watched its own source.
            if mod_name == "cash" or mod_name.startswith("cash."):
                continue
            try:
                if is_local_module(module):
                    ft.track_module(mod_name)
            except (AttributeError, OSError, TypeError, ValueError):
                logger.debug("Could not track '%s' at %%cash_on", mod_name)

    def _adopt_untracked_names(self, virtual_lineage: dict[str, str], simulation_trace: list) -> None:
        """Give names bound before cash was listening the simulation's lineage.

        A name bound in the ``%cash_on`` cell (``DATA = Path(...)``, ``N = 3``)
        has no runtime lineage: cash was not listening when that cell started.
        The first statement reading it was refused as "Input variable missing
        lineage" -- in the quickstart's own layout, that is the cell that loads
        the data (round 27, r27s1 and r27s4). The simulation reads the cell out
        of the .ipynb and has a lineage for it like any other, and it is the
        one the simulation keys the reader with, so adopting it is also what
        lets the runtime store under the key a restart will look up.

        Once per ``%cash_on``, at the first check after it, and only there:
        before anything is tracked no lineage can have been dropped ON PURPOSE
        (the module invalidator drops a from-import's lineage to force the
        next reader to miss), so there is nothing for this to undo. Names the
        runtime already has a lineage for are left alone; import-bound names
        already had theirs propagated and are not touched either.

        Only for a name whose binding statement cannot have read anything
        (``_binds_without_reading``). Nothing under tracking saw that statement
        run, so nothing recorded a file it read: adopt ``RAW = DATA.read_text()``
        and a changed file leaves every cell below serving the old text, where
        Restart & Run All would not
        (``test_a_value_read_in_the_cash_on_cell_still_follows_its_file``,
        which is how the first draft of this was caught). Such a name keeps the
        old behaviour -- refused once, then repaired under tracking.
        """
        if not self._adopt_untracked_pending:
            return
        self._adopt_untracked_pending = False
        user_ns = getattr(self.shell, "user_ns", None)
        if not user_ns:
            return
        runtime = self.tracking_state.variable_lineage
        imported = self.virtual_lineage.propagated_imports
        restores = self.virtual_lineage.restores
        binder: dict[str, str] = {}
        for entry in simulation_trace or ():
            for out in entry.outputs or ():
                binder[out] = entry.stmt_code
        adopted = []
        untracked = self.tracking_state.rerun_bindings
        untracked.clear()
        for name, lineage_hash in virtual_lineage.items():
            if not lineage_hash or name in runtime or name in imported or name.startswith("_") or name not in user_ns:
                continue
            code = binder.get(name)
            if code is None or not _binds_without_reading(code, user_ns):
                # Re-run under tracking instead, by the first cell that needs
                # it -- see TrackingState.rerun_bindings.
                untracked.add(name)
                continue
            restores.record_restore(var_name=name, lineage_hash=lineage_hash, value=user_ns[name])
            adopted.append(name)
        apply_collected_mutations(restores, self.tracking_state)
        if adopted and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[UPSTREAM_DEBUG] Adopted simulated lineage for names bound before %%cash_on: %s", sorted(adopted)
            )

    def _apply_phase_mutations(self) -> None:
        """Drain phase RestoreCollectors and apply buffered ops to TrackingState.

        Phases buffer mutations as ``CacheRestore`` / ``LineageReset`` ops and
        usually drain themselves at method boundaries (so direct callers and
        mid-simulation reads see writes immediately). This safety-net drain
        catches anything left over after the full pipeline runs.
        """
        apply_collected_mutations(self.virtual_lineage.restores, self.tracking_state)
        apply_collected_mutations(self.classifier.restores, self.tracking_state)

    # --- One statement or cell at a time, as a check does it ---

    def simulate_cell(
        self,
        cell_code: str,
        virtual_lineage: dict[str, str] | None = None,
        virtual_modules: set[str] | None = None,
    ) -> SimulationResult:
        """Simulate *cell_code* on its own, from *virtual_lineage*.

        Probes the cache and records what it learns for the live session (an
        imported module's lineage) exactly as a check does for a cell above.
        """
        sim = SimulationResult(virtual_lineage=dict(virtual_lineage or {}), virtual_modules=set(virtual_modules or ()))
        self.virtual_lineage.simulate_one_cell(sim, -1, cell_code)
        self._apply_phase_mutations()
        return sim

    def restore_statement(
        self,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        input_hashes: dict[str, str],
        virtual_modules: set[str] | None = None,
        expected_lineages: dict[str, str] | None = None,
    ) -> set[str]:
        """Restore *stmt_code*'s outputs from the cache entry its simulated
        inputs key; the names restored (none when the entry is missing, stale
        or for other lineages)."""
        restored, _restore_time, _saved_time = self.virtual_lineage.try_virtual_restore(
            stmt_code, outputs, inputs, input_hashes, virtual_modules, expected_lineages
        )
        self._apply_phase_mutations()
        return restored

    def record_replayed_file_deps(self, rerecorded: set[str]) -> None:
        self.virtual_lineage.record_replayed_file_deps(rerecorded)

    def _mark_stale_value_inputs_broken(
        self,
        required_inputs: set[str] | None,
        effects: CellEffects,
        broken_vars: set[str],
        notebook_cells: list[str] | None = None,
        current_cell_idx: int | None = None,
        virtual_lineage: dict[str, str] | None = None,
    ) -> None:
        """Flag self-modifying required inputs whose live value is stale.

        ``variable_lineage[var]`` can be reset to a pre-cell base (downstream
        advancement / forward simulation) WITHOUT touching the in-memory value,
        whose own ``_cash_lineage_hash`` still reflects the later (advanced)
        version. The recorded lineage then *looks* consistent with the virtual
        state, so Pass 2 does not flag the variable — yet the cell would
        re-execute on a stale value. This is the self-referential re-run case:
        ``df = df.sort(); ...; df = df.rename()`` re-run reads the
        already-renamed frame and raises ``KeyError``. Modelling ``df`` as
        distinct versions (df_in -> df_out), the cell consumes the input
        version; when the live value's lineage disagrees with the recorded one
        the input version is not actually present, so mark it broken and let the
        normal restore / upstream-re-execution machinery re-derive its base.

        Two regimes, split on whether the live value carries a
        ``_cash_lineage_hash``:

        * **Lineage-carrying objects** (DataFrame/Series — branches (a)/(b)
          below). Restricted to variables the current cell **reassigns** with a
          fresh value (a ``Name`` store, ``df = ...``). In-place mutation of a
          such a var (``df['c'] = ...`` / ``df.attr = ...``) is deliberately
          excluded: it replays through the mutation-lineage restoration path,
          and its live value legitimately runs ahead of the reset base —
          flagging it here would force needless recompute of the whole cell (and
          wrongly defeat per-statement cache hits, e.g. an unchanged
          ``df['VolAdj']`` when only a later ``df['SMA']`` window changed).
        * **No-lineage values** (primitives / builtin containers / ndarray —
          ``_mark_nolineage_self_write_broken``). These cannot hold the attribute
          and their self-modifying statements skip the per-statement cache for
          missing input lineage, so an isolated re-run computes on the cell's own
          prior output (``lst.append`` doubles, ``total = total + k`` doubles).
          Both pure reassignment *and* in-place mutation are eligible here —
          there is no per-statement cache to defeat.

        Builtins are skipped.
        """
        # A called function that carries mutable state on its own object (a
        # mutated mutable-default arg, a function-attribute counter) must have its
        # ``def`` re-run to recreate fresh state — force its producer to re-run by
        # marking it broken. On ``run_all`` the def re-runs to the same fresh
        # object first, so this only adds a cheap redundant redefine (B).
        for fn in effects.stateful_funcs:
            if fn in self.shell.user_ns:
                broken_vars.add(fn)
        if not required_inputs:
            return
        reassigned = effects.reassigned
        # A no-lineage var the current cell mutates in place (``lst.append`` /
        # ``arr += 1`` / ``d.update``) is *self-written* even though it is not a
        # ``Name``-store reassignment.  Treat both as self-modifying inputs.
        inplace_self = effects.mutated & required_inputs
        self_written = reassigned | inplace_self
        if not self_written:
            return
        # Vars that an *upstream* cell also mutates in place. A no-lineage var
        # mutated across several cells (``results.append(..)`` once per cell) has
        # no recorded per-cell base — current_session_hashes never advances past
        # the first assignment — so the content-base check below would wrongly
        # reset it to that assignment and drop the intermediate cells' mutations.
        # Computed lazily (only when an in-place no-lineage candidate exists).
        upstream_inplace_mutated: set[str] | None = None
        # Names this cell changes without producing them again; lazily too.
        lineage_invisible: set[str] | None = None
        for var_name in required_inputs:
            # A user variable shadowing a builtin name is tracked in
            # variable_lineage; only skip genuine (untracked) builtins.
            if var_name in BUILTIN_NAMES and var_name not in self.variable_lineage:
                continue
            if var_name not in self_written:
                continue
            # A swap / rotate / temp-swap target (``a, b = b, a``) reads its own
            # pre-cell value but on an isolated re-run holds the swapped OUTPUT,
            # whose content and lineage both equal the recorded output — so the
            # lineage-base and content-base signals below are BOTH fooled. The
            # staleness is lineage-invisible, so force the reset from the static
            # detector: mark broken and let the producers restore the cell-entry
            # base. On ``run_all`` the producers re-run to the same base first, so
            # this only adds a cheap redundant restore there.
            if var_name in effects.crossref_reassigned:
                broken_vars.add(var_name)
                continue
            live_value = self.shell.user_ns.get(var_name)
            live_lineage = getattr(live_value, "_cash_lineage_hash", None)
            # The value is the one the simulation of the cells above says this
            # cell starts from: current, and not this cell's own earlier output.
            # The checks below compare it with what the LAST statement writing
            # it read -- this cell's starting state only when that statement
            # is in this cell. When it is in a cell above (``df['b'] = ...``
            # there, ``df['a'] = ...`` here), a first run looked stale and the
            # value was rebuilt (r23s4: its article frame, every Run All).
            # Only when each write this cell makes moves the value's lineage:
            # ``del df['b']`` or ``lst.append(x)`` changes it in place and
            # leaves the lineage where it was, so a re-run would pass for a
            # first run -- those keep the checks below.
            if (
                live_lineage is not None
                and virtual_lineage is not None
                and live_lineage == virtual_lineage.get(var_name)
                and var_name not in effects.method_receivers
            ):
                if lineage_invisible is None:
                    lineage_invisible = self._lineage_invisible_writes(notebook_cells, current_cell_idx)
                if var_name not in lineage_invisible:
                    continue
            if live_lineage is None:
                # Primitives / builtin containers / ndarray carry no
                # ``_cash_lineage_hash`` and their self-modifying statements skip
                # the per-statement cache (missing input lineage), so an isolated
                # re-run would compute on the cell's own prior output. Restore the
                # cell-entry base by marking the input broken (producer re-runs).
                if upstream_inplace_mutated is None:
                    upstream_inplace_mutated = self._scan_upstream_inplace_mutations(
                        notebook_cells,
                        current_cell_idx,
                    )
                self._mark_nolineage_self_write_broken(
                    var_name,
                    live_value,
                    broken_vars,
                    upstream_inplace_mutated,
                )
                continue
            if var_name not in reassigned:
                # A lineage-carrying var mutated in place -- not a Name
                # reassignment -- whose isolated re-run is non-idempotent and would
                # accumulate unless restored to its cell-entry base. Two cases:
                #   * METHOD receivers (``b.items.append(..)``): no-output method
                #     statements skip the per-statement cache (``results.append(x)``
                # + ``obj.total += x``).
                #   * SELF-REFERENTIAL subscript/attr writes (``df['a']=df['a']*2``,
                # ``df['a']+=1``, ``df.iloc[i,j]+=x``).
                # Restore via the same content/lineage-base machinery used for
                # no-lineage self-writes. Scoped so that a write to a NEW column read
                # from OTHER columns (``df['VolAdj']=df.groupby('Close')..``) is NOT
                # included and keeps its per-statement cache (preserved).
                force_reset = var_name in effects.method_receivers or var_name in effects.selfref
                if force_reset:
                    before = var_name in broken_vars
                    # The live VALUE's own lineage (``_cash_lineage_hash``) reflects
                    # the object actually in memory. When the mutation is nested in a
                    # control structure (``if c: df['a']=df['a']*2``) the runtime
                    # advances it past the cell-entry base, but the downstream-
                    # advancement fallback collapses the recorded ``variable_lineage``
                    # back to the base via ``reset_to`` (which leaves the value's
                    # attribute intact). So compare the VALUE's lineage against the
                    # cell-entry base — it survives the collapse and still betrays the
                    # stale (own-prior-mutation) value on an isolated re-run, while a
                    # fresh forward run (producer restored the base) leaves them equal.
                    #
                    base_lineage = self.executed_input_lineages.get(var_name, {}).get(var_name)
                    if base_lineage is not None and live_lineage != base_lineage:
                        broken_vars.add(var_name)
                    else:
                        if upstream_inplace_mutated is None:
                            upstream_inplace_mutated = self._scan_upstream_inplace_mutations(
                                notebook_cells,
                                current_cell_idx,
                            )
                        self._mark_nolineage_self_write_broken(
                            var_name,
                            live_value,
                            broken_vars,
                            upstream_inplace_mutated,
                        )
                    trace_event("force_reset", var=var_name, broke=(var_name in broken_vars and not before))
                continue
            recorded = self.variable_lineage.get(var_name)
            if recorded is None:
                continue
            # A reassigned lineage-carrying input whose live value is a VALID
            # EXTENSION of the notebook state — executing its recorded producing
            # code on the simulated inputs reproduces the live lineage — is a
            # legitimate fresh value produced UPSTREAM, not the current cell's own
            # stale self-referential output. Pass 2 already kept it via this exact
            # check; the stale-value guard must not override that, or the forward
            # probe would restore a stale cache entry keyed on the outdated virtual
            # lineage (the unsaved cell edit routing through a user function).
            # Scoped to vars produced by an UPSTREAM cell: a genuine
            # self-modification whose producer is a statement of the CURRENT cell
            # (``df = df.iloc[1:]`` re-run) must still hit the guard, or its
            # isolated re-run would double-apply. [layer 2]
            if virtual_lineage is not None:
                prod_code = self.executed_cell_codes.get(var_name)
                produced_by_current_cell = True
                if prod_code:
                    # Strip cash's context markers (``# control_context: ...`` /
                    # ``# __iteration_context__: ...``) so a control-nested
                    # self-write still matches its cell's source text.
                    norm_prod = strip_markers(prod_code).strip()
                    cur_src = (
                        notebook_cells[current_cell_idx]
                        if notebook_cells is not None
                        and current_cell_idx is not None
                        and 0 <= current_cell_idx < len(notebook_cells)
                        else ""
                    )
                    produced_by_current_cell = (not norm_prod) or (norm_prod in cur_src)
                if (
                    prod_code is not None
                    and not produced_by_current_cell
                    and self.virtual_lineage.is_valid_extension(
                        prod_code, recorded, virtual_lineage, required_dependency=var_name
                    )
                ):
                    continue
            # (a) ``variable_lineage[var]`` was reset to a pre-cell base (the
            # downstream-advancement reset, e.g. test_134's multi-statement
            # chain) but the live value still carries the advanced lineage: the
            # recorded/live disagreement betrays the stale value directly.
            if live_lineage != recorded:
                logger.debug(
                    "[UPSTREAM_DEBUG] '%s' has a stale in-memory value "
                    "(recorded lineage %s but value lineage %s); marking broken "
                    "so its input version is restored before the cell re-runs.",
                    var_name,
                    recorded[:8],
                    live_lineage[:8],
                )
                broken_vars.add(var_name)
                continue
            # (b) Self-modifying single statement (``df = df.iloc[1:]``): the
            # forward simulation reproduces the advanced lineage exactly, so
            # recorded == live == virtual and the reset in (a) never fires.
            # ``executed_input_lineages[var][var]`` is the version this cell
            # *consumed* the last time it ran (its cell-entry base). When the
            # live value's lineage differs from that base, the namespace holds
            # the cell's own prior OUTPUT rather than the base it must
            # re-consume on an isolated re-run -- mark it broken so the same
            # restore machinery re-derives the base. Self-disables on the first
            # run (no recorded input version yet) and on legitimate forward runs
            # (live == base). Primitives carry no ``_cash_lineage_hash`` and were
            # already skipped above; in-place mutation is excluded via
            # ``effects.reassigned``.
            base_input = self.executed_input_lineages.get(var_name, {}).get(var_name)
            if base_input is not None and live_lineage != base_input:
                logger.debug(
                    "[UPSTREAM_DEBUG] '%s' holds its own prior output on re-run "
                    "(value lineage %s but cell-entry base %s); marking broken "
                    "so its base is restored before the cell re-runs.",
                    var_name,
                    live_lineage[:8],
                    base_input[:8],
                )
                broken_vars.add(var_name)

    def _mark_consumed_unrestorable_inputs_broken(
        self,
        required_inputs: set[str] | None,
        broken_vars: set[str],
        notebook_cells: list[str] | None = None,
        current_cell_idx: int | None = None,
    ) -> set[str]:
        """Flag consumed, unrestorable inputs whose live object is already drained.

        Returns the subset of vars flagged here, so the planner can also schedule
        the upstream statements that FILL them (a consumable's filler statements
        are usually not its trace ``outputs`` — see
        ``ReexecutionPlanner._schedule_consumable_producer_touches``).

        The sibling guard above only ever examines variables the current cell
        **writes** (``self_written``; it returns early otherwise). A drained
        ``queue.Queue`` or an exhausted generator is a READ-ONLY input, so it is
        never looked at — the cell re-runs against the leftovers of its own
        previous run and prints ``got=[]`` / ``total=0`` where ``run_all`` (which
        re-runs the producer first) prints ``got=[0, 1, 2]`` / ``total=55``.


        Neither existing staleness signal can see this. The var carries no
        ``_cash_lineage_hash`` and its lineage never advances (the producer's
        record still points at ``q = Queue()``), so the lineage-base check is
        blind; and because a consumable drains IN PLACE its identity is constant,
        so ``compute_hash``'s ``sha256(str(id(obj)))`` fallback returns the SAME
        hash before and after draining and the content-base check is blind too.
        Hence the dedicated per-type probes in ``consumables.py``.

        Marking the var broken is most of the fix: the planner's backward scan
        then re-executes the statements that OWN it as a trace output. The
        remainder — scheduling the statements that FILL it, which typically do
        not own it — is the returned set's job (see the planner method named
        above).

        Self-disabling by construction: the probe compares against a baseline
        recorded at this cell's ENTRY on its previous run, so a ``run_all``
        (producer re-ran, object fresh) compares equal and this is a no-op, and
        a first run has no baseline at all.

        The cross-cell-accumulator hazard does not apply: that reset
        re-derives an object that another cell also mutates in place, whereas
        here re-executing the producer chain is exactly what ``run_all`` does.
        """
        flagged: set[str] = set()
        if not required_inputs:
            return flagged
        cell_src = (
            notebook_cells[current_cell_idx]
            if notebook_cells is not None
            and current_cell_idx is not None
            and 0 <= current_cell_idx < len(notebook_cells)
            else None
        )
        if not cell_src:
            return flagged
        try:
            consumed = consumed_input_names(parse_cached(cell_src))
        except (SyntaxError, ValueError, TypeError):
            return flagged
        candidates = required_inputs & consumed
        if not candidates:
            return flagged
        bases = getattr(self.tracking_state, "consumable_bases", {})
        for var_name in candidates:
            if var_name in BUILTIN_NAMES and var_name not in self.variable_lineage:
                continue
            live_value = self.shell.user_ns.get(var_name)
            if live_value is None:
                continue
            try:
                if not is_consumable_unrestorable(live_value):
                    continue
                diverged = has_diverged(
                    live_value,
                    bases.get(var_name),
                    had_baseline=(var_name in bases),
                )
            except (TypeError, ValueError, AttributeError, RecursionError):
                continue
            if not diverged:
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[UPSTREAM_DEBUG] consumed unrestorable input '%s' (%s) is already "
                    "drained on re-run (cell-entry base %r but live %r); marking broken "
                    "so its producer re-runs.",
                    var_name,
                    type(live_value).__name__,
                    bases.get(var_name),
                    consumable_state(live_value),
                )
            broken_vars.add(var_name)
            flagged.add(var_name)
            trace_event("consumable_broken", var=var_name)
        return flagged

    def _mark_nolineage_self_write_broken(
        self,
        var_name: str,
        live_value: Any,
        broken_vars: set[str],
        upstream_inplace_mutated: set[str] | None = None,
    ) -> None:
        """Mark a no-lineage self-modifying input broken if its live value is stale.

        Values with no ``_cash_lineage_hash`` (int/str, builtin list/dict/set,
        ndarray) cannot be tracked by the lineage branches (a)/(b). Two staleness
        signals, each of which self-disables on a fresh forward run (where the
        producer has already restored the cell-entry base) and only fires on an
        isolated re-run (where it has not):

        * **Lineage-base** — the var is a self-modifying *output*
          (``total = total + k``, ``arr += 1``). ``executed_input_lineages[var]
          [var]`` is the lineage of the version this cell *consumed* last run (the
          cell-entry base). After the cell ran, ``variable_lineage[var]`` advanced
          to the output lineage; on an isolated re-run the producer has not reset
          it, so base != current betrays the stale value. (On ``run_all`` the
          producer runs first and resets ``variable_lineage[var]`` to the base, so
          base == current and nothing is flagged.)
        * **Content-base** — pure in-place mutation (``lst.append``) produces no
          output, so ``executed_input_lineages[var]`` is absent and
          ``current_session_hashes[var]`` still holds the upstream producer's
          *content* hash (the base, never advanced by the mutation). A live
          content hash that differs means the namespace holds this cell's own
          prior mutation.

        The content-base check is suppressed when an *upstream* cell also mutates
        the var in place: such a var is accumulated across cells
        (``results.append(..)`` once per cell), its ``current_session_hashes``
        entry never advances past the first assignment, and restoring it to that
        assignment would drop the intermediate cells' contributions.
        """
        base_lineage = self.executed_input_lineages.get(var_name, {}).get(var_name)
        if base_lineage is not None:
            current_lineage = self.variable_lineage.get(var_name)
            if current_lineage is not None and current_lineage != base_lineage:
                logger.debug(
                    "[UPSTREAM_DEBUG] no-lineage self-write '%s' holds its own prior "
                    "output on re-run (cell-entry base lineage %s but current %s); "
                    "marking broken so its base is restored before the cell re-runs.",
                    var_name,
                    base_lineage[:8],
                    current_lineage[:8],
                )
                broken_vars.add(var_name)
            return

        if upstream_inplace_mutated and var_name in upstream_inplace_mutated:
            return
        if self.compute_hash_fn is None:
            return
        session_hashes = getattr(self.tracking_state, "current_session_hashes", {})
        base_content = session_hashes.get(var_name)
        if base_content is None:
            return
        try:
            live_content = self.compute_hash_fn(live_value)
        except (TypeError, ValueError, AttributeError, RecursionError):
            return
        if live_content != base_content:
            logger.debug(
                "[UPSTREAM_DEBUG] no-lineage in-place mutation '%s' holds its own prior "
                "output on re-run (cell-entry base content %s but live %s); marking "
                "broken so its base is restored before the cell re-runs.",
                var_name,
                base_content[:8],
                live_content[:8],
            )
            broken_vars.add(var_name)

    @staticmethod
    def _lineage_invisible_writes(
        notebook_cells: list[str] | None,
        current_cell_idx: int | None,
    ) -> set[str]:
        """Names a statement of the current cell changes without producing them.

        ``del df['b']``, ``lst.append(x)``: the value changes in place and its
        lineage does not move, unlike ``df['b'] = ...``, which binds ``df`` as
        an output. Every simple statement is looked at on its own, so one
        nested in a loop or a branch counts; a ``def`` or ``class`` body is
        not the cell writing anything. A cell that does not parse never ran,
        so it has nothing to report.
        """
        if not notebook_cells or current_cell_idx is None or not 0 <= current_cell_idx < len(notebook_cells):
            return set()
        tree = parse_cached(notebook_cells[current_cell_idx].replace("\r\n", "\n"))
        if tree is None:
            return set()
        invisible: set[str] = set()
        pending: list[ast.stmt] = list(tree.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if hasattr(node, "body"):  # for / while / if / with / try / match
                for field in ("body", "orelse", "finalbody"):
                    pending.extend(getattr(node, field, ()) or ())
                for part in [*getattr(node, "handlers", ()), *getattr(node, "cases", ())]:
                    pending.extend(part.body)
                continue
            try:
                code = ast.unparse(node)
                _, outputs = CodeAnalyzer.analyze_code_block(code)
                mutated = analyze_statement(code, None).top_level_mutated_vars
            except (SyntaxError, ValueError, TypeError):
                continue
            invisible |= set(mutated) - set(outputs)
        return invisible

    def _scan_upstream_inplace_mutations(
        self,
        notebook_cells: list[str] | None,
        current_cell_idx: int | None,
    ) -> set[str]:
        """Return the set of names mutated in place by any cell *before* the current one.

        Used to suppress the content-base staleness check for variables
        accumulated across several cells (see ``_mark_nolineage_self_write_broken``).
        """
        if not notebook_cells or not current_cell_idx:
            return set()
        muts: set[str] = set()
        for code in notebook_cells[:current_cell_idx]:
            try:
                # top-level only: a cell that merely DEFINES a function whose body
                # mutates ``g`` (``def bump(): global g; g += 1``) does not itself
                # accumulate ``g`` across cells, so it must not suppress the
                # content-base reset for a later cell that CALLS it.
                muts |= set(analyze_statement(code, None).top_level_mutated_vars)
            except (SyntaxError, ValueError, TypeError):
                continue
        return muts

    def _persisted_reads(self, code: str) -> set[str] | None:
        """Files *code* read when it last ran, from the backend, or ``None``."""
        cash = getattr(self.virtual_lineage, "cash_instance", None)
        backend = getattr(cash, "backend", None) if cash is not None else None
        if backend is None:
            return None

        try:
            record = backend.get_metadata(read_provenance_key(code))
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        if not record or not record.get("read_provenance"):
            return None
        return set(record.get("paths") or ())

    @staticmethod
    def _statements_the_cell_depends_on(
        required_inputs: set[str] | None,
        simulation_trace: list,
    ) -> set[int]:
        """Trace positions whose outputs the current cell's inputs derive from.

        Only these can make a file the cell's reconstruction reads. Before this
        scope, one unresolvable read anywhere above -- a helper's
        ``pd.read_parquet(path)`` -- let every stale writer in the notebook
        re-fire, and with them the fits feeding their charts: a sanity-check
        cell reading only the loaded frame took 309 s (round 24, r24s1).
        """
        if required_inputs is None:
            return set(range(len(simulation_trace)))
        needed = set(required_inputs)
        relevant: set[int] = set()
        for i in range(len(simulation_trace) - 1, -1, -1):
            outputs, inputs = simulation_trace[i].outputs, simulation_trace[i].inputs
            if outputs & needed:
                relevant.add(i)
                needed |= set(inputs)
        return relevant

    def _defs_whose_callers_recorded_reads(
        self,
        simulation_trace: list,
        relevant: set[int],
        efd: dict,
    ) -> set[int]:
        """Relevant ``def`` statements whose reads are already known elsewhere.

        Defining a function reads nothing; its body reads when a statement
        calls it, and the tracker records that against the caller's outputs
        (or, after a restart, the caller's persisted reads). A path the body
        leaves unresolvable (``pd.read_csv(path)``) then says nothing unknown.
        A caller that is itself such a ``def`` counts when it is covered.
        """
        defs: dict[int, str] = {}
        for i in relevant:
            code = simulation_trace[i].stmt_code
            if not code.lstrip().startswith(("def ", "async def ", "@")):
                continue
            try:
                body = ast.parse(code).body
            except SyntaxError:
                continue
            if len(body) == 1 and isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[i] = body[0].name

        def recorded(i: int) -> bool:
            outputs = simulation_trace[i].outputs
            if outputs and all(efd.get(o) for o in outputs):
                return True
            return self._persisted_reads(simulation_trace[i].stmt_code) is not None

        covered: set[int] = set()
        changed = True
        while changed:
            changed = False
            for i, name in defs.items():
                if i in covered:
                    continue
                callers = [j for j in relevant if j > i and name in simulation_trace[j].inputs]
                if callers and all((j in covered) if j in defs else recorded(j) for j in callers):
                    covered.add(i)
                    changed = True
        return covered

    def _compute_relevant_read_paths(
        self,
        required_inputs: set[str] | None,
        simulation_trace: list,
        notebook_cells: list[str] | None,
        current_cell_idx: int | None,
    ) -> tuple[set[str], bool]:
        """File paths this cell's reconstruction actually READS.

        A file-writer is only worth re-firing during reconstruction when a
        consumer relevant to the current cell reads the file it writes. This
        collects those consumed paths from three sources:

        * the recorded file-deps of the current cell's required inputs (the
          within-session, already-propagated read edges), and
        * every file READ statically named by an upstream trace statement, and
        * every file READ statically named by the current cell itself (the only
          signal that survives a kernel restart, when the tracking dicts are
          empty and the reader is the cell the user ran).

        Returns ``(paths, fully_known)``. ``fully_known`` is ``False`` when any
        recognised read target could not be statically resolved (an f-string /
        computed path) — the caller must then suppress no writer, since it cannot
        prove the writer's output is unread. A writer whose resolvable output
        path is in none of these paths is an unrelated / terminal side-effect and
        must not be re-fired for THIS cell.
        """

        paths: set[str] = set()
        fully_known = True
        user_ns = getattr(self.shell, "user_ns", None)

        efd = getattr(self.tracking_state, "executed_file_deps", None) or {}
        for v in required_inputs or ():
            dep = efd.get(v)
            if not dep:
                continue
            # Recorded file deps are usually {path: snapshot} but some code paths
            # store a plain set/list of paths -- accept either shape.
            paths.update(dep.keys() if hasattr(dep, "keys") else dep)

        def _collect(src: str, outputs=(), namespace=None) -> None:
            nonlocal fully_known
            try:
                clean = CodeAnalyzer.strip_magics(src.replace("\r\n", "\n"))
            except (ValueError, TypeError):
                return
            if not clean.strip():
                return
            try:
                r = statement_read_paths(clean, namespace=user_ns if namespace is None else namespace)
            except (SyntaxError, ValueError, TypeError):
                r = None
            if r is None and outputs and all(o in efd for o in outputs):
                # Not resolvable from the code (``pd.read_csv(f)`` over a glob
                # result), but the statement ran this session and the tracker
                # recorded what fed its outputs -- a superset of what it read.
                # Without this one comprehension switched the scope gate off
                # for the whole notebook, and a chart nothing reads was re-drawn
                # for every downstream cell (round 21, R5).
                r = set()
                for o in outputs:
                    dep = efd[o]
                    r.update(dep.keys() if hasattr(dep, "keys") else dep)
            if r is None:
                # After a restart the session record is empty; what the
                # statement read when it last ran was persisted for this.
                r = self._persisted_reads(src)
            if r is None:
                trace_event("read_path_unknown", stmt=src[:90])
                fully_known = False
            else:
                paths.update(r)

        relevant = self._statements_the_cell_depends_on(required_inputs, simulation_trace)
        covered_defs = self._defs_whose_callers_recorded_reads(simulation_trace, relevant, efd)
        for i, entry in enumerate(simulation_trace):
            if i not in relevant:
                continue
            code = entry.stmt_code
            if i not in covered_defs and ("read" in code or "open(" in code or "load" in code):
                _collect(code, entry.outputs)
            # What the tracker recorded behind this statement's outputs counts
            # too, whatever the code looks like: a reader static analysis does
            # not recognise (``PIL.Image.open(p)``) must not make its file look
            # unread now that more write paths resolve.
            for o in entry.outputs:
                dep = efd.get(o)
                if dep:
                    paths.update(dep.keys() if hasattr(dep, "keys") else dep)

        if notebook_cells and current_cell_idx is not None and 0 <= current_cell_idx < len(notebook_cells):
            # One statement at a time, keyed as the runtime keys them, so a
            # statement's persisted read record is found after a restart (the
            # whole cell's text is no statement's key).
            # The cell has not run yet, so a path its own earlier statement
            # binds (``TF = [Path('other.csv')]``) is in no namespace; resolve
            # it from the code (round 30, r30s5).
            bound: dict = {}
            for stmt in _statement_codes(notebook_cells[current_cell_idx]):
                _collect(stmt, namespace=collections.ChainMap(bound, user_ns or {}))
                _bind_literal_paths(stmt, bound, collections.ChainMap(bound, user_ns or {}))

        return paths, fully_known

    def simulate_upstream(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        effects: CellEffects | None = None,
        cell_code: str | None = None,
    ) -> ReexecutionPlan:
        """What must re-run, and what can be restored, before cell
        *current_cell_idx* runs on the state memory holds.

        *effects* is what the current cell writes (see
        :func:`~cash.analysis.mutation_effects.cell_effects`); None when it is
        not known, which is not the same as a cell that writes nothing.
        *cell_code* is the source being run, when it may differ from the saved
        cell.
        """
        check = CellCheck(current_cell_idx, notebook_cells, required_inputs, effects, cell_code)
        effects = effects or CellEffects()
        trace_event(
            "simulate_enter",
            cell_idx=current_cell_idx,
            reassigned=set(effects.reassigned),
            mutated=set(effects.mutated),
            required_inputs=required_inputs or set(),
            selfref=set(effects.selfref),
            method_receivers=set(effects.method_receivers),
        )
        if self._adopt_untracked_pending:
            self._track_modules_bound_before_cash_on()

        sim = self.virtual_lineage.simulate(current_cell_idx, notebook_cells)

        # Hand the simulation's view of every name to the runtime about to
        # execute this cell. A control structure records the lineages of what
        # it read, and for a name bound in the `%cash_on` cell the runtime has
        # none to record -- see TrackingState.simulated_lineage. Taken here,
        # after pass 1 and before the cell runs, so it describes the state the
        # cell is about to start from.
        self.tracking_state.simulated_lineage = dict(sim.virtual_lineage)
        self._adopt_untracked_names(sim.virtual_lineage, sim.trace)

        self._settle_loop_trust(sim)

        result = self.classifier.classify(sim, check)
        broken_vars = result.broken_vars
        trace_event("broken_after_pass2", broken=broken_vars, tainted=result.tainted_vars)
        if is_tracing():
            # Every variable the two engines disagree on, relevant or not. In a
            # plain top-to-bottom run there must be none: each one is a spurious
            # "changed" waiting for a cell that reads it (round 21).
            recorded = self.variable_lineage
            virtual_lineage = sim.virtual_lineage
            trace_event(
                "lineage_disagreement",
                cell_idx=current_cell_idx,
                vars={
                    v: [str(virtual_lineage[v])[:12], str(recorded[v])[:12]]
                    for v in sorted(virtual_lineage.keys() & recorded.keys())
                    if virtual_lineage[v] != recorded[v]
                },
            )

        self._mark_stale_value_inputs_broken(
            required_inputs,
            effects,
            broken_vars,
            notebook_cells=notebook_cells,
            current_cell_idx=current_cell_idx,
            virtual_lineage=sim.virtual_lineage,
        )
        trace_event("broken_after_guard", broken=broken_vars)

        # Read-only consumable inputs (drained queue / exhausted generator) are
        # invisible to the guard above, which only examines self-WRITTEN vars.
        # Same ``broken_vars`` set, so the planner handles both identically.
        result.consumable_broken_vars = self._mark_consumed_unrestorable_inputs_broken(
            required_inputs,
            broken_vars,
            notebook_cells=notebook_cells,
            current_cell_idx=current_cell_idx,
        )
        trace_event("broken_after_consumables", broken=broken_vars)

        # A ``# @cash: no-cache`` statement opts the whole statement out of cash,
        # so its self-modified vars must behave like an uncached Jupyter cell:
        # re-running ACCUMULATES (advances), never resets. Pass 2 still flags such
        # a var stale (its runtime lineage advanced past the simulation's), which
        # would re-execute its producer and reset it -- so drop no-cache-written
        # vars from broken_vars here (the self-write-set exclusion only
        # covered the stale-value guard, not the pass-2 lineage mismatch).
        if effects.nocache:
            removed = broken_vars & effects.nocache
            if removed:
                broken_vars -= removed
                trace_event("broken_drop_nocache", dropped=removed, broken=broken_vars)

        # Scope the writer-scheduling to files THIS cell's reconstruction reads
        #: a writer whose output no relevant consumer reads is
        # an unrelated / terminal side-effect that must never be re-fired here.
        relevant_read_paths, relevant_read_paths_known = self._compute_relevant_read_paths(
            required_inputs,
            sim.trace,
            notebook_cells,
            current_cell_idx,
        )

        # File writes have no variable edge, so an edited/new upstream writer
        # statement leaves broken_vars empty while the on-disk state a reader
        # depends on is stale. The plan must still be built so
        # the planner can schedule the writer.
        has_stale_file_writers = bool(
            self.planner.find_stale_file_writer_indices(
                sim.trace,
                virtual_lineage=sim.virtual_lineage,
                relevant_read_paths=relevant_read_paths,
                relevant_read_paths_known=relevant_read_paths_known,
            )
        )

        if broken_vars:
            # A current-cell statement that is a cache hit restores what it
            # reads as well as what it writes: a broken ``df`` that the cell's
            # first ``df[...] = f(df)`` restores needs nothing upstream.
            self.virtual_lineage.eliminate_broken_vars_via_current_cell_probe(
                broken_vars,
                notebook_cells,
                current_cell_idx,
                sim.virtual_lineage,
                sim.virtual_modules,
            )
            if not broken_vars:
                logger.debug("[UPSTREAM] All broken vars resolved by current cell cache hits — skipping upstream")

        if not broken_vars and not has_stale_file_writers:
            self._apply_phase_mutations()
            return ReexecutionPlan([], [], 0.0)

        plan = self.planner.plan(
            sim,
            result,
            notebook_cells,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
        )
        self._apply_phase_mutations()
        return plan

    def _settle_loop_trust(self, sim: SimulationResult) -> None:
        """Decide which loop outputs memory is trusted for (``vars_mutated_by_loops``
        and ``vars_derived_from_loops`` of *sim*)."""
        # A reassignment accumulator that ALSO derives from an external input via
        # a non-loop producing statement (``result = np.zeros(N)``) must not join
        # the loop-trust set: editing that input and re-running the edited cell
        # first makes upstream_has_modifications False and every current-state
        # input lineage consistent, so the trust would serve a stale value. Drop
        # such accumulators so they follow the baseline lineage-mismatch path
        # (which re-executes correctly); a constant-init accumulator keeps the
        # new trust so one-shot iterables are not re-drained.
        externally_tainted = self.virtual_lineage.loop_accumulators_with_external_init(
            sim.vars_mutated_by_loops, sim.trace, sim.loop_target_vars
        )
        if externally_tainted:
            sim.vars_mutated_by_loops = sim.vars_mutated_by_loops - externally_tainted
            logger.debug(
                "[UPSTREAM_DEBUG] Dropped externally-dependent loop accumulators from loop-trust set: %s",
                externally_tainted,
            )

        sim.vars_derived_from_loops = loop_derived_vars(sim.vars_mutated_by_loops, sim.trace)

        # A loop whose data changed underneath it (a new file, not a code
        # edit) loses the trust, and so does everything built from it.
        changed_loops = self.virtual_lineage.loops_reading_changed_data(
            sim.vars_mutated_by_loops,
            sim.trace,
            sim.loop_target_vars,
            sim.vars_derived_from_loops,
        )
        if changed_loops:
            untrusted = loop_derived_vars(changed_loops, sim.trace)
            sim.vars_mutated_by_loops = sim.vars_mutated_by_loops - untrusted
            sim.vars_derived_from_loops = sim.vars_derived_from_loops - untrusted
            trace_event("loop_trust_dropped", vars=untrusted)

        if sim.loop_target_vars:
            logger.debug("[UPSTREAM_DEBUG] Loop target variables (iteration vars): %s", sim.loop_target_vars)


#: Builtins a ``%cash_on``-cell binding may call and still count as reading
#: nothing. Classes are judged separately (``_binds_without_reading``).
_PURE_BUILTINS = frozenset(
    {
        "len",
        "range",
        "min",
        "max",
        "abs",
        "round",
        "sorted",
        "sum",
        "zip",
        "enumerate",
        "reversed",
        "repr",
        "hash",
        "isinstance",
        "getattr",
    }
)

#: ``os.path`` functions that only compute a string -- ``DATA =
#: os.path.join(ROOT, "data")`` is as common in a setup cell as ``Path(...)``.
_PURE_PATH_FUNCS = frozenset(
    {
        "join",
        "dirname",
        "basename",
        "split",
        "splitext",
        "normpath",
        "abspath",
        "expanduser",
    }
)


def _binds_without_reading(code: str, user_ns: dict) -> bool:
    """True when *code* provably reads nothing outside the notebook.

    Every call must be a standard-library or builtin CLASS (``Path(...)``,
    ``datetime.date(...)``) or one of ``_PURE_BUILTINS``. A method call
    (``DATA.read_text()``), a third-party or user function, or anything that
    does not resolve fails -- deliberately: one ``load(DATA)`` wrongly adopted
    pins a stale value, while one wrongly refused only costs the first
    reader's cache.
    """

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = resolve_callee(node.func, user_ns, builtins_fallback=True)
        if callee is None:
            return False
        if isinstance(callee, type):
            root = (getattr(callee, "__module__", "") or "").split(".")[0]
            if root == "builtins" or root in stdlib:
                continue
            return False
        name = getattr(callee, "__name__", None)
        if name in _PURE_BUILTINS and getattr(builtins, name, None) is callee:
            continue
        if name in _PURE_PATH_FUNCS and getattr(os.path, name, None) is callee:
            continue
        return False
    return True
