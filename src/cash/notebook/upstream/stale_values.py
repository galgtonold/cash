"""Inputs whose live value is stale although their lineage says otherwise.

The classifier compares lineages. A value can be wrong with a lineage that
matches: a cell that changes its own input in place leaves the input's lineage
where the cell started, and a drained iterator keeps its identity. The
:class:`StaleValueGuard` finds those and adds them to the broken names.
"""

from __future__ import annotations

import ast
import logging
import weakref
from collections.abc import Callable
from typing import Any

from cash.control_markers import strip_markers

from ...analysis.ast_util import parse_cached
from ...analysis.cacheability import analyze_statement
from ...analysis.code_analyzer import CodeAnalyzer
from ...analysis.mutation_effects import CellEffects
from ...analysis.mutations import consumed_input_names
from ...lineage_tag import own_tag
from ...value_types import BUILTIN_NAMES
from .._protocols import ShellProtocol
from .._trace import trace_event
from ..consumables import consumable_state, has_diverged, is_consumable_unrestorable
from ..restored_var import hashed_by_lineage
from ..tracking_state import TrackingState
from .unsaved_edits import UnsavedEdits

__all__ = ["StaleValueGuard"]

logger = logging.getLogger(__name__)


class StaleValueGuard:
    """Marks broken the inputs whose live value no lineage shows as stale."""

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        unsaved_edits: UnsavedEdits,
        compute_hash_fn: Callable[[Any], str] | None,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.unsaved_edits = unsaved_edits
        self.compute_hash_fn = compute_hash_fn
        #: ``(cell index, name) -> (id, weak reference or None)`` of the object
        #: the cell last started from, for a value whose content is not
        #: compared (`hashed_by_lineage`). Never a strong reference: holding
        #: the value would keep it alive and count as a second holder.
        self._started_from: dict[tuple[int, str], tuple[int, Any]] = {}
        #: The cell the current check is for.
        self._cell_idx: int | None = None
        self._cells: list[str] | None = None

    def mark_stale_value_inputs_broken(
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
        self._cell_idx = current_cell_idx
        self._cells = notebook_cells
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
            if var_name in BUILTIN_NAMES and var_name not in self.tracking_state.variable_lineage:
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
            live_lineage = own_tag(live_value)
            # The value is the one the simulation of the cells above says this
            # cell starts from: current, and not this cell's own earlier output.
            # The checks below compare it with what the LAST statement writing
            # it read -- this cell's starting state only when that statement
            # is in this cell. When it is in a cell above (``df['b'] = ...``
            # there, ``df['a'] = ...`` here), a first run looked stale and the
            # value was rebuilt.
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
                    # Only a base this cell consumed: one a cell above
                    # consumed (``df['x'] += 1`` there) is passed rightly on
                    # a forward run (see `_mark_nolineage_self_write_broken`).
                    base_lineage = self.tracking_state.executed_input_lineages.get(var_name, {}).get(var_name)
                    if base_lineage is not None and self._written_above(var_name, notebook_cells, current_cell_idx):
                        base_lineage = None
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
            recorded = self.tracking_state.variable_lineage.get(var_name)
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
                prod_code = self.tracking_state.executed_cell_codes.get(var_name)
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
                    and self.unsaved_edits.is_valid_extension(
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
            base_input = self.tracking_state.executed_input_lineages.get(var_name, {}).get(var_name)
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

    def mark_consumed_unrestorable_inputs_broken(
        self,
        required_inputs: set[str] | None,
        broken_vars: set[str],
        notebook_cells: list[str] | None = None,
        current_cell_idx: int | None = None,
        cell_code: str | None = None,
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
        recorded at this cell's ENTRY on its previous run
        (:meth:`record_consumable_bases`), so a ``run_all`` (producer re-ran,
        object fresh) compares equal and this is a no-op, and a first run of
        the cell has no baseline at all. The baseline is this cell's own: a
        handle read by cell B and then by cell C reaches C further along than
        it reached B, which is a forward run, not a re-run.

        *cell_code* is the source being run (the saved cell when None); an
        edited cell has no baseline.

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
        bases = self.tracking_state.consumable_bases
        source = cell_code if cell_code is not None else cell_src
        for var_name in candidates:
            if var_name in BUILTIN_NAMES and var_name not in self.tracking_state.variable_lineage:
                continue
            live_value = self.shell.user_ns.get(var_name)
            if live_value is None:
                continue
            recorded = bases.get((current_cell_idx, var_name))
            had_baseline = recorded is not None and recorded[0] == source
            baseline = recorded[1] if had_baseline else None
            try:
                if not is_consumable_unrestorable(live_value):
                    continue
                diverged = has_diverged(live_value, baseline, had_baseline=had_baseline)
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
                    baseline,
                    consumable_state(live_value),
                )
            broken_vars.add(var_name)
            flagged.add(var_name)
            trace_event("consumable_broken", var=var_name)
        return flagged

    def record_consumable_bases(self, inputs: set[str], current_cell_idx: int, cell_code: str) -> None:
        """Record how far each consumable input of this cell has been drained.

        Called once the upstream repair has settled the namespace and before
        the cell body draws from it: the cell-ENTRY baseline that
        :meth:`mark_consumed_unrestorable_inputs_broken` compares against on
        the next run of this same cell. Equal means the producer handed the
        cell the same state as last time (``run_all``), different means the
        cell is looking at its own previous run's leftovers. Recording before
        the repair would store the drained state and lose the signal.

        Only consumable, unrestorable objects (generator / queue / file
        handle) get an entry. A name that holds anything else loses its entry,
        so a rebound variable is never compared against an unrelated
        predecessor's token.
        """
        bases = self.tracking_state.consumable_bases
        user_ns = self.shell.user_ns
        for var_name in inputs:
            key = (current_cell_idx, var_name)
            value = user_ns.get(var_name)
            token = None
            if value is not None:
                try:
                    if is_consumable_unrestorable(value):
                        token = consumable_state(value)
                except (TypeError, ValueError, AttributeError, RecursionError):
                    token = None
            if token is None:
                bases.pop(key, None)
            else:
                bases[key] = (cell_code, token)

    def _written_above(self, var_name: str, notebook_cells: list[str] | None, cell_idx: int | None) -> bool:
        """Whether the statement that last wrote *var_name* is found in a cell
        above *cell_idx* and not in it: the version it consumed is that
        cell's starting state, not this one's."""
        prod_code = self.tracking_state.executed_cell_codes.get(var_name)
        if not prod_code or notebook_cells is None or cell_idx is None or not 0 <= cell_idx < len(notebook_cells):
            return False
        norm_prod = strip_markers(prod_code).strip()
        if not norm_prod or norm_prod in notebook_cells[cell_idx]:
            return False
        return any(norm_prod in cell for cell in notebook_cells[:cell_idx])

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
        base_lineage = self.tracking_state.executed_input_lineages.get(var_name, {}).get(var_name)
        if base_lineage is not None and self._written_above(var_name, self._cells, self._cell_idx):
            # The base is what a statement of another cell consumed
            # (``c['n'] += 1`` above ``c.update()``): not this cell's entry
            # state, and on a forward run the value has moved past it
            # rightly. Re-running the producers then applied the update
            # above a second time, onto the object an alias (``c =
            # cfg['a']``) had kept.
            base_lineage = None
        if base_lineage is not None:
            current_lineage = self.tracking_state.variable_lineage.get(var_name)
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
        session_hashes = self.tracking_state.current_session_hashes
        base_content = session_hashes.get(var_name)
        if base_content is None:
            return
        if hashed_by_lineage(live_value):
            # Its session hash is a lineage, which no content hash equals, and
            # hashing it in full costs seconds. Its identity answers instead:
            # the value is this cell's own prior output when it is the very
            # object the cell started from last time. On a forward run the
            # producer above has bound a new one (or this cell never ran),
            # and rebuilding it would cut the views, aliases and containers
            # that share the object.
            if self._started_from_before(var_name, live_value):
                logger.debug(
                    "[UPSTREAM_DEBUG] in-place mutation '%s' holds its own prior output on "
                    "re-run (the cell already started from this object); marking broken.",
                    var_name,
                )
                broken_vars.add(var_name)
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

    def _started_from_before(self, var_name: str, live_value: Any) -> bool:
        """Whether the current cell started from *live_value* last time it ran.

        Fails closed: a recorded ``id`` without a weak reference (a list or
        dict takes none) that a new object happens to reuse reads as the same
        object, which costs a needless rebuild, never a doubled write.
        """
        if self._cell_idx is None:
            return True
        seen = self._started_from.get((self._cell_idx, var_name))
        if seen is None:
            return False
        seen_id, ref = seen
        if ref is not None:
            return ref() is live_value
        return seen_id == id(live_value)

    def note_cell_start(self, cell_idx: int, names: set[str]) -> None:
        """Record the objects cell *cell_idx* starts from, among *names*, once
        the upstream check has restored and re-run what it found stale.

        Only for values whose content is not compared (`hashed_by_lineage`),
        which :meth:`_started_from_before` asks about on the cell's next run.
        """
        user_ns = self.shell.user_ns
        for var_name in names:
            value = user_ns.get(var_name)
            if value is None or not hashed_by_lineage(value):
                self._started_from.pop((cell_idx, var_name), None)
                continue
            try:
                ref = weakref.ref(value)
            except TypeError:
                ref = None
            self._started_from[(cell_idx, var_name)] = (id(value), ref)

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
