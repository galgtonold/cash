"""Lineage computation for statement outputs.

Owns the operation "produce the lineage hash for each output variable
of a statement, plus all the bookkeeping that goes with it" — input
lineage assembly, function-source hashing, module-source hashing
(both ``import X`` and ``from X import Y`` cases), per-variable
content hashing, and granular module-attribute dependency tracking.

Single public entry: :meth:`StatementLineageBuilder.capture_and_track_variables`,
plus :meth:`build_output_lineages` for the cache-write side.

**Anti-god-class rule (load-bearing):** this module computes lineage
hashes and writes them through :class:`LineageStore` and the
tracking-state dicts.  It does **not** decide caching policy (skip
checks, freshness), execute statements, or replay output.  Those are
::class:`CacheFreshnessChecker`'s, ``StatementProcessor``'s, and
:class:`StatementRestorer`'s jobs, respectively.

All :class:`TrackingState` access happens through the ``tracking_state``
method parameter on the public entries.  The builder holds no aliased
dict references and has no ``set_tracking_state`` re-wiring step.
"""

from __future__ import annotations

import ast
import logging
import pickle
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...analysis.code_analyzer import CodeAnalyzer
from ...tracking.randomness import hidden_lineage_reads
from ..cache_key import is_cash_instrumentation, is_module_like, statement_source_hash
from ..lineage_formula import (
    callable_source_component,
    module_read_lineage,
    module_source_component,
    output_lineage,
)
from .derivation_edges import (
    bump_derived_lineages,
    clear_edges_for,
    detect_derivation_edges,
)
from .file_deps import compute_file_hash_component

if TYPE_CHECKING:
    from ...tracking.function_tracker import FunctionTracker
    from .._protocols import ShellProtocol, TrackingState
    from .file_deps import StatementFileDeps

logger = logging.getLogger(__name__)


class StatementLineageBuilder:
    """Compute + record lineage hashes for statement outputs.

    Holds permanent dependencies (shell, function tracker, file-deps
    sibling, optional content hasher) but no aliased dict references.
    All :class:`TrackingState` access flows through the
    ``tracking_state`` method parameter on the public entries.
    """

    def __init__(
        self,
        shell: "ShellProtocol",
        function_tracker: "FunctionTracker",
        file_deps: "StatementFileDeps",
        compute_hash: Callable[[Any], str] | None = None,
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self.function_tracker = function_tracker
        self._file_deps = file_deps
        self.compute_hash = compute_hash
        self.debug = debug

    # ------------------------------------------------------------------
    # Public entries
    # ------------------------------------------------------------------

    def capture_and_track_variables(
        self,
        tracking_state: "TrackingState",
        outputs: set[str],
        inputs: set[str],
        code: str,
        source_hash: str,
        cache_key: str,
        accessed_files: set[str] | None = None,
        tree: ast.Module | None = None,
        accessed_remote: set[str] | None = None,
    ) -> dict[str, Any]:
        """Capture output variables, compute their lineage, and update tracking state.

        *accessed_remote* holds object-storage URLs the statement read. They
        join the same lineage component as local files, contributing the store's
        validator instead of a stat — see
        :func:`~cash.notebook.statement.file_deps.compute_file_hash_component`.

        Returns ``{var_name: value}`` for the captured outputs.
        """
        captured_vars: dict[str, Any] = {}
        user_ns = self.shell.user_ns

        file_hash_component = ""
        if accessed_files or accessed_remote:
            file_hash_component = compute_file_hash_component(
                accessed_files or set(),
                accessed_remote,
            )
        if cache_key:
            tracking_state.statement_file_reads[cache_key] = (
                frozenset(accessed_files or ()),
                frozenset(accessed_remote or ()),
            )

        # A draw READS its module's hidden RNG variable (ADR-018): fold that
        # variable's lineage into every output's lineage so a re-seed upstream
        # propagates to everything cached downstream. Kept OUT of the plain
        # ``inputs`` set so cacheability / derivation-edge / function-source logic
        # never sees a phantom variable -- it only augments the lineage build.
        # NOTE: observed (AST-invisible) draws are deliberately NOT folded in
        # here, though they ARE folded into the cache key. Making a mutated
        # receiver's lineage fresh-per-run does not converge: the changed
        # lineage makes the reconstruction re-run the producing fit, which mints
        # yet another model, so a consumer never agrees with the value recorded
        # beside it -- measured, it broke even the first clean run.
        lineage_inputs = inputs | hidden_lineage_reads(code)

        # Derivation-alias edges. A fresh rebind (``g = ...`` — output not also
        # read as an input) drops the var's stale edges before we re-detect; an
        # in-place mutation (``df.iloc[...] = ...`` — output IS an input) keeps
        # them. All outputs are cleared BEFORE any is detected: clearing ``fig``
        # also drops edges INTO it, so ``fig, ax = plt.subplots()`` would lose
        # ``ax -> fig`` whenever the set happened to yield ``ax`` first.
        for var_name in outputs:
            if var_name in user_ns and var_name not in inputs:
                clear_edges_for(tracking_state.derivation_edges, var_name)

        for var_name in outputs:
            if var_name not in user_ns:
                continue
            value = user_ns[var_name]
            captured_vars[var_name] = value

            input_lineage_hashes, input_lineage_map = self._build_input_lineages(
                tracking_state, lineage_inputs, user_ns, code
            )
            tracking_state.executed_input_lineages[var_name] = input_lineage_map

            # The formula and its ingredients are shared with the simulator
            # (lineage_formula), which must arrive at the same hash.
            output_lineage_hash = output_lineage(
                source_hash,
                input_lineage_hashes,
                file_hash_component,
                callable_source_component(self.function_tracker, inputs, user_ns),
                self._compute_module_lineage_component(tracking_state, value, var_name, code, tree),
            )

            # Record via LineageStore so the dict entry and ``_cash_lineage_hash``
            # are written together and cannot drift.
            tracking_state.lineage.record(var_name, output_lineage_hash, value=value)

            detect_derivation_edges(tracking_state.derivation_edges, var_name, value, user_ns)

            self._apply_granular_module_update(tracking_state, var_name, value, output_lineage_hash)

            if var_name not in tracking_state.executed_cell_hashes:
                tracking_state.executed_cell_hashes[var_name] = set()
            tracking_state.executed_cell_hashes[var_name].add(source_hash)

            tracking_state.executed_cell_codes[var_name] = code

            self._update_module_attribute_deps(tracking_state, var_name, code, user_ns)
            self._update_variable_content_hashes(tracking_state, var_name, value, output_lineage_hash)

            tracking_state.variable_sources[var_name] = cache_key

            self._file_deps.update_for_var(
                tracking_state, var_name, accessed_files, inputs, value, rebind=var_name not in inputs
            )

        # After all outputs' lineages are recorded, replay derivation bumps:
        # a mutation of a base/frame bumps its live-alias derivatives
        # . Skip-inputs rule keeps view *creation* from
        # invalidating its base. Runtime attaches the live value so the bumped
        # var's ``_cash_lineage_hash`` stays paired with its dict entry.
        bump_derived_lineages(
            tracking_state.derivation_edges,
            tracking_state.variable_lineage,
            outputs,
            inputs,
            record=lambda t, h: tracking_state.lineage.record(t, h, value=user_ns.get(t)),
            present=lambda t: t in user_ns,
        )

        return captured_vars

    def build_output_lineages(self, tracking_state: "TrackingState", outputs: set[str]) -> dict[str, str]:
        """Collect ``{var: lineage_hash}`` for all outputs that have a lineage."""
        return {v: tracking_state.variable_lineage[v] for v in outputs if v in tracking_state.variable_lineage}

    def build_input_lineages(self, tracking_state: "TrackingState", inputs: set[str]) -> dict[str, str]:
        """Collect ``{var: lineage_hash}`` for the inputs this statement read.

        Stored on the entry so a RESTORED value can answer "has one of my
        inputs been rebuilt since?". Only an executed value could answer that
        before -- the classifier reads provenance from
        ``executed_input_lineages``, which execution alone writes -- so the
        guard against building on an older input passed vacuously for every
        restored value, which after a restart is most of them.

        Read from ``variable_lineage`` at save time, which is what the
        comparison reads at check time: an input's lineage is not changed by the
        statement that consumes it, so this is what the value was built on.
        """
        return {v: tracking_state.variable_lineage[v] for v in inputs if v in tracking_state.variable_lineage}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_input_lineages(
        self,
        tracking_state: "TrackingState",
        inputs: set[str],
        user_ns: dict,
        code: str | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """Build input lineage hashes list and map for a set of input variables.

        Mirrors the cache-key READ path (:func:`cache_key.is_module_like`): an
        untracked module contributes NOTHING. It previously fell through to
        ``compute_hash(module)``, which cannot pickle a module and so returns
        ``sha256(str(id(module)))`` -- a memory address, therefore a different
        value in every kernel. That volatile hash was folded into this
        statement's OUTPUT lineage, which is an input hash for every downstream
        statement's cache key, so after a restart the whole chain re-keyed,
        missed, recomputed, and wrote a duplicate entry.
        """
        input_lineage_hashes: list[str] = []
        input_lineage_map: dict[str, str] = {}
        for input_var in inputs:
            # A module read by plain attribute access is valued by what those
            # attributes reach, as the cache key values it -- see
            # `lineage_formula.module_read_lineage`. Recorded in the input map
            # too, which is what the upstream check compares against.
            narrowed = module_read_lineage(self.function_tracker, input_var, user_ns.get(input_var), code)
            if narrowed is not None:
                input_lineage_hashes.append(narrowed)
                input_lineage_map[input_var] = narrowed
                continue
            if input_var in tracking_state.variable_lineage:
                lineage = tracking_state.variable_lineage[input_var]
                input_lineage_hashes.append(lineage)
                input_lineage_map[input_var] = lineage
            elif input_var in user_ns:
                val = user_ns[input_var]
                if is_cash_instrumentation(val):
                    # cash's own I/O shim (e.g. the patched ``open``). Its
                    # identity is per-session, and without cash it would not be
                    # in user_ns at all.
                    continue
                if is_module_like(input_var, val, frozenset()):
                    # No tracked lineage for this module: contribute nothing,
                    # exactly as the read path does. Hashing it here would bake
                    # a memory address into a persisted key.
                    continue
                try:
                    lineage = self.compute_hash(val)
                    input_lineage_hashes.append(lineage)
                    input_lineage_map[input_var] = lineage
                except (TypeError, ValueError, AttributeError, pickle.PicklingError) as e:
                    if self.debug:
                        logger.warning("Warning: Could not hash input '%s' for lineage: %s", input_var, e)
        return input_lineage_hashes, input_lineage_map

    def _apply_granular_module_update(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        value: Any,
        output_lineage_hash: str,
    ) -> None:
        """Apply deferred granular lineage update when a tracked module is re-imported."""
        if isinstance(value, types.ModuleType) and var_name in tracking_state.granular_preserved_vars:
            preserved = tracking_state.granular_preserved_vars.pop(var_name)
            for pv in preserved:
                pv_inputs = tracking_state.executed_input_lineages.get(pv)
                if pv_inputs is not None and var_name in pv_inputs:
                    pv_inputs[var_name] = output_lineage_hash
                    if self.debug:
                        logger.debug(
                            "[GRANULAR] Deferred update: '%s'.'%s' -> %s...", pv, var_name, output_lineage_hash[:12]
                        )

    def _update_module_attribute_deps(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        code: str,
        user_ns: dict,
    ) -> None:
        """Update granular module attribute dependency tracking for *var_name*."""
        try:
            attr_accesses = self.function_tracker.extract_module_attribute_accesses(code)
            mod_deps: dict[str, set[str]] = {}
            for input_name, attrs in attr_accesses.items():
                input_val = user_ns.get(input_name)
                # Tracked under its REAL name; recorded under the name read,
                # which is what the invalidator looks it up by. Checking the
                # name read missed `import tickets_lib as tl` (see 9785293).
                if (
                    isinstance(input_val, types.ModuleType)
                    and getattr(input_val, "__name__", input_name) in self.function_tracker.tracked_modules
                ):
                    mod_deps[input_name] = attrs
            if mod_deps:
                tracking_state.module_attribute_deps[var_name] = mod_deps
            elif var_name in tracking_state.module_attribute_deps:
                del tracking_state.module_attribute_deps[var_name]
        except (AttributeError, TypeError, ValueError, SyntaxError):
            logger.debug(
                "[PROCESSOR] Module attribute tracking failed for '%s', falling back to full invalidation", var_name
            )
            tracking_state.module_attribute_deps.pop(var_name, None)

    def _update_variable_content_hashes(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        value: Any,
        output_lineage_hash: str,
    ) -> None:
        """Update variable_hashes and current_session_hashes for *var_name*."""
        type_name = type(value).__name__
        if type_name in ("DataFrame", "Series", "ndarray"):
            # For large objects, use lineage hash as proxy for content hash
            if var_name not in tracking_state.variable_hashes:
                tracking_state.variable_hashes[var_name] = set()
            tracking_state.variable_hashes[var_name].add(output_lineage_hash)
            tracking_state.current_session_hashes[var_name] = output_lineage_hash
        elif self.compute_hash:
            try:
                content_hash = self.compute_hash(value)
                if var_name not in tracking_state.variable_hashes:
                    tracking_state.variable_hashes[var_name] = set()
                tracking_state.variable_hashes[var_name].add(content_hash)
                tracking_state.current_session_hashes[var_name] = content_hash
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Could not hash captured variable '%s': %s", var_name, e)

    def lineage_if_rerun(self, tracking_state: "TrackingState", var_name: str, value: Any, code: str) -> str:
        """The lineage *var_name* would get if *code* ran again now: the same
        formula and ingredients as :meth:`capture_and_track_variables`, without
        running anything. For an import after its module was reloaded, so the
        name gets what a fresh kernel's import gives it (round 29, r29s1/r29s3:
        keyed with the reload's own hash, what the session computed after an
        edit was never restored the next morning)."""

        user_ns = self.shell.user_ns
        inputs, _outputs = CodeAnalyzer.analyze_code_block(code, user_ns=user_ns)
        input_lineage_hashes, _map = self._build_input_lineages(
            tracking_state, inputs | hidden_lineage_reads(code), user_ns, code
        )
        return output_lineage(
            statement_source_hash(code),
            input_lineage_hashes,
            "",
            callable_source_component(self.function_tracker, inputs, user_ns),
            module_source_component(self.function_tracker, value, var_name, code),
        )

    def _compute_module_lineage_component(
        self,
        tracking_state: "TrackingState",
        value: Any,
        var_name: str,
        code: str,
        tree: ast.Module | None = None,
    ) -> str:
        """The ``:mod_src:`` / ``:from_mod_src:`` fragment for *var_name*.

        See :func:`~cash.notebook.lineage_formula.module_source_component`;
        the runtime also records where a ``from`` import came from, for
        module invalidation.
        """
        component = module_source_component(
            self.function_tracker,
            value,
            var_name,
            code,
            tree,
            note_from_import=tracking_state.from_import_sources.__setitem__,
        )
        # Kept only when narrowed: it is the evidence the invalidator needs to
        # keep this name's lineage across a reload of its module. A whole-
        # module component, or none, must never vouch for it.
        if component.startswith(":from_sym_src:"):
            tracking_state.from_import_components[var_name] = component
        else:
            tracking_state.from_import_components.pop(var_name, None)
        return component
