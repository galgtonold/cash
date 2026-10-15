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
dict references, so there is nothing to re-wire when the state is replaced.
"""

from __future__ import annotations

import ast
import logging
import types
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ...analysis.code_analyzer import CodeAnalyzer
from ...tracking.module_symbols import static_attribute_reads
from ...value_hash import budgeted_hash, compute_hash, costly_after
from ..cache_key import statement_source_hash
from ..lineage_formula import (
    call_time_reads,
    callable_source_component,
    changed_module_environment,
    function_identity,
    input_lineage,
    lineage_hidden_reads,
    module_source_component,
    no_cache_value_component,
    no_cache_value_digest,
    note_function_identity,
    output_lineage,
    recorded_reads_lineage_component,
    setting_output_lineage,
    statement_environment_component,
)
from ..restored_var import hashed_by_lineage
from .derivation_edges import (
    assigned_names,
    bump_derived_lineages,
    clear_edges_for,
    detect_derivation_edges,
)
from .file_deps import compute_file_hash_component

if TYPE_CHECKING:
    from ...tracking.function_tracker import FunctionTracker
    from .._protocols import ShellProtocol
    from ..tracking_state import TrackingState
    from .file_deps import StatementFileDeps

logger = logging.getLogger(__name__)


def _record_file_reads(
    tracking_state: "TrackingState",
    cache_key: str,
    accessed_files: set[str] | None,
    accessed_remote: set[str] | None,
    probes: dict[str, str] | None = None,
) -> str:
    """Record the files and remote objects a statement read, and the paths it
    only checked for (*probes*), under its *cache_key*; returns their lineage
    component (empty when it read none)."""
    file_hash_component = ""
    if accessed_files or accessed_remote or probes:
        file_hash_component = compute_file_hash_component(
            accessed_files or set(),
            accessed_remote,
            probes,
        )
    if cache_key:
        tracking_state.statement_file_reads[cache_key] = (
            frozenset(accessed_files or ()),
            frozenset(accessed_remote or ()),
        )
        if probes:
            tracking_state.statement_probes[cache_key] = dict(probes)
        else:
            tracking_state.statement_probes.pop(cache_key, None)
    return file_hash_component


def clear_rebound_edges(
    tracking_state: "TrackingState",
    outputs: set[str],
    inputs: set[str],
    user_ns: dict[str, Any],
    tree: ast.AST | None = None,
) -> None:
    """Drop the derivation-alias edges of the outputs a statement rebinds.

    A fresh rebind (``g = ...`` — output not also read as an input) drops the
    var's stale edges before they are re-detected; an in-place mutation
    (``df.iloc[...] = ...`` — output IS an input) keeps them. All outputs are
    cleared BEFORE any is detected: clearing ``fig`` also drops edges INTO
    it, so ``fig, ax = plt.subplots()`` would lose ``ax -> fig`` whenever the
    set happened to yield ``ax`` first.
    """
    # ``x = x + [5]`` reads ``x`` but binds a new object all the same: the
    # edges of the old one are dropped too (`assigned_names`); its share
    # check links the new one again where it is shared.
    rebound = assigned_names(tree) & outputs
    for var_name in outputs:
        if var_name in user_ns and (var_name not in inputs or var_name in rebound):
            clear_edges_for(tracking_state.derivation_edges, var_name)


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
    ) -> None:
        self.shell = shell
        self.function_tracker = function_tracker
        self._file_deps = file_deps
        self.compute_hash = compute_hash

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
        no_cache: bool = False,
        replay_bumps: bool = True,
        lineage_reads: frozenset[str] = frozenset(),
        probes: dict[str, str] | None = None,
        settled: Mapping[int, tuple[Any, str]] | None = None,
    ) -> dict[str, Any]:
        """Capture output variables, compute their lineage, and update tracking state.

        *settled* holds content digests (`compute_hash`) the statement's
        check after it ran already took (`DigestHandoff.take_settled`), with
        no user code run since: an output among them is not read again.

        *lineage_reads* are outputs changed without being read as inputs
        (the current figure a ``plt.plot`` draws on): their lineage before
        joins every output's, as an input's would.

        *replay_bumps* False leaves the derivation bumps to the caller
        (:meth:`replay_derivation_bumps`), which knows by then which
        variables the statement's entry moves on itself.

        *no_cache* marks a ``# @cash:no-cache`` statement: each output's
        lineage then also carries a digest of its value
        (:func:`no_cache_value_digest`).

        *accessed_remote* holds object-storage URLs the statement read. They
        join the same lineage component as local files, contributing the store's
        validator instead of a stat — see
        :func:`~cash.notebook.statement.file_deps.compute_file_hash_component`.
        *probes* are the paths it only checked for, with their answers
        (:func:`~cash.tracking.file_dep_snapshot.probe_answers`).

        Returns ``{var_name: value}`` for the captured outputs.
        """
        captured_vars: dict[str, Any] = {}
        user_ns = self.shell.user_ns
        file_hash_component = _record_file_reads(tracking_state, cache_key, accessed_files, accessed_remote, probes)

        # A draw READS its module's hidden RNG variable: fold that
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
        lineage_inputs = inputs | lineage_hidden_reads(code) | lineage_reads
        # The environment and the data of the user's modules it read, as its
        # key folds them: a new value is a new lineage, so what is built on an
        # output misses too.
        environment = recorded_reads_lineage_component(tracking_state, cache_key, code, user_ns, outputs)
        value_digests: dict[str, str] = {}

        clear_rebound_edges(tracking_state, outputs, inputs | lineage_reads, user_ns, tree)

        # The inputs as the statement read them, taken once before any output
        # is recorded. Read inside the loop, an output that is also an input
        # (``M = enc.fit_transform(data)`` writes ``enc`` too) would hand its
        # NEW lineage to the outputs recorded after it, so each output's lineage
        # hung on set iteration order -- which string hashing randomises per
        # process. The simulation reads its inputs once, before the statement.
        input_lineage_hashes, input_lineage_map = self._build_input_lineages(
            tracking_state, lineage_inputs, user_ns, code
        )
        body_only = call_time_reads(code, user_ns)

        # All outputs of one statement share its formula lineage (``train,
        # test = ...``, or ``M = enc.fit_transform(X)`` writing ``enc`` too):
        # it names no single value, so none of them may be cut off to an
        # earlier value's lineage on its own, nor lend it to a later value.
        # Only all of them together, to the lineage they all shared when an
        # earlier statement bound the same names to the same values
        # (`ValueCutoff.lineage_for_outputs`).
        bound = {name: user_ns[name] for name in outputs if name in user_ns}
        rebuilt = not no_cache and not any(user_ns.get(name) is value for name in inputs for value in bound.values())
        cut_off = rebuilt and len(bound) == 1

        func_component = callable_source_component(self.function_tracker, inputs, user_ns)
        # The content digests the cutoff takes of the outputs, kept for the
        # session hash `_record_output` takes of the same values below: a
        # networkx graph has no cheaper digest and no sound "unchanged"
        # signal, so it is read once here, not twice.
        taken: dict[int, tuple[Any, str]] = {}
        if settled and (self.compute_hash or compute_hash) is compute_hash:
            taken.update(settled)
        content_hash = self._hash_once(taken)
        lineages: dict[str, str] = {}
        for var_name, value in bound.items():
            captured_vars[var_name] = value

            tracking_state.executed_input_lineages[var_name] = dict(input_lineage_map)

            # The formula and its ingredients are shared with the simulator
            # (lineage_formula), which must arrive at the same hash.
            components = (
                file_hash_component,
                func_component,
                self._compute_module_lineage_component(tracking_state, value, var_name, code, tree),
                changed_module_environment(var_name, value, code, environment),
                self._no_cache_value(value_digests, var_name, value) if no_cache else "",
            )
            # A module set by a setting that is the same when run again
            # (``pd.set_option('display.width', 120)``) keeps one run's
            # lineage however often it ran.
            output_lineage_hash = setting_output_lineage(
                var_name,
                value,
                code,
                input_lineage_map,
                lambda others: output_lineage(source_hash, others, *components),
            ) or output_lineage(source_hash, input_lineage_hashes, *components)
            # A new object only: ``t = raw`` binds the input itself, which
            # keeps the lineage the formula gives it.
            if cut_off:
                output_lineage_hash = tracking_state.value_cutoff.lineage_for(value, output_lineage_hash, content_hash)
            lineages[var_name] = output_lineage_hash

        if rebuilt and len(bound) > 1:
            lineages = tracking_state.value_cutoff.lineage_for_outputs(bound, lineages, content_hash)

        for var_name, value in bound.items():
            if body_only:
                # What a key reads for the function this ``def`` bound: its
                # code and what the ``def`` evaluated, not the globals only
                # its body reads (`lineage_formula.function_identity`).
                note_function_identity(
                    lineages[var_name],
                    code,
                    source_hash,
                    [function_identity(h) for name, h in input_lineage_map.items() if name not in body_only],
                    func_component,
                    user_ns,
                )
            self._record_output(
                tracking_state,
                var_name,
                value,
                lineages[var_name],
                code=code,
                source_hash=source_hash,
                cache_key=cache_key,
                inputs=inputs,
                accessed_files=accessed_files,
                taken=taken,
            )

        if cache_key:
            if value_digests:
                tracking_state.no_cache_values[cache_key] = value_digests
            else:
                tracking_state.no_cache_values.pop(cache_key, None)

        if replay_bumps:
            self.replay_derivation_bumps(tracking_state, outputs, inputs)

        return captured_vars

    def replay_derivation_bumps(
        self,
        tracking_state: "TrackingState",
        outputs: set[str],
        inputs: set[str],
        skip: set[str] | frozenset[str] = frozenset(),
    ) -> set[str]:
        """Move on the variables a statement with *outputs* changed through
        another name (`bump_derived_lineages`); the names moved.

        A mutation of a base/frame bumps its live-alias derivatives, a change
        to a list bumps every other variable bound to it or holding it. The
        skip-inputs rule keeps view *creation* from invalidating its base.
        The live value is attached so the bumped var's tag stays paired with
        its dict entry, and its session content hash follows the change: the
        value changed through the alias is the value a later cell starts
        from, not this cell's own prior output (`StaleValueGuard`).
        """
        user_ns = self.shell.user_ns

        def record(target: str, lineage_hash: str) -> None:
            value = user_ns.get(target)
            tracking_state.lineage.record(target, lineage_hash, value=value)
            if target in tracking_state.current_session_hashes:
                self._update_variable_content_hashes(tracking_state, target, value, lineage_hash)

        return bump_derived_lineages(
            tracking_state.derivation_edges,
            tracking_state.variable_lineage,
            outputs,
            inputs,
            record=record,
            present=lambda t: t in user_ns,
            skip=skip,
        )

    def _record_output(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        value: Any,
        output_lineage_hash: str,
        *,
        code: str,
        source_hash: str,
        cache_key: str,
        inputs: set[str],
        accessed_files: set[str] | None,
        taken: Mapping[int, tuple[Any, str]] | None = None,
    ) -> None:
        """Record everything the tracking state keeps about the output
        *var_name*, once its lineage is known. *taken* holds the content
        digests already taken of this statement's outputs (`_hash_once`)."""
        user_ns = self.shell.user_ns
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
        self._update_variable_content_hashes(tracking_state, var_name, value, output_lineage_hash, taken)
        cut = costly_after(value) if var_name in inputs else None
        if cut is not None:
            # Changed in place, or not: its check ran out of its budget, so
            # it counts as changed (`budgeted_hash`), and what reads it
            # misses for that. Said where it does (`input_change_reason`).
            tracking_state.budget_cut_lineages[var_name] = (output_lineage_hash, cut)
        else:
            tracking_state.budget_cut_lineages.pop(var_name, None)

        tracking_state.variable_sources[var_name] = cache_key

        self._file_deps.update_for_var(
            tracking_state, var_name, accessed_files, inputs, value, rebind=var_name not in inputs
        )

    def _hash_once(self, taken: dict[int, tuple[Any, str]]) -> Callable[[Any], str]:
        """The content hash (`compute_hash`, or the one this builder was
        given), noting each digest in *taken* by the value's id, with the
        value held so the id stays its own."""
        hash_fn = self.compute_hash or compute_hash

        def content_hash(value: Any) -> str:
            found = taken.get(id(value))
            if found is not None and found[0] is value:
                return found[1]
            digest = hash_fn(value)
            taken[id(value)] = (value, digest)
            return digest

        return content_hash

    @staticmethod
    def _no_cache_value(digests: dict[str, str], var_name: str, value: Any) -> str:
        """Record the digest of a no-cache output's value; return its component."""
        digests[var_name] = no_cache_value_digest(value)
        return no_cache_value_component(digests, var_name)

    def build_output_lineages(self, tracking_state: "TrackingState", outputs: set[str]) -> dict[str, str]:
        """Collect ``{var: lineage_hash}`` for all outputs that have a lineage."""
        return {v: tracking_state.variable_lineage[v] for v in outputs if v in tracking_state.variable_lineage}

    def build_output_digests(self, tracking_state: "TrackingState", outputs: set[str]) -> dict[str, str]:
        """Collect ``{var: content digest}`` for the outputs whose value the
        same-value cutoff saw (``ValueCutoff.digest_of`` their lineage), for
        the entry to keep (``StatementCacheMetadata.output_digests``)."""
        cutoff = tracking_state.value_cutoff
        lineages = tracking_state.variable_lineage
        found = {}
        for var in outputs:
            lineage = lineages.get(var)
            digest = cutoff.digest_of(lineage) if lineage is not None else None
            if digest is not None:
                found[var] = digest
        return found

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
        """Each input's lineage (``lineage_formula.input_lineage``), as a list
        for the output lineage and as a map for the upstream check."""
        input_lineage_hashes: list[str] = []
        input_lineage_map: dict[str, str] = {}
        for input_var in inputs:
            lineage = input_lineage(
                input_var,
                user_ns,
                (tracking_state.variable_lineage,),
                compute_hash=self.compute_hash,
                function_tracker=self.function_tracker,
                code=code,
            )
            if lineage:
                input_lineage_hashes.append(lineage)
                input_lineage_map[input_var] = lineage
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
        """Update granular module attribute dependency tracking for *var_name*.

        A module input gets an entry only when the statement does nothing with
        it but read plain attributes; any other use (passing it, a dynamic
        getattr) leaves no entry, so an edit to that module invalidates the
        variable whichever symbol changed. This is the same rule the cache key
        applies (``static_attribute_reads``).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tracking_state.module_attribute_deps.pop(var_name, None)
            return
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        mod_deps: dict[str, set[str]] = {}
        for input_name in names:
            input_val = user_ns.get(input_name)
            # Tracked under its REAL name; recorded under the name read,
            # which is what the invalidator looks it up by.
            if not (
                isinstance(input_val, types.ModuleType)
                and getattr(input_val, "__name__", input_name) in self.function_tracker.tracked_modules
            ):
                continue
            attrs = static_attribute_reads(code, input_name)
            if attrs is not None:
                mod_deps[input_name] = attrs
        if mod_deps:
            tracking_state.module_attribute_deps[var_name] = mod_deps
        else:
            tracking_state.module_attribute_deps.pop(var_name, None)

    def _update_variable_content_hashes(
        self,
        tracking_state: "TrackingState",
        var_name: str,
        value: Any,
        output_lineage_hash: str,
        taken: Mapping[int, tuple[Any, str]] | None = None,
    ) -> None:
        """Update variable_hashes and current_session_hashes for *var_name*.

        A digest in *taken* is the one this statement already took of this
        very *value*, after it ran and with no user code since: it is used
        rather than read again."""
        content_hash: str | None = None
        if not hashed_by_lineage(value):
            if not self.compute_hash:
                return
            found = taken.get(id(value)) if taken else None
            if found is not None and found[0] is value:
                content_hash = found[1]
            else:
                try:
                    # Within a budget: a value that takes longer to read is
                    # remembered as costly and has its lineage as its session
                    # hash, like a frame (`budgeted_hash`).
                    content_hash = budgeted_hash(value, self.compute_hash)
                except Exception as e:  # noqa: BLE001 - a user value that cannot be hashed
                    logger.debug("[CACHE DEBUG] Could not hash captured variable '%s': %s", var_name, e)
                    return
        # For large or costly objects the lineage stands for the content: no
        # content hash equals it, so a check comparing one reads "changed".
        session_hash = content_hash if content_hash is not None else output_lineage_hash
        tracking_state.variable_hashes.setdefault(var_name, set()).add(session_hash)
        tracking_state.current_session_hashes[var_name] = session_hash

    def lineage_if_rerun(self, tracking_state: "TrackingState", var_name: str, value: Any, code: str) -> str:
        """The lineage *var_name* would get if *code* ran again now: the same
        formula and ingredients as :meth:`capture_and_track_variables`, without
        running anything. For an import after its module was reloaded, so the
        name gets what a fresh kernel's import gives it.
        """

        user_ns = self.shell.user_ns
        inputs, _outputs = CodeAnalyzer.analyze_code_block(code, user_ns=user_ns)
        input_lineage_hashes, _map = self._build_input_lineages(
            tracking_state, inputs | lineage_hidden_reads(code), user_ns, code
        )
        return output_lineage(
            statement_source_hash(code),
            input_lineage_hashes,
            "",
            callable_source_component(self.function_tracker, inputs, user_ns),
            module_source_component(self.function_tracker, value, var_name, code),
            statement_environment_component(code, user_ns),
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
