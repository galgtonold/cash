"""Value objects passed between NotebookSimulator phases.

VirtualLineage emits SimulationResult.
MismatchClassifier consumes SimulationResult, emits ClassificationResult.
ReexecutionPlanner consumes both, emits ReexecutionPlan.
NotebookSimulator (orchestrator) applies RestoreOp to TrackingState.

These objects are the *interface* of each phase. Adding cross-phase data
means adding a field here, not threading a parameter through call chains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


class SimulationCacheEntry(NamedTuple):
    """Per-cell snapshot stored in the incremental simulation cache.

    Using a NamedTuple instead of a raw tuple makes the fields
    self-documenting and allows attribute access instead of magic indices.
    """

    cell_code_hash: str
    """SHA-256 hex digest of the cell's source code."""

    virtual_lineage: dict[str, str]
    """Snapshot of ``virtual_lineage`` after simulating this cell."""

    virtual_modules: set[str]
    """Snapshot of known module names after simulating this cell."""

    trace_segment: list[Any]
    """Simulation trace entries produced by this cell."""

    vars_mutated_by_loops: set[str]
    """Variables whose lineage was affected by loop mutations up to this cell."""

    vars_with_stale_files: set[str]
    """Variables depending on files whose mtime has changed."""

    cell_file_deps: dict[str, float]
    """``{filepath: mtime}`` for files read during this cell's simulation."""


class TraceEntry(NamedTuple):
    """A single entry in the simulation trace."""

    stmt_code: str
    outputs: set
    inputs: set
    input_hashes: list
    produced_lineages: dict
    files_stale: bool


class IncrementalStartResult(NamedTuple):
    """Result of VirtualLineage._find_incremental_start.

    Replaces a raw 9-element tuple with named fields so call sites are
    self-documenting.
    """

    first_changed_cell: int
    """Index of the first upstream cell that needs re-simulation."""

    had_prior_cache: bool
    """Whether a simulation cache existed before this call."""

    cache_had_hash_mismatch: bool
    """Whether any cached cell hash differed from the current notebook."""

    simulation_trace: list[Any]
    """Restored simulation trace entries from cached cells."""

    virtual_lineage: dict[str, str]
    """Restored variable lineage mapping from the cache boundary."""

    virtual_modules: set[str]
    """Restored set of known module names from the cache boundary."""

    new_cache_entries: list[Any]
    """Cache entries carried forward from unchanged cells."""

    vars_mutated_by_loops: set[str]
    """Variables whose lineage was affected by loop mutations."""

    vars_with_stale_files: set[str]
    """Variables depending on files whose mtime has changed."""


@dataclass
class RestoreOp:
    """A mutation NotebookSimulator applies to TrackingState after planning.

    Concentrating restores into explicit ops keeps the three phases pure.
    """

    var_name: str
    lineage_hash: str
    cache_key: str | None = None
    restored_code: str | None = None


@dataclass
class CacheRestore:
    """A var-restore mutation buffered by VirtualLineage during simulation.

    Captures one restore event. The orchestrator applies it to TrackingState
    after the phase completes.
    """
    var_name: str
    lineage_hash: str | None
    code: str | None = None
    code_hash: str | None = None
    input_lineages: dict[str, str] | None = None
    file_deps: set[str] | None = None


@dataclass
class LineageReset:
    """A LineageStore.reset_to mutation buffered by MismatchClassifier.

    Used when a current-cell output's lineage advances downstream after
    a mismatch resolution.
    """
    var_name: str
    lineage_hash: str


class RestoreCollector:
    """Buffer of TrackingState mutations from a phase, drained by the orchestrator.

    Phases call ``record_restore()`` / ``record_lineage_reset()`` instead of
    writing directly. After each phase, ``NotebookSimulator`` calls
    ``drain()`` to flush ops to TrackingState. This concentrates all
    phase-emitted mutations in one auditable site.
    """

    def __init__(self) -> None:
        self._restores: list[CacheRestore] = []
        self._resets: list[LineageReset] = []

    def record_restore(
        self,
        var_name: str,
        lineage_hash: str | None,
        *,
        code: str | None = None,
        code_hash: str | None = None,
        input_lineages: dict[str, str] | None = None,
        file_deps: set[str] | None = None,
    ) -> None:
        self._restores.append(CacheRestore(
            var_name=var_name, lineage_hash=lineage_hash,
            code=code, code_hash=code_hash,
            input_lineages=input_lineages, file_deps=file_deps,
        ))

    def record_lineage_reset(self, var_name: str, lineage_hash: str) -> None:
        self._resets.append(LineageReset(var_name=var_name, lineage_hash=lineage_hash))

    def drain(self) -> tuple[list[CacheRestore], list[LineageReset]]:
        """Return all buffered ops and clear the collector."""
        restores, resets = self._restores, self._resets
        self._restores, self._resets = [], []
        return restores, resets

    def __len__(self) -> int:
        return len(self._restores) + len(self._resets)


def apply_collected_mutations(collector: "RestoreCollector", state: Any) -> None:
    """Apply buffered phase mutations to *state* (a TrackingState).

    Single auditable site for phase-emitted writes. Called by
    NotebookSimulator after each phase, and by phase methods themselves
    at well-defined boundaries where mid-phase visibility is required
    (e.g. so subsequent statements see import lineages in cache-key
    computation).

    Preserves legacy ``executed_cell_hashes`` normalisation: bare-str
    entries are promoted to a set so subsequent ``.add`` calls succeed.
    """
    restores, resets = collector.drain()
    for op in restores:
        if op.lineage_hash is not None:
            state.variable_lineage[op.var_name] = op.lineage_hash
        if op.code is not None:
            state.executed_cell_codes[op.var_name] = op.code
        if op.code_hash is not None:
            existing = state.executed_cell_hashes.get(op.var_name)
            if existing is None:
                state.executed_cell_hashes[op.var_name] = {op.code_hash}
            elif isinstance(existing, str):
                state.executed_cell_hashes[op.var_name] = {existing, op.code_hash}
            else:
                existing.add(op.code_hash)
        if op.input_lineages is not None:
            state.executed_input_lineages[op.var_name] = dict(op.input_lineages)
        if op.file_deps:
            state.executed_file_deps.setdefault(op.var_name, set()).update(op.file_deps)
    for op in resets:
        state.lineage.reset_to(op.var_name, op.lineage_hash)


@dataclass
class SimulationResult:
    """Output of VirtualLineage.simulate (introduced in Task 2 of the simulator split).

    VirtualLineage gathers all data the downstream phases need from the
    forward pass and packs it here. Notably, ``loop_var_input_lineages``
    is computed by the planner between Pass 1 and Pass 2 in current code
    and is surfaced here for Task 3's MismatchClassifier to consume.
    """

    virtual_lineage: dict[str, str]
    virtual_modules: set[str]
    simulation_trace: list[TraceEntry]
    new_cache_entries: list[Any]
    vars_mutated_by_loops: set[str]
    vars_with_stale_files: set[str]
    vars_derived_from_loops: set[str]
    loop_target_vars: set[str]
    loop_var_input_lineages: dict[str, dict[str, str]]
    loop_derived_trust_overridden: bool
    upstream_has_modifications: bool
    first_changed_cell: int
    stmt_lookup_times: dict[str, float]
    restores_during_simulation: list[RestoreOp] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Output of MismatchClassifier.classify."""

    broken_vars: set[str]
    tainted_vars: set[str]
    directly_mismatched: set[str]
    simulation_trace_codes: set[str]
    additional_restores: list[RestoreOp] = field(default_factory=list)


@dataclass
class ReexecutionPlan:
    """Output of ReexecutionPlanner.plan."""

    stmts_to_run: list[str]
    restored_info: list[dict[str, Any]]
    total_restore_time: float
