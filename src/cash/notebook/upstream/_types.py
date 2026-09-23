"""Value objects passed between the upstream simulator's phases.

A check is a :class:`CellCheck`. :class:`VirtualLineage` simulates the cells
above it into a :class:`SimulationResult`; :class:`MismatchClassifier` reads
that and returns a :class:`ClassificationResult`; :class:`ReexecutionPlanner`
reads both and returns the :class:`ReexecutionPlan`. Data one phase hands the
next is a field here, not another parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from ...analysis.mutation_effects import CellEffects


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

    trace_segment: list[TraceEntry]
    """Simulation trace entries produced by this cell."""

    vars_mutated_by_loops: set[str]
    """Variables whose lineage was affected by loop mutations up to this cell."""

    vars_with_stale_files: set[str]
    """Variables depending on files whose mtime has changed."""

    cell_file_deps: dict[str, float]
    """``{filepath: mtime}`` for files read during this cell's simulation."""

    cell_environment: str = ""
    """What the environment reads written in the cell returned when it was
    simulated (``statement_environment_component``): empty when it reads none."""


@dataclass
class SimulationCache:
    """What one simulation leaves behind for the next one to start from."""

    entries: list[SimulationCacheEntry] = field(default_factory=list)
    """One snapshot per cell above the cell last checked, in notebook order."""

    cell_hashes: dict[int, str] = field(default_factory=dict)
    """Source hash of every cell seen so far, by index. Outlives *entries*,
    which are cut at the checked cell, so an edit below it is still noticed."""

    last_index_by_cell_id: dict[str, int] = field(default_factory=dict)
    """Where each cell id last sat in the notebook."""

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, idx: int) -> SimulationCacheEntry | None:
        return self.entries[idx] if 0 <= idx < len(self.entries) else None

    def reset(self) -> None:
        """Forget the snapshots and hashes; cell positions stay."""
        self.entries.clear()
        self.cell_hashes.clear()


@dataclass(slots=True)
class TraceEntry:
    """One statement of the simulation trace, in notebook order."""

    stmt_code: str
    """The statement as the runtime keys it."""

    outputs: set[str]
    """Names the statement binds or changes."""

    inputs: set[str]
    """Names the statement reads."""

    input_hashes: dict[str, str]
    """The simulated lineage of each input at this statement."""

    produced_lineages: dict[str, str]
    """The simulated lineage each output leaves with."""

    files_stale: bool
    """Whether a file the statement read changed since it last ran."""

    cell: int = -1
    """The notebook cell the statement belongs to; -1 when unknown."""


class IncrementalStartResult(NamedTuple):
    """Where the forward simulation starts: the first cell it must redo."""

    first_changed_cell: int
    """Index of the first upstream cell that needs re-simulation."""

    had_prior_cache: bool
    """Whether a simulation cache existed before this call."""

    cache_had_hash_mismatch: bool
    """Whether any cached cell hash differed from the current notebook."""

    new_cache_entries: list[SimulationCacheEntry]
    """Cache entries carried forward from unchanged cells."""

    simulation: SimulationResult
    """The simulation state at the end of the cells taken from the cache."""


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
    value: Any = None
    """When provided, apply uses lineage.record(...) instead of a direct dict
    write so ``_cash_lineage_hash`` is attached to the live object."""


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
        value: Any = None,
    ) -> None:
        self._restores.append(
            CacheRestore(
                var_name=var_name,
                lineage_hash=lineage_hash,
                code=code,
                code_hash=code_hash,
                input_lineages=input_lineages,
                file_deps=file_deps,
                value=value,
            )
        )

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
    """
    restores, resets = collector.drain()
    for op in restores:
        if op.lineage_hash is not None:
            if op.value is not None:
                state.lineage.record(op.var_name, op.lineage_hash, value=op.value)
            else:
                state.lineage.record(op.var_name, op.lineage_hash)
        if op.code is not None:
            state.executed_cell_codes[op.var_name] = op.code
        if op.code_hash is not None:
            existing = state.executed_cell_hashes.get(op.var_name)
            if existing is None:
                state.executed_cell_hashes[op.var_name] = {op.code_hash}
            else:
                existing.add(op.code_hash)
        if op.input_lineages is not None:
            state.executed_input_lineages[op.var_name] = dict(op.input_lineages)
        if op.file_deps:
            state.executed_file_deps.setdefault(op.var_name, set()).update(op.file_deps)
    for op in resets:
        state.lineage.reset_to(op.var_name, op.lineage_hash)


@dataclass
class CellCheck:
    """One upstream check: the cell about to run and the notebook above it."""

    current_cell_idx: int
    """Where the cell sits in *notebook_cells*."""

    notebook_cells: list[str]
    """Every cell's source, as saved."""

    required_inputs: set[str] | None = None
    """Names the cell reads that it does not bind first."""

    effects: CellEffects | None = None
    """What the cell writes; None when not known, which is not the same as
    a cell that writes nothing."""

    cell_code: str | None = None
    """The source actually being run, which may differ from the saved cell."""

    @property
    def current_cell_outputs(self) -> set[str] | None:
        return set(self.effects.outputs) if self.effects is not None else None


@dataclass
class SimulationResult:
    """The forward simulation's state after the cells it simulated.

    Pass 1 fills it cell by cell; the classifier and the planner read it.
    """

    virtual_lineage: dict[str, str] = field(default_factory=dict)
    """The lineage each name would have after a from-the-top run."""

    virtual_modules: set[str] = field(default_factory=set)
    """Names the simulation saw bound to a module."""

    trace: list[TraceEntry] = field(default_factory=list)
    """Every simulated statement that binds or writes something, in order."""

    vars_mutated_by_loops: set[str] = field(default_factory=set)
    """Names a control structure's body changes in place."""

    vars_with_stale_files: set[str] = field(default_factory=set)
    """Names built from a file that changed since it was read."""

    loop_target_vars: set[str] = field(default_factory=set)
    """Loop iteration variables (``item`` in ``for item in data``)."""

    stmt_lookup_times: dict[str, float] = field(default_factory=dict)
    """Seconds each statement's cache lookup took, by statement."""

    upstream_has_modifications: bool = False
    """A cell above was edited since the previous simulation (not merely
    new to it, and not merely reading a changed file)."""

    vars_derived_from_loops: set[str] = field(default_factory=set)
    """Names built, directly or not, from a trusted loop's output."""


@dataclass
class ClassificationResult:
    """Which names the in-memory state cannot be trusted for."""

    broken_vars: set[str]
    """Names whose live value is not what a from-the-top run would give."""

    tainted_vars: set[str]
    """Names whose lineage matches but whose producer was edited unsaved."""

    trace_codes: set[str]
    """Every statement of the trace, and of its control bodies, as text."""

    loop_derived_trust_overridden: bool = False
    """An unsaved edit to a loop's code withdraws the trust loop output gets."""

    consumable_broken_vars: set[str] = field(default_factory=set)
    """Broken names that are drained iterators or queues."""


class ReexecutionPlan(NamedTuple):
    """What the upstream repair runs and what it restored."""

    statements: list[str]
    """Statements to re-execute, in notebook order."""

    restored_info: list[dict[str, Any]]
    """One metric per statement restored from the cache or skipped."""

    restore_time: float
    """Seconds spent restoring."""
