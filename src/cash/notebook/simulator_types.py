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
class SimulationResult:
    """Output of VirtualLineage.simulate."""

    virtual_lineage: dict[str, str]
    virtual_modules: set[str]
    simulation_trace: list[TraceEntry]
    new_cache_entries: set[str]
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
