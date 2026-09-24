"""Value objects passed between the upstream simulator's phases.

A check is a :class:`CellCheck`. :class:`VirtualLineage` simulates the cells
above it into a :class:`SimulationResult`; :class:`MismatchClassifier` reads
that and returns a :class:`ClassificationResult`; :class:`ReexecutionPlanner`
reads both and returns the :class:`ReexecutionPlan`. Data one phase hands the
next is a field here, not another parameter. A trace entry's input lineages
(``InputHashes``) and the helpers that read them live here too, since every
part of the package reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from cash.control_markers import strip_markers

if TYPE_CHECKING:
    from ...analysis.mutation_effects import CellEffects


def normalize_stmt(s: str) -> str:
    """Strip iteration-context comments and whitespace for code comparison."""
    return strip_markers(s).strip()


class InputHashes(dict):
    """A trace entry's input lineages, plus what only its cache key reads.

    ``input_hashes`` names the statement's own inputs, and other code copies
    it as such (a restore records it as the variable's input lineages). Two
    more things belong in the key, at the statement's position, but nowhere
    else, so they ride alongside and are read back only when the key is
    rebuilt from the trace: the globals a simulated-only callee reads (see
    ``VirtualCallable``), and the hidden RNG variables the statement reads
    (``lineage_formula.key_hidden_reads``).
    """

    __slots__ = ("callee_lineages", "hidden_lineages")

    def __init__(
        self,
        own: dict[str, str],
        callee_lineages: dict[str, str] | None = None,
        hidden_lineages: dict[str, str | None] | None = None,
    ) -> None:
        super().__init__(own)
        self.callee_lineages = callee_lineages or {}
        self.hidden_lineages = hidden_lineages or {}


def key_lineages(input_hashes: dict[str, str]) -> dict[str, str]:
    """*input_hashes* plus the key-only lineages riding on it (``InputHashes``)."""
    callee = getattr(input_hashes, "callee_lineages", None) or {}
    hidden = {k: v for k, v in (getattr(input_hashes, "hidden_lineages", None) or {}).items() if v is not None}
    return {**callee, **hidden, **input_hashes} if callee or hidden else input_hashes


def key_inputs(inputs: set[str], input_hashes: dict[str, str]) -> set[str]:
    """The names a trace entry's cache key reads: its inputs plus the hidden
    variables riding on *input_hashes*, as the runtime keys it."""
    hidden = getattr(input_hashes, "hidden_lineages", None)
    return set(inputs) | set(hidden) if hidden else set(inputs)


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
