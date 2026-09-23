"""Protocol types and shared data structures for the notebook subsystem.

These protocols define the minimal interfaces that the notebook subsystem
requires from external objects (IPython shell, cache backend, etc.).
Using protocols instead of ``Any`` provides:

- Better IDE support (autocomplete, type checking)
- Documentation of expected interfaces
- Easier testing with mock objects
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cash.notebook.lineage_store import LineageStore


@runtime_checkable
class ShellProtocol(Protocol):
    """Minimal interface for an IPython-like shell.

    The notebook subsystem only requires ``user_ns`` (the user namespace
    dict) and ``run_cell`` for upstream re-execution.
    """

    user_ns: dict[str, Any]

    def run_cell(self, raw_cell: str, *, silent: bool = False) -> Any:
        """Execute a code cell in the shell."""
        ...


@runtime_checkable
class CacheBackendProtocol(Protocol):
    """Minimal interface for a cache backend used by the notebook subsystem.

    Matches the subset of :class:`cash.backends.CacheBackend` that
    ``StatementProcessor`` and ``UpstreamChecker`` actually use.
    """

    def get(self, key: str) -> tuple[dict[str, Any] | None, Any]:
        """Retrieve a cached value and its metadata by key."""
        ...

    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None, serializer: Any = None) -> None:
        """Store a value with optional metadata and serializer."""
        ...

    def delete(self, key: str) -> None:
        """Remove a cached entry by key."""
        ...


@runtime_checkable
class CashInstanceProtocol(Protocol):
    """Minimal interface for the ``Cash`` instance used by the notebook subsystem.

    ``StatementProcessor`` and ``UpstreamChecker`` access the ``Cash``
    object only through its ``.backend`` attribute.
    """

    backend: CacheBackendProtocol


@dataclass
class TrackingState:
    """Shared mutable state for variable lineage and dependency tracking.

    ``CashMagics`` creates one instance and hands the same object to the
    ``StatementProcessor``, the ``UpstreamChecker`` and the ``Restorer``, so
    every component reads and writes the same containers. Each field below
    says who writes it and who reads it; keep that comment current when a
    writer or reader is added. Lineage itself is written only through
    :attr:`lineage` (see :mod:`cash.notebook.lineage_store`).
    """

    # Variable -> the code of the statement that last produced it.
    # W: StatementLineageBuilder, both restorers, the simulator's restore drain.
    # R: UpstreamChecker (simulated vs executed code), ModuleInvalidator, %cash_status.
    executed_cell_codes: dict[str, str] = field(default_factory=dict)

    # Variable -> sha256 of every statement that defined it this session (a set:
    # a re-run cell, a loop body and a restore can each define it).
    # W: StatementLineageBuilder, both restorers, the simulator's restore drain.
    # R: UpstreamChecker, the RNG lineage helpers.
    executed_cell_hashes: dict[str, set[str]] = field(default_factory=dict)

    # Every variable's lineage hash; read it as ``variable_lineage``.
    # W: StatementProcessor, StatementLineageBuilder, the control-structure
    # handlers, both restorers, ModuleInvalidator, the upstream simulation
    # (resynchronising with memory). R: nearly everything.
    lineage: LineageStore = field(default_factory=LineageStore)

    # Variable -> the files its value was built from, read directly or inherited.
    # W: StatementFileDeps, the control-structure helpers, the simulator's
    # restore drain, ReexecutionPlanner. R: the upstream check, freshness
    # checks, %cash_provenance.
    executed_file_deps: dict[str, set[str]] = field(default_factory=dict)

    # Code of every statement that wrote a file this session. File writes have
    # no variable edge, so this is how the simulation tells an edited or new
    # writer from one that already ran.
    # W: StatementProcessor, ControlStructureProcessor. R: ReexecutionPlanner.
    executed_write_stmt_codes: set[str] = field(default_factory=set)

    # sha256 of every whole-cell source that ran this session, to tell an
    # edited-but-not-rerun seed() cell from one that ran.
    # W: CellExecutor. R: UpstreamChecker.
    executed_cell_source_hashes: set[str] = field(default_factory=set)

    # sha256(cell source) -> the global RNG state after that cell ran, so a
    # downstream draw can be restored to its position-correct state.
    # W: CellExecutor. R: UpstreamChecker.
    rng_post_states: dict[str, Any] = field(default_factory=dict)

    # sha256(cell source) -> the RNG state just before that cell ran, with the
    # seeds in force. Re-executing a draw reproduces it only by rewinding to
    # where it started. Kept only for cells that touched an RNG.
    # W: CellExecutor. R: UpstreamChecker.
    rng_pre_states: dict[str, Any] = field(default_factory=dict)

    # sha256(cell source) -> RNG modules the cell changed at runtime, including
    # draws inside called functions that static analysis cannot see.
    # W: CellExecutor. R: UpstreamChecker.
    observed_rng_cells: dict[str, set[str]] = field(default_factory=dict)

    # sha256(statement source) -> RNG modules the statement drew from without
    # the draw showing in its AST (``rf.fit(X, y)``). Makes the statement read
    # its module's virtual RNG variable, so a re-seed above it re-keys it.
    # W: StatementProcessor. R: the key and lineage builders on both engines.
    observed_rng_statement_draws: dict[str, set[str]] = field(default_factory=dict)

    # Variable -> every content hash seen for it.
    # W: StatementLineageBuilder, both restorers. R: Restorer, UpstreamChecker.
    variable_hashes: dict[str, set[str]] = field(default_factory=dict)

    # Variable -> the cache key that last produced it.
    # W: StatementLineageBuilder, StatementRestorer. R: Restorer, freshness
    # checks, end-of-cell persistence, UpstreamChecker.
    variable_sources: dict[str, str] = field(default_factory=dict)

    # The names the cells below the running one read: what a restart may need
    # restored from this cell. None when there is no notebook to read, and
    # then nothing is persisted ahead of need.
    # W: UpstreamChecker (per cell). R: StatementProcessor.end_cell_persistence (RebuildCostLedger).
    read_by_later_cells: frozenset[str] | None = None

    # The simulation's lineage for every name as of just before the current
    # cell ran. Used for one thing: a name a control structure read that the
    # runtime has no lineage for (bound in the ``%cash_on`` cell, before cash
    # was listening), so its recorded outcome can still match the simulation.
    # W: NotebookSimulator. R: ControlStructureProcessor.
    simulated_lineage: dict[str, str] = field(default_factory=dict)

    # Names in memory whose value can no longer be vouched for and must be
    # re-bound under tracking before a cell uses them: names bound untracked
    # before ``%cash_on`` by a statement that could have read something, and
    # variables built from an edited module. Never a from-imported name itself:
    # re-running the import would put back a key serving the pre-edit value.
    # W: NotebookSimulator, ModuleInvalidator. R/emptied: MismatchClassifier.
    rerun_bindings: set[str] = field(default_factory=set)

    # Bumped on every reload of a tracked module. The simulation caches per-cell
    # results by cell text and files read, which a helper-module edit does not
    # change, so this is part of that cache's key.
    # W: ModuleInvalidator. R: VirtualLineage.
    module_generation: int = 0

    # The names a notebook reaches a reloaded module through (the module, its
    # aliases, every name from-imported from it). The simulation re-simulates
    # from the first cell that reads or from-imports one of them.
    # W: ModuleInvalidator (with each generation). R: VirtualLineage.
    reloaded_names: set[str] = field(default_factory=set)

    # Variable -> its most recent content hash this session.
    # W: StatementLineageBuilder, StatementRestorer, StatementProcessor.
    # R: StatementProcessor (observed mutations), UpstreamChecker.
    current_session_hashes: dict[str, str] = field(default_factory=dict)

    # Variables mutated in place (``list.append``, ``d[k] = v``).
    # W: StatementProcessor, the control-structure helpers, UpstreamChecker.
    # R: the upstream check (skips lineage staleness for them).
    vars_with_mutation_lineage: set[str] = field(default_factory=set)

    # Variable -> the lineage of each input when its statement last ran.
    # W: StatementLineageBuilder, StatementRestorer, the simulator's restore
    # drain. R: the upstream check, ModuleInvalidator, miss attribution.
    executed_input_lineages: dict[str, dict[str, str]] = field(default_factory=dict)

    # Cache key -> (files, object-storage URLs) that statement itself read on
    # its last run. The simulation hashes exactly these to reach the lineage
    # the runtime recorded; ``executed_file_deps`` cannot stand in, because it
    # also holds what a variable inherited from its inputs.
    # W: StatementLineageBuilder. R: VirtualLineage, ControlStructureProcessor.
    statement_file_reads: dict[str, tuple[frozenset[str], frozenset[str]]] = field(default_factory=dict)

    # sha256(``ast.unparse`` of a top-level if/for/while/with/try) ->
    # ({input: lineage at entry}, {var: lineage it left behind}, files behind
    # those, their file-hash component). The runtime derives a control
    # structure's output lineages from values, which the simulation cannot
    # reproduce from code; reached again with the same inputs and file state,
    # the simulation takes what the runtime recorded.
    # W: ControlStructureProcessor. R: VirtualLineage.
    control_outcomes: dict[str, tuple[dict[str, str], dict[str, str], frozenset[str], str]] = field(
        default_factory=dict
    )

    # Module name -> variables whose stored input lineages need refreshing once
    # that module's import statement re-executes.
    # W: StatementLineageBuilder, ModuleInvalidator. R: ModuleInvalidator.
    granular_preserved_vars: dict[str, set[str]] = field(default_factory=dict)

    # Variable -> {module -> {attributes it read}}.
    # W: StatementLineageBuilder, StatementProcessor.forget_variable.
    # R: ModuleInvalidator, UpstreamChecker.
    module_attribute_deps: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    # Variable -> source module, for ``from X import Y`` bindings.
    # W: StatementLineageBuilder. R: ModuleInvalidator, UpstreamChecker.
    from_import_sources: dict[str, str] = field(default_factory=dict)

    # Variable -> the narrowed source component (only what Y reaches inside X)
    # a ``from X import Y`` name's lineage was built with. On a reload of X the
    # invalidator recomputes it and keeps Y's lineage when it is unchanged,
    # instead of dropping every name X exported.
    # W: StatementLineageBuilder, StatementProcessor.forget_variable.
    # R: ModuleInvalidator.
    from_import_components: dict[str, str] = field(default_factory=dict)

    # Statement source hash -> the receivers its bare method call mutates, so
    # the simulation, which never executes user code, reproduces the runtime's
    # mutation decision.
    # W: StatementProcessor (observed), VirtualLineage (read back from the
    # persisted verdict after a restart). R: both of them.
    mutation_verdicts: dict[str, set[str]] = field(default_factory=dict)

    # Variable -> the ``consumables.consumable_state`` token a consumable,
    # unrestorable input (generator, queue, file handle) held when the reading
    # cell last started. A consumable drains in place, so this is the only way
    # to tell a fresh object from the reader's own leftovers (``consumables.py``).
    # W: CellExecutor (cell entry). R: NotebookSimulator.
    consumable_bases: dict[str, Any] = field(default_factory=dict)

    # Source variable -> variables to bump when its lineage bumps through an
    # in-place mutation (a numpy view of it, a groupby holding it). The runtime
    # observes the object graph; the simulation only replays this map.
    # W: StatementLineageBuilder (derivation_edges). R: VirtualLineage.
    derivation_edges: dict[str, set[str]] = field(default_factory=dict)

    @property
    def variable_lineage(self) -> Mapping[str, str]:
        """A read-only live view of :attr:`lineage`; write through the store."""
        return self.lineage.view
