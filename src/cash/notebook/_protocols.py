"""Protocol types and shared data structures for the notebook subsystem.

These protocols define the minimal interfaces that the notebook subsystem
requires from external objects (IPython shell, cache backend, etc.).
Using protocols instead of ``Any`` provides:

- Better IDE support (autocomplete, type checking)
- Documentation of expected interfaces
- Easier testing with mock objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
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

    def set(
        self, key: str, value: Any, metadata: dict[str, Any] | None = None, serializer: Any = None
    ) -> None:
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

    This dataclass is the **single owner** of the tracking dictionaries that
    are shared between :class:`CashMagics`, :class:`StatementProcessor`, and
    :class:`UpstreamChecker`.  By passing a single ``TrackingState`` instance
    (rather than 6+ individual dicts) we:

    * Document exactly which dicts are shared and what they contain.
    * Give type checkers concrete types instead of ``dict[str, Any]``.
    * Make it impossible to accidentally pass the wrong dict to the wrong slot.

    All fields use ``default_factory`` so each ``TrackingState()`` starts empty.

    **Initialization order**: ``CashMagics`` creates the ``TrackingState``
    and passes the *same* instance to both ``StatementProcessor`` and
    ``UpstreamChecker`` via ``set_tracking_state()``.  Because the three
    components share mutable dict references, ``CashMagics`` **must** create
    the instance before constructing the processors.

    **Ownership summary** (W = writes, R = reads):

    +------------------------------+-----------------+-------------------+-------------------+
    | Field                        | CashMagics      | StatementProcessor| UpstreamChecker   |
    +==============================+=================+===================+===================+
    | executed_cell_codes          | —               | W (after exec)    | R (lineage check) |
    | executed_cell_hashes         | —               | W (after exec)    | R (rarely)        |
    | variable_lineage             | R (badge)       | W (after exec)    | R+W (reset/sync)  |
    | executed_file_deps           | —               | W (after exec)    | R (stale check)   |
    | executed_file_mtimes         | —               | W (after exec)    | —                 |
    | variable_hashes              | R (badge)       | W (after exec)    | —                 |
    | variable_sources             | R (badge)       | W (after exec)    | —                 |
    | current_session_hashes       | —               | W (after exec)    | —                 |
    | vars_with_mutation_lineage   | —               | W / ControlStruct | R (skip lineage)  |
    | executed_input_lineages      | —               | W (after exec)    | R (lineage check) |
    | granular_preserved_vars      | —               | W (lineage build) | R (module inv.)   |
    | module_attribute_deps        | —               | W (lineage build) | R (module inv.)   |
    | from_import_sources          | —               | W (lineage build) | R (module inv.)   |
    +------------------------------+-----------------+-------------------+-------------------+

    The last four fields used to live as instance attributes on
    ``StatementProcessor`` and were threaded into ``StatementFileDeps`` /
    ``StatementLineageBuilder`` by reference.  They were moved onto
    ``TrackingState`` so the four siblings of ``StatementProcessor`` can
    receive state as a method parameter rather than aliasing dict refs
    in their own ``set_tracking_state`` (which no longer exists on those
    siblings).
    """

    # Written by StatementProcessor after each statement execution.
    # Read by UpstreamChecker to compare simulated vs. executed statement code.
    executed_cell_codes: dict[str, str] = field(default_factory=dict)

    # Written by StatementProcessor after each statement execution.
    # Stores the SHA-256 of the defining statement code for fast change detection.
    executed_cell_hashes: dict[str, str] = field(default_factory=dict)

    # Written by StatementProcessor (and ControlStructureProcessor for mutations).
    # Read by UpstreamChecker to detect stale variables; occasionally reset by
    # UpstreamChecker when resynchronising simulation state with actual memory.
    variable_lineage: dict[str, str] = field(default_factory=dict)

    # Written by StatementProcessor (via FileAccessTracker) after each execution.
    # Read by UpstreamChecker to detect stale file dependencies.
    executed_file_deps: dict[str, set[str]] = field(default_factory=dict)

    # Written by StatementProcessor after executing a statement with a
    # file-WRITE side effect; read by ReexecutionPlanner (CAS-81/82).
    # File writes have no variable edge, so this code-text record is how the
    # simulation tells an edited/new writer statement from one that already
    # ran in this session.
    executed_write_stmt_codes: set[str] = field(default_factory=set)

    # Written by StatementProcessor; read by CashMagics for badge display.
    # Accumulates all content hashes seen for a variable across executions.
    variable_hashes: dict[str, set[str]] = field(default_factory=dict)

    # Written by StatementProcessor; read by CashMagics for badge display.
    # Records the cache key that last produced each variable.
    variable_sources: dict[str, str] = field(default_factory=dict)

    # Written by StatementProcessor after each execution.
    # Tracks the most recent content hash within the current session.
    current_session_hashes: dict[str, str] = field(default_factory=dict)

    # Written by StatementProcessor and ControlStructureProcessor for in-place
    # mutations (e.g., list.append, dict[k] = v).
    # Read by UpstreamChecker to skip lineage-based staleness checks for these vars.
    vars_with_mutation_lineage: set[str] = field(default_factory=set)

    # Written by StatementProcessor; read by UpstreamChecker (Pass 1 lineage check).
    # Stores the lineage snapshot of each input at the time of execution.
    executed_input_lineages: dict[str, dict[str, str]] = field(default_factory=dict)

    # Written by StatementFileDeps (via StatementProcessor) after each execution.
    # Per-variable mtime snapshots of accessed files; used together with
    # ``executed_file_deps`` for fast direct-file staleness detection.
    executed_file_mtimes: dict[str, dict[str, float]] = field(default_factory=dict)

    # Written by StatementLineageBuilder when a tracked module is re-imported.
    # Read by module_invalidator. Maps module_name -> {var_names} whose stored
    # input lineages need refreshing once the import statement re-executes.
    granular_preserved_vars: dict[str, set[str]] = field(default_factory=dict)

    # Written by StatementLineageBuilder; read by module_invalidator.
    # Per-variable record of which module attributes contributed: maps
    # var_name -> {module_name -> {attr1, attr2, ...}}.
    module_attribute_deps: dict[str, dict[str, set[str]]] = field(default_factory=dict)

    # Written by StatementLineageBuilder; read by module_invalidator.
    # Maps var_name -> source_module_name for ``from X import Y`` bindings.
    from_import_sources: dict[str, str] = field(default_factory=dict)

    # Written by StatementProcessor after observing a standalone method call;
    # read by VirtualLineage (upstream simulation). Maps a statement's
    # source_hash -> the set of receiver names that method call mutates (the
    # broad-precise mutation verdict). Lets the simulation, which never executes
    # user code, reproduce the runtime's mutation decision for a bare
    # ``obj.method()`` whose method is not statically known to mutate.
    mutation_verdicts: dict[str, set[str]] = field(default_factory=dict)

    # The single seam for reading/writing variable lineage. Wraps
    # ``variable_lineage`` as its backing dict so callers that still mutate
    # the dict directly during migration stay in sync. See ``CONTEXT.md``
    # entry: *LineageStore*.
    lineage: "LineageStore" = field(init=False)

    def __post_init__(self) -> None:
        from cash.notebook.lineage_store import LineageStore
        self.lineage = LineageStore(backing=self.variable_lineage)
