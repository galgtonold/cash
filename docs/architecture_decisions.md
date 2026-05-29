# Architecture Decision Records

## ADR-001: Statement-Level vs Cell-Level Caching

**Status:** Accepted  
**Date:** 2025-01-01  
**Context:** How granular should notebook caching be?

### Decision
Cash caches at the **statement level** rather than the cell level.

### Rationale
- **Partial reuse**: If a cell has 3 statements and only the 1st changes, statements 2 and 3 can still be restored from cache
- **Finer invalidation**: Changing one line doesn't invalidate an entire cell of expensive computations
- **Loop iteration caching**: Individual loop iterations can be cached separately
- **Real-world notebooks**: Data scientists often put multiple operations in one cell

### Consequences
- More complex implementation (AST parsing, statement decomposition)
- Cache key computation is per-statement, not per-cell
- Control structures (loops, conditionals) need special handling
- Higher overhead per cell execution (~5ms for AST parsing + lineage computation)

### Alternatives Considered
- **Cell-level caching**: Simpler but wastes cache when only part of a cell changes
- **Expression-level caching**: Too granular, excessive overhead for simple operations

---

## ADR-002: Lineage Hashing vs Timestamp-Based Invalidation

**Status:** Accepted  
**Date:** 2025-01-01  
**Context:** How to determine if a cached result is still valid?

### Decision
Cash uses **lineage hashing** - each variable gets a hash based on the code that produced it and all its inputs' lineage hashes.

### Format
```
lineage_hash(var) = SHA256(code + sorted(input_lineage_hashes) + file_dep_hashes)
```

### Rationale
- **Content-addressed**: Same code + same inputs → same hash → cache hit, regardless of time
- **Transitive invalidation**: If input A changes, everything downstream is automatically invalidated
- **Deterministic**: No race conditions or clock skew issues
- **File-aware**: Changing a CSV file invalidates all code that read it

### Consequences
- Must track full dependency graph (variable_lineage, executed_cell_codes, executed_input_lineages)
- Hash computation on every execution (~0.5ms per statement)
- Must handle circular dependencies (prevented by Python's linear execution model)

### Alternatives Considered
- **Timestamp-based**: Compare file mtimes. Simple but breaks with: same content different time, or content changes within same second
- **Content hashing of outputs**: Hash the result itself. Expensive for large DataFrames, and some objects aren't hashable

---

## ADR-003: AST Analysis vs Bytecode Inspection

**Status:** Accepted  
**Date:** 2025-01-01  
**Context:** How to determine which variables a statement reads and writes?

### Decision
Cash uses **AST (Abstract Syntax Tree) analysis** to detect inputs and outputs of each statement.

### Rationale
- **Reliable**: AST is the canonical representation of Python code
- **No execution needed**: Can analyze code before running it
- **Control flow support**: Can decompose loops and conditionals at the AST level
- **Standard library**: Uses Python's built-in `ast` module

### Consequences
- Must handle all Python syntax (assignments, augmented assignments, comprehensions, etc.)
- Some dynamic patterns are hard to detect statically (e.g., `setattr`, `globals()['x'] = 1`)
- AST parsing adds ~1-3ms overhead per cell

### Alternatives Considered
- **Bytecode inspection**: More precise but version-dependent and harder to understand
- **Runtime tracing (sys.settrace)**: Precise but very high overhead
- **Regex-based**: Simple but unreliable for complex Python syntax

---

## ADR-004: Upstream Simulation for Dependency Tracking

**Status:** Accepted  
**Date:** 2025-06-01  
**Context:** How to detect when upstream cells have changed without re-executing them?

### Decision
Before executing a cell, Cash **simulates** all upstream cells by parsing their code and computing virtual lineage hashes, without actually executing them.

### Rationale
- **No re-execution cost**: Simulation is orders of magnitude faster than execution
- **Change detection**: Compares simulated lineages with actual lineages in namespace
- **Selective re-execution**: Only re-executes cells whose simulated lineage differs
- **Handles notebook reordering**: Works correctly even if cells are rearranged

### Consequences
- Simulation must mirror execution's lineage computation exactly
- Performance critical: ~2ms per cell simulated (acceptable for <200 cells)
- Must cache simulation results for repeated runs (simulation cache added in Phase 3.7)

---

## ADR-005: Mutation Detection - Detection Only

**Status:** Accepted  
**Date:** 2026-02-06  
**Context:** How to handle in-place mutations like `list.append()` or `df.drop(inplace=True)`?

### Decision
Cash detects mutations via **AST analysis** and marks variables as mutated, but does **not** modify lineage for mutations.

### Rationale
- **Modifying lineage breaks accumulator patterns**: If Cell 1 does `items = []` and Cell 2 does `items.append(x)`, modifying the lineage of `items` in Cell 2 would make Cell 3's cache miss because `items` has a different lineage than when Cell 3 was first run
- **Detection is sufficient**: By knowing which variables are mutated, the skip optimization can be disabled for those variables, preventing stale cache reads
- **Safe default**: Better to not cache than to cache incorrectly

### Consequences
- Mutated variables are tracked in `vars_with_mutation_lineage`
- Skip optimization skips lineage validation for mutated vars
- Cache restoration still works (restores the post-mutation state)
- Future: could add snapshots or mutation replay

---

## ADR-006: Module-Qualified Function Keys

**Status:** Accepted  
**Date:** 2025-06-01  
**Context:** How to uniquely identify decorated functions for cache key generation?

### Decision
Cash identifies cached functions using `f"{func.__module__}.{func.__qualname__}"` instead of just `func.__qualname__`.

### Rationale
- **Collision avoidance**: Different modules can define functions with the same `__qualname__` (e.g., a notebook cell's `dep()` vs a helper module's `dep()`)
- **Using only qualname caused a critical bug**: Cache entries for different functions would collide, producing incorrect cache hits
- **Module path is stable**: `__module__` is set correctly by Python for all function types

### Format
```python
# Before (colliding):
"dep"                    # Both functions got the same key!

# After (unique):
"__main__.dep"           # Notebook-defined function
"my_utils.dep"           # Helper module function
```

### Consequences
- Cache keys from v0.2.0 are incompatible (one-time cache invalidation on upgrade)
- Slightly longer cache keys (negligible impact)
- `_func_key_cache` dictionary avoids repeated `__module__` + `__qualname__` lookups

---

## ADR-007: Unified Cache Key Computation

**Status:** Accepted  
**Date:** 2025-06-01  
**Context:** Cache keys are computed in 4+ different code paths. How to prevent divergence?

### Decision
All cache key computation goes through a single function: `compute_cache_key()` in `cash.notebook.cache_key`.

### Rationale
- **Past bugs**: Divergent cache key computation between runtime and simulation caused subtle bugs where cache misses occurred after kernel restart, or stale data was served
- **Single source of truth**: One function, one formula, one set of tests
- **Multiple call sites**: Used by `statement_processor.py` (runtime), `upstream.py` (simulation, virtual restore, skip checks)

### Consequences
- Any cache key change must be made in exactly one place
- All call sites automatically pick up changes
- Input lineage priority order is documented and enforced in one location
- Testing is centralized in `tests/test_notebook/test_virtual_restore_modules.py`

---

## ADR-008: Unified Lineage State via `LineageStore`

**Status:** Accepted
**Date:** 2026-05-15
**Context:** ADR-007 made cache-key *computation* a single seam, but the *lineage state* it reads (`variable_lineage`) was still mutated by several producers and paired with a separate `_cash_lineage_hash` attribute that had to be written in lockstep.

### Decision
All persistent lineage state lives behind one seam: `LineageStore` (in `cash.notebook.lineage_store`), held as `TrackingState.lineage`. Writes go through `record(var, hash, value=...)` (fresh computation / restore) or `reset_to(var, hash)` (downstream-advancement resync). Reads use `LineageStore.resolve(var, value=, virtual=, compute_hash_fn=)`, which is the canonical implementation of the priority ladder (virtual → store → `_cash_lineage_hash` → compute_hash → str fallback). `cache_key._resolve_input_lineage` delegates here.

### Rationale
- **Past bugs**: In ~6 sites the dict entry `variable_lineage[var] = h` and the value attribute `val._cash_lineage_hash = h` were written separately. Forgetting one (or guarding them differently, as at the old `upstream.py:3230–3235`) silently desynced the two views, producing cache misses after restore.
- **Named resync**: `reset_to` distinguishes downstream-advancement corrections from fresh writes — a distinction that previously lived only in comments.
- **Single read ladder**: extends ADR-007's "single source of truth" principle from cache-key computation to lineage resolution.

### Consequences
- New lineage writes call `state.lineage.record(...)` or `state.lineage.reset_to(...)`. Direct dict mutation (`state.variable_lineage[var] = h`) still works during migration because the store wraps the same dict, but is discouraged.
- `_cash_lineage_hash` is set by the store when `value` is passed to `record`. Standalone attribute writes outside the store are a code smell.
- `virtual_lineage` is *not* owned by the store — it is transient per-simulation state, passed by parameter into `resolve(virtual=...)`.
- `executed_input_lineages`, `executed_cell_codes`, `vars_with_mutation_lineage` remain separate — they carry different invariants (skip-check state, not current lineage) and may be folded in later if a natural seam emerges.
- The decorator path (`src/cash/core.py`) sets `_cash_lineage_hash` on cached function returns directly; that subsystem is out of scope for this ADR.

---

## ADR-009: Extract `NotebookSimulator` from `UpstreamChecker`

**Status:** Accepted
**Date:** 2026-05-15
**Context:** `UpstreamChecker` had grown to 96 methods / ~3800 lines, conflating two distinct concerns: **orchestration** (when to run what, callbacks into `process_statement`, badge metrics, two-phase coordination) and **simulation** (AST passes, virtual lineage tracking, cache probing, broken-var classification, re-execution-plan construction). Simulator behavior could only be tested by constructing the full orchestrator, which required a `Cash` instance, a backend, and a real-or-mocked IPython shell. The test suite for upstream behavior was scattered across 13 files (6 unit + 7 integration) because there was no clean test surface to attach to.

### Decision
Extract the simulator's transitive method closure (76 methods reachable from `_simulate_and_find_changes`) into a new `NotebookSimulator` class in `cash.notebook.notebook_simulator`. `UpstreamChecker` owns one via composition (`self.simulator`) and delegates to it. Shared mutable state (`TrackingState` dicts) is passed by reference so both views observe each other's writes. Simulator-owned caches (`_simulation_cache`, `_ast_cache`, `_simulation_cell_hashes`, `_cell_id_to_last_index`) move with the simulator.

### Rationale
- **Locality.** The simulator's helpers (~76 methods) live next to each other and the data they touch. Future changes to simulation logic don't risk perturbing the orchestrator and vice versa.
- **Test surface.** `NotebookSimulator` can be constructed with a `TrackingState()` and a `SimpleNamespace(user_ns={})` — no `Cash`, no backend, no IPython kernel. See `tests/test_notebook/test_notebook_simulator.py` for the new surface.
- **AI navigability.** Searching for "simulation" now finds one file with one class, not 76 methods scattered across a 3800-line class definition.
- **The 2-phase orchestrator stays intact.** `check_and_reexecute` still has Phase 1 (`_check_lineage_based`) and Phase 2 (`_check_notebook_based`); the 4-step pipeline inside Phase 2 is unchanged. We only split the *implementation*, not the algorithm.

### Consequences
- `UpstreamChecker` is now ~25 methods and ~840 lines (down from ~3800).
- The pre-existing duplicate `_handle_lineage_mismatch` definition (Phase 1's 7-param body was shadowed by Phase 2's 14-param body in the original class) is removed. Observable behavior preserved: the 7-arg call site in `_check_lineage_based` was always inside a `try/except (TypeError, ...)`, so it has been silently no-op'd for the entire history of the file. Filed as a latent bug for future work.
- Test code that previously reached into `UpstreamChecker._method(...)` was migrated to `UpstreamChecker.simulator._method(...)`. A transitional `__getattr__` / `__setattr__` shim was used during the initial extraction and then removed (~92 mechanical edits across 9 test files). Class-level method-existence tests (`hasattr(UpstreamChecker, '_validate_file_freshness')`) now point at `NotebookSimulator` — the new home.
- Tests that constructed an `UpstreamChecker` and then rebound shared-state fields (`checker.executed_file_deps = {}`, `checker.cash_instance = mock_cash`) were updated to pass a fresh `TrackingState()` or `cash_instance` through the constructor. Rebinding after construction would diverge the checker's view from the simulator's view, since both hold their own references.

---

## ADR-010: Co-locate the Upstream Subsystem as a Package

**Status:** Accepted
**Date:** 2026-05-29
**Context:** ADR-009 extracted `NotebookSimulator` from `UpstreamChecker` but left the six resulting files (`upstream.py`, `notebook_simulator.py`, `virtual_lineage.py`, `reexecution_planner.py`, `mismatch_classifier.py`, `simulator_types.py`) as flat peers among ~39 other top-level modules in `src/cash/notebook/`. The orchestrator-owns-simulator relationship documented in ADR-009 is invisible in the filesystem, and three of the six files carry disambiguator prefixes (`notebook_`, `simulator_`) that only made sense at the flat top level.

### Decision
Collapse the six files into one `upstream/` package. The package re-exports exactly three symbols — `UpstreamChecker`, `UpstreamResult`, `NotebookSimulator` — and treats the rest (`VirtualLineage`, `MismatchClassifier`, `ReexecutionPlanner`, the value-type IR) as package-internal. Files are renamed to drop the prefixes that the package path now provides:

```
upstream/__init__.py             # re-exports public + private-to-tests symbols
upstream/checker.py              # was upstream.py
upstream/simulator.py            # was notebook_simulator.py
upstream/virtual_lineage.py
upstream/reexecution_planner.py
upstream/mismatch_classifier.py
upstream/_types.py               # was simulator_types.py — leading _ marks IR as internal
```

### Rationale
- **Locality.** The simulator's helpers live one folder deep next to the simulator and the orchestrator, not scattered alphabetically among 39 unrelated modules. A reader searching for "simulation" or "virtual lineage" lands inside one folder.
- **Honest seam.** ADR-009 named `NotebookSimulator` as a deliberate test surface (constructible with just a `TrackingState` + `SimpleNamespace`); the package's `__init__.py` is the first place that statement is enforced in code. `VirtualLineage`, `ReexecutionPlanner`, and the IR types stop being importable from outside the package — they are implementation details, and now look like it.
- **Filename context.** `notebook_simulator.py` only carried the `notebook_` prefix because it sat next to `notebook_caching`-shaped peers. Inside `upstream/`, `simulator.py` is unambiguous.
- **Completes ADR-009.** The class extraction was Phase 1; the package extraction is Phase 2 of the same refactor.

### Consequences
- The public production import path (`from cash.notebook.upstream import UpstreamChecker`) is unchanged — `upstream` resolves to a package instead of a module.
- Three test files migrate `from cash.notebook.notebook_simulator import NotebookSimulator` → `from cash.notebook.upstream import NotebookSimulator`.
- One test (`tests/test_notebook/test_upstream_unit.py`) that imported `VirtualLineage` directly is rewritten to drive it through `NotebookSimulator`, the documented test seam.
- Three test files importing the private `_SimulationCacheEntry` / `_FORWARD_PROBE_PLACEHOLDER` symbols from `cash.notebook.upstream` continue to work — those symbols are re-exported (with their leading underscores intact) from `upstream/__init__.py`.
- Internal imports between the six files become relative (`from .virtual_lineage import VirtualLineage`).
- No backward-compatibility shims at the old paths. The subsystem has no external (non-test) callers beyond `UpstreamChecker`, so shims would be pure cost.
- Future architecture reviews that re-suggest flattening this back out should consult this ADR.

---

## ADR-011: Co-locate StatementProcessor and its Siblings as a Package

**Status:** Accepted
**Date:** 2026-05-29
**Context:** `CONTEXT.md` describes `StatementProcessor` as the owner of four sibling classes (`CacheFreshnessChecker`, `StatementFileDeps`, `StatementLineageBuilder`, `StatementRestorer`) sharing one `TrackingState` and one `ProcessResult` schema. The five files lived as flat peers in `src/cash/notebook/`, contradicting the documented ownership story. Additionally, `cache_freshness.py` carried two module-level helpers (`snapshot_file_deps`, `split_file_dep_value`) that are *not* sibling-of-StatementProcessor concerns — they are used by the decorator subsystem (`src/cash/core.py`), the variable-granular `Restorer`, and `upstream/virtual_lineage.py`. Bundling them into `statement/` would force unrelated callers to reach into a package they otherwise don't care about.

### Decision
Collapse the four sibling files plus `StatementProcessor` into one `statement/` package. The package re-exports exactly four production symbols — `StatementProcessor`, `ProcessResult`, `StatementCacheMetadata`, `DecoratorCallMetric` — plus two private symbols for test-patching paths (`_TeeWriter`, `_tee_output`). The siblings (`CacheFreshnessChecker`, `StatementFileDeps`, `StatementLineageBuilder`, `StatementRestorer`) are package-internal.

To keep the package honest, the two cross-cluster helpers in `cache_freshness.py` are extracted to a new top-level sibling module before the move:

```
notebook/file_dep_snapshot.py    # NEW — snapshot_file_deps + split_file_dep_value
                                  # Pure file-dep snapshot utilities used cross-subsystem.

notebook/statement/__init__.py
notebook/statement/processor.py   # was statement_processor.py
notebook/statement/lineage.py     # was statement_lineage.py
notebook/statement/restore.py     # was statement_restore.py
notebook/statement/file_deps.py   # was statement_file_deps.py
notebook/statement/freshness.py   # CacheFreshnessChecker class only; was cache_freshness.py
```

### Rationale
- **Locality.** Five files that already form a composition unit (one orchestrator + four owned siblings) become one folder. CONTEXT.md's "four siblings of StatementProcessor" framing now matches the filesystem.
- **Honest split.** `cache_freshness.py` was carrying two roles: the `CacheFreshnessChecker` class (a true StatementProcessor sibling) and two pure module-level helpers consumed by unrelated subsystems. Splitting the file *before* the move keeps `statement/` from accidentally becoming a dependency of the decorator path. Future maintainers won't see `from cash.notebook.statement.freshness import snapshot_file_deps` in `core.py` and wonder why the decorator is reaching into the notebook's statement package.
- **Filename context.** The `statement_*` prefixes on four of the files were disambiguators at the flat top level; once inside `statement/`, the parent package provides the context. The class names (`StatementProcessor`, `StatementFileDeps`, `StatementLineageBuilder`, `StatementRestorer`) are unchanged — only the filenames are shortened.
- **Consistency with ADR-010.** Same pattern: collapse a documented composition unit into a package, expose a small public surface, rename files for context.

### Consequences
- The public production import path becomes `from cash.notebook.statement import StatementProcessor` (instead of `cash.notebook.statement_processor`). All seven sibling notebook modules (`magics`, `cell_executor`, `restore`, `module_invalidator`, `control_for_handler`, `control_if_handler`, `control_try_handler`, `control_structures`) update their imports.
- `src/cash/core.py` updates its two `cache_freshness.snapshot_file_deps` imports to `file_dep_snapshot.snapshot_file_deps`. The decorator path no longer reaches into a notebook subpackage for this utility.
- `upstream/virtual_lineage.py` updates `from ..cache_freshness import split_file_dep_value` to `from ..file_dep_snapshot import split_file_dep_value`. (`upstream/` was extracted in ADR-010.)
- Three sites in `tests/test_notebook/test_metadata_only_persistence.py` that imported `StatementRestorer` directly migrate to the full internal path (`cash.notebook.statement.restore.StatementRestorer`). The class is internal but the static method `persist_metadata_only` has no clean wrapper on `StatementProcessor` today.
- Eight sites in `tests/test_notebook/test_control_structures.py` that imported `_TeeWriter` / `_tee_output` from `cash.notebook.statement_processor` migrate to `cash.notebook.statement` — both symbols are private re-exports.
- Internal cluster imports become relative (`from .lineage import StatementLineageBuilder`).
- No backward-compatibility shims at the old paths.
- The TypedDicts (`ProcessResult`, `StatementCacheMetadata`, `DecoratorCallMetric`, `_ProcessResultRequired`) stay inside `processor.py` for now. Extracting them into `statement/_types.py` (mirroring `upstream/_types.py`) is a separate, optional refactor.

---

## ADR-012: Co-locate the Control-Structure Strategy as a Package

**Status:** Accepted
**Date:** 2026-05-29
**Context:** `CONTEXT.md` described `ControlStructureProcessor` as the orchestrator of a strategy pattern: three handler classes (`ForLoopHandler`, `IfHandler`, `TryHandler`) plus a shared-helpers module. The five files (`control_structures.py`, `control_for_handler.py`, `control_if_handler.py`, `control_try_handler.py`, `control_structure_helpers.py`) lived as flat peers in `src/cash/notebook/`. The `control_*` prefix on four of them was doing the work a folder should do.

### Decision
Collapse the five files into one `control_structures/` package. Files are renamed to drop the `control_*` prefixes that the package path now provides:

```
control_structures/__init__.py        # re-exports public symbols
control_structures/processor.py       # was control_structures.py
control_structures/for_handler.py     # was control_for_handler.py
control_structures/if_handler.py      # was control_if_handler.py
control_structures/try_handler.py     # was control_try_handler.py
control_structures/helpers.py         # was control_structure_helpers.py
```

Public surface (re-exported from `__init__.py`):
- `ControlStructureProcessor` (orchestrator)
- `ControlStructureResult` (return type)
- Pure AST helpers used cross-cluster: `is_control_structure`, `get_control_structure_type`, `extract_target_names`, `contains_break_or_continue`, `bind_target_values`, `build_iteration_context`, `compute_context_hash`

Internal (not re-exported): the three handler classes, and everything in `helpers.py`.

### Rationale
- **Locality.** Five files documented as a composition unit become one folder. The orchestrator/handler/helpers relationship is now structural, not just naming convention.
- **Naming.** The `control_*` prefix on four files only existed because they sat among ~30 unrelated peers. Inside `control_structures/`, the parent path provides the disambiguation.
- **Public AST helpers stay re-exported.** Unlike `cache_freshness.py`'s `snapshot_file_deps` (which crossed *subsystems* and warranted extraction in ADR-011), the AST helpers here only cross *clusters* inside the notebook subsystem — `upstream/` and the handlers themselves use them. Re-exporting from `__init__.py` keeps existing import paths stable without leaking implementation details.
- **Consistency with ADR-010 and ADR-011.** Same pattern: collapse a documented composition unit into a package, expose a small public surface, rename files for context.

### Consequences
- Production import paths stay valid: `from cash.notebook.control_structures import is_control_structure` etc. work for callers that imported public AST helpers or `ControlStructureProcessor` from the old flat module.
- Cross-cluster callers (`cell_executor`, `magics`, `upstream/checker`, `upstream/simulator`, `upstream/virtual_lineage`) don't need import changes — they were already using public symbols at the package-resolved path.
- Three test files (`test_control_for_handler.py`, `test_control_if_handler.py`, `test_control_try_handler.py`) and one site in `test_notebook/test_control_structures.py` (importing `flush_metrics_output`) migrate to the full internal paths (`cash.notebook.control_structures.for_handler.ForLoopHandler`, etc.). The handler classes and `helpers.py` symbols are internal; this matches the precedent from ADR-011 for `StatementRestorer`.
- Internal cluster imports become relative (`from .processor import is_control_structure`, `from .helpers import ...`).
- No backward-compatibility shims at the old paths.
