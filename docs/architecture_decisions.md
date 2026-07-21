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

### Amendment (2026-07-15): the file-dependency path caught up with this decision

The two failure modes rejected above under *Timestamp-based* — "same content
different time" and "content changes within same second" — were not hypothetical.
They shipped, as bugs, in the **file-dependency** path: while variable lineage was
content-addressed from the start, file freshness was decided by `(mtime, size)`
until this wave.

- **CAS-98** = "same content, different time": a touch-only change (identical
  bytes, bumped mtime) forced a needless recompute — over-invalidation.
- **CAS-10** = "content changes within same second": a same-size edit under an
  mtime the coarse check couldn't distinguish was missed and served stale —
  under-invalidation.
- **CAS-119** carried the same fix into the `@cash.cache` decorator path, so the
  two subsystems share one helper and cannot drift.

**Resolution:** auto-tracked file deps now record a content hash alongside
`(mtime, size)` and treat **content as authoritative whenever the size matches**
(`file_dep_is_fresh`, `src/cash/notebook/file_dep_snapshot.py`). The cheap size
check runs first, so the "hashing a 2 GB parquet on every lookup" objection is
answered by never hashing when the size already proves staleness, and by sampling
files over 8 MiB (head/middle/tail) rather than reading them whole. The residual
tradeoff is a narrow one: a same-size edit confined to unsampled interior bytes of
a >8 MiB file is not detected.

**Still timestamp-based, deliberately:** the explicit `file_depends_on=` /
`FileDataSource` escape hatch folds the file's mtime into the cache key
(`state_token()` → `_get_mtime()`, `src/cash/data_source.py`). It is a
user-declared dependency on a *file*, not an observed read, and users can
subclass `DataSource` to return a digest instead. This ADR's reasoning applies to
the automatic path; the escape hatch keeps mtime for cost and predictability.

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
- **Multiple call sites**: Used by `statement/processor.py` (runtime), `upstream.py` (simulation, virtual restore, skip checks)

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
notebook/statement/processor.py   # was statement/processor.py
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

---

## ADR-013: Co-locate the IPython Adapter as a Package

**Status:** Accepted
**Date:** 2026-05-29
**Context:** `CONTEXT.md` repeatedly refers to "the IPython adapter (`CashMagics`)" as if it were a distinct architectural concept, but the four files that make up the adapter (`magics.py`, `magic_admin.py`, `cell_executor.py`, `error_display.py`) lived as flat peers in `src/cash/notebook/`. The adapter concept was named in prose but not in the filesystem.

### Decision
Collapse the four files into one `ipython/` package. Files are mostly unchanged in name; `magic_admin.py` drops its `magic_` prefix since the parent path now provides context:

```
ipython/__init__.py            # re-exports CashMagics (the only public symbol)
ipython/magics.py              # unchanged — CashMagics (the IPython Magics subclass)
ipython/admin.py               # was magic_admin.py — CashAdminMagicsMixin
ipython/cell_executor.py       # unchanged — CellExecutor, _PipelineCompleted, etc.
ipython/error_display.py       # unchanged — show_clean_error
```

Public surface: `CashMagics` only. Everything else (`CashAdminMagicsMixin`, `CellExecutor`, the value types `TimingBreakdown` / `StatementSummary` / `CellMetrics` / `CashSession`, `show_clean_error`, the internal sentinels `_EarlyReturn` / `_PipelineSyntaxError` / `_PipelineCompleted`) is package-internal.

### Rationale
- **Names a documented concept.** `CONTEXT.md` already says "the IPython adapter"; the package makes that statement structural.
- **Honest acknowledgement: one adapter, hypothetical seam.** By the *one adapter = hypothetical seam* rule, this is not yet a real seam — only `CashMagics` exists; no Marimo or headless adapter is in flight. Accepted because the **locality** win is real today (four files in one folder) and the **naming** win matches CONTEXT.md vocabulary.
- **Consistency with ADRs 010-012.** Same pattern: collapse a documented unit into a package, expose a small public surface, drop the `magic_` prefix where the package path provides context.

### Consequences
- The biggest test migration of the four refactors: ~30 test files migrate `from cash.notebook.magics import CashMagics` → `from cash.notebook.ipython import CashMagics`.
- Patch-target migrations: `@patch('cash.notebook.magics.display')`, `@patch('cash.notebook.magics.publish_display_data')`, `@patch('cash.notebook.magics.get_notebook_cells')` → repointed at `cash.notebook.ipython.magics.X`. `@patch('cash.notebook.cell_executor.CodeAnalyzer.analyze_code_block')` → `cash.notebook.ipython.cell_executor.CodeAnalyzer.analyze_code_block`.
- One test (`test_vscode_cell_id_path.py`) that imports `CellExecutor` directly migrates to the full internal path `cash.notebook.ipython.cell_executor.CellExecutor`.
- `src/cash/notebook/__init__.py`'s lazy `__getattr__` for `CashMagics` repoints at `from .ipython.magics import CashMagics`.
- `magic_admin.py`'s rename to `admin.py` matches the same convention as ADR-011 (`statement/processor.py` → `statement/processor.py`).
- No backward-compatibility shims at the old paths.
- The three metrics TypedDicts (`TimingBreakdown`, `StatementSummary`, `CellMetrics`) are extracted to `ipython/_types.py` (mirroring `upstream/_types.py`) so `magics.py` is just the orchestrator file. `CashSession` stays in `magics.py` for now — it carries instance state (`provenance`, `audit`) rather than being pure data, so it doesn't belong in `_types.py`.

---

## ADR-014: Cache Metadata as Frozen Dataclasses at the Edges (Placement B)

**Status:** Accepted
**Date:** 2026-05-29
**Context:** Cache-entry metadata was passed around as `TypedDict`s, so every consumer reached into it with string-keyed `.get('field', default)` calls. This was verbose, untyped at the access site, and easy to typo. We wanted typed attribute access (`meta.execution_time`) without giving up the backends' freedom to round-trip arbitrary keys.

### Decision
Metadata is modeled as two **sibling frozen dataclasses** — `CacheMetadata` (decorator layer, in `backends/_base.py`) and `StatementCacheMetadata` (notebook layer, in `notebook/statement/_metadata.py`) — but conversion happens **only at the cash-layer edges** ("Placement B"):

- **Producers** build a dataclass and call `.to_dict()` immediately before `backend.set(...)`.
- **Consumers** call `from_dict(...)` immediately after `backend.get(...)`.
- **Backends never see the dataclass.** They round-trip an opaque `MetadataDict = dict[str, Any]`. The channel is deliberately polymorphic: both dataclass shapes — plus backend-private keys (`compressed`, `size`, `created_at`, `last_access`, `access_count`, `serializer_cls`, `source`, `storage`) — flow through it as plain dicts.

Wire contract for both dataclasses:
- `to_dict()` **omits `None` fields**, preserving the historical "only-set-keys" dict shape so backend presence-checks (`'x' not in metadata`) keep working.
- `from_dict()` is **lenient**: unknown keys (legacy aliases like `cell_code`/`cell_hash`, backend-private keys) are ignored; missing keys default to `None`.

### Rationale
- **Dict-on-the-wire, dataclass-in-memory.** The persisted format stays a plain dict; we never pickle the dataclass, so on-disk caches written by older/newer versions stay readable.
- **Backends stay schema-agnostic.** A backend injecting its own bookkeeping keys (e.g. `FileBackend` stamping `compressed`) needs no schema change — those keys are simply dropped by `from_dict` at the cash layer, which doesn't need them.
- **Leaf module breaks the import cycle.** `StatementCacheMetadata` lives in a dependency-free `_metadata.py` so `processor.py` and its siblings (`freshness`, `file_deps`, `restore`) can all import it at runtime without re-introducing the `processor → freshness → processor` cycle.

### Consequences
- **Deliberate ttl normalization (behavior change).** The old producers stamped `ttl=None` into the wire dict when no per-call ttl was given, which left the key *present* and so suppressed `FileBackend`/`SqliteBackend` `default_ttl` (their `if 'ttl' not in metadata` / `metadata.get('ttl', default)` checks saw a `None` value). Because `to_dict()` now omits `None`, an unset ttl falls through to the backend default. **Decorator caches without an explicit `ttl` now expire under the backend's `default_ttl`** where previously they never expired. Guarded by `tests/test_core/test_ttl.py::test_default_ttl_applies_without_explicit_ttl`.
- **Storage back-propagation preserved.** The full-store path routes a mutable `wire = metadata.to_dict()` through `backend.set` (which may inject the resolved `storage` tiers) and re-wraps it via `from_dict(wire)` so the returned view carries `storage` on to the badge metrics.
- **Plain-dict consumers left as-is.** `notebook/restore.py` (reads legacy `cell_code`/`cell_hash` aliases outside the schema), `notebook/upstream/virtual_lineage.py` (own `_get_metadata_only` returning a dict), `notebook/ipython/admin.py` (verbatim export/import passthrough), and `ui/explorer.py`'s `list_entries()` (backend *listing* shape, which carries `size`) intentionally keep reading the opaque dict — they are not part of the typed surface.

### Alternatives Considered
- **Single shared dataclass for both layers**: rejected — the decorator and statement layers carry genuinely different field sets; a union type would be mostly-`None` either way and obscure which fields each layer owns.
- **Dataclass all the way into the backends**: rejected — it would force every backend to know the schema and would break the polymorphic channel that lets two unrelated shapes share one storage path.

---

## ADR-015: Consumable Producer Re-execution

**Status:** Accepted
**Date:** 2026-07-15
**Context:** An isolated re-run of a cell that drains a live object read the leftovers of its *own* previous run: a drained `queue.Queue` printed `got=[]` and an exhausted generator totalled `0`, where `run_all` — which re-runs the producer first — gives `got=[0, 1, 2]` / `total=55` (CAS-118 / CAS-50). Two existing guards could not catch it. The stale-value guard only ever examines variables the cell **writes** (it returns early when the self-written set is empty), and the object in question is a read-only **input**. The content-base staleness check could not have caught it either: a consumable drains *in place*, so its identity never changes and `compute_hash`'s `sha256(str(id(obj)))` fallback for unpicklable objects returns the same hash before and after draining.

### Decision
Add a `consumables.py` module that classifies an object as **consumable-unrestorable** only when **both** signals hold — it is a self-iterator (`iter(obj) is obj`) **and** it hits the cache store's by-ref fallback — then probes *divergence* per type against a baseline recorded at the consumer cell's **entry** (`TrackingState.consumable_bases`). A diverged input schedules its producer **and** the statements that fill it (`_schedule_consumable_producer_touches` in `upstream/reexecution_planner.py`).

### Rationale
- **Two signals, because each alone is wrong in a different direction.** "Self-iterator" alone over-classifies: `iter(range(6))` is a self-iterator but is perfectly restorable — it is literally probe #14 of the 14 generator over-invalidation probes. `io.StringIO` is deep-copyable too, so an `io.IOBase` type test over-classifies for the same reason. "Unpicklable" alone over-classifies the other way: a `dict.keys()` view is not deep-copyable but re-iterating it works fine. Only the AND flags exactly the objects the store hands back already-drained. `map` / `zip` / `filter` / `enumerate` / `iter(list)` / `reversed` must stay unflagged or their producers re-run for nothing.
- **`__reduce_ex__(4)` probing instead of `deepcopy`.** `InMemoryBackend._safe_deep_copy` decides the by-ref fallback with `copy.deepcopy`, but deep-copying is far too expensive as a *classifier*: a `map` over a 2M-element list deep-copies the whole list (~0.1s) where `__reduce_ex__` is ~3µs, because it only *describes* how to rebuild the object rather than rebuilding it. `__reduce_ex__(4)` is the protocol `deepcopy` itself consults for objects without `__deepcopy__` / `__copy__`, and it was verified to agree with `deepcopy` on every iterator type in remit. It disagrees on exactly one: `queue.Queue`, a plain Python object whose `reduce` succeeds while `deepcopy` chokes on its internal `threading.Lock` — handled by an explicit `isinstance` branch rather than by weakening the probe.
- **The cell-entry baseline is what makes this self-disabling.** The probe token is compared against the state the consumer cell saw on its *previous entry*. On `run_all` the producer re-runs first and hands the cell the same state (token == baseline → no-op); on a first run there is no baseline (→ no-op). Only an isolated re-run, where the object still holds the previous run's drained state, diverges. Recording the baseline beside `current_session_hashes` — i.e. at *post-execution* time — would have captured `qsize=0` (the drained state) and the check would never fire.
- **Marking the variable broken is insufficient on its own.** The statements that *fill* a consumable usually do not own it as a trace output. The canonical fill loop
  ```
  q = Queue()                  outputs={'q'}   <- the backward scan schedules this
  for i in range(3):
      q.put(i)                 outputs={'i'}   <- but NOT this
  ```
  has `outputs={'i'}` because `put` ∉ `MUTATING_METHODS` *and* the runtime's mutation verdict skips control bodies (the simulator treats loops as units), so the loop's mutation of `q` is invisible from both sides. Re-running only `q = Queue()` hands the consumer a fresh **empty** queue — turning `got=[]` into `got=[]` again. Hence the extra scheduling pass over statements that draw on or feed a flagged consumable.
- **Probes are kept out of cache-key derivation.** They never touch `source_hash` and never recompute a cache key, honouring the unified-cache-key rule (ADR-007).

### Consequences
- `TrackingState` gains `consumable_bases`, written by `CellExecutor` at cell entry and read by the simulator.
- The upstream channel now examines **read-only inputs**, breaking the previously-safe reading that the guard only looks at variables a cell writes. Any future work on the stale-value guard must not "optimize" the empty-self-written-set early return back into covering this path.
- Scoped to inputs the cell actually **consumes**: `n = q.qsize()` and `print(type(g))` are reporting reads and leave the producer alone.
- The scheduling pass is scoped to `consumable_broken_vars` — vars this run's probe actually flagged — so no other broken variable's plan changes.
- **Deliberate non-goal: opaque `itertools` cursors.** `cycle` / `chain` / `tee` keep their cursor entirely in C with no observable handle, so `consumable_state` returns `None` and the policy is "report NOT diverged, leave the producer alone". The only alternative would be to assume divergence and re-execute their producer on *every* isolated re-run — trading a silent wrong answer for unconditional recompute of anything that touches them. That is a real but narrower gap of the same family as CAS-50, tracked as CAS-122 rather than paid for by every cell here. (`itertools.count` is the exception: it renders its next value via `repr`, so it *is* probeable.)
- Regression corpus green at adoption: 2704 stress, 1634 unit, and the 14 generator over-invalidation probes.

### Alternatives Considered
- **Type allow-list (`isinstance` against a fixed set of iterator types)**: rejected — it cannot express the `StringIO`-is-fine / `Queue`-is-not distinction, which is about *restorability*, not type. An `io.IOBase` test flags `StringIO`, which restores correctly.
- **`deepcopy` as the classifier**: rejected on cost (~40000× slower on a `map` over a large list) for an identical verdict on every type in remit.
- **Content-hashing the consumable**: rejected — it cannot work by construction. Draining is in-place, so identity and therefore the `id()`-based hash fallback are unchanged.
- **Marking the variable broken and relying on the existing backward scan**: rejected — demonstrated insufficient (see Rationale); the fill statements are not scheduled.
- **Assume-diverged for unprobeable cursors**: rejected for now — unconditional producer recompute on every isolated re-run is a worse default than a narrow, documented gap. See CAS-122.

---

## ADR-016: Derivation/Alias Edge Store

**Status:** Accepted
**Date:** 2026-07-15
**Context:** Some objects hold a *live* reference to another object that lineage tracking never models: a numpy **view** (`v = a[100:200]`, where `v.base is a`) — mutating `v` in place mutates `a` — and a pandas **ref-holder** (`g = df.groupby('k')`, `r = df.rolling(3)`), where `g.obj is df` — mutating `df` in place changes what `g` aggregates. Lineage freezes each variable's hash at *creation*, so a later in-place mutation of one side never bumps the other and a downstream consumer serves a stale cached result (CAS-115 / CAS-89).

### Decision
Keep an explicit **derivation edge store** on `TrackingState` (`derivation_edges`), shaped `bump_source_var -> {vars_to_bump_when_source_bumps}`, in `statement/derivation_edges.py`. Split the work in two: **detection** at runtime only (it must observe live `.base` / `.obj` identity) and **replay** in both the runtime and the upstream simulator (which never executes user code and only reads the recorded edges).

### Rationale
- **The existing lineage graph structurally cannot carry this.** The `LineageStore` resolve ladder is `virtual → store → value._cash_lineage_hash → compute_hash_fn → sha256(str(value))`. For a tracked variable `g`, the store hit returns `variable_lineage[g]` and the ladder **short-circuits before it would ever content-hash** — so a content-based check can never observe that `df` was mutated underneath `g`. The alias relationship has to be recorded out-of-band at creation time, when `g.obj is df` is observable, or it is unrecoverable later.
- **Identity, not type, is the signal.** Detection walks `.base` / `.obj` by object identity. A `.copy()` gives numpy `base is None` and pandas an independent frame, so no edge is recorded — that is the over-invalidation guard.
- **Replay must be byte-identical across runtime and simulator.** The bump derives the new hash *deterministically from existing lineage strings only* (`sha256(f"{old}:{src_lineage}")`) and never recomputes `source_hash`, so the two paths cannot drift. Lineage hashes are a separate artifact already computed outside `compute_cache_key()`, so this does not violate ADR-007.
- **Creation is not mutation.** A bump is suppressed when the target is an *input* of the current statement: at creation (`v = a[...]`) the base `a` IS an input, so a plain view creation does not invalidate the base; at mutation (`v[:] = 9`) the base is not an input, so it is bumped.
- **The `93979b7` narrowing to named-base views is load-bearing.** The first cut treated any `value.base is not None` as an uncacheable alias. But `np.linspace` — and in some builds `np.arange` and ufunc results — carry a non-`None` `.base` pointing at an **anonymous internal buffer** that no user variable references. Restoring an independent copy of those is perfectly correct, so the crude test wrongly forced `skip_cache` on plain arrays and broke legitimate caching (it regressed `test_cache_plot_data`). An array is only an uncacheable alias when its `.base` chain — or a groupby/rolling `.obj` — resolves to a **named** `user_ns` variable that something else could mutate.

### Consequences
- `TrackingState` gains `derivation_edges`; `statement/lineage.py`, `statement/processor.py`, and `upstream/virtual_lineage.py` all read or write it.
- Views and ref-holders whose base is a named live variable are **never restored from cache** — they must re-derive from the live base, because pickling and restoring them breaks the reference identity (the restored object would alias a stale *copy*).
- `bump_derived_lineages` returns the set of bumped vars, which the simulator unions into the statement's `outputs`. Without that, the reexecution planner would not record the mutation statement as a producer of the aliased base, and an isolated re-run would restore the base to its stale pre-mutation cache while orphaning the mutation statement.
- Edges are cleared on reassignment (`clear_edges_for`), or `g = other` would keep a dead `df -> g` edge pointing at an object that is no longer a live alias.
- Bumps cascade transitively with a visited set (view-of-view, groupby-of-…); targets no longer present are pruned lazily.
- numpy and pandas stay **soft dependencies** — both are lazy-imported inside the detectors.

### Alternatives Considered
- **Ride the existing lineage graph**: rejected — structurally impossible; the resolve ladder returns the stored lineage before any content check could notice the mutation (see Rationale).
- **Content-hash views/ref-holders on every access**: rejected — it would pay a hash on every statement to catch a rare case, and for a large base array the hash is sampled anyway, so it is not even reliably correct.
- **Treat any non-`None` `.base` as an alias**: rejected — over-classifies fresh arrays with anonymous internal buffers (`np.linspace`, ufunc results) and breaks legitimate caching. This was tried and reverted in `93979b7`.
- **Make views uncacheable wholesale**: rejected — far too broad; most views are created and read without anyone mutating the base.

---

## ADR-017: Track Global-RNG Dependencies by Runtime Observation, Not Static Analysis

**Status:** Accepted — implemented (`5c5fdf2`)
**Date:** 2026-07-21
**Context:** How should cash know which cells depend on a global random seed, so that editing a `np.random.seed(...)` / `random.seed(...)` cell invalidates and correctly refreshes the draws that depend on it?

This ADR is the chosen direction for CAS-225. It is a design, not yet implemented.

### The problem

Two independent round-11 testers hit the same defect. Given:

<!-- test:skip reason="illustrative: two-cell notebook fragment, not runnable as one block" -->
```python
np.random.seed(0)          # cell 1 — edit to seed(1), do NOT re-run it
x = np.random.rand(10**6)  # cell 2 — run this alone
```

editing cell 1 and running only cell 2 returns a value matching **no** clean run (seed-1's *second* draw, or a stale seed-0 value), silently. A clean top-to-bottom run with `seed(1)` would give seed-1's *first* draw.

This is worse than running with caching off, and the "worse" is cash-specific. Cash's core pitch is *"you don't need to re-run upstream cells — just run the one you care about."* That pitch trains the user into exactly the workflow that breaks here; a caching-off user has no such pitch and habitually re-runs the edited seed cell, getting the right answer. So cash's promise makes the wrong outcome **more likely**.

### Why the current design cannot fix it

Randomness is handled by **static AST analysis** (`RandomnessVisitor` / `get_drawing_rng_modules` scan the cell source for `np.random.*` calls) plus post-hoc **state capture** (`capture_rng_state` snapshots the global RNG after a statement, for replay on a cache hit — CAS-90). CAS-223 additionally keys a draw on the *seed epoch* — the cache key of the last-executed seeding statement.

Static analysis has a coverage hole that no amount of more analysis closes: it only sees `np.random.rand()` written **directly in the cell**. It is blind to draws **inside called functions** — `model.fit(X, y)` (sklearn draws internally), or any helper `def simulate(): return np.random.rand(...)`. The calling cell's AST shows `fit(...)` / `simulate()` and no RNG, so cash does not even know the cell consumes randomness.

Two fixes were considered and rejected by experiment:
- **CAS-223's seed epoch** keys the draw on the *last-executed* seed statement. When the seed cell is edited but **not** re-run, the epoch is unchanged, so the draw is not invalidated.
- **Injecting a variable edge** (making a bare `seed()` behave like it binds a variable the draw reads) was prototyped with a *real, visible* variable edge — the draw literally read a variable set in the seed cell. It still failed: editing the seed and running only the draw returned seed-1's second draw (`0.7203244934421581`). cash re-executed the **downstream draw** but not the **seed cell**, so the global RNG was never re-seeded. This proves the missing piece is not the edge but the **replay of the seed's side effect**.

This is inconsistent with how cash handles **files**, which are runtime-instrumented: `file_tracker` patches `open`/`read_csv`/etc., and records path + content-hash as a dependency when the cell actually reads the file. Files get an observer; randomness does not.

### Decision

Adopt a **runtime-observer** model for global-RNG dependency tracking, mirroring the file tracker, in two halves. Both are required — the probe above proves half 1 alone is inert.

**Half 1 — observe the seed → draw dependency at runtime.**
- **Patch only `seed()`** (`numpy.random.seed`, `random.seed` — two functions, same lifecycle as `tracked_open`, installed on `%cash_on` and removed on `%cash_off`). On call, record the *active seed per module*: `(module, seed_value, seeding_cell_key)`.
- **Detect draws by state-diff, not by patching every draw function.** cash already snapshots the global RNG state *after* each statement; add a *before* snapshot. `before != after` ⇒ the statement consumed randomness, wherever the draw physically happened — direct **or inside a called function**. This is what closes the coverage hole.
- **Record a new dependency kind** on the drawing statement's metadata: a reference to the active seed `(seed_value, seeding_cell_key)`, stored the way a file dependency stores path + content-hash. Editing the seed cell changes the seeding statement's source, hence its key, so the stored dependency no longer matches → the draw is invalidated. This is the "hidden variable, observed at runtime" idea.

**Half 2 — replay the seed's side effect during reconstruction.**
- When the reconstruction planner schedules a draw whose seed dependency is stale, it must re-establish the seed *before* the draw. The seeding statement is a pure side effect (it binds nothing to restore), so it must be **re-executed**, not restored. The observed `(seeding_cell_key)` gives the planner the exact producer statement to replay in order.
- Storing the *observed* seed value does not shortcut this: the observed value is the **old** seed; the user edited it to a new one, which only re-running the edited seed statement produces.

### Rationale

- **Closes the coverage hole.** Runtime observation catches draws inside `fit()` and user helpers that static analysis structurally cannot see. This is a correctness win independent of the edit-seed bug.
- **Consistent with files.** Same observer pattern, same patch lifecycle, same "record what actually happened at runtime" philosophy (ADR-002's spirit: observe reality, don't predict it from source).
- **Precise.** Only cells that *actually* drew become consumers; only the cell that *actually* set the active seed becomes the producer. No over-approximation from "any cell mentioning np.random."
- **Cheap.** The extra work is one ~2.5 KB state memcpy per statement (the "before" snapshot) and a two-function patch. Draw functions are not wrapped.

### Consequences

- **Statement metadata gains a new dependency kind** (`rng_seed_dep` or similar) alongside `file_dependencies`. Freshness checking (`statement/freshness.py`) grows a branch that compares the stored seed reference against the currently-active one, exactly parallel to the file-dep branch.
- **The reconstruction planner (`upstream/reexecution_planner.py`) gains a "replay this statement for its side effect, in order" capability.** This is the load-bearing, fragile half — the same subsystem whose reset-lineage branch is load-bearing for 500+ integration tests. It must be built baseline-first with the real-driver oracle and a full before/after integration run.
- **Interacts with, and largely subsumes, CAS-223's seed epoch.** The runtime dependency is a superset of the static epoch; CAS-223 should be kept as the same-cell / re-run fast path and reconciled so the two do not double-invalidate.
- **Global RNG only.** Named `np.random.default_rng(SEED)` generators already work through ordinary variable lineage (verified round 11) and are out of scope. So is a generator's **stream position across multiple drawing cells**, which remains the documented Generator limitation — this ADR fixes seed *changes*, not stream bookkeeping.
- **Patching risk** is the same class cash already accepts for files: thread-safety of the global singleton, correct un-patching on `%cash_off`, and not disturbing user code that introspects `np.random.seed`.

### Interim behaviour until implemented

Docs corrected (`81eb312`) to describe the gap accurately. Three workarounds hold: re-run the seed cell after editing it (correct via CAS-223), seed in the same cell as the draw, or use a named `default_rng(SEED)`. A cheaper partial step, if the full replay proves too risky, is to use half 1's precise dependency to at least **warn** ("a draw depends on a seed cell you edited but did not re-run") instead of silently serving a wrong value — strictly better than today even without half 2.

### Alternatives Considered

- **Keep static analysis, extend the AST walker**: rejected — cannot see draws inside called functions, which is the dominant real case (sklearn, helpers). No amount of static work closes it.
- **CAS-223 seed epoch alone**: rejected — keys on the *last-executed* seed, so edit-without-re-running is invisible. Kept as a fast path, not the whole answer.
- **Inject a synthetic variable edge and rely on existing reconstruction**: rejected by experiment — even a real variable edge did not make reconstruction re-run the seed; the draw recomputed on stale global state.
- **Patch every draw function** instead of state-diffing: rejected — numpy exposes dozens of draw entry points; state-diff observes the same fact (the stream advanced) with a two-line snapshot and no per-function wrapping.
- **Warn-only, never reconstruct**: rejected as the *final* design (it leaves the value wrong), but accepted as a legitimate **interim** step because a precise runtime dependency makes the warning reliable, and a loud wrong-value beats a silent one.

### Implementation progress

- **Detection core landed** (`aeeabdc`): `randomness.seed_cells_not_yet_run(drawing_modules, notebook_cells, executed_cell_hashes)` — pure, cell-granular, 11 unit tests, zero integration surface. This is the testable heart of half 1.
- **Feasibility settled two things by experiment.** (1) Correctness genuinely needs half 2: an expensive cached draw *with a real variable edge* to the seed cell still returned a third wrong value (`0.848…`) on edit-without-rerun — reconstruction replays the draw, not the seed. (2) The wiring must cross a **granularity seam**: the processor hashes per *statement*, the notebook/checker sees whole *cells*. `executed_cell_hashes` must therefore be a set of whole-cell source hashes recorded by the cell executor — cash does not track that yet (only per-statement and per-variable hashes exist). Adding it is the first wiring step.
- **Implemented (`5c5fdf2`).** The correctness fix landed directly (not the warn-only interim). The cell executor records `sha256(raw_cell)` into `TrackingState.executed_cell_source_hashes`; the checker's `_prepend_stale_seed_cells` runs after `simulate_upstream`, and when the current cell draws and an **upstream** seed cell's source is absent from that set, it prepends the seed cell to `statements_to_reexecute` so the seed's side effect is re-established before the draw.
- **Correction to the assumption above:** nb_runner is **not** blind to this bug class. It writes a real `.ipynb` and `set_cell_source` persists an edit without running the cell, so the reconstruction path is reachable — only the *discovery* bugs (CAS-218) are invisible to nb_runner (it injects the path). So verification is plain pytest with an in-process oracle (the same cell sources run without cash), not the real driver. That correction is what made the fix landable with confidence.
- **Not done:** the general runtime *observer* (patching `seed()`, state-diff draw detection) for draws **inside called functions** is still future work — the current fix uses the existing static draw/seed detection, which covers the reported cases. And CAS-226 (position-unaware epoch) is a related follow-up.
- **Superseded by ADR-018** for the general position-awareness problem. ADR-017's seed-replay (`5c5fdf2`) stays as a working point fix, but CAS-226 and CAS-227 showed the side-channel model is the root issue; ADR-018 replaces it with a position-aware virtual variable.

---

## ADR-018: Model the Global RNG as a Position-Aware Virtual Variable

**Status:** Accepted — first increment implemented (`66e32dc`); full virtual-variable model still to do
**Date:** 2026-07-21
**Context:** CAS-226 (a draw above a later seed keys on that later seed) and CAS-227 (re-executing an edited draw uses the current stream position, not the position it holds top-to-bottom) are both symptoms of one thing: the global RNG state — *seed epoch and stream position* — is tracked by **runtime side-channels** instead of cash's position-aware reconstruction.

* `_rng_seed_epochs` (CAS-223) records the *last-executed* seed — time-ordered, not position-ordered.
* `capture_rng_state`/`restore_rng_state` (CAS-90) store each statement's POST state and replay it on a cache *hit*. On a full in-order Run All that keeps the stream coherent; on a partial/out-of-order re-execution the live state is whatever was last left.

Ordinary variables never have this problem, because they flow through the lineage graph: the simulator, before re-executing a statement, reconstructs each input to the value it holds *at that position*. The RNG was bolted on beside that machinery instead of into it.

### Decision

Model the global RNG state, per module, as a **virtual variable** that rides the existing variable machinery.

* A **seed** statement *defines* it (output only): `__cash_rng_<module>__ = <fresh state>`.
* A **draw** statement *reads and modifies* it (input **and** output): consumes the state, advances it.
* Its **value is the RNG state**; its accessor is `capture_rng_state` and its mutator is `restore_rng_state` — **not** a `user_ns` slot. This is the one place the virtual variable differs from a real one, and the one real wiring gap (below).
* **Detection:** static for direct calls (`RandomnessVisitor` already sees `np.random.rand()` / `.seed()` in the cell); **runtime state-diff** for draws inside called functions (`model.fit()`), observed once and recorded like a file dependency. This is why the dependency "is only known after it executes at least once" for the indirect case — exactly the file-tracker pattern.

Then the position-aware machinery does the rest for free:

* The virtual variable's **lineage chains in notebook order**, so a draw keys on the RNG state governing *its* position — **CAS-226 dissolves**.
* Reconstruction **restores the virtual variable to its at-position value before re-executing a consumer**, so an edited draw re-runs from the correct stream position — **CAS-227 dissolves**.
* CAS-225's seed-replay becomes the ordinary "reconstruct the producer of the virtual variable."

### The one confirmed gap

The upstream reconstruction path (`virtual_lineage._restore_vars_from_cache`) restores `user_ns` variables; it does **not** apply `rng_state` when restoring a *producer* as an upstream dependency (RNG state is restored only on a statement's own hit today). So the make-or-break work is teaching the restore path that the virtual RNG variable's "restore" means `restore_rng_state(value)`, not a namespace write. Once that exists, the variable rides lineage + reconstruct-inputs-before-executing unchanged, and CAS-223's global epoch + CAS-90's on-hit-only replay both fold into this one model.

### Consequences

* Subsumes CAS-223 and CAS-90 into a single model; the epoch dict and the on-hit replay become special cases of "restore the virtual variable at its position."
* A long draw chain (draw3←draw2←draw1←seed) is a linear dependency, but each restore is O(1) (the preceding state is cached, never recomputed) — no performance cliff.
* The virtual variable must never surface in user-visible state (`%who`, saved vars).
* Unseeded draws still track the variable but remain non-reproducible across kernels (matches today's warning).

### Rationale

Reuses the position-aware reconstruction cash already trusts for variables, instead of a parallel RNG code path. Fixes CAS-226 and CAS-227 uniformly rather than as separate patches, and the indirect-draw coverage hole (ADR-017 half 1) is closed by the same runtime state-diff.

### Alternatives considered

* **Keep the side-channels, patch each symptom** — rejected: whack-a-mole, and the stream-position case (CAS-227) needs a position-aware restore regardless, which is 90% of this work.
* **Store each draw's PRE state and restore it on re-execution** — rejected as the model: a cache *miss* doesn't load the old entry, so the pre-state isn't available; and a stale pre-state (upstream changed) would be wrong. Reconstructing from the producer is the correct source of the pre-state. (Usable only as a spike shortcut, see below.)

### Spike (before committing to the refactor)

Prove the core claim: restoring the position-correct RNG state before a *recomputing* draw makes CAS-227 green **without** disturbing the CAS-223 or CAS-225 suites. Cheapest proof — record each random cell's POST state in memory (keyed by cell source) and, before a drawing cell re-executes, restore the POST state of the immediately-preceding upstream random cell. If that holds, build the real virtual-variable wiring; if it breaks the CAS-223 suite, the model needs rework before any refactor.

### First increment implemented (`66e32dc`)

The spike held and shipped as the first increment: `TrackingState.rng_post_states` (cell executor snapshots the RNG after each random cell), and `checker._restore_position_rng_state` restores the nearest upstream random cell's post-state before a drawing cell re-executes. **Fixes CAS-226 and CAS-227.** Verified with nb_runner + the same-content-no-cash oracle; notebook+core 2509 passed; full deterministic integration clean (only the pre-existing zzprobe records).

The spike also **found its own limitation**, exactly as the "store the PRE state" alternative predicted: the post-state table cannot reconstruct across a *seed change*. So the combined case — edit a seed **and** have intervening draws before the re-run draw — can't be fully corrected by a snapshot (the intervening draws' post-states are stale under the old seed). A guard makes `_restore_position_rng_state` **defer to the CAS-225 reseed path when an upstream seed is stale**, so that case falls back to the new seed's stream and is never made *worse*, but it is not yet fully correct (it lands on the new seed's position 0, not the position the intervening draws would have advanced it to).

**Still to do (the actual virtual variable):** make each draw depend on the per-module RNG variable in the *lineage graph*, so a seed edit invalidates the whole downstream chain and reconstruction re-runs the intervening draws in order. That closes the combined case, makes CAS-226's key position-aware (turning its spurious miss into a hit), and — via the runtime state-diff detector — extends coverage to draws inside called functions. The `66e32dc` increment is the position-aware *restore*; the remaining work is the position-aware *invalidation + chain reconstruction*.
