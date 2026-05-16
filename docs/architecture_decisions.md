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
- A transitional `__getattr__` / `__setattr__` pair on `UpstreamChecker` forwards method calls and known shared-state assignments to the simulator. This preserves the existing test API (`checker._update_virtual_lineage(...)`, `checker.executed_file_deps = {}`) without rewriting dozens of tests. Tests should migrate to `checker.simulator.<method>` over time; once they have, the shim can be removed.
- A handful of class-level method-existence tests (`hasattr(UpstreamChecker, '_validate_file_freshness')`) were updated to point at `NotebookSimulator`. Those are *the* tests that should always look at the new home.
