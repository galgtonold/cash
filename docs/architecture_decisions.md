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
