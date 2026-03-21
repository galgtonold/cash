# Decorator-Notebook Bridge: Feature Roadmap & Progress

**Created:** 2026-02-10
**Goal:** Bridge `@cash.cache` decorator with notebook caching for seamless integration, plus production-level hardening features.

---

## Overview

The `@cash.cache` decorator and notebook statement caching (`%cash_on`) are currently two completely separate systems:
- **Decorator** (`core.py`): Function-level caching via `pickle.dumps((args, kwargs))` → SHA256 key
- **Notebook** (`statement_processor.py`): Statement-level AST-based caching with lineage tracking

This initiative bridges them and adds production-level features.

---

## Feature Roadmap

### Phase A: Decorator-Notebook Integration
| # | Feature | Status | Tests |
|---|---------|--------|-------|
| A1 | Decorator call tracking in notebooks | ✅ DONE | ✅ 8 unit + 2 integration |
| A2 | Badge display for `@cash.cache` calls | ✅ DONE | ✅ 7 unit + 1 integration |
| A3 | Condensed badge for many decorator calls | ✅ DONE | ✅ in A2 tests |
| A4 | Decorator metrics in cell summary | ✅ DONE | ✅ in A2 tests |

### Phase B: Non-Hashable Parameter Handling  
| # | Feature | Status | Tests |
|---|---------|--------|-------|
| B1 | Custom hasher registry for types | ✅ DONE | ✅ 4 unit + 1 integration |
| B2 | Built-in hashers (DataFrame, ndarray, Series) | ✅ DONE | ✅ 4 unit + 2 integration |
| B3 | Graceful fallback for unhashable types | ✅ DONE | ✅ 4 unit |

### Phase C: Production Hardening
| # | Feature | Status | Tests |
|---|---------|--------|-------|
| C1 | Source-aware cache invalidation | ✅ DONE | ✅ 5 unit + 1 integration |
| C2 | DataFrame library onboarding (polars, modin, dask) | ✅ DONE | ✅ 10 unit |
| C3 | File dependency tracking for decorator | ✅ DONE | ✅ 5 unit |
| C4 | Thread safety for decorator call log | ✅ DONE | ✅ 2 unit |
| C5 | Decorator cache statistics & introspection API | ✅ DONE | ✅ 6 unit |

---

## Detailed Design

### A1: Decorator Call Tracking in Notebooks

**Problem:** When a notebook cell calls `result = my_func(df)` where `my_func` is decorated with `@cash.cache`, the notebook's statement processor treats this as a regular function call. The decorator cache hit/miss is invisible to the notebook caching system.

**Solution:** Add a notification mechanism to the `Cash.cache` wrapper that records decorator call events. The `StatementProcessor` can then collect these events after executing a statement.

**Implementation:**
1. Add `_decorator_call_log` list to `Cash` class
2. In `wrapper()`, append call event with: func_name, cache_hit/miss, execution_time, args_hash, cache_key
3. In `StatementProcessor._execute_statement()`, check for decorator events after execution
4. Include decorator events in statement metrics for badge rendering

### A2-A3: Badge Display for Decorator Calls

**Problem:** Decorator cache hits are invisible in the notebook badge.

**Solution:** Add a new badge row type for decorator calls. When a statement has many decorator calls (e.g., in a loop), condense them into a collapsible group.

**Implementation:**
1. New metrics key `decorator_calls` in statement metrics
2. Badge renderer handles `decorator_calls` list
3. If >3 calls to same function, condense into "⚡ my_func: 10 calls (8 cached, 2 computed)"
4. Expandable details show individual calls

### B1-B3: Custom Hasher Registry

**Problem:** `_serialize_args` uses `pickle.dumps` which fails for many types (e.g., generators, file handles, database connections, custom classes without `__reduce__`).

**Solution:** Type-based hasher registry with fallback chain.

**Implementation:**
1. `TypeHasherRegistry` class with `register(type, hasher_fn)` method
2. Built-in hashers for: DataFrame (use `_cash_hash` or hash columns+shape+dtypes+sample), ndarray (tobytes for small, shape+dtype+sample for large), Series, dict (sorted items), set (sorted), list (recursive)
3. Fallback chain: `_cash_hash` attr → registered hasher → `pickle.dumps` → `repr()` → skip cache
4. `Cash.register_hasher(type, fn)` public API

### C1: Operation-Based Lineage

**Problem:** DataFrames lose lineage after operations not tracked by cash.

**Solution:** Track operation history on DataFrames via a lightweight wrapper.

### C2: DataFrame Library Onboarding

**Problem:** Only pandas DataFrames get `_cash_hash` attribute attachment. Other DF libraries (polars, modin) are not supported.

**Solution:** Extend `_attach_lineage` to detect and handle multiple DataFrame types.

### C3: File Dependencies for Decorator

**Problem:** `FileDataSource` works but requires manual `depends_on=[FileDataSource("path")]`. 

**Solution:** Already works - document and test more thoroughly.

---

## Progress Log

### Session 1 (2026-02-10)
- [x] Explored full codebase architecture
- [x] Identified the gap: decorator and notebook systems are completely separate
- [x] Created this roadmap document
- [x] Implemented Phase A (decorator-notebook integration): call tracking, badge display, condensed view, text mode
- [x] Implemented Phase B (non-hashable params): custom hasher registry, built-in hashers, graceful fallback
- [x] 30 unit tests + 10 integration tests, all passing
- [x] Full regression: 966 unit + 143 integration tests pass
- [x] Committed as b693610
- [ ] Starting Phase C: source-aware invalidation, DataFrame library onboarding, file deps, thread safety, stats API

### Session 2 (continued)
- [x] Phase C1: Source-aware cache invalidation
  - Root cause: inspect.getsource() fails for functions defined in IPython cells intercepted by %cash_on
  - Fix: bytecode fallback using __code__ object (co_code, co_consts, co_names, co_varnames)
  - Fixed in both CodeAnalyzer.get_source_hash() and FunctionTracker.get_function_source_hash()
  - Also fixed cache eviction in bytecode fallback path
- [x] Phase C2: DataFrame library onboarding
  - Extended _builtin_type_hash for: polars Series, polars LazyFrame, PyArrow Table/RecordBatch, modin DataFrame, dask DataFrame
  - Changed type detection from type_name to module-based dispatch (more reliable)
  - Extended _attach_lineage for: polars Series, modin DataFrame, PyArrow Table
- [x] Phase C3: file_depends_on parameter for @cash.cache
  - New `file_depends_on` parameter: shorthand for `depends_on=[FileDataSource("path")]`
  - Accepts single string or list of strings
  - Auto-creates FileDataSource and registers in dependency graph
- [x] Phase C4: Thread safety for _decorator_call_log
  - Added threading.Lock to _decorator_call_log
  - _log_decorator_call() and drain_decorator_calls() are now thread-safe
- [x] Phase C5: cache_info() and cache_clear() introspection API
  - cache_info() returns {hits, misses, hit_rate, total_time_saved}
  - cache_clear() resets stats and removes entries from backend
  - __wrapped__ attribute preserved for inspect compatibility
- [x] Updated API stability test for register_hasher export
- [x] Full regression: 994 unit + 143 integration tests pass
- [x] Committed as Phase C (ea7175f)

### Session 3 (continued)
- [x] Added 5 Phase C integration tests: polars, file deps, cache_info, cache_clear
- [x] Fixed integration test bugs:
  - f-string escaping in file deps test
  - cache_info test: statement-level caching intercepts before decorator
- [x] Final regression: 994 unit + 328 integration + 95 auxiliary = **1417 total tests passing**
- [x] Committed as f482e22

## Final Summary

### All Phases Complete ✅

| Phase | Features | Unit Tests | Integration Tests |
|-------|----------|------------|-------------------|
| A | Decorator-Notebook Integration (4 features) | 15 | 8 |
| B | Non-Hashable Parameter Handling (3 features) | 12 | 3 |
| C | Production Hardening (5 features) | 31 | 4 |
| **Total** | **12 features** | **58** | **15** |

### Bug Fixes (Post-Completion)
| # | Bug | Commit | Fix |
|---|-----|--------|-----|
| 1 | Transitive invalidation: changing `dep()` in module didn't invalidate `fun()` which calls `dep()` | 71b7024 | Added `expand_changed_symbols_transitively()` BFS reverse closure in FunctionTracker; moved `recently_reloaded_modules.clear()` to start-of-next-cell |
| 2 | Qualname collision: notebook `dep()` and module `dep()` shared dict keys, causing source hash overwrites and skipped analysis | 9b016f0 | Introduced `Cash._get_func_key(func)` returning `f"{func.__module__}.{func.__qualname__}"` as unique key; added `_analyzed.discard()` on source hash change |

### Files Modified
- `src/cash/core.py` — Custom hasher registry, built-in type hashers, call tracking, thread safety, file_depends_on, cache_info/cache_clear, module-qualified keys
- `src/cash/__init__.py` — Exported register_hasher
- `src/cash/notebook/statement_processor.py` — Drains decorator calls into metrics
- `src/cash/notebook/magics.py` — Badge rendering for decorator cache events
- `src/cash/notebook/analysis.py` — Bytecode fallback for source hashing, module-qualified find_called_functions
- `src/cash/notebook/function_tracker.py` — Bytecode fallback + eviction fix, intra-module call deps, transitive expansion

### Files Created
- `DECORATOR_NOTEBOOK_BRIDGE.md` — This progress tracking file
- `tests/test_notebook/test_decorator_notebook_bridge.py` — 58 unit tests
- `tests/test_notebook_integration/test_decorator_bridge_integration.py` — 15 integration tests
- `tests/test_notebook_integration/test_decorator_module_reload.py` — 7 integration tests (transitive invalidation + qualname collision)

