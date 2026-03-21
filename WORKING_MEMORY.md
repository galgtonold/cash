# Working Memory - Current State

## Last Updated: 2026-02-12

## Current Status

### Bug Fix F6: Cell Edit Cache Invalidation (RESOLVED)
**Symptom:** Editing a cell to add a mutation call (e.g., `bus.on(handler)`) didn't invalidate
cached results for other statements in the same cell. The `bus.emit('hello')` line would get
a cache hit and restore stale pre-mutation results.

**Root Cause (Two Layers):**
1. **ALREADY_EXECUTED optimization** skipped statements whose individual code matched their
   previous version, even though the overall cell had changed.
2. **Cache hits** restored stale values because mutation calls (e.g., `bus.on(...)`) don't
   change the object's lineage hash. The cache key for downstream statements matched.

**Fix:** Added `force_recompute` flag in `statement_processor.py`:
- Set to `True` when `cell_code_changed=True` (tracked via `_executed_cell_raw_codes` set in `magics.py`)
- Disables **both** ALREADY_EXECUTED skip **and** cache lookups for all statements in the edited cell
- Cache **storage** is still enabled so new results get cached for future runs
- Files modified: `magics.py`, `statement_processor.py`, `conftest.py`
- Unit tests: `tests/test_notebook/test_cell_code_changed.py`

### All Major Features Complete ✅
- **Decorator-Notebook Bridge** (12 features): Committed across b693610, ea7175f, f482e22
- **Bug Fix #1**: Transitive invalidation (71b7024)
- **Bug Fix #2**: Qualname collision via module-qualified keys (9b016f0)
- **Roadmap Phases 1-6**: All implementable items complete

### Test Stats
- **Total tests**: 2308 collected
- **Unit tests**: 1063 passed, 3 skipped
- **Integration tests**: 334 passed, 1 xfail (design limitation)
- **Coverage**: 77% overall (notebook modules)

### Coverage by Module
| Module | Coverage | Target | Gap |
|--------|----------|--------|-----|
| mutation_detector.py | 98% | - | ✅ |
| annotations.py | 91% | - | ✅ |
| purity.py | 90% | - | ✅ |
| side_effects.py | 90% | - | ✅ |
| function_tracker.py | 88% | 92% | 4% |
| cache_key.py | 87% | - | ✅ |
| file_tracker.py | 87% | - | ✅ |
| analysis.py | 86% | - | ✅ |
| randomness.py | 86% | - | ✅ |
| statement_processor.py | 81% | 85% | 4% |
| control_structures.py | 79% | 80% | 1% |
| magics.py | 76% | 80% | 4% |
| upstream.py | 62% | 70% | 8% |
| audit.py | 39% | 50% | 11% |
| provenance.py | 34% | 50% | 16% |

### Remaining Roadmap Items (All Deferred/External)
- JupyterLab/VS Code extensions (separate repos)
- Distributed caching, smart cache warming, incremental computation
- Multi-tenancy, cost optimization
- Communication channels, marketing materials
- Statement-level caching inside function definitions
- Parallel simulation

## Next Priority: Test Coverage Improvements
Focus on upstream.py (62% → 70%) and magics.py (76% → 80%) as highest-impact targets.
