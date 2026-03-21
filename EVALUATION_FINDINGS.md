# Cash Caching Framework - Evaluation Findings

**Date**: 2025-02-19  
**Methodology**: Systematic MCP-based Jupyter notebook testing (61 scenarios across 2 rounds)  
**Version**: cash v0.2.0, Python 3.14.3, JupyterLab 4.4.1  

## Summary

| Finding | Severity | Category | Status |
|---------|----------|----------|--------|
| F1 - File dep not propagated through simulation cache | CRITICAL | File Dependencies | ✅ FIXED & VERIFIED |
| F2 - Duplicate expression statements share cache key | HIGH | Caching Correctness | ✅ FIXED & VERIFIED |
| F3 - Badge stuck | LOW | UI/Badge | NOT REPRODUCED |
| F4 - Decorator not in badge | LOW | UI/Badge | NOT REPRODUCED (works correctly) |

### Resolution Summary
- **F1 FIXED**: Modified `upstream.py` to store file deps and stale-file vars in 7-element simulation cache tuple, re-check mtimes during validation. 5 integration tests in `test_file_dep_simulation_cache.py`.
- **F2 FIXED**: Added `occurrence_index` parameter through `cache_key.py` → `statement_processor.py` → `magics.py` → `upstream.py`. 6 integration tests in `test_duplicate_expression_caching.py`.
- **Full regression**: 1009 unit tests + 346 integration tests = 0 failures.
- **Live verification**: Both fixes verified working in real Jupyter notebooks with fresh kernels.

### Round 1 Tests (33 scenarios) — PASSED: 28/33
### Round 2 Tests (28 scenarios) — PASSED: 28/28 (including F1 live verification)

### Tests that PASSED (56/61 total, all 5 failures were F1/F2 pre-fix)
- Basic variable caching (skip/restore) ✅
- Dependency chain propagation ✅
- Direct file dependency detection ✅
- Module import/reload detection ✅
- Loop caching (for, nested, while) ✅
- @cash.cache decorator in notebook ✅
- Out-of-order execution / upstream auto-detection ✅
- Function definition change propagation ✅
- Try/except caching ✅
- Conditional if/else caching ✅
- Accumulator pattern (list.append in loop) ✅
- Counter self-increment pattern ✅
- Dict mutation tracking ✅
- Tuple unpacking ✅
- Comprehensions and complex expressions ✅
- Lambda and higher-order functions ✅
- Variable deletion (del) ✅
- Error handling (exception in cell) ✅
- HTML badge rendering ✅
- Numpy operations with seed ✅
- Cross-cell mutable state propagation ✅
- JSON serialization/deserialization ✅
- Long-running tasks with sleep (badge progress) ✅
- Upstream cell modification detection ✅

---

## F1: File Dependency Changes Not Propagated Through Simulation Cache (CRITICAL)

**Category**: File Dependencies / Upstream Simulation  
**Severity**: CRITICAL  
**Status**: ✅ FIXED & VERIFIED  
**File**: `src/cash/notebook/upstream.py` lines 438-468  
**Fix**: Simulation cache extended from 5 to 7 elements; file mtimes re-checked during validation  
**Tests**: `tests/test_notebook_integration/test_file_dep_simulation_cache.py` (5 tests, all pass)  
**Live Verified**: CSV change (a=[1,2,3]→a=[10,20,30]) correctly detected in downstream cell (mean=20.0)  

### Description
When a file dependency changes (e.g., CSV file is modified), the upstream simulation cache does NOT invalidate. This means downstream cells don't detect the stale data and return incorrect cached results. The bug specifically occurs when:
1. There is a chain: Cell A (read file) → Cell B (self-assignment transform) → Cell C (downstream computation)
2. The file changes
3. Only Cell C is re-executed

The simulation cache uses only cell CODE hash for invalidation. Since no cell code changed (only the file), all cached simulation results are reused, and `vars_with_stale_files` is never populated.

### Root Cause
In `upstream.py:_simulate_and_find_changes()`:
- **Line 430**: `vars_with_stale_files = set()` — starts empty every time
- **Lines 440-461**: Incremental simulation cache checks only `cell_hash` (code content), not file mtimes
- **Lines 457-463**: When restoring from cache, `virtual_lineage`, `virtual_modules`, `simulation_trace`, and `vars_mutated_by_loops` are restored, but **`vars_with_stale_files` is NOT cached or restored**
- **Lines 515-524**: `vars_with_stale_files` is only built during simulation loop — skipped cells never populate it

### Steps to Reproduce
```
Cell 1: fdf = pd.read_csv('test.csv')  # values: 10, 20, 30
Cell 2: fdf = fdf.sort_values('val')   # self-assignment
Cell 3: fdf_sum = fdf['val'].sum()     # downstream
```
1. Run all three cells → fdf_sum = 60 ✅
2. Modify test.csv to contain 100, 200, 300
3. Run ONLY Cell 3 → Expected: 600, Actual: 60 ❌

**Note**: Without Cell 2 (self-assignment), the bug does NOT occur — direct dependency works.

### Expected Behavior
Running Cell 3 should detect that test.csv changed, re-execute Cell 1 and Cell 2 upstream, then compute fdf_sum with new data (600).

### Actual Behavior
Cell 3 uses the old cached fdf value (sum=60) because the simulation cache is reused entirely (no code changed), so `vars_with_stale_files` remains empty, and upstream re-execution is not triggered.

### Fix Plan
1. Include `vars_with_stale_files` in the simulation cache entries (element [5])
2. When restoring from cache, rebuild `vars_with_stale_files` from all cached entries
3. Add file mtime check during cache validation: if ANY cached cell has file dependencies whose mtimes changed, invalidate the cache from that cell onwards
4. Alternative: always re-check file deps during cache restoration, not just during fresh simulation

---

## F2: Duplicate Expression Statements Share Same Cache Key (HIGH)

**Category**: Caching Correctness  
**Severity**: HIGH  
**Status**: ✅ FIXED & VERIFIED  
**Files**: `src/cash/notebook/cache_key.py`, `src/cash/notebook/statement_processor.py`, `src/cash/notebook/magics.py`, `src/cash/notebook/upstream.py`  
**Fix**: Added `occurrence_index` parameter to `compute_cache_key()`, threaded through entire processing chain  
**Tests**: `tests/test_notebook_integration/test_duplicate_expression_caching.py` (6 tests, all pass)  
**Live Verified**: `c.increment()` × 3 → c.value=13 (correct), each call gets unique cache key

### Description
When the same expression statement appears multiple times in a cell (e.g., `c.increment()` called 3 times), the 2nd and 3rd calls are RESTORED from cache instead of COMPUTED. This is because all identical statements produce the same cache key — there is no occurrence counter or positional differentiator.

For **pure side-effect-free** duplicates (like `print('hello')`), this is cosmetically wrong but functionally correct since cached stdout is replayed. For **stateful method calls** (like `counter.increment()`), this causes **data corruption** — the object's state is only modified once instead of the expected number of times.

### Root Cause
In `cache_key.py:compute_cache_key()`:
- Cache key = `stmt:{sha256(code + input_lineages + file_hash)}`
- No occurrence index is included
- When an expression has no outputs (no assignment), the lineage state doesn't change between calls
- So 2nd and 3rd calls compute the exact same cache key → cache HIT from 1st call

### Steps to Reproduce
```python
class Counter:
    def __init__(self, start=0):
        self.value = start
    def increment(self):
        self.value += 1
        return self.value

c = Counter(10)
c.increment()  # COMPUTED → c.value = 11
c.increment()  # RESTORED (BUG) → c.value stays 11
c.increment()  # RESTORED (BUG) → c.value stays 11
print(c.value)  # Expected: 13, Actual: 11
```

### Expected Behavior
Each `c.increment()` call should be COMPUTED because it mutates `c`'s internal state.

### Actual Behavior
Only the first call is COMPUTED. The second and third are RESTORED from cache (same cache key).

### Fix Plan
Add a per-cell statement occurrence counter to the cache key computation. In `magics.py` or `statement_processor.py`, track how many times each unique statement code has appeared within the current cell, and pass the count as an additional component to `compute_cache_key()`.

---

## F3: Badge Gets Stuck (NOT REPRODUCED)

**Category**: UI/Badge  
**Severity**: LOW (could not reproduce)  
**Status**: NOT REPRODUCED  

### Description
User reported that the badge "gets stuck" during execution. Testing 33 scenarios with both HTML and print badge modes, including error handling, long-running tasks, and rapid cell execution, the badge always rendered correctly.

The issue may be:
- Specific to JupyterLab UI rendering (not detectable via MCP)
- Related to a specific sequence of operations not covered in testing
- Intermittent/timing-dependent

### Recommendation
Monitor in future testing. If reproducible, investigate the `display_id` update mechanism in `magics.py`.

---

## F4: Decorator Caching Not in Badge (NOT REPRODUCED)

**Category**: UI/Badge  
**Severity**: LOW (could not reproduce)  
**Status**: NOT REPRODUCED  

### Description
User reported that @cash.cache decorated functions defined in the notebook don't show in the badge. Testing showed that decorator calls ARE properly displayed in print mode: `@cash.cache: slow_compute(): 1/3 cached (1.001s)`.

May be specific to HTML badge mode or a specific notebook configuration.

---

## Test Matrix

| # | Scenario | Result | Notes |
|---|----------|--------|-------|
| 1 | Basic variable assignment | PASS | Skip/restore works |
| 2 | Dependency chain | PASS | z=y+10, w=z*3 propagated |
| 3 | File dep (read_csv) | PASS | Direct file dep detected |
| 4 | File dep downstream | PASS | Changes propagate to df2 |
| 5 | File dep + self-assignment | **FAIL** | F1 bug |
| 6 | Module import/reload | PASS | VERSION change detected |
| 7 | Loop caching (for) | PASS | Per-iteration caching |
| 8 | @cash.cache decorator | PASS | Shown in badge |
| 9 | Out-of-order execution | PASS | Upstream auto-detected |
| 10 | Function def change | PASS | Propagated to callers |
| 11 | Self-assignment file dep (3 cells) | **FAIL** | F1 bug |
| 12 | Try/except | PASS | Branch caching works |
| 13 | If/else conditional | PASS | Branch caching works |
| 14 | Accumulator (list.append in loop) | PASS | Mutations detected |
| 15 | Counter self-increment | PASS | Self-assignment tracked |
| 16 | Nested loops | PASS | Matrix built correctly |
| 17 | Class method calls (assigned) | PASS | r1, r2, r3 correct |
| 18 | Class method calls (unassigned) | **FAIL** | F2 bug |
| 19 | Dict mutation | PASS | += tracked |
| 20 | HTML badge | PASS | Renders correctly |
| 21 | Error in cell | PASS | Badge handles error |
| 22 | Loop with sleep (badge progress) | PASS | Progress shown |
| 23 | Duplicate print statements | PASS* | Output correct, badge misleading |
| 24 | Tuple unpacking | PASS | Multi-assignment works |
| 25 | Upstream modification detection | PASS | base_val change propagated |
| 26 | Variable deletion (del) | PASS | NameError raised correctly |
| 27 | Comprehensions | PASS | List/dict comp work |
| 28 | JSON serialize/deserialize | PASS | Complex types work |
| 29 | File dep 4-cell chain | **FAIL** | F1 bug (clean repro) |
| 30 | Lambda/higher-order functions | PASS | map/filter work |
| 31 | Try/except in loop | PASS | Mutations detected |
| 32 | Numpy with seed | PASS | Reproducible |
| 33 | Cross-cell mutable state | PASS | Mutations propagated |

## Round 2 Test Matrix (Post-Fix Verification)

All tests run on a fresh kernel with fixes applied.

| # | Scenario | Result | Notes |
|---|----------|--------|-------|
| R2-F1 | File dep change detected downstream | PASS | CSV a=[1,2,3]→[10,20,30], mean=20.0 ✅ |
| R2-5 | Nested function with closure (make_adder) | PASS | result5=15 |
| R2-6 | Generator expression (sum of squares) | PASS | total6=285 |
| R2-7 | Tuple unpacking | PASS | result7=6 |
| R2-8 | String formatting with f-string | PASS | msg8='hello world' |
| R2-9 | Exception handling (try/except) | PASS | x9=-1, branch caching works |
| R2-10 | Dict comprehension | PASS | d10={'a':1,'b':4,'c':9} |
| R2-11 | Walrus operator (`:=`) | PASS | result11=42 (note: n11 has no tracked lineage) |
| R2-12 | Sorted list | PASS | Correct ordering |
| R2-13a | Mutable list with .append() | PASS | Mutation detected (🚫 No Cache) |
| R2-13b | Cross-cell read of mutated list | PASS | mutation-updated lineage respected |
| R2-14 | Upstream change detection (x14=10→20) | PASS | y14=60, lineage change detected |
| R2-15 | Re-execution skip optimization | PASS | val15=42^10 SKIPPED on 2nd run |
| R2-16 | Random with seed (no-cache) | PASS | Executed correctly |
| R2-17 | Numpy broadcasting & normalization | PASS | normalized[0]=-1.4142 |
| R2-18 | @cash.cache decorator in notebook | PASS | Badge shows DECORATOR CACHE section |
| R2-19 | DataFrame operations chain | PASS | z_mean=33.0 |
| R2-20 | Nested list comprehension | PASS | flat20=[1,2,3,2,4,6,3,6,9] |
| R2-21 | For loop accumulator | PASS | total21=20, per-iteration caching shown |
| R2-22 | While loop | PASS | count22=6 |
| R2-23 | Class with inheritance (super()) | PASS | result23=20 |
| R2-24 | Lambda in map/filter | PASS | squares24=[0,4,16,36,64] |
| R2-25 | Closure with mutable state | PASS | [1,2,3] |
| R2-26 | String method chaining | PASS | 'Hello, Python!' |
| R2-27 | Enumerate + sorted + zip | PASS | Ranked correctly |
| R2-28 | Multiline string parsing | PASS | CSV-like data parsed |
