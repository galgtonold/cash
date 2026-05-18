# Notebook Cache Overhead – Raw Analysis Notes (Task 10)

**Date:** 2026-05-19  
**Method:** In-process pyinstrument profiling (interval=0.001s) + JSON result analysis  
**Platform:** Windows 11, Python 3.14, cash v0.5.0b1

---

## Step 1: Profile Attribution

All four notebooks were profiled in-process by wrapping `benchmarks._overhead_driver.run_notebook`
with pyinstrument at 1 ms sampling. Each run used a fresh `cache_dir` (cold equivalent).

### Notebook: synthetic_micro

100 trivial `a_i = i + 1` statements in one cell. 101 statement_metrics (100 + implicit `pass`).  
Profile duration: 2.545s, CPU time: 1.078s, samples: 1169.

```
Total time in StatementProcessor.process_statement: ~1828ms (71.9% of total profile)
  Of which:
    _diagnose_miss -> TieredBackend.list_entries -> FileBackend.list_entries: 1603ms (87.7% of process_statement)
      └─ open (built-in): 1487ms  (92.8% of _diagnose_miss)
      └─ load (msgpack/pickle): 39ms
      └─ BufferedReader.__exit__: 33ms
      └─ glob._iglob: 31ms
    _post_execute / _save_to_cache / _store_in_cache: 195ms (10.7%)
      └─ InMemoryBackend._get_object_size (recursive genexpr): 83ms
      └─ FileBackend.set_metadata_only (open): 57ms
      └─ TieredBackend.set: 103ms total
    capture_output enter+exit: not visible in profile (below 1ms threshold)
    FileAccessTracker.__enter__ / _apply_patches / _patch_module / _patch_user_ns: not visible
    FileAccessTracker.__exit__ / _unpatch: not visible
    ast.parse (redundant inside _execute_statement): not visible
    exec/eval (actual user code): not visible
    _analyze_and_hash: not visible
    _do_cache_lookup: not visible
Badge rendering (_render_interactive_badge + view_builder): 436ms (17.1% of total)
  └─ render_html + _statement_row_html: 205ms
  └─ build_interactive_badge: 147ms
Badge init (_init_cell_timing_and_badge) per-cell: 120ms (4.7%)
  └─ get_notebook_cells -> ipykernel import chain: 108ms (one-off import cost)
```

**Key finding:** `_diagnose_miss` dominates entirely. This is called after every cache MISS to
generate the "why did I miss" diagnostic, and it reads **every file in the cache directory**
(`FileBackend.list_entries` iterates and opens all cache entries). With 100 statements each
producing a cache entry, by statement 100 the list walk reads 100 files. This is O(n^2) I/O.

---

### Notebook: file_tracking_demo

20 cells, 55 statement_metrics (COMPUTED + RESTORED + SKIPPED), real pandas/numpy/pathlib I/O.  
Profile duration: 4.728s (includes real sleeps totalling ~3.1s), CPU time: 0.906s, samples: 897.

```
Total time in StatementProcessor.process_statement: ~4195ms
  Of which:
    _execute_and_drain -> _execute_statement (real user code): 3580ms
      └─ sleep (built-in): 3102ms  [actual user work, not cash overhead]
      └─ _PatchingLoader.exec_module -> pandas import: 416ms  [FileAccessTracker overhead]
    _diagnose_miss -> FileBackend.list_entries: 505ms
      └─ open (built-in): 494ms
    _post_execute / _save_to_cache: 74ms
Badge rendering per-cell: 183ms total (AnalyticsManager.flush + Connection.commit: 161ms)
Badge init: 136ms (ipykernel import chain: 122ms, one-off)
```

**Key finding for file_tracking_demo:** The `_PatchingLoader.exec_module` hook fires on every
`import pandas` (and its submodules). Pandas' import tree traverses dozens of submodules, each
going through `_PatchingLoader.exec_module`. This adds ~416ms to a single cell. The `_diagnose_miss`
cost is lower here (505ms vs 1603ms) because fewer statements execute before errors halt some cells.

---

### Notebook: financial_analysis_demo

32 cells, 186 statement_metrics (148 COMPUTED, 32 RESTORED, 1 SKIPPED).  
Profile duration: 14.895s (includes real sleeps totalling ~13s in `@cash.cache`-decorated functions).  
CPU time: 1.859s, samples: 1830.

```
Total time in StatementProcessor.process_statement: ~12497ms
  Of which:
    _execute_and_drain -> user code (sleep-heavy @cash.cache decorated): 11111ms
      [This is real user computation; cash overhead is the remainder]
    _diagnose_miss -> FileBackend.list_entries: 311ms + 457ms = 768ms across two call sites
      └─ open: 279ms + 442ms respectively
    _post_execute / _save_to_cache: 74ms
Badge rendering: not top-level visible (below threshold)
Error display (AutoFormattedTB / showtraceback): 549ms  [from cells that fail]
  └─ getmodule (inspect.py): 232ms
```

**Key finding for financial_analysis_demo:** The `_diagnose_miss` path appears in TWO distinct
branches — once from `_process_regular_stmt` (0.457s) and once buried inside `_execute_cell_statements`
(0.311s). This suggests `_diagnose_miss` is called multiple times per failed cell. Error display
via `AutoFormattedTB` adds substantial overhead (549ms) when statements raise `NameError`.

---

### Notebook: cfd_simulation_demo

Multiple cells, many with NameError (scipy not installed). 128+ COMPUTED statements, nested loops.  
Profile duration: 4.038s, CPU time: 2.078s, samples: 1994.

```
Total time in StatementProcessor.process_statement: ~3280ms
  Of which:
    ControlStructureProcessor (nested for+if loop): 2029ms
      └─ ForLoopHandler._process_one_iteration: 2002ms
         └─ _execute_loop_body_statement -> _diagnose_miss -> FileBackend.list_entries: 869ms
            └─ open (built-in): 767ms
         └─ _execute_loop_body_nested_control (for-in-for + if) -> _diagnose_miss: 563ms
            └─ open: 523ms
    _process_regular_stmt -> _diagnose_miss: 697ms
      └─ FileBackend.list_entries -> open: 668ms
    _post_execute / _save_to_cache: 112ms + 63ms = 175ms
      └─ InMemoryBackend._get_object_size (recursive): 49ms+
      └─ TieredBackend.set: 61ms
Error display (AutoFormattedTB from NameErrors): 325ms
_resolve_upstream_state (upstream recheck): 282ms
capture_output enter+exit: not visible
FileAccessTracker enter/exit: not visible (except via _PatchingLoader: 77ms for numpy import)
ast.parse, _analyze_and_hash, _do_cache_lookup: not visible
```

**Key finding for cfd_simulation_demo:** The nested loop amplifies `_diagnose_miss` cost. Each
iteration of the outer loop calls inner-loop statements, each of which triggers `list_entries`
on a growing cache. The cumulative `open` time across all loop iterations is 767ms + 523ms + 668ms
= ~1958ms out of 4.038s total, representing 48.5% of total profile time.

---

## Step 2: Cross-Check Against Per-Statement Cash-Declared Metrics

### synthetic_micro

Source JSONs: `synthetic_micro-off-1.json`, `synthetic_micro-cold-1.json`  
n_statements = 100 (a_i assignments only)

| | Value |
|---|---|
| off cell wall_seconds (median of rep-1, rep-2) | 1.76 ms total / 0.0176 ms per stmt |
| cold cell wall_seconds (median of rep-1, rep-2) | 2074.67 ms total / 20.747 ms per stmt |
| **A: real per-stmt overhead (cold wall - off wall) / 100** | **20.729 ms** |
| **B: cash declared overhead (median total_time - execution_time)** | **14.811 ms** |
| **C: hidden overhead in execution_time (median exec_time - per_stmt_off)** | **0.095 ms** |
| B + C | 14.906 ms |
| Unaccounted (A - B - C) | 5.823 ms (28.1% of A) |
| C as fraction of A | 0.5% |

**The hypothesis:** C is NOT substantial. The badge UI is NOT hiding meaningful overhead inside
the "Compute" field for the synthetic case. `execution_time` (which maps to actual exec time) is
close to the true per-statement baseline (~0.018ms off + ~0.095ms hidden = ~0.113ms per stmt
in cold, consistent with the JSON median of 0.113ms).

The dominant issue is the 28.1% unaccounted overhead — time spent at the cell/statement-dispatch
level outside the `process_statement` boundary. This is likely time in `_execute_cell_statements`
looping, IPython overhead between statements, and badge rendering that doesn't show in any single
statement's `total_time`.

**B (14.811ms) accounts for the declared overhead, but where does it go?**  
Per the profile: `_diagnose_miss` takes 1603ms for 101 statements = **15.9ms per statement**,
which matches B almost exactly. The declared overhead is real and is almost entirely `_diagnose_miss`.

**Additional note on repeat-0 vs repeat-1:**  
cold-0: 1998ms, cold-1: 1836ms, cold-2: 2386ms. Variability is ±30%. Using median of rep-1 and
rep-2 avoids both first-run startup cost and one-off outlier.

---

### file_tracking_demo

Source JSON: `file_tracking_demo-cold-1.json`  
Statement count: 55 total, 31 COMPUTED non-sleep non-pass statements used for analysis.

| | Value |
|---|---|
| B: median declared overhead (COMPUTED non-sleep stmts) | 13.18 ms |
| Mean declared overhead | 13.15 ms |
| Max declared overhead | 30.79 ms |

**A/B/C breakdown not computed** for file_tracking_demo because `cold < off` in wall time —
cold saves ~2.4s by restoring `time.sleep()` calls from cache. The wall-clock comparison is
dominated by the 5.3s of `sleep()` calls in the off baseline, not by cash overhead.

The per-statement declared overhead (B = 13.18ms) is consistent with synthetic_micro (B = 14.81ms),
confirming that cash overhead per COMPUTED statement is approximately **13-15ms across both notebooks**.

---

## Step 3: Summary Table — All Four Notebooks

| Notebook | Median B (declared overhead, COMPUTED stmts) | Profile dominant cost | `_diagnose_miss` share |
|---|---|---|---|
| synthetic_micro | 14.81 ms/stmt | `_diagnose_miss` → `FileBackend.list_entries` | 87.7% of `process_statement` |
| file_tracking_demo | 13.18 ms/stmt | `_diagnose_miss` (505ms) + `_PatchingLoader` (416ms) | ~12% of total (dominated by real work) |
| financial_analysis_demo | 18.62 ms/stmt (COMPUTED) | Real user code (sleep-heavy), then `_diagnose_miss` | ~6% of total |
| cfd_simulation_demo | 14.83 ms/stmt (COMPUTED) | `_diagnose_miss` in nested loops | ~48% of total profile |

---

## Headline Numbers (A / B / C) for synthetic_micro

```
A = 20.73 ms  (real per-statement overhead, wall clock)
B = 14.81 ms  (cash declared overhead in total_time - execution_time)
C =  0.09 ms  (hidden inside execution_time — negligible)

B/A = 71.5%   Cash correctly attributes most of the overhead in its declared metrics
C/A =  0.5%   Badge UI is NOT hiding meaningful overhead inside "Compute"
Unaccounted = 28.1% of A  (cell-level dispatch overhead, badge rendering, not in any stmt's total_time)
```

---

## Profile Limitations and Caveats

1. **1ms sampling misses sub-millisecond operations.** `capture_output`, `FileAccessTracker.__enter__/exit`,
   `ast.parse`, `_analyze_and_hash`, and `_do_cache_lookup` are all below the sampling threshold
   in the synthetic profile. Their contributions are real but summed into the 28.1% unaccounted
   residual or embedded in the measured totals.

2. **`_PatchingLoader.exec_module` is a one-time cost per import** (modules are cached in
   `sys.modules`). In the file_tracking_demo profile it shows 416ms because pandas is imported
   for the first time inside the profiled run. Real user notebooks would not pay this cost
   repeatedly — but the `_PatchingLoader` hook intercepts every `import` execution, so it does
   add a few ms to each `import` statement even after the first run.

3. **cfd_simulation_demo produced NameErrors** because `scipy` is not installed in the profiling
   environment. This means the profile reflects a partially-broken run; the `_resolve_upstream_state`
   and error-display costs inflate the numbers.

4. **The financial_analysis_demo profiling environment** found an existing cache with 6 entries
   from a previous benchmark run. This means some statements RESTORED from cache rather than
   computing cold, slightly underrepresenting `_diagnose_miss` for that notebook.

5. **`_diagnose_miss` is O(n) per invocation** (reads all files in cache dir). After n statements
   each producing a cache entry, statement k's `_diagnose_miss` reads k-1 files. Total I/O is
   O(n^2). For 100 statements: ~5050 file reads.

6. **InMemoryBackend._get_object_size recursion** is visible in the profile. It recursively
   computes the size of nested Python objects using a generator expression, which itself triggers
   another `_get_object_size` call. This infinite-recursion-safe pattern is still O(depth) CPU.

7. **Badge rendering (17% of synthetic_micro profile)** is not a per-statement cost — it fires
   once per cell. For a cell with 100 statements, the badge renders once with 100 rows, so the
   cost scales with statement count only once per cell, not linearly. Still, `render_html` +
   `_statement_row_html` take 205ms for 100 statements, which is ~2ms/statement amortized.

---

## Key Observations for Task 11 (Remediation)

1. **Primary bottleneck:** `_diagnose_miss` → `FileBackend.list_entries` → `open` for every
   cache entry. This is called on every COMPUTED statement and is O(n) in cache size.
   Fix candidate: index-based lookup instead of directory scan; or defer `_diagnose_miss` until
   user explicitly requests it (it's diagnostic, not on the hot path).

2. **`_PatchingLoader.exec_module`** wraps every module import. Costly on first import but
   memoized by `sys.modules`. Not a repeated overhead in steady state.

3. **Badge rendering** is substantial (17%) but scales once per cell, not per statement. For
   cells with many statements it amortizes. Fix: lazy rendering / incremental update.

4. **Unaccounted 28% in synthetic_micro** suggests significant overhead in `_execute_cell_statements`
   dispatch loop and statement-boundary work that no single metric captures. The profile sampler
   likely attributes this to IPython internals or the tee wrapper.

5. **C ≈ 0** confirms the "Compute" column in the badge is trustworthy. The badge is NOT
   mis-attributing overhead as user computation.
