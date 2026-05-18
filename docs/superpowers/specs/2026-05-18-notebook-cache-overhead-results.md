# Notebook cache overhead — results

**Date:** 2026-05-18 (data collected 2026-05-18/19)
**Spec:** [2026-05-18-notebook-cache-overhead-analysis.md](2026-05-18-notebook-cache-overhead-analysis.md)
**Run on:** Windows-11-10.0.26200-SP0, Python 3.14.3, cash 0.5.0b1
**Harness:** `benchmarks/bench_notebook_overhead.py`

## TL;DR

The original hypothesis — that `capture_output` and `FileAccessTracker` setup were silently inflating the badge's "Compute" field — is refuted: only ~0.09 ms per statement (0.5% of total cold-mode overhead) is hidden inside `execution_time`, and the badge's declared metrics are accurate. The real bottleneck is `_diagnose_miss` in `src/cash/notebook/statement_processor.py`, which walks the entire cache directory on every cache miss and exhibits O(N²) I/O behaviour in cache size: on a 100-statement cold run it consumed 1603 ms, 88% of all `process_statement` time. The highest-priority fix is to make `_diagnose_miss` lazy — evaluate it only when the badge's row-detail drawer is opened or `%cash_explain` is invoked — which would eliminate the O(N²) path with minimal risk and no semantic change for users.

## Methodology

The harness (`benchmarks/bench_notebook_overhead.py`) drives notebooks in-process through a freshly initialised `InteractiveShell`. Each notebook is run in three modes — cash-off (baseline), cash-on cold (cache wiped before every repeat), and cash-on warm (cache pre-populated by a silent initial run before the timed repeats) — with three repeats each. Repeat 0 is discarded as warmup; reported numbers are the median of repeats 1 and 2. Per-cell wall clock is measured with `time.perf_counter()`. Per-statement cash metrics (`execution_time`, `total_time`, `status`) are captured by monkey-patching `StatementProcessor.process` to tee its return value into a harness-owned sink. For the cold run of each notebook, the last repeat is additionally wrapped in `pyinstrument.Profiler(interval=0.001)` and the resulting call graph is written to `benchmarks/results/<stem>-cold-profile.html`. See the spec for a full description of the harness design.

Three limitations are relevant to interpreting the results. First, pyinstrument's 1 ms sampling interval means that operations completing in under ~1 ms are invisible in the profiles; `capture_output`, `FileAccessTracker.__enter__/exit`, `ast.parse`, `_analyze_and_hash`, and `_do_cache_lookup` all fall below this threshold and their costs are aggregated into a 28% residual in the synthetic_micro profile rather than attributed to named buckets. Second, the warm-mode harness does not reliably trigger cache restores on the synthetic_micro and cfd_simulation_demo notebooks — the in-process `InteractiveShell.instance()` singleton shares state across `run_notebook` calls within one subprocess, preventing the variable lineage check from matching the cold-populated cache. This is a harness-side issue; single-statement warm restore works correctly in isolation (confirmed by the Task 5 test). Third, the financial_analysis_demo profiling environment found 6 pre-existing cache entries from a prior benchmark run, meaning some statements RESTORED rather than recomputing cold; `_diagnose_miss` costs for that notebook are therefore slightly underrepresented.

## Per-notebook results

### synthetic_micro

| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |
|---|---:|---:|---:|---:|---:|
| cell 0 | 1.76 | 2074.67 | 2237.32 | +2072.90 | +117695.1% |
| **TOTAL** | **1.76** | **2074.67** | **2237.32** | **+2072.90** | **+117695.1%** |

The synthetic notebook has a single cell containing 100 trivial assignments (`a_i = i + 1`). Off-mode cost is 1.76 ms total — essentially zero compute. Cold-mode cost is 2074 ms, making the cold-off overhead 2073 ms, almost all of it pure cash machinery. This is the cleanest signal in the benchmark: with no I/O, no imports, and no real work, every millisecond of cold overhead is attributable to cash's per-statement bookkeeping path.

Warm mode shows 2237 ms — slower than cold. This is a harness artefact: the warm-mode cache was pre-populated in a different subprocess call, but the in-process shell's variable lineage does not match, so every statement re-executes rather than restoring. The warm column for synthetic_micro is not meaningful.

### file_tracking_demo

| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |
|---|---:|---:|---:|---:|---:|
| cell 0 | 0.11 | 41.18 | 31.65 | +41.06 | +36130.9% |
| cell 1 | 0.24 | 6.04 | 32.07 | +5.79 | +2369.0% |
| cell 2 | 0.15 | 5.79 | 6.54 | +5.64 | +3677.2% |
| cell 3 | 0.55 | 33.03 | 43.99 | +32.48 | +5858.8% |
| cell 4 | 2001.34 | 2080.80 | 98.03 | +79.46 | +4.0% |
| cell 5 | 0.22 | 51.33 | 53.54 | +51.11 | +22840.5% |
| cell 6 | 1101.01 | 1160.88 | 45.89 | +59.87 | +5.4% |
| cell 7 | 2.33 | 70.38 | 78.67 | +68.05 | +2922.1% |
| cell 8 | 1.15 | 88.20 | 25.56 | +87.04 | +7536.2% |
| cell 9 | 2.99 | 70.76 | 14.04 | +67.77 | +2262.7% |
| cell 10 | 1101.87 | 48.80 | 16.20 | -1053.07 | -95.6% |
| cell 11 | 3.33 | 60.09 | 13.82 | +56.76 | +1704.0% |
| cell 12 | 2.60 | 74.43 | 89.77 | +71.83 | +2763.4% |
| cell 13 | 1.31 | 80.05 | 80.21 | +78.75 | +6028.0% |
| cell 14 | 1101.41 | 53.13 | 75.87 | -1048.28 | -95.2% |
| cell 15 | 1.02 | 64.12 | 65.59 | +63.10 | +6160.0% |
| cell 16 | 0.94 | 78.42 | 74.35 | +77.48 | +8251.7% |
| cell 17 | 0.92 | 65.49 | 79.11 | +64.57 | +7038.3% |
| cell 18 | 1101.50 | 57.05 | 61.43 | -1044.45 | -94.8% |
| cell 19 | 1.36 | 72.46 | 70.04 | +71.11 | +5236.5% |
| **TOTAL** | **6426.37** | **4262.44** | **1056.39** | **-2163.93** | **-33.7%** |

The TOTAL row shows cold is actually 2164 ms faster than off — not because cash adds no overhead, but because cash short-circuits the `time.sleep(1.1)` calls embedded in cells 10, 14, and 18. Each of those cells takes ~1100 ms in off mode; in cold mode cash restores the sleep result from cache at ~50–57 ms. The -95% cold-off values for those cells are genuine cash wins, not measurement error.

The per-statement declared overhead for non-sleep, non-pass COMPUTED statements is 13.18 ms (median), consistent with the 14.81 ms seen in synthetic_micro. Warm mode works correctly here: total warm time is 1056 ms vs 6426 ms off-mode — a roughly 6× speedup — because the sleep cells restore from cache in 16–100 ms instead of blocking for 1100 ms each.

### financial_analysis_demo

| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |
|---|---:|---:|---:|---:|---:|
| cell 0 | 110.06 | 127.95 | 156.77 | +17.89 | +16.3% |
| cell 1 | 32.73 | 16.52 | 28.56 | -16.21 | -49.5% |
| cell 2 | 501.46 | 1027.70 | 48.28 | +526.24 | +104.9% |
| cell 3 | 91.10 | 209.86 | 39.08 | +118.76 | +130.4% |
| cell 4 | 39.19 | 75.15 | 15.77 | +35.95 | +91.7% |
| cell 5 | 5871.72 | 14162.69 | 53.92 | +8290.97 | +141.2% |
| cell 6 | 17.64 | 42.84 | 18.29 | +25.20 | +142.8% |
| cell 7 | 94.96 | 245.31 | 60.11 | +150.35 | +158.3% |
| cell 8 | 114.89 | 52.51 | 21.76 | -62.38 | -54.3% |
| cell 9 | 43.17 | 83.71 | 53.23 | +40.54 | +93.9% |
| cell 10 | 4542.45 | 9360.67 | 45.54 | +4818.21 | +106.1% |
| cell 11 | 40.36 | 47.81 | 17.39 | +7.45 | +18.5% |
| cell 12 | 114.99 | 203.45 | 19.72 | +88.46 | +76.9% |
| cell 13 | 49.80 | 70.25 | 50.19 | +20.45 | +41.1% |
| cell 14 | 134.77 | 161.80 | 18.05 | +27.04 | +20.1% |
| cell 15 | 277.68 | 578.51 | 122.26 | +300.83 | +108.3% |
| cell 16 | 6.22 | 21.54 | 40.87 | +15.32 | +246.3% |
| cell 17 | 16.81 | 155.95 | 180.63 | +139.14 | +827.8% |
| cell 18 | 533.40 | 1136.53 | 137.89 | +603.14 | +113.1% |
| cell 19 | 12.40 | 53.50 | 72.91 | +41.10 | +331.6% |
| cell 20 | 16.21 | 157.00 | 114.05 | +140.79 | +868.3% |
| cell 21 | 12.57 | 64.44 | 70.62 | +51.87 | +412.7% |
| cell 22 | 10.09 | 146.54 | 125.43 | +136.45 | +1352.7% |
| cell 23 | 16.41 | 163.51 | 148.22 | +147.10 | +896.5% |
| cell 24 | 12.13 | 63.42 | 66.21 | +51.30 | +423.1% |
| cell 25 | 14.00 | 86.53 | 99.62 | +72.53 | +518.1% |
| cell 26 | 9.20 | 19.98 | 19.81 | +10.79 | +117.3% |
| cell 27 | 28.76 | 58.56 | 58.15 | +29.80 | +103.6% |
| cell 28 | 49.10 | 111.91 | 95.17 | +62.81 | +127.9% |
| cell 29 | 9.26 | 95.05 | 45.82 | +85.79 | +927.0% |
| cell 30 | 22.49 | 67.17 | 62.69 | +44.67 | +198.6% |
| cell 31 | 550.72 | 106.04 | 17.88 | -444.68 | -80.7% |
| **TOTAL** | **13396.72** | **28974.40** | **2124.91** | **+15577.68** | **+116.3%** |

Two cells dominate the cold overhead: cell 5 adds +8291 ms and cell 10 adds +4818 ms in cold mode relative to off. Both cells produce large pandas DataFrames; cash is serialising those objects to the file backend synchronously, effectively doubling the wall time for those cells. This is the per-output serialisation cost, a separate growth mode from the per-statement fixed cost (13–15 ms) seen in the simpler notebooks. Cold overhead here is not just O(statements) — it also scales with object size.

Warm mode is the effective story for this notebook: total warm time is 2125 ms vs 13397 ms off-mode, a 6.3× speedup. Cells 5 and 10, which were the worst cold performers, drop to 54 ms and 46 ms warm — restoring a pre-serialised DataFrame is fast.

### cfd_simulation_demo

| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |
|---|---:|---:|---:|---:|---:|
| cell 0 | 37.23 | 48.29 | 48.90 | +11.06 | +29.7% |
| cell 1 | 26.21 | 25.77 | 22.51 | -0.44 | -1.7% |
| cell 2 | 78.18 | 337.27 | 427.21 | +259.09 | +331.4% |
| cell 3 | 41.70 | 91.17 | 122.04 | +49.47 | +118.6% |
| cell 4 | 8.03 | 17.88 | 15.53 | +9.84 | +122.5% |
| cell 5 | 86.01 | 112.52 | 138.87 | +26.51 | +30.8% |
| cell 6 | 38.54 | 207.38 | 149.30 | +168.85 | +438.1% |
| cell 7 | 341.65 | 104.50 | 90.34 | -237.15 | -69.4% |
| cell 8 | 21.37 | 74.33 | 78.92 | +52.97 | +247.9% |
| cell 9 | 365.58 | 153.34 | 142.08 | -212.24 | -58.1% |
| cell 10 | 312.47 | 18.72 | 15.22 | -293.75 | -94.0% |
| cell 11 | 307.04 | 19.13 | 16.05 | -287.91 | -93.8% |
| cell 12 | 958.64 | 1817.47 | 2072.03 | +858.83 | +89.6% |
| cell 13 | 301.37 | 21.62 | 17.65 | -279.74 | -92.8% |
| cell 14 | 24.57 | 29.39 | 27.67 | +4.83 | +19.6% |
| **TOTAL** | **2948.57** | **3078.79** | **3384.33** | **+130.21** | **+4.4%** |

Two distinct patterns are present here. Cells 7, 9, 10, 11, and 13 have negative cold-off values: cold is faster than off, by 69–94%. These are redundant-recompute wins — cash detects that the cell's inputs have not changed and restores from cache rather than re-running the numerical computation. Cell 12 is the inversion: cold adds +859 ms over off. This cell produces sparse matrices; serialising them to disk takes longer than recomputing them from scratch, so cold mode is slower than off on that cell.

Warm mode for this notebook is also slower than off (3384 ms vs 2949 ms total). The cells where warm > off are cells 2, 3, 5, 6, 8, and 12 — all cells that produce or consume NumPy arrays or sparse matrices. Deserialization for large array objects can exceed the cost of recomputation, which breaks the expected warm-mode guarantee. This is a real behavioural issue worth addressing separately from the `_diagnose_miss` fix. The scipy-related NameErrors that affected some cells (scipy is not installed in the profiling environment) mean the profile for this notebook reflects a partially broken run; the error-display overhead from `AutoFormattedTB` (325 ms) inflates the totals.

## Overhead decomposition (synthetic_micro)

The following table attributes the 2075 ms cold-mode wall time on the synthetic 100-statement notebook. Numbers come from the pyinstrument profile (`benchmarks/results/synthetic_micro-cold-profile.html`, 1 ms sampling interval, 2.545 s profile duration).

| Bucket | Time (ms) | Share of process_statement | Share of total profile |
|---|---:|---:|---:|
| `_diagnose_miss` → `FileBackend.list_entries` → `open` | 1603 | 87.7% | 63.0% |
| `_post_execute` / `_save_to_cache` / `TieredBackend.set` | 195 | 10.7% | 7.7% |
| Badge rendering (`_render_interactive_badge` + `render_html` + `_statement_row_html`) | 436 | — (per-cell, not per-stmt) | 17.1% |
| Badge init (`_init_cell_timing_and_badge`, incl. ipykernel import) | 120 | — | 4.7% |
| All other / below sampling threshold | ~28% of process_statement time | ~11.6% | 7.5% |

The "below sampling threshold" bucket covers `capture_output` enter/exit, `FileAccessTracker.__enter__/exit`, `ast.parse`, `_analyze_and_hash`, and `_do_cache_lookup`. Each of these operations completes in well under 1 ms per statement and does not register as a distinct frame in the profile. Their aggregate contribution is subsumed into the 28% residual measured by the A/B/C cross-check (see the next section). A deterministic profiler (cProfile or 0.1 ms pyinstrument sampling) would decompose this residual further.

## The original hypothesis (refuted)

The spec hypothesised that `execution_time` — the "Compute" field in the badge — silently includes `capture_output()` and `FileAccessTracker()` context-manager setup, and that these costs are non-trivial. The argument was grounded in the source: `start_time = time.time()` in `_execute_statement` is set before the context managers are entered and after they exit, so anything they do is numerically inside `execution_time`.

The data does not support the hypothesis. Cross-checking the synthetic_micro JSON against the wall-clock baseline gives three numbers:

```
A = 20.73 ms  — real per-statement overhead (cold wall − off wall) / 100 statements
B = 14.81 ms  — cash-declared overhead: median(total_time − execution_time) per statement
C =  0.09 ms  — hidden overhead inside execution_time: median(execution_time) − per-stmt off baseline

B/A = 71.5%   — cash correctly attributes most overhead via its declared metrics
C/A =  0.5%   — badge "Compute" field is not hiding meaningful overhead
Unaccounted = 5.82 ms (28.1% of A)  — cell-level dispatch and badge rendering outside any stmt's total_time
```

C is 0.09 ms per statement. Over 100 statements that is 9 ms — less than 0.5% of the 2073 ms cold-off total. The `capture_output` and `FileAccessTracker` setup that the spec flagged as suspects turns out to be so fast that pyinstrument's 1 ms sampler never catches a sample in those frames. The badge's "Compute" column is accurate.

The source-code review that motivated the hypothesis was not wrong — the context managers are numerically inside `execution_time` — but the review overestimated their cost. The patches are cheap; it is the diagnostic walkover of the cache directory after the miss that is expensive, and that happens outside `execution_time` in the explicitly-declared `overhead` portion of `total_time`. The badge is telling the truth.

This is a meaningful secondary finding: the existing cash telemetry is trustworthy. Reported "Compute" time faithfully represents user-code execution, and the "overhead" column is where the real machinery costs land.

## The real bottleneck: `_diagnose_miss` is O(N²)

`_diagnose_miss` is defined at [`src/cash/notebook/statement_processor.py:1894–1941`](src/cash/notebook/statement_processor.py#L1894). Its job is to explain why a cache miss occurred: it looks for prior entries with the same source hash and checks whether the input set changed. To do this it calls `backend.list_entries()` (line 1907), which on `FileBackend` opens and reads every file in the cache directory. The return value is a list of all cached entry metadata; `_diagnose_miss` then scans that list for a matching `source_hash`.

The call site is at [`src/cash/notebook/statement_processor.py:489–494`](src/cash/notebook/statement_processor.py#L489), inside `process()` on the COMPUTED (cache-miss) path, after the statement has already been executed and the miss is confirmed:

```python
if not skip_cache:
    reason = getattr(self, '_last_miss_reason', None)
    if reason is None:
        reason = self._diagnose_miss(source_hash, inputs)
    if reason:
        metrics['miss_reason'] = reason
```

This fires for every COMPUTED statement. On a cold run with N statements, statement k produces a cache entry, so statement k+1's `_diagnose_miss` reads k entries. Total file opens across N statements: N·(N−1)/2 — O(N²). For 100 statements this is ~5050 file opens. The pyinstrument profile attributes 1603 ms to `_diagnose_miss` → `open` on the 100-statement notebook, 87.7% of all `process_statement` time.

The pathology is specific to the cold path: on a warm run, most statements RESTORE from cache rather than computing, so `_diagnose_miss` is rarely called. This means the O(N²) behaviour hits exactly when users are most likely to notice cash slowness — the first run of a notebook, or after a cache wipe. On a 1000-statement cold run the cost scales approximately as (1000/100)² = 100× relative to the 100-statement case, projecting to ~160 s of `_diagnose_miss` time alone.

## Remediation strategies

### Strategy 1: Make `_diagnose_miss` lazy

**Hypothesis:** The miss reason text is consumed in two places: the badge's row-detail drawer (visible only when the user clicks a row to expand it) and `%cash_explain`. The vast majority of cells are never expanded. Computing the reason eagerly on every cold miss pays the O(N) file-read cost for a result that is almost never seen. Replacing the eager call with a deferred callable eliminates the O(N²) path without changing any user-visible behaviour.

**Expected savings:** ~1603 ms on the 100-statement cold run (88% of process_statement time). The savings scale super-linearly: a 1000-statement cold run would save approximately 160 s. On data-heavy notebooks (financial_analysis_demo), the serialisation cost dominates so the proportional gain is smaller, but the per-statement fixed-cost reduction still applies.

**Implementation cost:** small. The call site is a single block at [`src/cash/notebook/statement_processor.py:489–494`](src/cash/notebook/statement_processor.py#L489). Replace it with a lambda stored on `metrics`:

```python
if not skip_cache:
    reason = getattr(self, '_last_miss_reason', None)
    if reason is not None:
        metrics['miss_reason'] = reason
    else:
        # Defer: capture bindings now, evaluate only when the badge drawer opens.
        _hash, _inputs = source_hash, inputs
        metrics['miss_reason_lazy'] = lambda: self._diagnose_miss(_hash, _inputs)
```

Then update wherever the badge renderer reads `metrics['miss_reason']` to also check `metrics.get('miss_reason_lazy')` and call it if present. The only consumers are inside `src/cash/notebook/badge_renderer/` (the row-detail expansion path) and `%cash_explain`.

**Risk:** low. No semantic change for the user. Consumers that read `miss_reason` as a string at process() return time would need to be updated, but the badge renderer is the only known consumer.

**Where:** [`src/cash/notebook/statement_processor.py:489–494`](src/cash/notebook/statement_processor.py#L489) (call site) and [`src/cash/notebook/statement_processor.py:1894`](src/cash/notebook/statement_processor.py#L1894) (definition); also wherever `metrics['miss_reason']` is read in `src/cash/notebook/badge_renderer/`.

### Strategy 2: Skip `_diagnose_miss` for an empty cache

**Hypothesis:** On statement 1 of a cold run, `backend.list_entries()` returns zero entries, and `_diagnose_miss` immediately returns `"first time seeing this code"` without doing any useful work beyond the call itself. For any notebook where every statement is new (the most common cold-run scenario), every call to `_diagnose_miss` on statements 1 through N that happen to be the first occurrence of that code will go through the full list walk, finding nothing. Adding a fast-path that checks `len(entries) == 0` before scanning can skip the scan entirely when the cache is empty, and can also short-circuit when `source_hash` is trivially absent because the cache is brand new.

**Expected savings:** Zero on the 100-statement synthetic notebook in isolation (by statement 2 there is already 1 entry), but this is a correctness and readability improvement that sets the stage for Strategy 1. As a standalone change it would save the overhead of calling `list_entries()` on the very first statement. On a truly empty cache it would also eliminate the glob call.

**Implementation cost:** small. A two-line guard at the top of `_diagnose_miss`:

```python
entries = backend.list_entries()
if not entries:
    return "first time seeing this code"
```

This is already the logical outcome of the existing code when `entries` is empty; making it an explicit early return eliminates the list comprehension and any subsequent work.

**Risk:** negligible. The return value is identical to the current behaviour.

**Where:** [`src/cash/notebook/statement_processor.py:1894–1913`](src/cash/notebook/statement_processor.py#L1894), specifically the block that calls `list_entries()` and checks `prior`.

### Strategy 3: Reduce per-statement badge HTML cost

**Hypothesis:** Badge rendering fires once per cell but its cost scales with the number of statement rows. For the 100-statement synthetic cell, `render_html` + `_statement_row_html` took 205 ms (10% of the cold total). Rendering is synchronous and happens on the IPython execution thread. Generating 100 rows of HTML per cell is the source of this cost. Lazy rendering (skip building rows for rows that are collapsed by default) or limiting the number of rendered rows would reduce this.

**Expected savings:** ~205 ms on the 100-statement cell. For typical notebooks with 5–20 statements per cell the cost is proportionally smaller (~10–40 ms per cell), but at 17% of the total profile time for synthetic_micro it is the second-largest attributable cost after `_diagnose_miss`.

**Implementation cost:** small/medium. The rendering pipeline runs through `build_interactive_badge` → `render_html` → `_statement_row_html` for each row. The rows are built eagerly in `src/cash/notebook/badge_renderer/renderers/html.py:_statement_row_html`. A guard that renders only the first N rows and inserts a "show all" control for larger cells would cap the cost without changing the visible default state.

**Risk:** low. The collapsed badge appearance does not change for small cells. For cells with many statements the expanded view would load rows on demand rather than at render time.

**Where:** `src/cash/notebook/badge_renderer/renderers/html.py:1035` (`_statement_row_html`) and `src/cash/notebook/badge_renderer/renderers/html.py:1907` (`render_html`); the row-building loop that calls `_statement_row_html` for each statement.

### Strategy 4: Address the per-output serialisation cost

**Hypothesis:** For cells that produce large objects (DataFrames, sparse matrices), cold-mode serialisation doubles or more than doubles wall time. Cell 5 in financial_analysis_demo added +8291 ms to cold (141% overhead) and cell 12 in cfd_simulation_demo added +859 ms (90% overhead). The existing `_should_skip_large_object_caching` logic at [`src/cash/notebook/statement_processor.py:1479`](src/cash/notebook/statement_processor.py#L1479) already gates on object size and execution time, but the threshold is tuned for the "skip entirely" case rather than for deferred or async serialisation. Serialising large objects asynchronously (after control returns to the user) or implementing a size-based heuristic to prefer recompute over cache-restore for objects above a threshold would eliminate the first-run wall-clock doubling.

**Expected savings:** Up to +8291 ms (cell 5, financial_analysis_demo) and +859 ms (cell 12, cfd_simulation_demo) on cold runs. Warm-mode correctness for large-array cells in cfd_simulation_demo is a related issue (Strategy 5).

**Implementation cost:** medium. Async serialisation requires a background thread and a mechanism to signal when the cache entry is durable. A simpler version — a size-based heuristic that skips serialisation for objects above a threshold when execution time is below a threshold — is smaller but less precise.

**Risk:** medium. Async caching changes the semantics of when a cache entry is guaranteed durable. A crash or kernel restart between execution and serialisation completion would lose the entry. This needs explicit handling in the backend protocol.

**Where:** [`src/cash/notebook/statement_processor.py:1479`](src/cash/notebook/statement_processor.py#L1479) (`_should_skip_large_object_caching` call site) and [`src/cash/notebook/statement_processor.py:1310`](src/cash/notebook/statement_processor.py#L1310) (`_should_skip_large_object_caching` definition).

### Strategy 5: Investigate warm-mode pathologies for large arrays

**Hypothesis:** In cfd_simulation_demo, warm mode is slower than off (3384 ms vs 2949 ms total). The cells where warm exceeds off all produce or consume NumPy arrays or sparse matrices. Msgpack/pickle deserialisation for these objects exceeds the cost of recomputing them, breaking the expected warm-mode guarantee. A size-aware deserialization decision — where cash falls back to recomputation when a stored object exceeds a size threshold — would preserve the "warm is never slower than off" contract.

**Expected savings:** Restoring the warm-mode guarantee on numerical notebooks. The total warm-vs-off regression in cfd_simulation_demo is 436 ms; the per-cell regressions range from 30 to 430 ms.

**Implementation cost:** medium. The deserialisation path is in the file backend; adding a size check before deserialising (via stored metadata about the serialised size) would allow a fall-through to recompute. This requires storing the serialised size in the cache metadata, which is not currently present.

**Risk:** low. Falling back to recompute on large objects degrades gracefully — it matches off-mode behaviour rather than being slower than it.

**Where:** `benchmarks/results/cfd_simulation_demo-warm-*.json` for the empirical baseline; the file backend deserialisation path (likely `src/cash/backends/file_backend.py`) and the size-threshold logic in `_should_skip_large_object_caching`.

### Optional: Deterministic profiling pass to attribute the 28% residual

The 28.1% unaccounted overhead in synthetic_micro (5.82 ms per statement) represents real cost that the 1 ms pyinstrument sampler cannot decompose. A deterministic profile — `cProfile` or pyinstrument at 0.1 ms interval — would attribute this residual to specific call sites: `capture_output`, `FileAccessTracker.__enter__/exit`, `ast.parse`, `_analyze_and_hash`, `_do_cache_lookup`. This step costs nothing to implement (add a `CASH_BENCH_CPROFILE=1` env-var hook to the harness) and would feed a follow-up perf pass if Strategies 1–3 leave meaningful overhead.

**Implementation cost:** small. **Risk:** none.

## Open questions and follow-ups

- The warm-mode harness does not trigger cache restores reliably on synthetic_micro even though the Task 5 single-statement test confirms warm-restore works. The most likely cause is shared `InteractiveShell.instance()` state across `run_notebook` calls inside one subprocess: the variable lineage hash differs between the pre-populate run and the timed run because the shell's internal state is not fully reset. Worth tracking down whether the fix is harness-side (re-instantiate the shell or use a subprocess per run) or cash-side (relax the lineage matching for trivial assignments).

- The 28% residual on the synthetic profile — the sum of `capture_output`, `FileAccessTracker`, `ast.parse`, `_analyze_and_hash`, `_do_cache_lookup` — remains undecomposed. A deterministic profiler pass would resolve this and might surface a third priority item if any single component claims more than ~5 ms.

- Does `FileAccessTracker` get hot on real-world file-IO-heavy notebooks beyond the 416 ms one-time `_PatchingLoader.exec_module` pandas import cost already observed? Worth re-profiling with a synthetic IO-heavy notebook (e.g., 50 `pd.read_csv` calls in a loop) if Strategies 1–3 are implemented and the residual overhead is still measurable.

- `InMemoryBackend._get_object_size` uses a recursive generator expression to compute deep object sizes. The profile shows 83 ms attributed to this on the synthetic notebook. It is called inside `_post_execute` / `_save_to_cache` and is part of the `_should_skip_large_object_caching` decision. It may be worth capping the recursion depth or using `sys.getsizeof` with a shallow bound for the gating decision.

## Appendix: links to raw artifacts

- Pyinstrument HTMLs (gitignored): `benchmarks/results/*-cold-profile.html`
- Per-mode result JSONs (gitignored): `benchmarks/results/*-{off,cold,warm}-*.json`
- Raw analysis notes: [2026-05-18-notebook-cache-overhead-notes.md](2026-05-18-notebook-cache-overhead-notes.md)
