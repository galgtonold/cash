# Notebook cache overhead analysis — design

**Status:** Design, awaiting implementation
**Date:** 2026-05-18
**Owner:** Performance investigation

## Background

The cash notebook integration reports two timing fields per statement, exposed
on `ProcessResult` and displayed in the badge UI:

- `execution_time` — labelled as "Compute" in the badge
- `total_time − execution_time` — labelled as overhead

Reading the source confirms that `execution_time`, as currently defined in
[`statement_processor.py:_execute_statement`](src/cash/notebook/statement_processor.py),
starts at `start_time = time.time()` **before** the
`capture_output()` and `FileAccessTracker(...)` context managers are entered
and ends after they exit. It therefore silently includes:

- `capture_output()` enter/exit (IPython stdout/stderr/display interception)
- `FileAccessTracker.__enter__` — monkey-patches builtins, pandas, numpy,
  polars, joblib, pickle, json, and installs a `PostImportHook` on
  `sys.meta_path`
- `FileAccessTracker.__exit__` — reverses all patches
- A defensive fallback `ast.parse(code)` on a tree that was already parsed
  once at the entry of `process()`

The hypothesis driving this investigation: a non-trivial fraction of what cash
currently presents as "user-code compute time" is in fact cash machinery, and
the user-facing "overhead" metric understates the real cost.

This document defines the benchmark needed to confirm or refute that
hypothesis and to point at where to fix it.

## Goals

1. Quantify cash-on-cold vs cash-off overhead at notebook level (per-cell
   wall clock).
2. Decompose where overhead lives via pyinstrument, focusing on time inside
   `_execute_statement.execution_time` that is cash machinery rather than
   user code.
3. Produce a prioritized list of remediation strategies, grounded in the
   measured data.

## Non-goals

- No changes to cash source code in this phase. Pure measurement and
  analysis.
- No comprehensive warm-cache (cache-hit) benchmarking. We include one warm
  column for completeness; the focus is the cold path because that is where
  every first-time user lives.
- No `nb_runner` / kernel-subprocess measurement. The cash code path being
  measured is identical whether IPython runs in-process or as a kernel; the
  kernel boundary only adds a constant ZMQ cost that cancels out of any
  cash-on vs cash-off comparison.

## Harness

### `benchmarks/bench_notebook_overhead.py`

Single CLI entrypoint:

```
python benchmarks/bench_notebook_overhead.py <notebook.ipynb> \
    --mode {off,cold,warm} \
    [--profile] \
    [--repeats N] \
    [--cache-dir PATH]
```

Per invocation:

1. Spawn a fresh subprocess so module state is clean (no monkey-patch
   carryover from prior runs).
2. Inside the subprocess, instantiate a fresh `InteractiveShell`
   (`IPython.core.interactiveshell.InteractiveShell` or
   `TerminalInteractiveShell`).
3. If mode is `cold` or `warm`: enable cash via the public magics, pointing
   at a temp cache dir. For `cold`, the dir is wiped before **each repeat**
   so every repeat starts empty. For `warm`, the harness runs the notebook
   once first to populate the cache, then re-runs the notebook `--repeats`
   times against the populated dir and reports only those.
4. For each code cell: record `t0 = time.perf_counter()`, call
   `shell.run_cell(src)`, record `t1 = time.perf_counter()`, push
   `(cell_index, t1 - t0)` to the result list. Also capture cash-emitted
   `ProcessResult` fields (`execution_time`, `total_time`) by monkey-patching
   `StatementProcessor.process` so its return value is appended to a
   harness-owned sink before being returned to the caller. This gives us
   per-statement (not just per-cell) timing for the cold runs.
5. If `--profile` is set, wrap the per-cell loop in
   `pyinstrument.Profiler(interval=0.001)` and dump HTML to
   `benchmarks/results/<notebook-stem>-<mode>.html`.
6. Write `benchmarks/results/<notebook-stem>-<mode>.json` with per-cell
   timings and run metadata (Python version, cash version, platform, cache
   dir bytes after run, repeat index).

### `benchmarks/compare_modes.py`

Reads the JSON artifacts and emits a markdown table per notebook:

```
| cell | off (ms) | cold (ms) | warm (ms) | cold-off (ms) | (cold-off)/off |
```

Plus a row of totals. This is the per-cell overhead view.

### Profile interpretation

For each notebook, the cold-mode pyinstrument HTML is the primary artifact.
We attribute time under `_execute_statement` to:

- `capture_output.__enter__` / `__exit__`
- `FileAccessTracker.__enter__` (and its children: `_apply_patches`,
  `_patch_module`, `_patch_user_ns`, `sys.meta_path.insert`)
- `FileAccessTracker.__exit__` (and `_unpatch`)
- `ast.parse` (the redundant fallback inside `_execute_statement`)
- `exec` / `eval` (the actual user code)

Pyinstrument's statistical sampling (1ms interval) makes small per-statement
overheads visible because the synthetic notebook calls them ~100x.

## Notebook matrix

Four notebooks, each run in three modes, three repeats (first repeat
discarded as warmup) = 36 timing collections.

| # | Notebook | Purpose |
|---|---|---|
| 1 | `benchmarks/synthetic_micro.ipynb` (generated on first run) — 100 statements like `a_i = i + 1`, zero I/O | Isolates per-statement overhead from compute noise. The notebook that answers "how much hidden cost is in `execution_time`?" |
| 2 | `examples/file_tracking_demo.ipynb` — 20 cells, many small file-IO calls | Maximizes `FileAccessTracker` work. If patch/unpatch dominates, this surfaces it. |
| 3 | `examples/financial_analysis_demo.ipynb` — 32 cells, real data pipeline | Representative typical user workload; tells us the impact users feel. |
| 4 | `examples/cfd_simulation_demo.ipynb` — 15 cells, 16 for-loops, compute-heavy | Tests whether per-iteration loop caching adds overhead beyond linear statements. |

### Fallbacks

If a notebook needs external data we don't have locally, skip and note it.
If runtimes are too short (< 50 ms wall clock per cell) to produce signal
above timer noise, fall back to one or more notebooks from
`examples/large_scale_projects/` (the 10 real-world analysis notebooks
`01_nyc_taxi_analysis.ipynb` … `10_us_flights.ipynb`) as supplementary
sources. These are larger and exercise realistic data-pipeline code paths.

## Metrics and methodology

For each `(notebook, mode, cell)` tuple:

- Wall clock per cell, via `time.perf_counter()`
- Total notebook wall clock
- pyinstrument call graph (cold mode only)
- Cash-internal metrics where available: `ProcessResult.execution_time`,
  `total_time` — captured by hooking the badge render path or by reading
  the magic's last-result accessor

Runs: 3 repeats per (notebook, mode), median reported with min/max range.
First repeat discarded (import / JIT warmup). Single-threaded, no background
processes, system load disclosed in the report header.

## Deliverable

1. `benchmarks/bench_notebook_overhead.py` — harness script
2. `benchmarks/compare_modes.py` — JSON → markdown table emitter
3. `benchmarks/results/` (gitignored) — raw artifacts: per-notebook
   per-mode JSON and HTML
4. `docs/superpowers/specs/2026-05-18-notebook-cache-overhead-results.md` —
   analysis report, structured as:
   1. Methodology and environment disclosure
   2. Per-notebook results tables (off vs cold vs warm)
   3. Overhead decomposition (with flame-graph references)
   4. The hidden-overhead-inside-`execution_time` finding, quantified
   5. Prioritized remediation strategies — each with hypothesis,
      expected savings, implementation cost, risk
   6. Open questions and follow-ups

The remediation list is data-driven; strategies that the source review
already suggests as candidates (subject to confirmation by the numbers):

- Skip or lazy-install `FileAccessTracker` for statements with no I/O risk
  per AST analysis
- Cache the per-module list of patch targets across statements instead of
  re-discovering via `dir(module)` each call
- Fast-path past `capture_output()` when no stdout/stderr/display output is
  produced
- Deduplicate the multiple `ast.parse(code)` calls per statement
  (currently at `process()` entry, inside `_execute_statement`, inside
  `_analyze_and_hash`, and inside `analyze_statement`)
- Move the `start_time` measurement inside the context managers so reported
  `execution_time` reflects user code only — and add an explicit
  `interception_overhead` field so the cost is visible, not hidden

The last item is bookkeeping rather than perf, but the numbers in the
report will make the case for or against doing it.

## Open questions

- Does the pooled-kernel optimization in `nb_runner` already amortize some
  of the per-statement setup we're about to measure? (Probably not because
  pooling reuses kernels across notebooks, not statements within one — but
  worth confirming.)
- Does `PostImportHook` add measurable cost on its own (insertion into
  `sys.meta_path`) when no imports happen during the statement? Need a
  micro-bench to isolate.
- Does the badge's HTML rendering itself contribute non-trivial time? It's
  outside `_execute_statement` but inside `total_time` and may inflate the
  declared "overhead" number in a way that masks the truly-hidden overhead
  inside `execution_time`.
