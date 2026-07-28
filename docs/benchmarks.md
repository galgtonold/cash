# Benchmarks

Cash does **not** speed up the first run — it pays a small cost to fill the
cache. What it speeds up is every **re-run** afterwards: restoring a result that
hasn't changed instead of recomputing it. So these numbers are all *reuse*
speedups — how much faster the **second** run of an unchanged statement is.

How big the saving is depends almost entirely on **the ratio of compute cost to
result size**. A statement that was expensive to compute and is cheap to store
is where a restore wins big; a statement whose result is nearly as slow to load
from disk as to recompute is where it wins little — or loses, in which case the
[cost model](cost-model.md) tries to decline to persist it.

!!! warning "These figures are being re-validated — do not cite them yet"
    The numbers below came from external tester sessions. They are **not**
    currently reproducible from the harness in `benchmarks/`, and the archived
    results that ship with the repo do not support them:

    - The archived run (`cash_version: 0.5.0b1`) recorded heavy per-statement
      **errors** in warm mode — 7 of 11 statements in `synthetic_heavy`, 22 of
      50 in `01_nyc_taxi_analysis`. A notebook that dies partway does less work,
      so its warm time is artificially low and the ratio is inflated.
    - A clean re-run of all ten reference notebooks on the current code tops out
      at **5.9x**, not ~190x.
    - That re-run is not a fair measure either: the reference notebooks are
      built around per-object RNG generators, whose stream position cash does
      not track, so warm mode restores almost nothing (3.63 s recomputed vs
      0.03 s restored on `synthetic_heavy_xl`) for reasons unrelated to how
      cash performs on ordinary work.

    Both the suite and the figures need rebuilding before this page can claim
    reproducibility. Until then, measure your own workload — the methodology
    below is sound even though the headline numbers are not yet backed.

## How much a re-run saves

<div class="cash-bench" role="img" aria-label="Re-run (reuse) speedups by workload on a log scale: re-running an expensive loop about 190 times faster, a cached model fit 9 to 11 times, a restarted Monte Carlo 4 to 5.5 times, a cache-friendly ETL 1.4 to 1.6 times, a naive big-frame ETL 1.2 times.">
  <div class="cash-bench-row">
    <span class="cash-bench-label">Re-run an expensive loop<br><span class="cash-bench-note">backtest windows, per-entity API calls</span></span>
    <span class="cash-bench-track"><span class="cash-bench-bar cash-bench-hi" style="width:99%"></span></span>
    <span class="cash-bench-val">~190×</span>
  </div>
  <div class="cash-bench-row">
    <span class="cash-bench-label">Re-run a cached model fit <span class="cash-bench-note">(<code>@cash.cache</code>)</span></span>
    <span class="cash-bench-track"><span class="cash-bench-bar cash-bench-hi" style="width:44%"></span></span>
    <span class="cash-bench-val">~9–11×</span>
  </div>
  <div class="cash-bench-row">
    <span class="cash-bench-label">Restart, then re-run a Monte Carlo</span>
    <span class="cash-bench-track"><span class="cash-bench-bar cash-bench-hi" style="width:30%"></span></span>
    <span class="cash-bench-val">~4–5.5×</span>
  </div>
  <div class="cash-bench-row">
    <span class="cash-bench-label">Re-run a cache-friendly big-frame ETL</span>
    <span class="cash-bench-track"><span class="cash-bench-bar cash-bench-lo" style="width:8%"></span></span>
    <span class="cash-bench-val">~1.4–1.6×</span>
  </div>
  <div class="cash-bench-row">
    <span class="cash-bench-label">Re-run the same ETL, written naturally</span>
    <span class="cash-bench-track"><span class="cash-bench-bar cash-bench-lo" style="width:3.5%"></span></span>
    <span class="cash-bench-val">~1.2×</span>
  </div>
</div>

<p class="cash-bench-note">Each bar is how much faster the <strong>re-run</strong> is, on a <strong>log scale</strong> — a ~190× win and a ~1.2× win are both shown honestly. Green = a big win (was expensive to compute, cheap to store); ochre = marginal (the result is nearly as slow to reload as to recompute).</p>

Two honest caveats that the table can't show on its own:

- **Run #1 is slower** — around 1.3× on a big-frame ETL. Cash pays to fill the
  cache before it can pay you back; the win is on iteration and restart, not the
  first execution.
- **The loop number depends on how the loop is written.** Assign the result
  (`out = [f(e) for e in items]`) and it caches at any length; append into a list
  inside a long, multi-statement loop body and it can stop caching entirely. The
  [badge](glossary.md#badge) says when that happens — see
  [known limitations](known-limitations.md#a-long-for-append-loop-can-stop-caching).

`%cash_stats` reports the totals for *your* session — hits, misses, and time
saved — and will say so plainly when cash cost you time rather than saving it.

## Cold-run overhead

On a cold run cash adds roughly **5–30 ms per cached statement** (lineage
computation, cache-key hashing, and the write). For real notebook work that
overhead is dwarfed by the work being cached; it only dominates when the
statements themselves are trivially cheap, in which case the size-aware floor
keeps those results in RAM rather than paying disk I/O. The
[cost model](cost-model.md) lays out the maths.

## Measuring your own workload

You don't need the repo to get numbers for the thing you actually care about.
`%cash_benchmark` arms the **next** cell to run N timed iterations:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_benchmark 5 --compare
# next cell — this is what gets measured:
df = pd.read_csv("big.csv").groupby("region").sum()
```

`--compare` also runs the cell N times with caching off, so the output includes a
cached-vs-uncached speedup for *your* data; `--cold` clears the cache before each
cached iteration to measure cold-start instead. Timing uses `perf_counter`, so
fast cells aren't distorted by Windows' ~16 ms clock granularity. See
[`%cash_benchmark`](magics.md#cash_benchmark) for the full flag list.

That number is the one worth quoting internally — the table above is shape and
order of magnitude, not a promise about your hardware.

## Reproducing these numbers

The `benchmarks/` directory in the repository holds the harness:

- `benchmarks/bench_core.py` — core decorator + hashing microbenchmarks.
- `benchmarks/bench_notebook_overhead.py` — per-statement notebook overhead.
- `benchmarks/compare_modes.py` — end-to-end cash-on vs cash-off comparison.
- `benchmarks/synthetic_heavy.ipynb`, `synthetic_micro.ipynb`,
  `synthetic_heavy_xl.ipynb` — representative notebook workloads spanning the
  compute-to-size spectrum above.

Results land in `benchmarks/results/`. Because the numbers depend on hardware,
storage speed, and the exact workload, treat the table above as *shape and order
of magnitude*, and run the harness on your own machine and data for figures you
can act on.

## Related

- [Cost model](cost-model.md) — how cash predicts restore vs recompute cost and
  decides what reaches disk.
- [Why Cash?](why-cash.md) — the capability comparison against other caching
  tools.
