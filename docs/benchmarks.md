# Benchmarks

How much cash speeds things up depends almost entirely on **the ratio of compute
cost to result size**. A statement that is expensive to compute and cheap to
store is where cash wins big; a statement whose result is nearly as slow to load
from disk as to recompute is where it wins little — or loses, in which case the
[cost model](cost-model.md) tries to decline to persist it.

The numbers below are measured by independent testers on real workloads, not
projections. They are the same figures shown in the project README; this page
adds the methodology and how to reproduce them.

## Measured speedups

| Workload | Measured |
|---|---|
| An expensive loop body (backtest windows, per-entity API calls) | **~190×** on the loop |
| Model training via `@cash.cache` | **~9–11×** |
| Monte Carlo / restart-and-re-run | **~4–5.5×** |
| Big-frame pandas ETL, written the way people naturally write pandas | **~1.2×** |
| The same ETL, restructured cache-friendly | **~1.4–1.6×** |

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
