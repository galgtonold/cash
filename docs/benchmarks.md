# Benchmarks

Cash does **not** speed up the first run — it pays a small cost to fill the
cache. What it speeds up is every **re-run** afterwards: restoring a result that
hasn't changed instead of recomputing it.

## Why this page doesn't lead with an "N× faster" number

The obvious headline would be a speedup. We've deliberately not written one,
because the number would mostly not be about cash:

```
speedup  =  your compute time  ÷  cash's restore cost
```

The denominator is a property of cash and your result size. The numerator is a
property of *your workload*. So a quoted speedup is dominated by the term cash
has nothing to do with — pick a slower computation and the number goes up, with
no change to the tool. It is a claim that can't really be wrong, which makes it
close to useless.

It also **inverts** the thing you care about:

| workload | speedup | actually saved |
|---|---|---|
| a 2-second cell | 190× | 2 seconds |
| a 10-hour pipeline | 1.4× | 3 hours |

The bigger ratio is the smaller saving. So instead of asserting a multiplier,
this page publishes the half of the equation cash actually determines — what a
restore costs — and lets you supply the other half.

## What a restore costs

Measured by `benchmarks/measure_ser_deser.py` across result types and sizes. The
frozen matrix is committed at `benchmarks/results/ser_deser_matrix.frozen.csv`,
so these figures are reproducible from a clean checkout rather than quoted from
a session someone else ran.

**Deserialise time — restoring a cached value back into your session:**

| Result size | DataFrame (RAM) | DataFrame (disk) | ndarray (disk) | raw bytes (disk) |
|---|---|---|---|---|
| 1 KB | 0.01 ms | 0.5 ms | 1.4 ms | 0.5 ms |
| 1 MB | 0.18 ms | 11 ms | 14 ms | 21 ms |
| 10 MB | 1.6 ms | 22 ms | 26 ms | 13 ms |
| 100 MB | 16 ms | 166 ms | 164 ms | 52 ms |

Two things to read off it:

- **Restore cost is roughly linear in size, and modest in absolute terms.** A
  100 MB DataFrame comes back from disk in about a sixth of a second.
- **The RAM tier is one to two orders of magnitude cheaper than disk.** Which
  tier a value lands in is the [cost model's](cost-model.md) decision, not one
  you have to make.

## Working out your own number

Take your compute time and divide:

```
your compute time  ÷  restore cost from the table  =  your speedup
```

Three workloads, same arithmetic:

| Your work | Result size | Restore | Reclaimed per re-run |
|---|---|---|---|
| A 0.5 s groupby | 10 MB | 22 ms | 0.48 s — real, but you won't feel it |
| A 4-minute feature build | 100 MB | 166 ms | ~4 minutes, every iteration |
| A 30-minute metric pass over a 1 GB input | ~1 GB | ~1.7 s | ~30 minutes (≈1000×) |

That last row is where this kind of caching earns its keep, and it is worth
being explicit about why. If a pipeline computes expensive per-item metrics and
*later* stages aggregate across many items, then every time you change the
aggregation — a different grouping, one more chart, a fixed bug in the report —
you would otherwise re-run the whole metric pass. With the per-item results
cached, the aggregation becomes something you can iterate on directly.

The multiplier isn't really the point there. **The threshold crossing is**: work
that took long enough to context-switch away from becomes work you can sit in
front of. That is a different activity, not just a faster one.

Note that you didn't need us to assert ≈1000× — it fell out of your compute time
and our restore cost. That is the arithmetic worth trusting, because you supplied
the part that varies.

## What cash costs you

The honest other direction:

- **The first run is slower.** Cash pays to fill the cache before it can pay you
  back — around 1.3× on a big-frame ETL. The win is on iteration and restart,
  never on first execution.
- **Roughly 5–30 ms per cached statement** on a cold run: lineage computation,
  cache-key hashing, and the write.
- **On trivially cheap cells, cash is a net loss.** In our own sweep a notebook
  of sub-second cells (`benchmarks/synthetic_micro.ipynb`) ran about **35×
  slower** with caching on — 7.5 ms of work became 261 ms. Nothing is malfunctioning there — per-statement
  bookkeeping simply outweighs microsecond statements. The
  [cost model](cost-model.md) declines to persist such results, but it cannot
  make the bookkeeping free.

If your notebook is made of fast cells, cash is not the tool for it — and
`%cash_stats` will say so plainly rather than reporting a phantom win.

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

That number is the one worth quoting internally.

## Reproducing these numbers

The `benchmarks/` directory holds the harness:

- `benchmarks/measure_ser_deser.py` — the restore-cost matrix above.
- `benchmarks/bench_core.py` — core decorator + hashing microbenchmarks.
- `benchmarks/bench_notebook_overhead.py` — per-statement notebook overhead.
- `benchmarks/_rerun_sweep.py` — the full notebook sweep, all modes.
- `benchmarks/compare_modes.py` — reads a sweep's results into a per-cell table.

!!! note "The end-to-end notebook sweep is not currently a fair measure"
    The reference notebooks in `benchmarks/` are built around per-object RNG
    generators (`np.random.default_rng(...)`), whose stream position cash does
    not track. Warm runs of that suite therefore restore very little for reasons
    unrelated to how cash performs on ordinary work, and its end-to-end ratios
    should not be read as representative in either direction. The restore-cost
    matrix above does not have that problem — it measures serialisation
    directly.

Restore costs depend on hardware and storage speed, so treat the table as shape
and order of magnitude, and run the harness on your own machine for figures you
can act on.

## Related

- [Cost model](cost-model.md) — how cash predicts restore vs recompute cost and
  decides what reaches disk.
- [Why Cash?](why-cash.md) — the capability comparison against other caching
  tools, and the same arithmetic as a slider you can drag.
