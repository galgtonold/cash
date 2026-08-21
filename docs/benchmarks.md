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

**The first run is slower**, because cash has to fill the cache before it can
pay you back. The cost is not a multiplier — it is roughly **5–30 ms per cached
statement** for lineage and cache-key hashing, plus the write, which scales with
your result size. Those are the terms to think in:

- On a notebook of small results, the overhead is the per-statement figure and
  little else. Tens of milliseconds across a cell you wait seconds for is not
  something you will notice.
- On a notebook of large results, the write dominates, and it is the same
  serialisation cost as the restore-cost table above — read it in reverse. A
  100 MB DataFrame costs about a sixth of a second to write, once.

Quoting a single "first run is N× slower" number would be the same mistake as
quoting a speedup: the ratio is set by how cheap your compute is, not by how
expensive cash is.

**Where it buys you nothing: notebooks made of very fast statements.** Our
own negative control (`benchmarks/synthetic_micro.ipynb`) is 100 statements of
microsecond work. With caching on it goes from 7.5 ms to 261 ms — a quarter of
a second added, on a notebook that finishes instantly either way. That is the
per-statement bookkeeping and nothing is malfunctioning; the
[cost model](cost-model.md) correctly declines to store any of it, but it
cannot make the decision itself free.

Read that as neither "35× slower" — which sounds alarming and means very
little — nor as a cost you are paying. A quarter of a second spread across a
whole notebook sits below what anyone perceives interactively, where you are
already absorbing kernel round-trips of the same order. Cash gives you nothing
here, but it does not take anything you would feel either.

The same per-statement cost *does* become visible at scale: tens of thousands
of statements, or `@cash.cache` on a function called in a hot loop. There the
bookkeeping stops being noise — in that control it is ~2.5 ms per statement
even though the cost model stores none of them, and a statement that *is*
stored pays the 5–30 ms above instead. That is a reason to put
[`# @cash:no-cache`](annotations.md) on the cheap parts, not a reason to leave
caching off.

**The shape that genuinely costs you is the opposite one: results that are
large but cheap to produce.** A few hundred megabytes that took a second to
generate costs more to write and read back than to recompute, so there is
nothing for caching to win. Cash's [cost model](cost-model.md) declines to
*persist* results like these — which is what stops the loss compounding —
but the work of sizing and hashing them still happens.

The pattern behind both: caching pays when compute is expensive relative to
the result's size. Cheap-and-small and cheap-and-huge are the two ways to be
on the wrong side of that — but they are not equally bad. Cheap-and-small
wastes nothing you can feel; cheap-and-huge spends real I/O on results you
would have been better off recomputing. If your notebook is mostly the second
shape, cash is not the tool for it — and `%cash_stats` will say so plainly
rather than reporting a phantom win. Measure your own case with
[`%cash_benchmark`](magics.md#cash_benchmark) rather than trusting a figure
from someone else's machine.

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

### The two warm modes

"Re-running" means two different things, and they give different numbers:

| mode | models | what can come back |
|---|---|---|
| `warm-session` | same kernel, run the cells again | RAM **and** disk |
| `warm-restart` | kernel restarted, or a fresh process | disk only |

The distinction is the [cost model's](cost-model.md) tiering. A result whose
compute time is under ~0.1s is kept in memory and never written to disk —
cheap to recompute, not worth the write. Those values survive re-running a cell
in a live kernel and do not survive a restart.

So `warm-restart` is the pessimistic bound and `warm-session` the optimistic
one; your day is somewhere between. If you quote one number, say which.

Across our reference suite the gap is wide wherever results are mostly
sub-0.1 s — these are counts of statements restored rather than recomputed, on
the same notebook in the same sweep:

| notebook | warm-session | warm-restart |
|---|---|---|
| `cfd_simulation_demo` | 71 of 286 | 1 of 286 |
| `financial_analysis_demo` | 65 of 141 | 16 of 141 |
| `bench_cost_model_validation` | 13 of 44 | 5 of 44 |

**A restore count is not a speedup.** `cfd_simulation_demo` restores 71
statements against 1 and takes the same wall-clock time either way, because
its cost is a single sequential solver loop that cash declines to cache at
all. Restoring seventy cheap statements around an uncacheable ten-second loop
saves nothing you can feel. Count restores to understand *what the tiers do*;
look at wall clock to decide whether you care.

**Treat end-to-end ratios from this suite as orders of magnitude.** The
cash-off baseline for `synthetic_heavy` has measured between 3.9 s and 7.3 s
across repeated sweeps of identical code on one machine — enough to move its
warm-session ratio between roughly 20× and 45×. Nothing changed but the
weather. This is the same reason the page leads with restore cost: that number
is stable and it is the half cash actually determines.

!!! note "Read the end-to-end sweep per notebook, not as one number"
    The reference notebooks differ enormously in how much of their work is
    cacheable at all, so averaging them produces a figure that describes no
    workload. `synthetic_heavy` restores nearly everything; `synthetic_micro`
    restores nothing *by design* (it is the negative control — every statement
    is below the cost-model floor); and the solver notebooks are dominated by
    sequential loops that accumulate with `.append()`, which cash declines to
    cache and says so.

    Each run's JSON records `uncacheable_reasons` per statement, so a notebook
    that restores little will tell you why rather than leaving you to guess.
    Check that field before reading an end-to-end ratio as a statement about
    cash.

Restore costs depend on hardware and storage speed, so treat the table as shape
and order of magnitude, and run the harness on your own machine for figures you
can act on.

## Related

- [Cost model](cost-model.md) — how cash predicts restore vs recompute cost and
  decides what reaches disk.
- [Why Cash?](why-cash.md) — the capability comparison against other caching
  tools, and the same arithmetic as a slider you can drag.
