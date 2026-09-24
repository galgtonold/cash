# Benchmarks

!!! info "Applies to: both paths"
    Anyone deciding whether caching will pay off for their workload.

Cash does not speed up the first run; it pays a small cost to fill the cache.
It speeds up every later run, by restoring a result instead of recomputing it.
What you gain is:

```
your compute time  −  restore cost
```

Your compute time is a property of your workload, so this page does not quote
an "N× faster" figure. It publishes the half Cash controls, what a restore
costs, and shows how to measure the other half.

## What a restore costs

**Deserialise time — restoring a cached value back into your session:**

| Result size | DataFrame (RAM) | DataFrame (disk) | ndarray (disk) | raw bytes (disk) |
|---|---|---|---|---|
| 1 KB | 0.01 ms | 5.8 ms | 6.1 ms | 6.9 ms |
| 1 MB | 0.23 ms | 6.7 ms | 5.9 ms | 7.9 ms |
| 10 MB | 2.2 ms | 13 ms | 12 ms | 13 ms |
| 100 MB | 21 ms | 70 ms | 79 ms | 72 ms |

Source: `python benchmarks/measure_ser_deser.py`, three repeats per cell,
committed as `benchmarks/results/ser_deser_matrix.frozen.csv`. The file does
not record the machine it was measured on, so read the table as shape and
order of magnitude, and run the script on your own machine for figures you can
act on. A disk restore is timed in a fresh process reading an entry no process
has read before, as a later session would.

- **Below about 1 MB, a disk restore costs what opening the file costs:**
  about 6 ms, whatever the size.
- **Above that it grows with the size:** a 100 MB DataFrame comes back in
  about 70 ms.
- **The memory tier is cheaper, most of all for small values** (0.23 ms
  against 6.7 ms at 1 MB), but it lasts only as long as the process. A `@cash.cache` result is always written to disk too; for a
  notebook statement, the [cost model](cost-model.md) decides.

Writing costs more than reading: the same 100 MB DataFrame took about 180 ms
to write to disk in that matrix. You pay the write once, on the run that
computes the value, and the read on every run after it.

## Working out your own number

Subtract the restore cost from your compute time:

| Your work | Result size | Restore (disk) | Saved per re-run |
|---|---|---|---|
| A 0.5 s groupby | 10 MB | 13 ms | about 0.49 s |
| A 4-minute feature build | 100 MB | 70 ms | about 4 minutes |
| A 30-minute metric pass | about 1 GB | about 0.7 s (estimated) | about 30 minutes |

The 1 GB row goes past the measured sizes; its restore cost extends the
100 MB figure linearly and is an estimate.

Caching pays when compute is expensive relative to the result's size. Results
that are large but quick to produce (a few hundred megabytes made in a second)
cost more to write and read back than to recompute. The notebook's cost model
declines to write such results to disk; a `@cash.cache` result is written
anyway, so do not decorate such a function.

## What Cash costs you

Each cached call or statement costs some bookkeeping (hashing, lineage, the
key lookup) on top of the write. On a notebook of trivially fast statements
there is nothing to win, and the bookkeeping is all you see:
`benchmarks/synthetic_micro.ipynb` is 100 one-line statements such as
`a_0 = 0 + 1`.

```bash
python benchmarks/bench_notebook_overhead.py benchmarks/synthetic_micro.ipynb --mode off --repeats 5
python benchmarks/bench_notebook_overhead.py benchmarks/synthetic_micro.ipynb --mode cold --repeats 5
```

On a 4-core 2.1 GHz Xeon virtual machine (Linux, Python 3.11, cash 0.11.0), it
took about 17 ms with Cash off and 0.2 to 0.7 s on a first run with Cash on:
2 to 7 ms per statement. Nothing is stored, because every statement is below
the cost model's floor. That is below what you notice in an interactive
session, but it adds up for tens of thousands of statements, or for a
`@cash.cache` function called in a hot loop. Put
[`# @cash:no-cache`](annotations.md) on cheap statements, and decorate
functions whose work is worth more than a few milliseconds.

## Measuring your own workload

=== "Decorator"

    Time a call that computes and a call that restores:

    ```python
    import time

    import cash

    @cash.cache
    def build(n):
        time.sleep(0.2)          # stands in for your real work
        return list(range(n))

    start = time.perf_counter()
    build(1000)                  # first call: cache miss
    compute = time.perf_counter() - start

    start = time.perf_counter()
    build(1000)                  # second call: cache hit
    restore = time.perf_counter() - start

    print(f"compute {compute:.3f} s, restore {restore:.4f} s")
    ```

    The second call is served from memory. For the restore cost a later run
    pays, run the script a second time: its first call reads from disk.
    `build.cache_info()["total_time_saved"]` adds up the compute time recorded
    when each entry was written, so treat it as an upper bound, not a
    measurement.

=== "Notebook"

    Time the cell once without Cash, then let Cash cache it:

    <!-- test:skip reason="IPython magic command — requires kernel context" -->
    ```python
    %cash_off
    ```

    <!-- test:skip reason="IPython magic command — requires kernel context" -->
    ```python
    %%time
    df = pd.read_csv("big.csv").groupby("region").sum()   # uncached cost
    ```

    <!-- test:skip reason="IPython magic command — requires kernel context" -->
    ```python
    %cash_on
    ```

    Now run the same cell, without `%%time`, twice. The first run computes and
    stores the result, the second restores it, and the badge shows each
    statement's time. `%cash_stats` reports the session's time saved, net of
    Cash's own overhead, counting only savings it measured.

## Re-running in the same session or after a restart

"Re-running" means two things, and they give different numbers:

| Case | What can come back |
|---|---|
| The same kernel or process, again | memory and disk |
| After a restart, or in a new process | disk only |

In a notebook, statements that took under 0.1 s stay in memory and do not
survive a restart. So a notebook of many quick statements restores far more
in the same session than after a restart. Say which case you measured when
you quote a number. `benchmarks/bench_notebook_overhead.py` measures both
(`--mode warm-session` and `--mode warm-restart`).

A restore count is not a speedup either. A notebook that restores seventy
cheap statements around one uncacheable ten-second loop saves nothing you can
feel. Look at wall-clock time to decide whether caching pays.

## Reproducing these numbers

The `benchmarks/` directory holds the harness:

- `benchmarks/measure_ser_deser.py`: the restore-cost matrix above.
- `benchmarks/bench_notebook_overhead.py`: one notebook in `off`, `cold`,
  `warm-session` or `warm-restart` mode; each result records the Python
  version, cash version and platform.
- `benchmarks/bench_core.py`: decorator and hashing microbenchmarks.
- `benchmarks/_rerun_sweep.py` and `benchmarks/compare_modes.py`: a sweep over
  the reference notebooks and a per-cell comparison of its results.

## Related

- [Cost model](cost-model.md): how Cash predicts restore and recompute cost
  and decides what reaches disk in a notebook.
- [Why Cash?](why-cash.md): how Cash compares with other caching tools.
