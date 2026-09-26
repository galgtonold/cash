# Restarts and persistence

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. What survives a kernel restart, and how to keep
    more of it.

cash keeps every cached statement in memory for the rest of the session and
writes the ones worth keeping to disk. After a kernel restart, only what is on
disk comes back.

## What survives a restart

A statement's result goes to disk when it took more than **0.1 s** to compute
and reading it back is clearly faster than computing it again. Quicker results
stay in memory, and results under 10 ms are not stored at all. The
[cost model](../../cost-model.md) has the exact rules and the settings that tune
them.

With `%cash_badge print` each row shows where its result went. Here `slow`
takes 0.3 s and `mid` 0.05 s:

```text title="Output"
[Cash] EXECUTED (0.37s)
  EXECUTED: a = slow(1)  (0.30s) -> RAM+DISK
    sub-call slow(1): 0/1 hit
  EXECUTED: b = mid(2)  (0.05s) -> RAM
    sub-call mid(2): 0/1 hit
  EXECUTED: c = a + 1  (0.00s)
  @cash.cache:
    slow() [intercepted]: 0/1 cached (0.300s)
    mid() [intercepted]: 0/1 cached (0.050s)
```

After a restart, the same cell restores `a` and runs `b` and `c` again:

```text title="Output"
[Cash] CACHED (1 restored, 2 ran; 0.06s, saved 0.29s)
  CACHED: a = slow(1)  (saved 0.30s)
  EXECUTED: b = mid(2)  (0.05s) -> RAM
    sub-call mid(2): 0/1 hit
  EXECUTED: c = a + 1  (0.00s)
  @cash.cache:
    mid() [intercepted]: 0/1 cached (0.050s)
```

The HTML badge shows the same with the storage dots on each row.

## How a restart picks up

Run any cell after a restart. cash works out which variables the cell needs,
checks that the code and files behind each one are unchanged, and restores them
from disk instead of running the cells that built them. A deep pipeline comes
back in the time it takes to read the results. Whatever it cannot restore, such
as a value that was kept in memory only, it computes by running the statements
that produced it. See
[Picking up after a kernel restart](../../how-it-works/notebook-path.md#picking-up-after-a-kernel-restart).

<!-- claim: cash/notebook/statement/rebuild_cost.py:RebuildCostLedger.end_cell_persistence @b5fce68e -->
**A cheap value over expensive inputs is written too.** `latest =
sales["week"].max()` takes milliseconds, but after a restart `sales` may be gone,
and so is everything it was built from. So at the end of each cell, cash adds up
what rebuilding each value would cost after a restart (the statement plus every
statement behind it that is not on disk) and writes the value when reading it
back is cheaper. Only the version the cell ends with is written: a name the cell
assigns three times is stored once.

<!-- claim: cash/notebook/upstream/simulator.py:NotebookSimulator.plan_cell_run @e6aa3247 -->
Running such a cell again after a restart restores the last versions it has on
disk and runs only the statements they do not cover, in order.

## Keeping more

To store a result on disk however quick it was, mark the statement:

<!-- test:skip reason="illustrative: build_lookup and raw are the reader's own" -->
```python { .nb-cell }
# @cash:persist
lookup = build_lookup(raw)  # 50 ms, but needed after every restart
```

`%cash_persist on` does the same for every statement until `%cash_persist off`.
It suits a benchmark or a reproducible run; for everyday work it fills the disk
with values that are faster to compute than to read.

A slow cell made of cheap parts is a common surprise: a cell of 120 independent
statements at 0.05 s each takes six seconds and stores nothing on disk, because
the threshold applies to each statement on its own. Mark the ones you need with
`# @cash:persist`, or move the expensive work into one statement.

## Checking what was written

- The badge: `-> RAM+DISK` in the text badge, the storage dots in the HTML
  badge.
- `%cash_debug on` logs `[STORAGE] Stored in: RAM, DISK` for each write, and
  `[SIZE_AWARE] ... below 10ms floor` for a result too cheap to store. See
  [Debugging](debugging-and-monitoring.md).
- `cash inspect` in a terminal lists what is on disk, with the time each entry
  saves.

## Related

- [Cost model](../../cost-model.md): the thresholds and how to tune them.
- [Notebook guide](../../notebook_caching_api.md#where-the-cache-is): where the
  cache folder is and how to clear it.
- [Where your cache lives](../../how-it-works/storage.md): memory and disk tiers,
  size caps and eviction.
