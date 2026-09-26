# Reading the badge

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. The badge is how you see what cash did in each cell.

Every cell cash runs gets a badge above its output: one summary line, and a panel
with one row per statement. Each row says whether the statement ran or came from
the cache, how long it took or how much time it saved, and, when it ran again or
was not stored, why.

!!! warning "Read the badge, not your prints"
    A restored statement replays what it printed last time. A marker you add to
    see what ran (`print("loading...")`, a timestamp) shows up on a cache hit
    exactly as it did on the run that produced it. (A `@cash.cache` function
    that hits prints nothing: only restored statements replay output.) Only the
    badge tells you what actually ran.

!!! tip "Running headless? Use the text badge"
    The HTML badge is about 20 kB per cell, most of it styling. `nbconvert`, CI
    logs, a diff or a coding agent reading the output see only that styling.
    `%cash_badge print` prints the same information as plain text, one line per
    statement. Use it for headless runs and for notebooks you commit.

## Anatomy

In this cell `df`, from an earlier cell, and `features` came from the cache,
and `preds` ran again because its input changed. The badge is shown open here,
with its parts numbered as in the list below; in a notebook it starts closed.

<iframe class="cash-badge" title="cash badge example: a CACHED cell, opened, parts numbered" src="/_badges/anatomy_open.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<!-- claim: cash/notebook/badge_renderer/view_builder.py:_compute_stats @f3f0282b, cash/notebook/badge_renderer/view_builder.py:_overhead_section @4ab027d7 -->
1. **Header.** The cell's status (`CACHED` here), total time and **net** saving,
   and counters (`exec 1`, `cached 2`). The net saving subtracts cash's own
   overhead for this cell, so it can read a little below the sum of the rows.
   Click the header to open the panel.
2. **Upstream context.** Statements from earlier cells that cash re-checked or
   restored so this cell could run correctly. Collapsed by default.
3. **Current cell.** One row per statement in this cell. Click a row for its
   detail: cache key, storage tiers and the reason it ran.
4. **Cached calls.** Hits and misses of the calls cached inside the statements,
   such as `predict_one()` here: two hits, one miss.
5. **Row layout.** A coloured rail for the status (green: cached, ochre: ran),
   the code, one dot per storage tier (filled: the value is there; ring:
   restored from there; outline: skipped, for example a value over that tier's
   size cap), a timing bar and the time.
6. **Overhead.** cash's own time for the cell: `upstream` (re-checking earlier
   cells), `cache` (hashing and storing results), `badge` and `other`.
7. **Report incorrect caching.** Opens a pre-filled GitHub issue with the badge
   attached.

## Statuses

A row and the cell header use the same word for the same state.

| Label | Rail | Meaning |
|---|---|---|
| **CACHED** | green | The value came from the cache. On the header: nothing ran, or restoring saved more time than running took (`12 restored, 1 ran`). |
| **EXECUTED** | ochre | It ran. On the header: running took more time than restoring saved. |
| **NOT CACHED** | ochre | It ran and cash did not store it, so it runs every time. The row gives the reason. |
| **SKIPPED** | green | Nothing to do, such as an `import` whose names are already bound. |
| **MIXED** | — | A loop where some iterations came from the cache and some ran. |
| **FUNC CHANGED** | red | A function this statement calls was edited. |
| **MODULE RELOADED** | red | A local module you import was edited, and cash reloaded it. |
| **WARNING** | red | Something to look at, such as a notebook file older than what the kernel ran. |
| **ERROR** | red | The statement raised. |
| **BYPASSED** | red | Header only. cash handed the cell straight to IPython, usually because of a syntax error. If the cell looks fine, the kernel log names the error after `Cash auto-caching failed`. |

The header's counters (`EXEC`, `NOT CACHED`, `CACHED`, `WARN`) count rows, so
`EXEC 1 · CACHED 1` under an `EXECUTED` header means one statement ran and one
came from the cache. [`%cash_status`](magics.md#cash_status) reports the same
states by their enum names: `COMPUTED` for EXECUTED and `RESTORED` for CACHED.

<iframe class="cash-badge" title="cash badge example: CACHED" src="/_badges/status_restored.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<p class="cash-badge-caption">Click a badge to open it.</p>

<iframe class="cash-badge" title="cash badge example: EXECUTED" src="/_badges/status_computed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" title="cash badge example: SKIPPED" src="/_badges/status_skipped.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" title="cash badge example: MIXED loop" src="/_badges/status_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" title="cash badge example: WARNING" src="/_badges/status_warning.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" title="cash badge example: ERROR" src="/_badges/status_error.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

### Randomness pills

A statement that touches a random-number generator gets a pill on its row:
`seed` (it sets a seed), `random` (a seeded draw) or `unseeded` (a draw, or an
estimator `.fit()`, with no seed). An unseeded value is cached and replayed, not
redrawn. The text badge shows the same pills as `[seed]`, `[random]` and
`[random: unseeded]`. What to do about them is in
[Randomness](known-limitations.md#randomness).

## Why did this run again?

### First time seeing this code

<iframe class="cash-badge" title="cash badge example: first time seeing this code" src="/_badges/miss_first_time.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

cash has no entry for this statement yet. Any edit to a statement gives it a new
key. The next run with the same code shows `CACHED`.

### Input lineage changed

<iframe class="cash-badge" title="cash badge example: input lineage changed" src="/_badges/miss_input_lineage.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

A variable this statement reads (`features` here) was produced by a different
run than last time. If the upstream statement ran again, everything that reads
it runs again, even when the new value is equal. If you did not expect the
upstream to run, open its row: its reason says what it reacted to.

### A cached call ran again

The row's `sub-call` line reads, for example, `sub-call fit(rows, k): 0/6 hit -
changed: rows`. The name after `changed:` is the part of that call's key that
moved: an argument, a variable it reads, or a global its function uses. Nothing
is named on the first run in a kernel, or when what moved was a file the call
reads or the function's own code.

### File changed

<iframe class="cash-badge" title="cash badge example: file changed" src="/_badges/miss_file_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<!-- claim: cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @b9d64ecd, cash/tracking/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES_DEFAULT == 268435456 -->
A file the statement read (through `pd.read_csv`, `np.load`, `open` and the
other [tracked readers](how-it-works/invalidation.md#what-counts-as-a-change))
has different contents. A new timestamp alone does not count for a file up to
256 MiB (`file_hash_full_max_bytes`): cash compares contents. A bigger file is
hashed in three sampled regions, so touching it or re-downloading an identical
copy runs the statement once more.

### Evicted to make room

<!-- claim: cash/notebook/statement/evictions.py:EVICTED_MISS_REASON @ed666be3, cash/notebook/statement/evictions.py:EvictedRecomputes.attribute @c24d956b -->
The row's reason reads `evicted to make room when the disk cache reached its
size cap`: the value was on disk, and the cache's size cap removed it to make
room for newer results. It shows after a restart, when only the disk copy could
have restored it. A recompute of two seconds or more also warns once
([`CACHE-EVICTED-RECOMPUTE`](warnings.md#cache-evicted-recompute)), naming the
cap and how to raise it.

### Function source changed

<iframe class="cash-badge" title="cash badge example: function source changed" src="/_badges/miss_function_source_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

A function the statement calls, directly or through other functions, was
edited. The `FUNCTION_CHANGED` upstream row names it (`score_rows` here).

### Module reloaded

<iframe class="cash-badge" title="cash badge example: module reloaded" src="/_badges/miss_module_reloaded.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

A local `.py` module you import was edited, and cash reloaded it in your kernel
(plain Jupyter needs `%autoreload` for that). Only statements that use code that
changed run again. If one runs although nothing it uses changed, its function
probably reaches into the module dynamically; the
[invalidation page](how-it-works/invalidation.md) lists those cases.

## Why wasn't this cached?

A **NOT CACHED** row ran, and cash did not store the result, so it runs again
every time. The row names the reason.

Two things that are not cached do **not** show as NOT CACHED: a statement too
cheap to be worth an entry (under 10 ms, see the [cost model](cost-model.md)),
and one marked `# @cash:no-cache`. Both show as a plain `EXECUTED` row with no
reason.

### Side effects

<iframe class="cash-badge" title="cash badge example: NOT CACHED, side effect" src="/_badges/not_cached_side_effect.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<!-- claim: cash/notebook/badge_renderer/view_builder.py:map_status @b33b69ae -->
A cache hit would skip the effect, so cash runs the statement every time. The
reason names the call and its kind: `Side effect: session.post() (network)`,
`Side effect: subprocess.run() (system)`, `Side effect: plt.plot() (display)`,
`Calls save(), which writes files`. Network reads, database reads and prints
are cached. The full list is in
[What gets cached](notebook_caching_api.md#what-gets-cached).

The expensive call **inside** such a statement is still cached, and the row's
`sub-call` line shows its hits. Only the effect runs again.

To cache a statement whose effect is harmless to skip, such as a POST that only
runs a search, mark it [`# @cash:assume-safe`](annotations.md#cashassume-safe).

### In-place mutation

<iframe class="cash-badge" title="cash badge example: NOT CACHED, in-place mutation" src="/_badges/not_cached_mutation.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

The statement changes an object made in an earlier cell (`out.append(...)`,
`d[k] = v`, `df.sort_values(inplace=True)`) instead of producing a new value.
There is nothing to restore that would repeat the change, so the statement runs
every time and cash marks everything downstream as changed. The same holds for a
bare call that changes its argument and returns nothing, such as
`sc.pp.calculate_qc_metrics(adata, inplace=True)`: cash compares the arguments
before and after, and runs the call every time from then on.

**Fix:** assign a new value instead: `out = [f(e) for e in items]`,
`df = df.sort_values(...)`. For estimators, see
[`# @cash:cache-fit`](annotations.md#cashcache-fit).

### Unstable key

```text title="Output"
  NOT CACHED: model = fit(features, y)  (4.20s) - unstable key (`features` changed each run)
```

<!-- claim: cash/notebook/statement/miss_guard.py:GUARD_AFTER_CONSECUTIVE_CHURN_MISSES == 5, cash/notebook/statement/miss_guard.py:REPROBE_EVERY_N_RUNS == 10 -->
The statement ran five times in a row with a different cache key each time and
no hit, so storing it could never pay back. cash stops writing it, keeps looking
it up, and lets one write through every 10 runs or as soon as the key repeats.

<!-- claim: cash/notebook/statement/miss_guard.py:MissGuard.cause @ce6fa03a -->
The row names the input that kept changing, or says "something outside its
inputs" (a file it reads, or the code of a function it calls). The cause is
almost always **upstream**: a statement above that runs every time instead of
being restored, such as a loop that re-runs some iterations, an unseeded draw, or
a value built from the clock or an unordered directory listing. Look at the
badges of the cells above for an `EXECUTED` row you expected to be `CACHED`, and
fix that one.

### Input variable missing lineage

```text title="Output"
  NOT CACHED: segments = load(DATA)  (2.10s) - Input variable missing lineage
```

The statement reads a name cash never saw assigned, so it cannot tell an old
value from a new one. This happens when the name came from `%run`, `exec` or an
IPython startup file. Bind the name in a notebook cell below `%cash_on` instead.

Names bound in the `%cash_on` cell itself are handled: imports, constants and
paths get their lineage from the notebook, and a load such as
`df = pd.read_parquet(...)` is run once more under tracking by the first cell
that needs it (an `^EXECUTED` upstream row). To avoid reading that file twice,
load data in a cell below the first cell.

## See also

- [Debugging](tutorials/feature-guides/debugging-and-monitoring.md): the badge
  together with `%cash_debug` and `%cash_stats`.
- [Writing cache-safe cells](known-limitations.md): cases where a `CACHED` badge
  can still hide a stale value, such as
  [a file read cash cannot see](known-limitations.md#reads-through-a-loader-cash-cannot-see).
