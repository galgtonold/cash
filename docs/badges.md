# Reading the Cash badge

Every cell Cash touches gets a badge above its output — a one-line summary plus an expandable detail panel. The badge is the answer to "what did cash do, and why?". This page walks through what the badge shows, then runs through the most common questions it gets used to answer.

## 1. Anatomy

This is a badge from a cell where the upstream `df` was restored, the intermediate `features` was restored, and the final `preds` was recomputed because a new `features` lineage invalidated its cache:

<iframe class="cash-badge" src="/_badges/anatomy_hero.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

What you're looking at:

<!-- claim: cash/notebook/badge_renderer/view_builder.py:_compute_stats @bceb6587, cash/notebook/badge_renderer/view_builder.py:_overhead_section @67bf94ef -->
1. **Header line** — the collapsed view. Shows the cell-level status (`EXECUTED` here), the total time and the **net** saving (`0.42s · saved 3.38s`), a tiny sparkline, and counter chips (`exec 1`, `cached 2`). The header saving is net: it subtracts this cell's own Cash overhead from the gross recompute the restores avoided, so it can read a little below the sum of the per-row savings (that difference is the overhead row at the bottom). Click the header to expand and see the panel below.
2. **Upstream context** — a collapsed pill at the top of the panel labeled "upstream context · 1 step · ↑2.85s". Click to expand and see the upstream rows (statements from earlier cells that Cash had to re-check or re-restore for this cell to be valid).
3. **Current cell** — the section labeled `CURRENT CELL`. Each row is a statement in *this* cell with its per-row status — `restored` (green rail) for `features = encode(df)`, `computed` (ochre rail) for `preds = decorated_predict(features)`. Click any row for a detail tooltip showing the cache key, storage tiers, miss reason, and `@cash.cache` hit ratio (`2/3 cache hits` here).
4. **Decorator cache** — the section labeled `DECORATOR CACHE (@cash.cache)`. One row per `@cash.cache` call during this cell. The example shows three calls to `predict_one()`: two HITs and one MISS.
5. **Per-row anatomy** — left vertical rail color is the status (green = cached, ochre = computed), then the syntax-highlighted code, then storage-tier dots — **one dot per configured backend tier** in order, where a filled (solid) dot means the value is present in that tier, a hollow ring means the value was restored from that tier this run, and an empty outline means the tier was skipped (e.g. Redis bypassed an object over its 10 MiB cap). Hover any dot for the tier name. Then a timing bar scaled to the cell max, then a time chip on the right.
6. **Overhead** — the bottom row labeled `overhead`, split into up to four short sub-parts (hover any part for the full description): `upstream` (re-checking upstream cells), `cache` (Cash hashing and serialising this cell's results into the cache — the dominant overhead for large objects), `badge` (building and updating the badge display), and `other` (everything else). Each per-row time chip above shows a statement's pure compute (or restore) time, so the `cache` cost lives here rather than being folded into those chips. Shows when the breakdown is meaningful relative to cell time.
7. **Report incorrect caching** (footer link) — if a row looks wrong, this opens a pre-filled GitHub issue with the badge output included.

## 2. Status reference

Every row and the overall badge use one of these statuses. The badge identifies status by the **colored left rail** on each row and by the header chip text — not by an icon glyph.

| Status | Rail color | Header label | When you see it |
|---|---|---|---|
| **RESTORED** | green | `CACHED` chip | Row's value came from the cache. |
| **COMPUTED** | ochre | `EXEC` chip | Row ran. May or may not have been stored — check the row detail for "NOT CACHED". |
| **SKIPPED** | green | `cached` chip | Row was unreachable on this run (downstream of a branch not taken) or its value isn't needed. |
| **MIXED** | (cell header only) | `EXECUTED` summary with both `EXEC` and `CACHED` chips | Cell-level only: some rows restored, some computed. |
| **FUNCTION_CHANGED** | red | warning chip | A helper function this row calls had its source change since the last run. |
| **MODULE_RELOADED** | red | warning chip | A tracked local import was edited; everything downstream re-runs. |
| **WARNING** | red | warning chip | Something to look at (e.g. an opaque call Cash can't see through); the row still ran. |
| **ERROR** | red | (error label) | The statement raised. |

Individual examples:

<iframe class="cash-badge" src="/_badges/status_restored.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_computed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_skipped.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_function_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_module_reloaded.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_warning.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_error.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

### Randomness markers

A statement that touches a random-number generator carries a small text pill on its row, so you can see at a glance which cells control reproducibility and which produce randomness:

| Pill | Meaning |
| --- | --- |
| `seed` | The statement sets a global RNG seed (`np.random.seed(0)`, `random.seed(0)`). Neutral — informational. |
| `random` | A **seeded** draw (`np.random.rand()` after a seed). Neutral — its cached value is reproducible. |
| `unseeded` | A draw or estimator `.fit()` with **no frozen seed**. Warn-coloured: cash still caches it, but the cached value is a **frozen replay**, not a fresh draw — re-running won't change it. An unseeded row also bumps the header's warning count. Seed the RNG to make it reproducible, or use `# @cash:no-cache` to redraw every run. |

A seed that cash re-ran to re-establish the random stream (after you edited an upstream input) shows the row-detail reason **"re-run to restore the random stream"**, so an unchanged seed cell re-executing is explained rather than mysterious.

These markers render in the **text badge** too (`%cash_badge print`) as `[seed]` / `[random]` / `[random: unseeded]` — which matters when you're driving the notebook headlessly (papermill, nbconvert, an agent), where the HTML badge above isn't visible. Per-object generators (`np.random.default_rng()`) are only partially covered here — see [Known limitations](known-limitations.md).

## 3. Why did this re-run?

Five common causes, each with the badge you'll see and the one-line fix.

### First time seeing this code

<iframe class="cash-badge" src="/_badges/miss_first_time.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** Cash has no record of this exact statement having been computed before. Editing a statement (even whitespace, in some cases) makes a new cache key.

**Fix:** Nothing to fix — this is expected. The next run with unchanged code will show as `RESTORED` (green rail).

### Input lineage changed

<iframe class="cash-badge" src="/_badges/miss_input_lineage.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** One of the variables this statement reads (`features` here) has a different lineage hash than the last time this statement ran. Cash tracks every assignment's lineage, so a re-run upstream — even if the *value* is identical — invalidates downstream caches that read it.

**Fix:** If the upstream re-ran legitimately (you changed it), there's nothing to fix; let the downstream catch up. If you didn't expect the upstream to re-run, jump to that row's badge — its `miss_reason` will explain what it was reacting to.

### File changed

<iframe class="cash-badge" src="/_badges/miss_file_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** Cash tracks files passed to common I/O calls (`pd.read_csv`, `np.load`, `open`, `joblib.load`, `pickle.load`, `json.load`, and others) and records each file's size and a content hash. The file's **contents** differ from what was recorded when the cache was populated. The size is checked first because it proves a change cheaply; when the size matches, the content hash decides.

**Fix:** If you changed the file on purpose, the recompute is correct. A bumped mtime alone will *not* trigger this — a sync tool or a notebook autosave plugin that rewrites the file byte-for-byte leaves the cache valid, so there's nothing to exclude. If you see this badge without having changed the file, the bytes really did move: check for a process rewriting it with different content.

### Function source changed

<iframe class="cash-badge" src="/_badges/miss_function_source_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** A helper function called from this statement had its source change since the last run. The dedicated `FUNCTION_CHANGED` upstream row identifies which one (`score_rows` here).

**Fix:** Expected when you edit a helper. If you didn't edit it, you may be re-importing across kernel sessions where the source bytes differ trivially — `%cash_track` and `%cash_verify` help diagnose this.

### Module reloaded

<iframe class="cash-badge" src="/_badges/miss_module_reloaded.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** A tracked local module (one you `import` from a local `.py` file) was edited. Everything downstream of the import re-runs.

**Fix:** Expected when you edit the module. If you want a module *not* to invalidate caches, declare its functions `@cash.pure` so Cash only tracks the relevant function bodies rather than the whole module.

## 4. Why wasn't this cached?

A `COMPUTED` row (ochre rail) that also says **NOT CACHED** ran but Cash refused to store the result. Four common causes:

!!! note "Unseeded randomness is *not* one of them"
    A statement that draws from an unseeded RNG is still cached — Cash warns
    ([`CashRandomnessWarning`](annotations.md#cashallow-random-alias-allowrandom))
    but does not refuse to store it. If you want it to re-run every time, say so
    explicitly with `# @cash:no-cache`.

### Side effects

<iframe class="cash-badge" src="/_badges/not_cached_side_effect.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<!-- claim: cash/notebook/badge_renderer/view_builder.py:map_status @4b2a5f04 -->
**Why:** The statement writes to a file, sends a network request, mutates a database, or prints/plots — Cash never caches statements with observable side effects because restoring from cache would skip the side effect.

**Fix:** Split the side effect off into its own statement and let the *value-producing* statement above it cache. If the side effect *is* the point of the cell, leave it uncached and use `@cash:no-cache` to suppress the warning.

### Cost model: too cheap to cache

<iframe class="cash-badge" src="/_badges/not_cached_too_cheap.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** Cash's cost model predicted that restoring this value from cache would be *slower* than recomputing it. By default a statement must take longer than `min_execution_time_to_cache_seconds` (0.01 s) and the predicted restore must save at least `min_cache_savings_pct` of that time.

**Fix:** If you actually want it cached (e.g. you're benchmarking restore overhead, or the statement is downstream of something more expensive that needs the value early), force it with `# @cash:persist`.

See [Cost model and smart persistence](cost-model.md) for how this decision is made and how to override it.

### Explicit `# @cash:no-cache`

<iframe class="cash-badge" src="/_badges/not_cached_explicit.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** You (or the previous author) annotated the statement to opt out.

**Fix:** Delete the annotation if you no longer want the opt-out.

### In-place mutation

<iframe class="cash-badge" src="/_badges/not_cached_mutation.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** The statement mutates an object that already existed — `out.append(...)`, `d[k] = v`, `df.sort_values(inplace=True)` — rather than producing a new value. There is no snapshot to restore that would reproduce the mutation, so Cash bumps the receiver's lineage (everything downstream stays correct) and re-executes the statement each run.

**Fix:** Assign the result instead of mutating in place — `out = [f(e) for e in items]` caches at any length where the append loop does not. See [A long `for`-append loop can stop caching](known-limitations.md#a-long-for-append-loop-can-stop-caching).

!!! info "Not on this list: a file read through a loader Cash doesn't intercept"
    Cash does **not** refuse to cache a statement because it couldn't see the
    file it read — it caches it like any other statement, and simply has no file
    dependency recorded for it. That is a staleness risk rather than a badge
    state, and it is described under
    [Reads through a loader cash cannot see](known-limitations.md#reads-through-a-loader-cash-cannot-see).

## See also

- [Debugging and monitoring](tutorials/feature-guides/debugging-and-monitoring.md) —
  using the badge alongside `%cash_debug`, `func.explain()`, and `cache_info()`
  to diagnose unexpected misses and silent skips.
