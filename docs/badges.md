# Reading the Cash badge

Every cell Cash touches gets a badge above its output — a one-line summary plus an expandable detail panel. The badge is the answer to "what did cash do, and why?". This page walks through what the badge shows, then runs through the most common questions it gets used to answer.

!!! warning "Read the badge, not your prints"

    **A cache hit replays what the code printed last time.** Both a cached
    `@cash.cache` call and a restored statement write their recorded stdout
    and stderr back out, so a marker you added to see what ran —
    `print("RUN load", file=sys.stderr)`, a counter, a timestamp — appears
    on a hit exactly as it did on the run that produced it.

    Three of five people in one round of user testing nearly reported "it never caches"
    because of this, and one built an entire alternative instrument to get
    around it. The badge is the instrument: its status tells you what ran,
    and your own prints cannot.

    The one thing that really is suppressed is *other cells'* output while
    Cash repairs upstream state for the cell you ran — deliberately, so
    re-running one cell does not replay half the notebook.

!!! tip "Driving Cash headlessly? Switch the badge to text"

    The HTML badge opens with ~700 characters of CSS and costs about **9 kB
    per cell**, so anything that is not a browser — an agent reading captured
    output, `nbconvert`, a diff — sees CSS and no information. It also lands
    in the `.ipynb`, which bloats notebooks you commit.

    `%cash_badge print` gives the same information as one line per cell.
    Prefer it for headless runs and for notebooks that go into version
    control.

## 1. Anatomy

This is a badge from a cell where the upstream `df` was restored, the intermediate `features` was restored, and the final `preds` was recomputed because a new `features` lineage invalidated its cache:

<iframe class="cash-badge" src="/_badges/anatomy_hero.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

What you're looking at:

<!-- claim: cash/notebook/badge_renderer/view_builder.py:_compute_stats @059abbbf, cash/notebook/badge_renderer/view_builder.py:_overhead_section @d2e9d4c3 -->
1. **Header line** — the collapsed view. Shows the cell-level status (`EXECUTED` here), the total time and the **net** saving (`0.42s · saved 3.38s`), a tiny sparkline, and counter chips (`exec 1`, `cached 2`). The header saving is net: it subtracts this cell's own Cash overhead from the gross recompute the restores avoided, so it can read a little below the sum of the per-row savings (that difference is the overhead row at the bottom). Click the header to expand and see the panel below.
2. **Upstream context** — a collapsed pill at the top of the panel labeled "upstream context · 1 step · ↑2.85s". Click to expand and see the upstream rows (statements from earlier cells that Cash had to re-check or re-restore for this cell to be valid).
3. **Current cell** — the section labeled `CURRENT CELL`. Each row is a statement in *this* cell with its per-row status — `restored` (green rail) for `features = encode(df)`, `computed` (ochre rail) for `preds = decorated_predict(features)`. Click any row for a detail tooltip showing the cache key, storage tiers, miss reason, and `@cash.cache` hit ratio (`2/3 cache hits` here).
4. **Decorator cache** — the section labeled `DECORATOR CACHE (@cash.cache)`. One row per `@cash.cache` call during this cell. The example shows three calls to `predict_one()`: two HITs and one MISS.
5. **Per-row anatomy** — left vertical rail color is the status (green = cached, ochre = computed), then the syntax-highlighted code, then storage-tier dots — **one dot per configured backend tier** in order, where a filled (solid) dot means the value is present in that tier, a hollow ring means the value was restored from that tier this run, and an empty outline means the tier was skipped (e.g. Redis bypassed an object over its 10 MiB cap). Hover any dot for the tier name. Then a timing bar scaled to the cell max, then a time chip on the right.
6. **Overhead** — the bottom row labeled `overhead`, split into up to four short sub-parts (hover any part for the full description): `upstream` (re-checking upstream cells), `cache` (Cash hashing and serialising this cell's results into the cache — the dominant overhead for large objects), `badge` (building and updating the badge display), and `other` (everything else). Each per-row time chip above shows a statement's pure compute (or restore) time, so the `cache` cost lives here rather than being folded into those chips. Shows when the breakdown is meaningful relative to cell time.
7. **Report incorrect caching** (footer link) — if a row looks wrong, this opens a pre-filled GitHub issue with the badge output included.

## 2. Status reference

Every row and the overall badge use one of these statuses. The badge identifies status by the **colored left rail** on each row and by the header chip text — not by an icon glyph.

One word per state, and the **same** word whether you are reading a single row or the cell header above it.

| Label | Rail color | When you see it |
|---|---|---|
| **CACHED** | green | The value came from the cache. On the cell header this means nothing ran — or, with counts beside it (`12 restored, 1 ran`), that restoring saved more time than running took. |
| **EXECUTED** | ochre | It ran. On the cell header this means at least one statement ran and running took more time than restoring saved; when some were restored too, the counts say how many of each (`2 ran, 3 restored`). |
| **NOT CACHED** | ochre | It ran and Cash did *not* store the result, so it will run again every time. The row names the reason; the cell header counts these in a `not cached` chip. |
| **SKIPPED** | green | There was nothing to do: a redundant `import` whose names are already bound, or an upstream statement the simulation found already satisfied. Not a branch that didn't run — an untaken branch gets no row at all. |
| **MIXED** | (loop aggregates only) | Some iterations came from cache, some ran. |
| **FUNC CHANGED** | red | A helper function this row calls had its source change since the last run. |
| **MODULE RELOADED** | red | A tracked local import was edited; everything downstream re-runs. |
| **WARNING** | red | Something to look at (e.g. an opaque call Cash can't see through); the row still ran. |
| **ERROR** | red | The statement raised. |
| **BYPASSED** | red | *Cell header only.* Cash stepped aside before processing the cell at all and handed it straight to IPython, so nothing was cached and there are no rows to show. Distinct from `NOT CACHED`, which is about a statement that **did** run under Cash. Usually a syntax error in the cell; if the cell looks fine, check the kernel log for `Cash auto-caching failed` — that names the internal error Cash swallowed on the way out. |

The cell header also carries counter chips — `EXEC`, `NOT CACHED`, `CACHED`, `WARN` — each with a count. They tally rows by family rather than naming this cell's state, so `EXEC 1 · CACHED 1` under an `EXECUTED` header means one statement ran and one came from cache.

`CACHED` and `EXECUTED` are the badge's words. The `CacheStatus` and `BadgeStatus` enums in the API still spell the underlying members `RESTORED` and `COMPUTED`; only the display vocabulary was unified.

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
| `unseeded` | A draw or estimator `.fit()` with **no frozen seed**, in a *notebook statement*. Warn-coloured: cash still caches it, but the cached value is a **frozen replay**, not a fresh draw — re-running won't change it. An unseeded row also bumps the header's warning count. Seed the RNG to make it reproducible, or use `# @cash:no-cache` to redraw every run. A `@cash.cache` function has no badge row; it warns instead ([`RANDOM-UNSEEDED`](warnings.md#random-unseeded)), including when it *returns* an estimator with `random_state=None`. |

A seed that cash re-ran to re-establish the random stream (after you edited an upstream input) shows the row-detail reason **"re-run to restore the random stream"**, so an unchanged seed cell re-executing is explained rather than mysterious.

These markers render in the **text badge** too (`%cash_badge print`) as `[seed]` / `[random]` / `[random: unseeded]` — which matters when you're driving the notebook headlessly (papermill, nbconvert, an agent), where the HTML badge above isn't visible. Per-object generators (`np.random.default_rng()`) are only partially covered here — see [Known limitations](known-limitations.md).

## 3. Why did this re-run?

Five common causes, each with the badge you'll see and the one-line fix.

### First time seeing this code

<iframe class="cash-badge" src="/_badges/miss_first_time.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** Cash has no record of this exact statement having been computed before. Editing a statement (even whitespace, in some cases) makes a new cache key.

**Fix:** Nothing to fix — this is expected. The next run with unchanged code will show as `CACHED` (green rail).

### Input lineage changed

<iframe class="cash-badge" src="/_badges/miss_input_lineage.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** One of the variables this statement reads (`features` here) has a different lineage hash than the last time this statement ran. Cash tracks every assignment's lineage, so a re-run upstream — even if the *value* is identical — invalidates downstream caches that read it.

**Fix:** If the upstream re-ran legitimately (you changed it), there's nothing to fix; let the downstream catch up. If you didn't expect the upstream to re-run, jump to that row's badge — its `miss_reason` will explain what it was reacting to.

### A cached call re-ran

**Why:** the statement's `sub-call` line reads `sub-call fit(rows, k): 0/6 hit`,
and when Cash can say what moved it adds it: `- changed: rows`. The name is a
part of that call's own key — an argument, a variable it reads, a global its
callee reaches — that differs from the last run of the same statement.

**Fix:** the same as for a statement. If that input changed because you meant
it to, the re-run is right. Nothing is added when nothing named moved, when
it is the first run in this kernel, or when the key moved with no visible
part moving (a file the callee reads, or the callee's own source) — naming
nothing beats naming the wrong thing.

### File changed

<iframe class="cash-badge" src="/_badges/miss_file_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** Cash tracks files passed to common I/O calls (`pd.read_csv`, `np.load`, `open`, `Path.read_text`, `joblib.load`, `pickle.load`, `json.load`, and others) and records each file's size and a content hash. The file's **contents** differ from what was recorded when the cache was populated. The size is checked first because it proves a change cheaply; when the size matches, the content hash decides.

<!-- claim: cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @3dd62608, cash/tracking/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES_DEFAULT == 268435456 -->
**Fix:** If you changed the file on purpose, the recompute is correct. For a file up to 256 MiB (`file_hash_full_max_bytes`), a bumped mtime alone will *not* trigger this — a sync tool or a notebook autosave plugin that rewrites the file byte-for-byte leaves the cache valid, so there's nothing to exclude. If you see this badge for such a file without having changed it, the bytes really did move: check for a process rewriting it with different content.

A larger file is different. Cash hashes three regions of it rather than every byte, and the timestamps are what cover the rest — so touching it, or re-downloading an identical copy, recomputes once. See [large objects are hashed by sampling](known-limitations.md#large-objects-are-hashed-by-sampling); raise `file_hash_full_max_bytes` if a big input is rewritten unchanged often enough to matter.

### Function source changed

<iframe class="cash-badge" src="/_badges/miss_function_source_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** A helper function called from this statement had its source change since the last run. The dedicated `FUNCTION_CHANGED` upstream row identifies which one (`score_rows` here).

**Fix:** Expected when you edit a helper. If you didn't edit it, you may be re-importing across kernel sessions where the source bytes differ trivially.

### Module reloaded

<iframe class="cash-badge" src="/_badges/miss_module_reloaded.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Why:** A tracked local module (one you `import` from a local `.py` file) was edited, and Cash **reloaded it in your kernel** -- which plain Jupyter does not do without `%autoreload`. Statements that use what changed re-run; a statement reading only functions you did not touch stays cached, because Cash keys it on the code those functions reach, not on the whole file (see [Invalidation](how-it-works/invalidation.md)).

**Fix:** Expected when you edit the module. If a statement re-runs although nothing it uses changed, its function reaches the module's namespace dynamically or something run at import time differs from run to run; the invalidation page lists what makes Cash fall back to the whole module. If you compare against a plain kernel, give that kernel `%load_ext autoreload` and `%autoreload 2`, or restart it after each edit.

## 4. Why wasn't this cached?

A row labelled **NOT CACHED** (ochre rail) ran but Cash refused to store the result, so it will run again on every future run. The cell header counts these in a `not cached` chip. Six common causes:

!!! note "Unseeded randomness is *not* one of them"
    A statement that draws from an unseeded RNG is still cached — Cash warns
    ([`CashRandomnessWarning`](annotations.md#cashallow-random))
    but does not refuse to store it. If you want it to re-run every time, say so
    explicitly with `# @cash:no-cache`.

### Side effects

<iframe class="cash-badge" src="/_badges/not_cached_side_effect.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<!-- claim: cash/notebook/badge_renderer/view_builder.py:map_status @4b2a5f04 -->
**Why:** The statement writes to a file, sends a network request, mutates a database, or prints/plots — Cash does not cache *statements* with observable side effects, because restoring from cache would skip the side effect.

!!! note "This badge is about the statement, not everything inside it"

    By default Cash still caches an expensive **call inside** such a statement
    — see [Call-level
    caching](annotations.md#call-level-caching-default-and-cashno-cache-calls).
    A call's stdout is captured and replayed on a hit, files it reads are
    tracked as dependencies, and globals its body writes are captured and
    restored with it (which is also why a statement calling such a helper is
    itself left uncached — it re-executes so the write really happens, while
    the call inside it still caches). Effects Cash *cannot* replay — mutating
    an argument, drawing from an RNG — make a call ineligible, so it re-runs.
    A global mutated from inside a callee *is* tracked, at cell level and in a
    loop body alike — but such a call is re-executed rather than served, so it
    does not get the order-independent reuse an ordinary call does ([Mutating
    global state inside a
    function](known-limitations.md#mutating-global-state-inside-a-function)).

    `# @cash:no-cache-calls` turns call-level caching off for a statement or
    cell; `# @cash:no-cache` covers the statement *and* everything in it.

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

That includes a function you pass the object to, when it changes the object in place and returns nothing. This is how scanpy works (`sc.pp.calculate_qc_metrics(adata, inplace=True)`, `sc.tl.leiden(adata)`), and so does a helper like `add_qc(df)`. Cash checks the objects a bare call like that is given before and after it runs, and if one changed, the statement is treated as an in-place mutation from then on. It re-runs every time, so the change is made every time — including after a kernel restart, where the object is restored from the cache and the call is replayed over it before anything below reads it. It does this for a bare call only: `res = f(df)`, where `f` also changes `df`, is not covered, so return the changed object instead.

**Fix:** Assign the result instead of mutating in place — `out = [f(e) for e in items]` caches at any length where the append loop does not. See [A long `for`-append loop can stop caching](known-limitations.md#a-long-for-append-loop-can-stop-caching).

### Unstable key

```text
  NOT CACHED: model = fit(features, y)  (4.20s) - unstable key (`features` changed each run)
  1 statement stopped caching: `model = fit(features, y)`
  (unstable key: the cache key changed every run, so storing
  the value could never pay back). They still run normally; ...
```

<!-- claim: cash/notebook/statement/miss_guard.py:GUARD_AFTER_CONSECUTIVE_CHURN_MISSES == 5, cash/notebook/statement/miss_guard.py:REPROBE_EVERY_N_RUNS == 10 -->
**Why:** The same statement, with the same source, ran five times in a row
and got a different cache key every time, without a single hit. A key that
never repeats can never hit, so writing the value is pure cost, and Cash
stops writing it. Nothing is stale: the statement still runs and its result
is correct. Cash keeps looking the key up, and lets one write through every
10 runs, or at once if the key repeats, so the statement starts caching again
as soon as its key settles.

The key churns because one of the statement's **inputs** gets a new lineage
on every run. That almost always means an upstream value is **recomputed on
every run instead of being restored**. Find that upstream statement and you
have found the cause. The flagged statement is only where it shows up, which
is why one upstream problem can flag a whole chain below it. The usual
suspects are:

- a loop upstream that re-runs some of its iterations each time,
- an unseeded random draw upstream,
- a value built from something that differs between runs, such as the
  current time or a directory listing whose order changes,
- a local module you keep editing, when the statement uses it directly.

<!-- claim: cash/notebook/statement/miss_guard.py:MissGuard.cause @ce6fa03a -->
The row names what kept changing, in brackets: the input or inputs whose
lineage moved most often between those runs (`` `features` changed each run``),
or "something outside its inputs" -- a file the statement reads, or the code
of a function it calls -- when the key moved while every input stayed put.

**Fix:** Look at the badges of the cells *above* the flagged statement, and
find a row that shows `EXECUTED` when you expected `CACHED`. Its miss reason
([§3](#3-why-did-this-re-run)) tells you what it reacted to, and
`%cash_debug on` prints the lineage of every input. Fixing that upstream row
fixes the flagged ones too. Don't rewrite the flagged statement itself.
Rebinding a name (`df = df.assign(c=...)`) does *not* make a key unstable:
re-running such a cell on its own keeps its key.

### Input variable missing lineage

```text
  NOT CACHED: segments = load(DATA)  (2.10s) - Input variable missing lineage
```

**Why:** The statement reads a name that Cash never saw being assigned, so
there is no lineage to put into the key. A key without it could not tell the
old value from a new one, so Cash refuses to store rather than risk a wrong
hit. The statement still runs normally.

This happens when the name was bound somewhere Cash was not watching and
that isn't in the notebook: `%run`, `exec`, or an IPython startup file.
Cash cannot re-run what it cannot see. To fix it, bind the name in a notebook
cell below `%cash_on`.

Names bound in the `%cash_on` cell itself are handled for you:

- **Setup lines** get the lineage the notebook implies: imports, constants,
  literals, `Path(...)`, `os.path.join(...)`. Statements reading them cache
  normally.
- **Anything that could have read something**, such as
  `df = pd.read_parquet(...)` or `RAW = DATA.read_text()`, is re-run once
  under tracking by the first cell that needs it. You'll see it as an
  `^EXECUTED` upstream row. From then on it's tracked against its file like
  any other load. After a restart that re-run is served from the cache, as
  long as the load is expensive enough to be cached at all.

That one re-run is why it's still better to load data in the cells *below*
`%cash_on`. In a fresh session a load in the `%cash_on` cell reads its file
twice: once when that cell runs, and once when the first cell needing it
re-runs it under tracking.

<!-- test:skip reason="illustrative — references missing data/segments.parquet" -->
```python
# cell 1
import cash
%cash_on
import pandas as pd
from pathlib import Path
DATA = Path("data")

# cell 2 -- cached, and tracked against the file
segments = pd.read_parquet(DATA / "segments.parquet")
```

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
