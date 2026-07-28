# Debugging and monitoring — figuring out why Cash did what it did

When Cash isn't behaving the way you expect — missing when you thought it'd hit, hitting when you expected fresh — there are four tools for figuring out why: `f.explain()` for per-call introspection, `%cash_debug` for verbose tracing, `%cash_stats` / `cache_info()` for aggregate health, and the `cash` CLI for inspecting on-disk state.

This guide walks through all four, plus the common diagnostic patterns and the cache-management commands you reach for once you've found the problem.

## Why this exists

Cash makes a *lot* of decisions per call: build the cache key from the function source, dependency state, and arguments; look it up in the backend; check the TTL; check file-dependency freshness; serve the cached value or recompute. When something goes wrong the symptom is usually "it ran when I expected a hit" or "it hit when I expected a re-run" — each with several possible causes. The tools below let you match a symptom to its actual cause instead of guessing.

## Quick start

```python
import cash

@cash.cache
def expensive(x):
    return x ** 2

expensive(5)                       # compute
expensive.explain(5)               # why would the next call hit?
# [HIT] __main__.expensive — hit
#   cache_key: __main__.expensive:9a3c...:...
#   cached_at: 1779637032.79
#   cache_age_seconds: 0.05
#   execution_time_saved: 0.0008

expensive.cache_info()
# {'hits': 0, 'misses': 1, 'hit_rate': 0.0, 'total_time_saved': 0.0, 'warnings': []}
```

That's the decorator path. In a notebook the equivalents are `%cash_debug on`, `%cash_stats`, and the badge above each cell. On disk, `cash inspect` and `cash clear` cover everything from outside the kernel.

## Tool 1: `f.explain()` — the diagnostic API

Every function wrapped with `@cash.cache` gets an `explain` attribute. Call it with the same args you'd pass to the function and it tells you exactly what would happen on the next real call — without computing anything, without mutating stats, without touching the backend.

```python
import cash

@cash.cache(ttl=60)
def fetch_user(uid):
    return {"id": uid, "name": "..."}

fetch_user.explain(42)              # no entry yet
fetch_user(42)                      # compute and store
fetch_user.explain(42)              # hit
```

<!-- claim: cash/core.py:CacheExplanation @28df5e84 broad="the field list and reason set are a claim about the whole dataclass", cash/core.py:Cash._explain_call @02d3d980 -->
The return value is a `CacheExplanation` dataclass (`would_hit`, `reason`, `func_name`, `cache_key`, `details`) with five fields and one of five reason codes:

| `reason` | Meaning | Key `details` |
|---|---|---|
| `hit` | Next call returns cached value. | `cached_at`, `cache_age_seconds`, `execution_time_saved` |
| `no_entry` | No matching cache entry — first call with these args, the cache was cleared, or the function source / a tracked dependency changed since the last write. | `hint` |
| `ttl_expired` | Entry exists but the configured `ttl` has elapsed. | `ttl_seconds`, `age_seconds`, `cached_at` |
| `file_changed` | An auto-tracked file dependency changed. Invalidation is decided by **content**: size first, then a content hash when the size matches — a touch alone is not a change. | `changed_files: {path: reason}` |
| `key_uncomputable` | The args couldn't be hashed (unpicklable type, custom hasher needed). | `arg_type`, `error`, `hint` |

`Cash._explain_call` walks the same code path as a real call up to "would I get a hit?", then returns the verdict instead of executing. Its file-dependency arm delegates to the shared content-authoritative `file_dep_is_fresh`, the same helper the real lookup uses, so the explanation and the call cannot disagree — a **touch** (identical bytes, bumped mtime) explains as `hit`. The `changed_files` values are `'content changed'`, `'size changed'`, or `'file missing'`.

## Tool 2: `%cash_debug on` / `%cash_debug off`

Inside a notebook, `%cash_debug on` raises the cash logger to DEBUG and prints labelled lines from each subsystem as cells execute. Turn it off with `%cash_debug off` (or pipe to JSON with `%cash_debug json`, or to a file with `%cash_debug file <path>`).

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @ce13e22b -->
The five log prefixes you'll see most:

| Prefix | What it tells you |
|---|---|
| `[CACHE_KEY]` | How the cache key for a statement was constructed (source hash, dependency hashes, args hash). |
| `[CACHE_HIT_DEBUG]` | Why a lookup hit or missed, including which validation step failed (TTL, file deps, …). |
| `[UPSTREAM_DEBUG]` | What made an upstream cell invalidate, cascading into a downstream re-run. |
| `[LINEAGE_DEBUG]` | Which inputs were detected for a statement and what their resolved lineage hashes are. |
| `[STATE]` | The tracking state at each step of cell execution. |

```python { .nb-cell }
%cash_debug on

# Run a cell that you expected to hit but didn't:
result = featurize(df)
# [CACHE_KEY]      featurize: state=b7e2... args=4a91... key=featurize:b7e2:...:4a91
# [CACHE_HIT_DEBUG] featurize: lookup miss — no entry for this key
# [UPSTREAM_DEBUG] featurize: upstream df changed (state hash b7e2 vs prev 9a3c)
# [LINEAGE_DEBUG]  featurize: inputs={df: lineage=...}, depends_on=[_normalize]
# [STATE]          tracked_vars: {df: ..., result: ...}

%cash_debug off
```

The output is verbose by design — leave it on only long enough to diagnose. JSON mode (`%cash_debug json`) gives structured records that are easier to filter; file mode persists them past the notebook session.

## Tool 3: `%cash_stats` and `cache_info()`

For health checks rather than per-call diagnostics, you want aggregates.

### In a notebook — `%cash_stats`

```python { .nb-cell }
%cash_stats
```

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @851d8352 -->
Prints a session-wide summary: cells executed, statements computed / restored / skipped, hit rate, and a time ledger of gross saved, cash overhead, and net saved. The net line is the honest headline, and it is **not** gross minus overhead: it credits only savings this session *verified* by recomputing the same statement, minus the measured overhead. Gross is printed beside it and labelled *(estimated)*, because it values each restore at what the entry cost when first written and nothing re-measures that. The consequence is deliberate understatement — an overstatement would be the bug — and a real loss prints as one ("cash cost you Xs this session"). `%cash_stats json` returns the same numbers as a dict (including `total_overhead`, `net_time_saved`, and `net_time_saved_upper_bound`); `%cash_stats reset` zeros the counters.

!!! warning "It only sees decorator hits on the *default* Cash instance"
    `%cash_stats` drains the per-instance decorator call log of the notebook's
    own `Cash` (or the module-level `cash.cache` singleton). A hit on a
    separately-constructed instance — `c = cash.Cash()`, the pattern several
    guides on this site use for `register_hasher`, custom backends and
    `file_depends_on` — is **not** counted. Measured, same workload both ways:

    | decorator | `cache_info()['total_time_saved']` | `%cash_stats` gross |
    |---|---|---|
    | `@cash.cache` (global) | 0.401 s | 0.401 s |
    | `@c.cache` on `c = cash.Cash()` | 0.403 s | **0.0 s** |

    Both genuinely hit the cache. If you use a custom instance, trust
    `cache_info()` and `explain()` for that function; `%cash_stats` will
    understate your session.

### On a decorated function — `cache_info()`

Each `@cash.cache` wrapper carries a per-function counter:

```python
@cash.cache
def expensive(x):
    return x ** 2

for x in range(10):
    expensive(x % 3)

expensive.cache_info()
# {'hits': 7, 'misses': 3, 'hit_rate': 0.7,
#  'total_time_saved': 0.0021, 'warnings': []}
```

The full shape and field meanings:

- `hits`, `misses`, `hit_rate` — counters since the wrapper was created (not since process start).
- `total_time_saved` — sum of execution times avoided on hits.
- `warnings` — rolling log of the last 20 `CashWarning` emissions for this function. Each entry has `category`, `message`, `timestamp`. Survives `warnings.simplefilter('ignore')` so you can find silent misbehavior after the fact.

### Reading hit rate

- **>80%** — healthy. Your code is deterministic enough that the cache is doing useful work.
- **40–80%** — varies by workload. A research notebook with frequent re-runs against new arguments lives here.
- **<40%** — something is fighting the cache. Walk through the patterns below.

## Tool 4: CLI `cash inspect` and `cash clear`

Outside a notebook (CI, scripts, postmortem), the `cash` CLI inspects and manages cache directories on disk:

```bash
cash inspect                          # summarise ./.cash
cash inspect ./notebooks/analysis.ipynb   # inspect the .cash next to a notebook
cash inspect /tmp/some-cache-dir      # any directory

cash clear --all                      # nuke ./.cash (no confirmation)
cash clear ./notebooks/analysis.ipynb # nuke the sibling .cash
```

`cash inspect` reports total size, entry count, file-extension breakdown, and the five most recent entries with timestamps and cache-key prefixes. See the [CLI reference](../../cli.md) for the full output and flag list.

**When to reach for the CLI vs notebook magics** — the CLI when you can't (or don't want to) start a kernel: post-incident inspection on a CI machine, clearing a runaway cache directory on a teammate's box, or scripted size monitoring. Inside an active notebook, `%cash_stats` and `f.cache_info()` are faster.

## Common patterns — what each diagnostic tells you

### "Hit rate is low"

<!-- claim: cash/core.py:Cash._wrap_with_stats.cache_info @b3cd263b -->
Start with `cache_info()['warnings']` (decorator) or `%cash_stats` (notebook) — `cache_info()` returns a plain dict, so subscript it; `.warnings` raises `AttributeError`. Look for:

- `CashRandomnessWarning` — unseeded RNG; pass `random_state=42` (or whatever) to make calls reproducible.
- `CashImpurityWarning` — analyzer found `requests.get` / `datetime.now()` / similar in the function body. See [Purity Decorators](purity-decorators.md).
- `CashCacheIneffectiveWarning` — args weren't hashable, or the value was too big to promote past RAM, or a `cache_if` predicate excluded the call.

Once the obvious culprits are gone and the rate is still low, run `f.explain(...)` for a sample call and check whether the `reason` is mostly `no_entry` (you're editing helpers between runs, args genuinely vary, or the cache was cleared) or `key_uncomputable` (need a custom hasher).

### "Cell I didn't change is recomputing"

Turn on `%cash_debug on` and re-run. The `[UPSTREAM_DEBUG]` lines show which upstream variable's state hash flipped — typically a notebook-level variable that an upstream cell rewrote in place (`df.sort_values(inplace=True)`), or a function whose source you edited indirectly via auto-reload. Fix by isolating the mutation (`df = df.sort_values(...)` returns a new frame) or by marking the helper `@pure` if its body really is deterministic.

### "Cell I changed isn't recomputing"

The opposite mystery: you edited code, but Cash is serving a stale value. Call `f.explain(...)` with the args you expect to hit. If `reason == "hit"`, Cash's view of the source genuinely hasn't changed — that usually means the edit was inside a helper that isn't in `depends_on` and isn't auto-tracked (cross-package, or behind a dynamic import). Either add the helper to `depends_on=[...]` on the decorator, or call `f.cache_clear()` to force the next call to recompute and re-key.

### "Cache is huge on disk"

```bash
cash inspect ./.cash
```

<!-- claim: cash/__main__.py:_inspect_cache_dir @be5b4bea -->
The output gives the entry count, the total size, and a breakdown by file type — then lists the five **most recently written** entries. Note that those five are sorted by modification time, not by size, so they are not necessarily the ones filling your disk; use the total and the file-type breakdown for that. If a single statement is responsible for most of the size, consider `# @cash:no-cache` on cheap statements you don't need to cache, or pick a different backend (`SQLiteBackend` is more efficient for thousands of small entries — see [Choosing a backend](choosing-a-backend.md)).

<!-- claim: cash/analytics.py:AnalyticsManager.__init__ @db5dc0e8 -->
!!! note "The `~/.cash/analytics.db` telemetry file"
    Separate from the project-local `./.cash/` cache, Cash keeps a small global
    SQLite file at `~/.cash/analytics.db` recording per-session hit/miss/timing
    events (this is what `%cash_stats` reads). It is **best-effort observability,
    never correctness** — deleting it is always safe and loses only telemetry, no
    cached results. Cash recreates it automatically if it is missing, corrupt, or
    oversized, so you should never see an error about it; if you want to reset the
    telemetry, just delete the file.

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_repair @2cfd33e0, cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_export @1ce0818b, cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_import @1a4d7a1b -->
## Cache management — export, import, clear

When diagnosis is done and you need to *act*, four notebook magics and one CLI command cover the lifecycle:

### Export

```python { .nb-cell }
%cash_export project_cache.cache                       # all cached data (pickle)
%cash_export model_cache.cache --vars model,features   # specific variables only
%cash_export lineage.json --json                       # lineage graph as JSON
```

The `.cache` file is a portable bundle; the `--json` variant is human-readable and useful for code review or dependency-graph inspection.

### Import

```python { .nb-cell }
%cash_import project_cache.cache              # restore from file
%cash_import teammate_cache.cache --merge     # merge with the current cache instead of replacing
```

Use `--merge` when pulling in a teammate's cache without losing your own entries.

### Clear

From inside a notebook, `%cash_repair` covers the two flavors of reset:

```python { .nb-cell }
%cash_repair             # remove corrupted entries, keep healthy ones
%cash_repair --state     # reset in-memory tracking, keep on-disk cache
%cash_repair --full      # clear cache + reset state — start clean
```

From outside, the CLI:

```bash
cash clear --all                      # delete ./.cash
cash clear ./notebooks/analysis.ipynb # delete the sibling .cash
```

The CLI has no confirmation prompt; double-check the path before pressing enter.

## Experimental UI tools

Two interactive helpers live under `cash.experimental` for richer inspection. Both emit a `FutureWarning` on import — the API may change between releases.

```python
from cash.experimental import CacheExplorer, CacheDebugger
import cash

c = cash.Cash()
explorer = CacheExplorer(c)            # also available as c.explorer()
explorer.list_entries()                # every cache entry with metadata
explorer.to_dataframe()                # same as a pandas DataFrame
explorer.get_preview(key)              # peek at a stored value
explorer.clear_function("mod.func")    # surgical per-function clear
```

<!-- claim: cash/ui/explorer.py:CacheExplorer @326b80ba broad="the listed method set is a claim about the whole class", cash/core.py:Cash.explorer @599913c8 -->
`CacheExplorer` is the read-side: list, search, preview, and surgically clear entries by function name without touching the rest of the cache. `CacheDebugger` is a step-through inspector for the notebook decision pipeline — drives the same machinery `%cash_on` uses but stops between phases so you can see what Cash sees.

Both are experimental: stick to `f.explain()` and `%cash_debug` for anything that needs to survive a version bump.

## API reference

| Tool | Surface | Import / invocation | Effect |
|---|---|---|---|
| `f.explain(*args, **kwargs)` | Decorator | attribute on `@cash.cache`-wrapped function | Returns `CacheExplanation` for the next call. No execution, no stats mutation. |
| `f.cache_info()` | Decorator | attribute on `@cash.cache`-wrapped function | Returns `{hits, misses, hit_rate, total_time_saved, warnings}` per function. |
| `f.cache_clear()` | Decorator | attribute on `@cash.cache`-wrapped function | Wipes backend entries for this function; resets stats + warnings. |
| `%cash_stats` | Notebook | line magic | Session-wide aggregate counters. `json` → dict, `reset` → zero. |
| `%cash_debug on/off/json/file <path>` | Notebook | line magic | Toggles DEBUG-level cash logging with five labelled prefixes. |
| `%cash_export <file> [--vars X,Y] [--json]` | Notebook | line magic | Dump cache (or lineage) to a portable file. |
| `%cash_import <file> [--merge]` | Notebook | line magic | Load cache from a file; `--merge` preserves existing entries. |
| `%cash_repair [--state] [--full]` | Notebook | line magic | Clear corrupted entries (default), reset state only (`--state`), or full reset (`--full`). |
| `cash inspect [path]` | CLI | shell command | Summarise a cache dir or notebook's sibling `.cash`. Read-only. |
| `cash clear [path] [--all]` | CLI | shell command | Delete a cache directory. **No confirmation prompt.** |
| `CacheExplanation` | Type | `from cash import CacheExplanation` | Frozen dataclass returned by `explain()`. Fields: `would_hit`, `reason`, `func_name`, `cache_key`, `details`. |
| `cash.experimental.CacheExplorer` | UI | `from cash.experimental import CacheExplorer` | List/preview/clear backend entries. Experimental. |
| `cash.experimental.CacheDebugger` | UI | `from cash.experimental import CacheDebugger` | Step-through inspector for the notebook pipeline. Experimental. |

## Related

- [Cost Model](../../cost-model.md) — how cache size, hit rate, and time-saved compose into the value of caching at all.
- [Smart Persistence](smart-persistence.md) — tune what hits disk vs stays in RAM when `cache inspect` says you're disk-heavy.
- [Purity Decorators](purity-decorators.md) — fix "low hit rate" caused by impurity warnings the analyzer is raising.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — `# @cash:no-cache` and friends for the statements diagnostics flag as wasteful.
- [CLI](../../cli.md) — full reference for `cash inspect`, `cash clear`, and the rest of the command-line surface.
