# Seeing what Cash did

You don't have to take Cash's word for any of this. Every decision it makes is
observable — at a glance after each cell, and in depth on demand. This page is
a tour of those lenses, and the end of the "how it works" journey: by now
you've seen the whole loop, from key to verdict to storage; here's how to watch
it run.

## The execution badge

Every cell you run under `%cash_on` paints an **execution badge**: a compact
summary of what just happened — which statements computed, which were restored
from cache, the time each took, and any decorator hit/miss metrics folded in
from the [decorator path](decorator-path.md). It's the first thing you see, and
usually all you need. The full field-by-field guide lives in
[Reading the Cash Badge](../badges.md).

## Where a value came from

Cash records the **provenance** of every variable — the code that produced it,
its inputs, and how long it took. Ask for the dependency graph of any variable:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_provenance df --graph
```

```
📋 Provenance for 'df':
  Last computed: 12:34:57
  Status: restored
  Code: df = clean_data.merge(other)
  Inputs: clean_data, other
  File deps: data.csv
  Duration: 2.3ms
  History: 1 records

  Dependency Graph:
    ├─ clean_data ← clean_data = raw_data.dropna()
    │  └─ raw_data ← raw_data = pd.read_csv('data.csv')
    └─ other (external)
```

The graph follows inputs recursively (to a depth of 5), walking the *union* of
inputs across every history record rather than only the latest — so a `df`
created in one cell and mutated in three others still shows its creation chain.
Names with no provenance record of their own — imported modules, built-ins,
anything the AST picked up but Cash never produced — render as `(external)`
leaves instead of being expanded. Files appear on the `File deps:` line, not in
the tree.

Add `--time` (or `--timeline`) for the last ten records chronologically, with
timing and whether each step **computed**, **restored**, or was **skipped**:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_provenance df --time
```

```
  Timeline:
    12:34:56 🔧 computed (45.2ms)
    12:34:57 📦 restored (2.3ms)
```

That `restored (2.3ms)` line is the payoff made visible: a step that would have
taken seconds, served from cache in milliseconds. `--json` gives the same
records machine-readably, `%cash_provenance --all` lists every tracked variable,
and `%cash_provenance --clear` drops the history.

## Going deeper: debug and log

When a cache decision surprises you, turn on debug mode to see Cash's
reasoning — key computations, lineage calculations, upstream detection, and
restoration decisions:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_debug on            # human-readable (default)
%cash_debug json          # structured JSON output
%cash_debug file log.txt  # also log to a file (JSON format)
%cash_debug off
```

Once debug logging is on, `%cash_log` prints the structured event buffer —
recent events by default, a custom count, or the whole thing as JSON:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_log             # recent events (last 20)
%cash_log 50          # show the last 50 events
%cash_log json        # output as a JSON array
%cash_log clear       # clear the buffer
```

Both magics are documented in full under [Magic Commands](../magics.md).

## Asking a decorated function directly

For a `@cash.cache`-wrapped function, `explain()` answers "would the next call
with these arguments hit, and why?" without calling the function, mutating
stats, or writing anything:

```python
import cash

@cash.cache
def load(n):
    return list(range(n))

load(1000)
print(load.explain(1000))
```

```
[HIT] __main__.load - hit
  cache_key: __main__.load:ca32787f...::0bba688a...
  cached_at: 1784739785.759226
  cache_age_seconds: 0.0006113052368164062
  execution_time_saved: 0.0020235000120010227
```

`reason` is a short stable string: `hit`, or one of the four ways a call misses —
`no_entry`, `ttl_expired`, `file_changed`, `key_uncomputable` — each carrying its
own `details` (which files changed, which argument type couldn't be hashed). The
full shape is in the [`CacheExplanation`](../api/cash.md) reference.

!!! warning "`cache_info()` is not the surface to trust in a notebook"
    The wrapper also exposes `cache_info()`, but its `hits` / `misses` counters
    live on the **wrapper object** and count only since that wrapper was
    created. Re-running the cell that defines your function re-runs the
    decorator and produces a fresh wrapper with zeroed counters — so
    `cache_info()` can read `{'hits': 0, 'misses': 0}` forever while caching is
    working perfectly. Even in the transcript above it reports one miss and zero
    hits, because the `explain()` call is not a call.

    `total_time_saved` is weaker still: it sums the execution time recorded when
    each entry was *first written*, so it is an estimate of the original compute
    cost, not a measurement of what recomputing would cost now. On a workload
    with warm imports and a warm page cache it can report a large saving on a
    session that measurably lost time. Treat it as an upper bound.

    `explain()` and `%cash_stats` both read through to the real cache. Prefer
    them. `cache_info()` is reliable in scripts and long-lived processes, where
    the wrapper is built once. See [The decorator](../decorator.md) for the
    full API.

## Session-wide: `%cash_stats`

`%cash_stats` reports the whole session: cells executed, statements computed /
restored / skipped, hit rate, and the time ledger. It is deliberately careful
about what it claims:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_stats           # human-readable
%cash_stats json      # same numbers as JSON
%cash_stats reset     # zero the session counters
```

Two details worth knowing. First, the hit rate is reported **over statements
that were worth caching**, with the all-statements rate shown beside it — a
notebook of `print()` calls should not be scored as if Cash missed on every one
of them, and when nothing cleared the floor it says `n/a` rather than `0%`.
Second, "Net time saved" is gross savings minus Cash's own measured overhead,
and it will print a **loss** when there is one. When only the overhead half was
actually measured it reports a range ("at least … at best …") rather than
passing off an unverified estimate as a fact.

`%cash_stats` deliberately does not walk the backend — on a disk cache with
thousands of entries that is an O(N) scan that opens every metadata file, and
paying it every time you want a hit rate is not a trade worth making.

## From a terminal: the `cash` CLI

Anything that requires touching the cache directory itself lives in the CLI, not
in a magic:

```bash
cash version            # installed version
cash info               # resolved config + where it came from
cash inspect [path]     # entry count, total size, file breakdown
cash clear [path]       # drop a notebook's cache (--all for the directory)
cash autoload on|off    # load cash in every new kernel via an IPython startup hook
```

For the thresholds that actually drive persistence, see
[Where your cache lives](storage.md).

## Programmatic inspection

For everything you'd want to do from code rather than a magic, Cash ships a set
of (experimental) tools:

| Tool | What it's for |
|------|---------------|
| `CacheExplorer` | Browse, search, and clear cache entries |
| `CacheDebugger` | Step through the cell-cache pipeline |
| `visualize_notebook` | Render the dependency graph as a graphic |
| `AnalyticsManager` | Hit-rate and timing analytics behind the dashboard |

```python
from cash.experimental import CacheExplorer, CacheDebugger, visualize_notebook
```

See [Experimental API](../api/experimental.md) for the full surface.

---

That's the whole loop. You've followed a value from a [cache key](cache-keys-and-lineage.md),
through the [invalidation](invalidation.md) and [safety](safety.md) checks that
keep it honest, down both the [notebook](notebook-path.md) and
[decorator](decorator-path.md) paths, into [tiered storage](storage.md) — and
now you can watch every step of it happen. If you're ready to use it, start with
[Quick Start](../getting-started/quickstart.md).
