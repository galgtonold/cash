# Seeing what Cash did

You don't have to take Cash's word for any of this. Every decision it makes is
observable — at a glance after each cell, and in depth on demand. This page is
a tour of those lenses, and the end of the "how it works" journey: by now
you've seen the whole loop, from key to verdict to storage; here's how to watch
it run.

## The execution badge

<!-- claim: cash/notebook/badge_renderer/view_builder.py:_statement_row_from_metric @16e6b771, cash/notebook/badge_renderer/renderers/html.py:_rowtip_html @4a2577a0 -->
Every cell you run under `%cash_on` paints an **execution badge**: a compact
summary of what just happened — which statements ran, which came from the
cache, which ran without being stored, the time each took, and any decorator
hit/miss metrics folded in from the [decorator path](decorator-path.md). It's
the first thing you see, and
usually all you need. The full field-by-field guide lives in
[Reading the Cash Badge](../badges.md).

## Where a value came from

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_provenance @c880acac, cash/notebook/provenance.py:ProvenanceTracker.format_provenance @0d2c524a -->
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

<!-- claim: cash/notebook/provenance.py:ProvenanceTracker._format_graph_section @a22103af -->
The graph follows inputs recursively (to a depth of 5), walking the *union* of
inputs across every history record rather than only the latest — so a `df`
created in one cell and mutated in three others still shows its creation chain.
Names with no provenance record of their own — imported modules, built-ins,
anything the AST picked up but Cash never produced — render as `(external)`
leaves instead of being expanded. Files appear on the `File deps:` line, not in
the tree.

<!-- claim: cash/notebook/provenance.py:ProvenanceTracker._format_timeline_section @b6526b4c -->
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

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @ce13e22b, cash/logging.py:setup_logging @bdde1e72 -->
<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_debug on            # human-readable (default)
%cash_debug json          # structured JSON output
%cash_debug file log.txt  # also log to a file (JSON format)
%cash_debug off
```

`%cash_log` prints the structured event **buffer** — but only the `json` and
`file` modes create one. Plain `%cash_debug on` prints its records straight to
the cell and keeps nothing, so `%cash_log` after it reports *"No log handler
active"* (and unhelpfully suggests the `%cash_debug on` you just ran). Start
from `%cash_debug json` if you want a buffer to query:

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_log @ce466107, cash/notebook/ipython/admin.py:CashAdminMagicsMixin._find_cash_log_handler @44e1fb94, cash/notebook/ipython/admin.py:_parse_log_args @613d5159 -->
<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_debug json      # the buffer exists from here on
%cash_log             # recent events (last 20)
%cash_log 50          # show the last 50 events
%cash_log json        # output as a JSON array
%cash_log clear       # clear the buffer
```

Both magics are documented in full under [Magic Commands](../magics.md).

## The audit log: every cache operation, in order

The badge shows one cell and `%cash_stats` shows session totals. Between them
sits `%cash_audit` — an append-only record of each individual cache operation,
with its variable, timestamp, and the statement that caused it. It answers
"what did cash actually do, in what order?" rather than "how did this one cell
end up".

<!-- claim: cash/notebook/ipython/magics.py:_OP_MAP @46199615 -->
Each notebook statement records exactly one of three operations, named for the
cache's point of view rather than the statement's: a **restored** statement
logs `cache_hit`, a **computed** one logs `cache_miss`, and a skipped one logs
`cache_skip`. There is no separate `cache_restore` or `cache_store` to filter
on — those names exist in the entry format but nothing in the notebook path
emits them.

<!-- claim: cash/notebook/audit.py:AuditLogger.__init__ @86911f4a -->
Audit logging is **off by default**; turn it on and it records from that point:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_audit on                    # start recording (--file <path> also mirrors to disk)
%cash_audit summary               # totals: entries, unique variables, time range, per-operation counts
%cash_audit show                  # the last 50 entries
%cash_audit show cache_hit        # filter to one operation type
%cash_audit clear                 # drop the entries
%cash_audit off
```

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin._audit_cmd_summary @9bb89b1a, cash/notebook/audit.py:AuditLogger.get_summary @1c4a6aab -->
A `summary` reads like this — two runs of the same statement, one miss then one
hit:

```
Audit Summary:
  Total entries: 2
  Unique variables: 1
  Time range: 2026-07-27 12:34:56.021 to 2026-07-27 12:35:01.884
  Operations:
    cache_hit: 1
    cache_miss: 1
```

<!-- claim: cash/notebook/audit.py:AuditLogger.__init__ @86911f4a, cash/notebook/audit.py:AuditLogger.log @ad7de6ca, cash/notebook/audit.py:AuditLogger.get_entries @6323a20a -->
Reach for it when a *sequence* is the question — a cell that behaves differently
on the third run, or a session where you want a compliance-style record of what
was served from cache versus computed. Entries live in memory unless you pass
`--file`, and the in-memory buffer keeps only the most recent 5,000 — a long
session silently drops the oldest, so pass `--file` for anything you need to
keep. `show` displays the last 50. Full flag reference in
[Magic Commands](../magics.md#cash_audit).

## Asking a decorated function directly

<!-- claim: cash/core.py:Cash._explain_call @135def80 -->
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

<!-- claim: cash/core.py:EXPLAIN_NO_ENTRY == "no_entry", cash/core.py:EXPLAIN_TTL_EXPIRED == "ttl_expired", cash/core.py:EXPLAIN_FILE_CHANGED == "file_changed", cash/core.py:EXPLAIN_KEY_UNCOMPUTABLE == "key_uncomputable" -->
`reason` is a short stable string: `hit`, or one of the four ways a call misses —
`no_entry`, `ttl_expired`, `file_changed`, `key_uncomputable` — each carrying its
own `details` (which files changed, which argument type couldn't be hashed). The
full shape is in the [`CacheExplanation`](../api/cash.md) reference.

<!-- claim: cash/core.py:Cash._wrap_with_stats @a28b87be, cash/core.py:Cash._wrap_with_stats.cache_info @b3cd263b, cash/core.py:Cash._log_decorator_call @55a1f795 -->
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
that were worth caching**, with the all-statements rate printed beneath it
whenever some statements were too cheap to cache — a notebook of `print()`
calls should not be scored as if Cash missed on every one of them, and when
nothing cleared the floor it says `n/a` rather than `0%`.

Second, the **"Net time saved"** headline is credited only from savings this
session *verified* by computing the same statement itself, minus Cash's own
measured overhead. It is deliberately not gross-minus-overhead: "Gross time
saved" is printed separately and labelled *(estimated)*, because it credits
each restore with the compute time recorded when that value was **first**
cached, and nothing re-measures it. So the ledger reads:

| Line | Evidence behind it |
|------|--------------------|
| `Compute time` | measured this session |
| `Gross time saved` | estimated — the cost of each value *when first cached* |
| `Cash overhead` | measured this session |
| `Net time saved` | verified savings − measured overhead |

The consequence is that Cash *understates* a session that really did save time
but never re-measured a baseline. That is the intended direction of error: an
overstatement would be the bug. When the verified savings don't cover the
overhead but the gross figure would, it prints a range ("at least … at best …")
instead of picking the flattering end; and when even the gross reading is a
loss, it says so plainly.

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @851d8352 -->
`%cash_stats` deliberately does not walk the backend — on a disk cache with
thousands of entries that is an O(N) scan that opens every metadata file, and
paying it every time you want a hit rate is not a trade worth making. For size
and entry counts it points you at `cash info` in a terminal.

## From a terminal: the `cash` CLI

Anything that requires touching the cache directory itself lives in the CLI, not
in a magic. These five subcommands are the whole surface:

<!-- claim: cash/__main__.py:main @9184b728, cash/__main__.py:cmd_info @c6a1b14a, cash/__main__.py:_inspect_cache_dir @be5b4bea -->
```bash
cash version            # installed version
cash info               # resolved config + where it came from
cash inspect [path]     # entry count, total size, file breakdown, recent entries
cash clear [path]       # delete a cache directory (see below)
cash autoload on|off    # load cash in every new kernel via an IPython startup hook
```

<!-- claim: cash/__main__.py:cmd_clear @23843bc2 -->
!!! warning "`cash clear` deletes a whole directory"
    `cash clear` is directory-granular, not notebook-granular. Pointing it at a
    notebook (`cash clear analysis.ipynb`) removes the entire `.cash/` folder
    sitting next to that notebook — including the entries belonging to every
    *other* notebook in the same folder, since they all share it. `--all` does
    the same to `./.cash` in the current working directory. There is no
    per-notebook or per-variable clear from the CLI.

For the thresholds that actually drive persistence, see
[Where your cache lives](storage.md).

## Programmatic inspection

<!-- claim: cash/experimental/__init__.py:_LOADERS @e8607bb8, cash/experimental/__init__.py:_warn_experimental @5dcce1c0 -->
For everything you'd want to do from code rather than a magic, Cash ships a set
of (experimental) tools — importing any of them raises a `FutureWarning`, which
is the API-stability promise being kept rather than a problem to suppress:

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
