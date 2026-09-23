# Seeing what Cash did

You don't have to take Cash's word for any of this. Every decision it makes is
observable — at a glance after each cell, and in depth on demand. This page is
a tour of those lenses, and the end of the "how it works" journey: by now
you've seen the whole loop, from key to verdict to storage; here's how to watch
it run.

## The execution badge

<!-- claim: cash/notebook/badge_renderer/view_builder.py:_statement_row @38bc6e18, cash/notebook/badge_renderer/renderers/html.py:_rowtip_html @2fa08537 -->
Every cell you run under `%cash_on` paints an **execution badge**: a compact
summary of what just happened — which statements ran, which came from the
cache, which ran without being stored, the time each took, and any decorator
hit/miss metrics folded in from the [decorator path](decorator-path.md). It's
the first thing you see, and
usually all you need. The full field-by-field guide lives in
[Reading the Cash Badge](../badges.md).

<!-- claim: cash/notebook/ipython/cell_executor.py:_statement_source @dc85d1e5, cash/notebook/badge_renderer/renderers/html.py:_row_code_html @7ebcdfa1 -->
The code shown for each row is **your own source**, laid out across its
original lines — not the single-line, normalized form Cash always hashes for
the cache key. That original text is also what actually gets compiled and
run, whenever Cash can recover one. `def`/`class` statements are the one
exception to what the row **shows**: a row for one of those still clips to
its first line (`def foo(x):` … `+N lines`), because defining a function only
*binds* the name — the body doesn't run until something calls it, so unlike
every other row, it isn't "the code that ran".

Whether a `def`/`class` is also an exception to what actually **runs**
depends on its body. The common case — nothing the purity analyzer
recognises as an `# @cash:assume-safe` waiver anywhere in it — still compiles
from the normalized, comment-free form, same as its clipped row summary.
Only a function or class whose body the analyzer *does* recognise as
carrying that waiver is compiled from its original text, comments included,
so a per-line `# @cash:assume-safe` in its body is honoured instead of
silently lost. The gate checks the same thing the analyzer itself checks,
not merely whether the text `@cash:` appears somewhere — a docstring, a
string literal, or an ordinary comment that happens to *mention* `@cash:`
without being a real waiver does not qualify, and doesn't affect what gets
compiled. That split isn't cosmetic: recovering a function's original text
too eagerly — unconditionally, or on any looser a check — would let its
compiled-source identity hash depend on *which* internal path (re)compiled
it — the per-cell path, which can offer the original text, versus a
same-session replay that re-executes an earlier definition and never could —
silently moving the cache key for calls to that function even though nothing
about it changed. Gating recovery on a real waiver keeps the overwhelming
majority of functions on the single, stable path they always had.

Because the cache key is always built from the normalized form with
comments already stripped, editing a comment on an otherwise-unchanged
statement changes what the row displays (and, for a `def`/`class` whose body
carries a recognised waiver, what actually runs) without changing its key —
the next run still hits.

## Where a value came from

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_provenance @efdb75ad, cash/notebook/provenance.py:ProvenanceTracker.format_provenance @67f073a8 -->
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

<!-- claim: cash/notebook/provenance.py:ProvenanceTracker._format_graph_section @8089ee0d -->
The graph follows inputs recursively (to a depth of 5), walking the *union* of
inputs across every history record rather than only the latest — so a `df`
created in one cell and mutated in three others still shows its creation chain.
Names with no provenance record of their own — imported modules, built-ins,
anything the AST picked up but Cash never produced — render as `(external)`
leaves instead of being expanded. Files appear on the `File deps:` line, not in
the tree.

<!-- claim: cash/notebook/provenance.py:ProvenanceTracker._format_timeline_section @a165df2e -->
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

## Going deeper: debug output

When a cache decision surprises you, turn on debug mode to see Cash's
reasoning — key computations, lineage calculations, upstream detection, and
restoration decisions:

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @5f76702f, cash/_log.py:setup_logging @1d39182c -->
<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_debug on            # human-readable (default)
%cash_debug json          # structured JSON output
%cash_debug file log.txt  # also log to a file (JSON format)
%cash_debug off
```

The records go to the cell as they happen; `json` formats each one as a JSON
object, and `file` also appends them to the file, one JSON object per line.

The flags are documented in full under [Magic Commands](../magics.md).

## Asking a decorated function directly

<!-- claim: cash/decorator/explain.py:ExplainMixin._explain_call @2a140859 -->
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
  cache_dir: /home/me/project/.cash
  cache_key: __main__.load:ca32787f...::0bba688a...
  entry_id: 5d0e9a1c44b7
  cached_at: 1784739785.759226
  cache_age_seconds: 0.0006113052368164062
  execution_time_saved: 0.0020235000120010227
```

<!-- claim: cash/decorator/explain.py:EXPLAIN_NO_ENTRY == "no_entry", cash/decorator/explain.py:EXPLAIN_TTL_EXPIRED == "ttl_expired", cash/decorator/explain.py:EXPLAIN_FILE_CHANGED == "file_changed", cash/decorator/explain.py:EXPLAIN_KEY_UNCOMPUTABLE == "key_uncomputable", cash/decorator/explain.py:EXPLAIN_DISABLED == "disabled" -->
`reason` is a short stable string: `hit`, or one of the ways a call misses —
`no_entry`, `ttl_expired`, `file_changed`, `key_uncomputable`, or `disabled` when
caching is switched off — each carrying its own `details` (which files changed,
which argument type couldn't be hashed). The
full shape is in the [`CacheExplanation`](../api/cash.md) reference.

<!-- claim: cash/core.py:Cash._wrap_with_stats @8d5c519b, cash/core.py:Cash._wrap_with_stats.cache_info @4765fb6e, cash/core.py:Cash._log_decorator_call @1746a43f -->
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
| `Compute time` | measured this session — **your code's**, with cash's own time inside the statement taken out |
| `Gross time saved` | estimated — the cost of each value *when first cached* |
| `Cash overhead` | measured this session, including what cash spends inside a statement keying and hashing the calls it routes |
| `Net time saved` | verified + measured savings − measured overhead |

A saving counts when a measurement backs it: one *this* session took
(**verified**), or the least an earlier kernel on this machine ever measured
(**measured** — a minimum, so a cold first run's timing cannot be paid out
forever). The label on the line says which.

The consequence is that Cash *understates* a session that really did save time
but has no measurement to point at — a cache built on another machine, say.
That is the intended direction of error: an overstatement would be the bug.
There it prints a range ("at least … at best …") instead of picking the
flattering end; and when even the gross reading is a loss, it says so plainly.

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @711be826 -->
`%cash_stats` deliberately does not walk the backend — on a disk cache with
thousands of entries that is an O(N) scan that opens every metadata file, and
paying it every time you want a hit rate is not a trade worth making. For size
and entry counts it points you at `cash info` in a terminal.

## From a terminal: the `cash` CLI

Anything that requires touching the cache directory itself lives in the CLI, not
in a magic. These five subcommands are the whole surface:

<!-- claim: cash/__main__.py:main @fc218b1b, cash/__main__.py:cmd_info @f8794ec8, cash/__main__.py:_inspect_cache_dir @79ce6b6f -->
```bash
cash version            # installed version
cash info               # resolved config, where it came from, what the cache holds
cash inspect [path]     # total size + a per-function table, sorted by size
cash clear [path]       # delete a cache directory (see below)
cash autoload on|off    # load cash in every new kernel via an IPython startup hook
```

<!-- claim: cash/__main__.py:cmd_clear @a2a0458b -->
!!! warning "`cash clear` deletes a whole directory"
    `cash clear` is directory-granular, not notebook-granular. Pointing it at a
    notebook (`cash clear analysis.ipynb`) removes the entire `.cash/` folder
    sitting next to that notebook — including the entries belonging to every
    *other* notebook in the same folder, since they all share it. `--all` does
    the same to the cache in use — the directory `cash info` reports, which
    follows the project you are standing in rather than being `./.cash` — and
    `--tool NAME` to an installed tool's per-user cache. There is still no
    per-notebook or per-variable clear.

    `--function NAME` is the one exception: it deletes just that function's
    entries and leaves the rest of the directory alone. It works on decorator
    entries, which carry the owning function in their key; notebook statements
    do not, and group under `(notebook statements)` — addressable as
    `--function notebook`, so they can be dropped without touching the
    decorator entries beside them. `--entry ID` goes one level finer and
    deletes a single entry.

For the thresholds that actually drive persistence, see
[Where your cache lives](storage.md).

## Programmatic inspection

<!-- claim: cash/core.py:Cash.explorer @599913c8, cash/ui/explorer.py:CacheExplorer @7abe3173 broad="the table is a claim about what the class offers" -->
For everything you'd want to do from code rather than a magic, Cash ships a
few inspection tools:

| Tool | What it's for |
|------|---------------|
| `CacheExplorer` | Browse, preview, and clear cache entries (`Cash.explorer()` returns one) |
| `AnalyticsManager` | Hit-rate and timing analytics behind the dashboard |

```python
from cash.ui.explorer import CacheExplorer
from cash.analytics import AnalyticsManager
```

See [Inspection tools](../api/inspection.md) for the full surface.

---

That's the whole loop. You've followed a value from a [cache key](cache-keys-and-lineage.md),
through the [invalidation](invalidation.md) and [safety](safety.md) checks that
keep it honest, down both the [notebook](notebook-path.md) and
[decorator](decorator-path.md) paths, into [tiered storage](storage.md) — and
now you can watch every step of it happen. If you're ready to use it, start with
[Quick Start](../getting-started/quickstart.md).
