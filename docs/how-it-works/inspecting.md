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
df
├── raw_data (via: raw_data = pd.read_csv('data.csv'))
│   └── [file: data.csv]
├── clean_data (via: clean_data = raw_data.dropna())
│   └── raw_data
└── df (via: df = clean_data.merge(other))
    ├── clean_data
    └── other
```

The graph is a transitive closure — it follows inputs recursively, so you see
the full chain back to the source files. Add `--time` to view a variable's
history chronologically, with timing and whether each step **computed** or
**restored**:

<!-- test:skip reason="IPython magic command — requires kernel context" -->
```python
%cash_provenance df --time
```

```
[12:34:56] COMPUTED  raw_data    (45.2ms)  raw_data = pd.read_csv('data.csv')
[12:34:57] COMPUTED  clean_data  (12.1ms)  clean_data = raw_data.dropna()
[12:34:57] RESTORED  df          (2.3ms)   df = clean_data.merge(other)
```

That `RESTORED ... (2.3ms)` line is the payoff made visible: a step that would
have taken seconds, served from cache in milliseconds.

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
