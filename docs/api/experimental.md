# Experimental

<!-- claim: cash/experimental/__init__.py:_LOADERS @e8607bb8, cash/experimental/__init__.py:_warn_experimental @5dcce1c0 -->
APIs under `cash.experimental` are useful but their surfaces are
still evolving — minor versions may rearrange them. Importing emits
a `FutureWarning` so you can spot the dependency in any project that
upgrades cash unexpectedly.

## Imports

```python
from cash.experimental import (
    CacheExplorer,        # browse / search / clear cache entries
    CacheDebugger,        # step through the cell-cache pipeline
    DependencyGraph,      # the structure behind Cash.graph
    AnalyticsManager,     # rolling cache analytics db
    visualize_notebook,   # render the dependency graph as a graphic
)
```

`RedisBackend`, `S3Backend`, and `TieredBackend` also live here —
see [Backends](backends.md) for those.

---

## CacheExplorer

The user-facing entry point for poking around a populated cache.
Normally you don't construct this directly — `Cash.explorer()`
returns an instance bound to your `Cash` object.

::: cash.ui.explorer.CacheExplorer
    options:
      members:
        - __init__
        - list_entries
        - clear_function
        - get_preview
        - to_dataframe

---

## DependencyGraph

The structure Cash uses to track which `@cash.cache`-decorated
functions and `DataSource`s feed which others. `Cash.graph` is an
instance; this page documents what you can do with it.

```python
import cash
c = cash.Cash()

# After registering some cached functions...
c.graph.get_dependencies("my_module.feature")
# {'my_module.load_raw_data'}

c.graph.get_dependents("my_module.load_raw_data")
# {'my_module.feature', 'my_module.summary'}
```

::: cash.graph.DependencyGraph
    options:
      members: true

---

## CacheDebugger

Step-through debugger for the notebook cache decision pipeline.
Useful when the badge isn't detailed enough to explain a stubborn
miss. Requires an active IPython shell.

::: cash.ui.debugger.CacheDebugger
    options:
      members:
        - __init__

---

## visualize_notebook

Render the current notebook's dependency graph as a visualization
(uses `graphviz` if available). Useful for sanity-checking the
lineage Cash inferred.

::: cash.ui.visualizer.visualize_notebook

---

## AnalyticsManager

Tracks per-function hit/miss/time-saved metrics in a small SQLite
database alongside the cache. Drives the analytics dashboard
(`cash.show_stats()`).

::: cash.analytics.AnalyticsManager
    options:
      members:
        - __init__
        - record_event
        - get_function_stats
        - get_session_summary
