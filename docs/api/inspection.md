# Inspection tools

<!-- claim: cash/core.py:Cash.explorer @599913c8, cash/ui/explorer.py:CacheExplorer @29620f36 broad="the page documents the class as a whole" -->
Three classes for looking at a cache from code rather than from a magic or
the badge. Each has one import path:

```python
from cash.ui.explorer import CacheExplorer   # browse / preview / clear cache entries
from cash.graph import DependencyGraph       # the structure behind Cash.graph
from cash.analytics import AnalyticsManager  # rolling cache analytics db
```

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
```

::: cash.graph.DependencyGraph
    options:
      members: true

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
        - get_session_stats
        - get_stats_for_session
        - get_global_stats
        - get_daily_savings
        - flush
