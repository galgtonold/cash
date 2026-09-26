---
search:
  boost: 0.5
---

# Inspection tools

For both paths: browsing a cache from code. For the command line, see
[`cash inspect`](../cli.md#cash-inspect-path).

<!-- claim: cash/core.py:Cash.explorer @c62dd5f2, cash/ui/explorer.py:CacheExplorer @7abe3173 broad="the page documents the class as a whole" -->
`Cash.explorer()` returns a `CacheExplorer` bound to that instance; you do
not create one yourself.

```python
import cash

explorer = cash.Cash().explorer()
entries = explorer.list_entries()
```

::: cash.ui.explorer.CacheExplorer
    options:
      members:
        - list_entries
        - clear_function
        - get_preview
        - to_dataframe
