# Exceptions & warnings

The exceptions and warnings Cash exports from the top-level `cash` module.

A few internal warnings live on their own submodule and are not re-exported —
`CashNotebookDiscoveryWarning` (raised when Jupyter path discovery fails and
upstream tracking degrades) is importable from
`cash.notebook.server_discovery`, not from `cash`.

This page is the class hierarchy. For what an individual warning *means* and
what to do about it, every warning carries a diagnostic code — look it up in
[Warnings](../warnings.md).

## Imports

```python
from cash import (
    # Exceptions
    CashError,
    CacheBackendError, CacheSerializationError, CacheExpiredError,
    DependencyNotFoundError,
    AmbiguousCellError, UpstreamStateError, CacheKeyComputationError,
    CashImpureFunctionError,
    # Warnings
    CashWarning,
    CashCacheIneffectiveWarning, CashImpurityWarning,
    CashCacheStoreFailedWarning,
    CashRandomnessWarning, CashUpstreamSyntaxWarning,
)
```

Every exception and warning Cash produces is reachable from the
top-level `cash` module. `CashError` and `CashWarning` are the
inheritance roots — see the hierarchies below for filtering recipes.

<!-- claim: cash/exceptions.py:__all__ @6574cfb2 broad="the page claims to be the COMPLETE hierarchy, which is a claim about the whole module's exports" -->
## Exception hierarchy

```text
Exception
└── CashError                          — base for everything Cash raises
    ├── CacheBackendError              — backend I/O failures
    ├── CacheSerializationError        — pickle / serialize round-trip failed
    ├── CacheExpiredError              — entry TTL exceeded (caught internally)
    ├── CacheKeyComputationError       — cache key build failed (notebook)
    ├── CashImpureFunctionError        — @cash.cache(strict=True) saw issues
    ├── AmbiguousCellError             — notebook cell cannot be identified
    ├── UpstreamStateError             — upstream cell state unrestorable
    └── DependencyNotFoundError        — optional backend extra missing
        (also ImportError so existing handlers still match)
```

A single `except cash.CashError` catches everything cash throws.

::: cash.CashError

::: cash.CacheBackendError

::: cash.CacheSerializationError

::: cash.CacheExpiredError

::: cash.DependencyNotFoundError

::: cash.AmbiguousCellError

::: cash.UpstreamStateError

::: cash.CacheKeyComputationError

::: cash.CashImpureFunctionError

---

## Warning hierarchy

```text
UserWarning
└── CashWarning                        — base for everything Cash warns about
    ├── CashCacheIneffectiveWarning    — cache won't help this call
    │   └── CashImpurityWarning        — purity analyzer found issues
    ├── CashCacheStoreFailedWarning    — compute OK but backend rejected store
    ├── CashRandomnessWarning          — unseeded draw; the cached value is frozen
    ├── CashUpstreamSyntaxWarning      — an upstream cell won't parse, so it
    │                                    could not be simulated
    └── CashNotebookDiscoveryWarning   — no notebook path; upstream tracking is
                                         off (not re-exported from `cash`)
```

`CashImpurityWarning` deliberately subclasses `CashCacheIneffectiveWarning`
so existing filters that catch the parent also catch impurity warnings.
Filter more precisely with `CashImpurityWarning` directly.
`CashNotebookDiscoveryWarning` is the class behind
[`NOTEBOOK-NOT-FOUND`](../warnings.md#notebook-not-found); it hangs off
`CashWarning` like the two above it, so only a filter on `CashWarning` itself
catches all five branches.

::: cash.CashWarning

::: cash.CashCacheIneffectiveWarning

::: cash.CashImpurityWarning

::: cash.CashCacheStoreFailedWarning

::: cash.CashRandomnessWarning

::: cash.CashUpstreamSyntaxWarning

---

## Common filter recipes

```python
import warnings
import cash

# Suppress all Cash warnings:
warnings.filterwarnings("ignore", category=cash.CashWarning)

# Fail CI on any purity warning:
warnings.filterwarnings("error", category=cash.CashImpurityWarning)

# Suppress just the "the analyzer found issues" warning:
warnings.filterwarnings("ignore", category=cash.CashImpurityWarning)

# Suppress just the store-failed warnings:
warnings.filterwarnings("ignore", category=cash.CashCacheStoreFailedWarning)

# Stop being told a cached random draw is a replay (it still is one):
warnings.filterwarnings("ignore", category=cash.CashRandomnessWarning)
```

Note that `CashRandomnessWarning` and `CashUpstreamSyntaxWarning` hang
directly off `CashWarning`, *not* off `CashCacheIneffectiveWarning` — a filter
on the ineffective-cache branch will not catch them.
