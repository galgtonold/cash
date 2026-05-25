# Exceptions & warnings

The complete hierarchy of things Cash raises or warns about.

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
)
```

Every exception and warning Cash produces is reachable from the
top-level `cash` module. `CashError` and `CashWarning` are the
inheritance roots — see the hierarchies below for filtering recipes.

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
    └── CashCacheStoreFailedWarning    — compute OK but backend rejected store
```

`CashImpurityWarning` deliberately subclasses `CashCacheIneffectiveWarning`
so existing filters that catch the parent also catch impurity warnings.
Filter more precisely with `CashImpurityWarning` directly.

::: cash.CashWarning

::: cash.CashCacheIneffectiveWarning

::: cash.CashImpurityWarning

::: cash.CashCacheStoreFailedWarning

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
```
