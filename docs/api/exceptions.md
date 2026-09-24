# Exceptions and warnings

For both paths: every exception and warning class, and how to filter
warnings. What each warning code means, and what to do about it, is on
[Warnings](../warnings.md).

```python
from cash import (
    CashError, CacheBackendError, CacheSerializationError,
    DependencyNotFoundError, CashImpureFunctionError,
    AmbiguousCellError, UpstreamStateError, ForwardReferenceError,
    CacheKeyComputationError,
    CashWarning, CashCacheIneffectiveWarning, CashImpurityWarning,
    CashCacheStoreFailedWarning, CashRandomnessWarning,
    CashUpstreamSyntaxWarning,
)
```

<!-- claim: cash/exceptions.py:__all__ @7e2c7a88 broad="the page claims to be the COMPLETE hierarchy, which is a claim about the whole module's exports" -->
## Exceptions

Every exception derives from `CashError`, so `except cash.CashError`
catches them all.

**Raised to your code** (decorator path, or backend methods you call):

| Exception | Cause | Fix |
|---|---|---|
| `CashImpureFunctionError` | First call of a cached function that picks what to call at run time (`getattr(obj, name)()`, `eval`, `importlib.import_module`), or any purity finding under `strict=True`. | Call the code directly, mark an audited helper with `pure`, or pass `assume_safe=True`. |
| `DependencyNotFoundError` | A backend needs a package that is not installed. Also an `ImportError`. | Run the `pip install` the message names. |
| `CacheBackendError` | A backend method you called could not reach its storage. A cached function warns instead. | Check the disk, server or credentials. |
| `CacheSerializationError` | A stored entry could not be turned back into a value. | Clear the entry or the function; the next call recomputes. |

**Shown as a notebook cell's error** (with `%cash_on`):

| Exception | Cause | Fix |
|---|---|---|
| `ForwardReferenceError` | The cell reads a name only a later cell defines. | Move the definition above the cell. |
| `UpstreamStateError` | An earlier statement the cell needs failed when cash re-ran it. | Fix that statement, or run the notebook from the top. |
| `AmbiguousCellError` | The cell's code appears more than once and cash cannot tell which copy is running. | Save the notebook, or make the copies differ. |
| `CacheKeyComputationError` | No key could be built for a statement. Not shown as an error: the statement runs uncached and cash warns [`NOTEBOOK-BAILOUT`](../warnings.md#notebook-bailout). | See that code. |

::: cash.CashError

::: cash.CashImpureFunctionError

::: cash.DependencyNotFoundError

::: cash.CacheBackendError

::: cash.CacheSerializationError

::: cash.ForwardReferenceError

::: cash.UpstreamStateError

::: cash.AmbiguousCellError

::: cash.CacheKeyComputationError

## Warnings

```text
CashWarning
├── CashCacheIneffectiveWarning     caching is not working or not paying off
│   └── CashImpurityWarning         the function has side effects a hit skips
├── CashCacheStoreFailedWarning     a result was computed but not stored
├── CashRandomnessWarning           an unseeded random draw was cached
├── CashUpstreamSyntaxWarning       an earlier notebook cell does not parse
└── CashNotebookDiscoveryWarning    the notebook file was not found
```

`CashNotebookDiscoveryWarning` is not importable from `cash`; filter it by its
code, [`NOTEBOOK-NOT-FOUND`](../warnings.md#notebook-not-found). The
[Warnings index](../warnings.md#index) gives the class of every code.

::: cash.CashWarning

::: cash.CashCacheIneffectiveWarning

::: cash.CashImpurityWarning

::: cash.CashCacheStoreFailedWarning

::: cash.CashRandomnessWarning

::: cash.CashUpstreamSyntaxWarning

## Filtering warnings

Filter by class with the standard `warnings` module. A filter on a class
also catches its subclasses, so a filter on `CashCacheIneffectiveWarning`
catches `CashImpurityWarning` but none of the other branches.

```python
import warnings
import cash

# Silence every cash warning.
warnings.filterwarnings("ignore", category=cash.CashWarning)

# Fail CI on any side effect in a cached function.
warnings.filterwarnings("error", category=cash.CashImpurityWarning)

# Silence failed stores only.
warnings.filterwarnings("ignore", category=cash.CashCacheStoreFailedWarning)
```

To silence a single code, see
[Silencing one code](../warnings.md#silencing-one-code).
