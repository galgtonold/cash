# Migration Guide

This guide helps you migrate from other Python caching solutions to Cash. It
focuses on the `@cash.cache` **decorator**, the closest analogue to the tools
below. In a notebook you usually don't migrate function-by-function at all — you
add `%cash_on` once and your existing cells cache themselves; see the
[Quick Start](getting-started/quickstart.md).

For a capability-by-capability comparison against these tools, see the
[matrix in Why Cash?](why-cash.md#cash-vs-the-alternatives-youve-tried).

## From functools.lru_cache

**Before (lru_cache):**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_compute(data_hash):
    # ... expensive computation
    return result
```

**After (Cash decorator):**
```python
import cash

@cash.cache
def expensive_compute(data):
    # Cash handles hashing automatically
    return result
```

**Key differences:**
- `lru_cache` requires hashable arguments; Cash handles DataFrames, numpy arrays, polars, etc. natively
- `lru_cache` is memory-only; Cash persists to disk automatically via TieredBackend
- Cash tracks file dependencies (if your function reads files)
- Cash provides `cache_info()` and `cache_clear()` per function, similar to `lru_cache`
- Cash supports `register_hasher()` for custom types that can't be pickled

**For notebooks (recommended):**
```python
import cash
%cash_on
# All cells are cached automatically - no decorator needed
```

## From joblib.Memory

**Before (joblib):**
<!-- test:skip reason="joblib.Memory tries to pickle the stub df argument" -->
```python
from joblib import Memory

memory = Memory("/tmp/joblib-cache", verbose=0)

@memory.cache
def slow_function(data):
    return data.apply(complex_transform)

result = slow_function(df)
```

**After (Cash):**
```python
import cash

@cash.cache
def slow_function(data):
    return data.apply(complex_transform)

result = slow_function(df)
```

**Or in notebooks:**
```python
import cash
%cash_on

# Just write your code normally - it's cached automatically
result = df.apply(complex_transform)
```

**Key differences:**

- **Transitive helper invalidation — the headline difference.** joblib hashes
  only the decorated function's *own* body. Edit a helper it calls and joblib
  keeps serving the old result. Cash folds the source of the functions you call —
  transitively, within the module — into the cache key, so editing a callee
  (even a few levels down) invalidates the cache.
- joblib doesn't track upstream state; Cash invalidates on changed files
  (automatically), declared `depends_on=`, and module globals the function reads.
- joblib requires a decorator per function; Cash notebook mode caches every
  statement automatically, at statement-level granularity.

## From diskcache

**Before (diskcache):**
<!-- test:skip reason="diskcache is not a doc-test dependency" -->
```python
from diskcache import Cache

cache = Cache("/tmp/diskcache")

@cache.memoize()
def slow_function(data):
    return data.apply(complex_transform)
```

**After (Cash):**
```python
import cash

@cash.cache
def slow_function(data):
    return data.apply(complex_transform)
```

**Key differences:**
- `diskcache.memoize` keys on the function's **name and arguments** (like
  `lru_cache`), so editing the function body does *not* invalidate a cached
  entry. Cash hashes the function source, so a body edit recomputes.
- diskcache has no notebook awareness, dependency lineage, file tracking, or
  mutation detection — it's a fast key→value store, not a "know when to
  recompute" layer. (Cash can use a disk backend for that same storage job.)

## From `%store` (IPython)

`%store result` / `%store -r result` is a manual save/load primitive — no
automatic invalidation and no signal when an upstream variable or file changes.
Replace it with `%cash_on`: cash restores your variables on re-run *and*
recomputes them when their code or inputs change.

```python
import cash
%cash_on
# no %store / %store -r needed — cells restore and invalidate automatically
```

## From jupyter-cache

`jupyter-cache` (the engine behind Jupyter Book / MyST-NB) caches a notebook's
executed outputs so a **build** can skip re-running an unchanged notebook — it
matches at whole-notebook granularity and is designed for CI/book builds, not
the interactive edit-and-re-run loop. There's nothing to "migrate": keep
`jupyter-cache` for reproducible book builds, and add `%cash_on` for the
interactive statement-level caching cash provides while you work.

## From Manual Pickle Caching

**Before (manual):**
```python
import pickle
import os

cache_file = "result.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        result = pickle.load(f)
else:
    result = expensive_computation()
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
```

**After (Cash notebook mode):**
```python
import cash
%cash_on

# That's it - just run your code
result = expensive_computation()
```

**Key differences:**
- No manual cache file management
- Automatic invalidation when code or inputs change
- No stale cache bugs (Cash tracks exact inputs and code)

## Feature comparison

The full capability matrix — cash versus `lru_cache`, `joblib.Memory`,
`diskcache`, `jupyter-cache`, `%store`, and manual pickling — lives on one page
so it stays in sync: see
[Cash vs. the alternatives](why-cash.md#cash-vs-the-alternatives-youve-tried).

## Related

- [Decorator (`@cash.cache`)](decorator.md) — the full decorator reference.
- [FAQ](faq.md) and [Glossary](glossary.md) — quick answers and vocabulary.
- [Why Cash?](why-cash.md) — when cash is (and isn't) the right fit.
