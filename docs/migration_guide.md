# Migration Guide

This guide helps you migrate from other Python caching solutions to Cash.

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
- joblib requires explicit caching per function; Cash notebook mode caches everything
- joblib doesn't track upstream changes; Cash invalidates when dependencies change
- Cash provides statement-level granularity in notebooks

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

## Feature Comparison

| Feature | lru_cache | joblib | Manual pickle | **Cash** |
|---------|-----------|--------|---------------|----------|
| Disk persistence | ❌ | ✅ | ✅ | ✅ |
| Auto-invalidation | ❌ | ❌ | ❌ | ✅ |
| Notebook support | ❌ | ❌ | ❌ | ✅ |
| File dep tracking | ❌ | ❌ | ❌ | ✅ |
| Statement-level | ❌ | ❌ | ❌ | ✅ |
| DataFrame support | ❌ | ✅ | ✅ | ✅ |
| Custom type hashers | ❌ | ❌ | ❌ | ✅ |
| Per-function stats | ✅ | ❌ | ❌ | ✅ |
| Zero config | ✅ | ❌ | ❌ | ✅ |
