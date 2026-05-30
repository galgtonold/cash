# Dynamic dependencies — invalidation that depends on the call's arguments

Most cache invalidation triggers are static: argument values, file mtimes, function source. Sometimes the *set* of dependencies you need to track is itself a function of the call. The file path you want to watch isn't known at decoration time — it's computed from the arguments. That's what `dynamic_depends_on=` is for.

## Why this exists

`file_depends_on=` and `depends_on=` are both decided when the decorator runs. They take fixed paths, fixed `DataSource` instances, fixed upstream callables. That covers the case where you write `@cash.cache(file_depends_on="data/features.csv")` and the path is the same on every call.

The case it doesn't cover is when the path (or the data source identity, more generally) varies with the arguments:

- A loader called as `load(dataset_name)` that should track `f"data/{dataset_name}.parquet"`.
- A reader that mounts a different config file per tenant.
- A function that watches an mtime on a file whose name was computed from a hash of the input.

You could expand each case into one `@cash.cache` per dataset, but that doesn't scale. `dynamic_depends_on=` lets you express "compute the dependency from the arguments, then track it" in one decorator.

## Quick start

```python
import cash
from cash import FileDataSource

def file_for(name: str) -> FileDataSource:
    return FileDataSource(f"data/{name}.parquet")

@cash.cache(dynamic_depends_on=file_for)
def load(name: str):
    import pandas as pd
    return pd.read_parquet(f"data/{name}.parquet")

load("features")          # First call: cache miss — compute + record mtime
load("features")          # cache hit
load("labels")            # cache miss — different args and different tracked file
# test:inject: import os, time as _t; _ts = _t.time() + 1; os.utime("data/features.parquet", (_ts, _ts))
load("features")          # cache miss — dynamic dep mtime moved
```

The resolver receives **the same positional and keyword arguments as the decorated function** and must return a `DataSource` instance (or a list of them, or a single one wrapped in a list). On every cache lookup Cash calls the resolver, hashes the returned source(s), and folds the digest into the cache key.

## How it works

The resolver lives in `_resolve_dynamic_dependencies` at `src/cash/core.py:1501-1542`. The path is:

1. The resolver is called as `resolver(*args, **kwargs)` — same signature as the decorated function.
2. The return value is normalised to a list: `dss = ds_result if isinstance(ds_result, list) else [ds_result]` (`src/cash/core.py:1520`).
3. Each entry that **is a `DataSource`** contributes a state string: `str(ds._get_mtime())` if the subclass exposes one, otherwise `str(ds.has_changed())` (`src/cash/core.py:1522-1527`).
4. The collected strings are sorted and SHA-256'd to produce a `dynamic_state_hash` that is mixed into the cache key alongside the args hash and the static dependency hash (`src/cash/core.py:1539-1541`).

Two consequences of step 3 worth pinning down:

- **Only `DataSource` instances count.** A raw string, int, or dict returned from the resolver is silently dropped — the `isinstance(ds, DataSource)` gate at `src/cash/core.py:1523` filters everything else out, and the resulting `dynamic_state_hash` is `""`. The call still caches; the dynamic dependency just doesn't contribute to the key. There's no warning for this — it's an easy way to think you've enabled a dependency that isn't actually being tracked.
- **Sorting makes the result order-independent.** Two resolvers that return the same set of states in different orders produce the same hash.

## Returning a single source vs. a list

The resolver can return:

<!-- test:skip reason="illustration: `def load(...)` literal ellipsis params is a SyntaxError; references undefined file_resolver, schema_resolver" -->
```python
# One source
def one(filename):
    return FileDataSource(filename)

# Several
def many(filename, schema):
    return [FileDataSource(filename), FileDataSource(schema)]

# Pass a list of resolvers — same effect as one resolver returning a list
@cash.cache(dynamic_depends_on=[file_resolver, schema_resolver])
def load(...):
    ...
```

The list-of-resolvers form is handled at `src/cash/core.py:1512`: each resolver is called independently and the results are pooled before sorting. Either shape is fine — pick whichever reads better at the call site.

## Combining with other invalidation triggers

`dynamic_depends_on=` is *additive*. The cache key is the digest of `(function source, static deps, dynamic deps, args)`. Anything else that already invalidates continues to invalidate:

- **With `file_depends_on=`** — the static file list is folded into the dependency state hash via `_register_func` (`src/cash/core.py:669-673`); the dynamic resolver is folded into the separate dynamic state hash. Both must hold for a hit; either drifting forces a miss.
- **With `ttl=`** — TTL is checked *after* the key matches (`src/cash/core.py:862-876`). A dynamic dep change misses immediately; a TTL expiry misses on the next call after the timestamp passes. Whichever triggers first wins on any given lookup.
- **With `depends_on=`** — the static `DataSource` and upstream-function entries contribute to `current_state_hash` (computed by `DependencyStateHasher.compute` in `src/cash/dependency_state.py`, invoked as `self._state_hasher.compute(func_name)` in `_resolve_cache_key`), which is independent of the dynamic state hash.

## What you can return from the resolver

The resolver must return one of:

- A single `DataSource` instance (`FileDataSource`, or your own subclass).
- A list of `DataSource` instances.
- An empty list, `None`, or any non-`DataSource` value — but these are silently dropped from the key. The call still caches; the dynamic dep just doesn't contribute.

To track something other than an mtime, write a `DataSource` subclass. The interface is three methods (`src/cash/data_source.py:10-23`):

```python
import hashlib
from cash import DataSource

class EnvVarSource(DataSource):
    def __init__(self, var: str):
        self.var = var
        self._last = None

    def get_id(self) -> str:
        return f"env:{self.var}"

    def has_changed(self) -> bool:
        import os
        return os.environ.get(self.var) != self._last

    def update_state(self) -> None:
        import os
        self._last = os.environ.get(self.var)
```

`has_changed()` is what gets hashed into the key when the source has no `_get_mtime` method (`src/cash/core.py:1527`). Return a stable string representation of whatever you want to track — a version, a config digest, a tenant id — and the call invalidates when the string moves.

## What NOT to do

### Don't return raw values expecting them to be hashed

```python
# test:inject: import os
# test:inject: import cash
@cash.cache(dynamic_depends_on=lambda: os.environ.get("MODEL_VERSION", "v1"))
def predict(features):
    ...
# Resolver returns a str. Cash silently drops it. The cache key has
# no dynamic component. MODEL_VERSION changes do NOT invalidate.
```

This is a common trap. The fix is to wrap the value in a `DataSource` subclass (see above) so that `has_changed()` reports the change.

### Don't put side effects or expensive work in the resolver

The resolver runs on **every cache lookup**, including hits. Anything slow — a database query, a network call, an expensive hash over a large object — directly inflates lookup latency. Keep it to a constructor call and let the `DataSource` do its own cheap stat on the next step.

### Don't capture mutable state that won't actually change

```python
state = {"v": 1}

def resolver(x):
    return FileDataSource(f"data/{state['v']}.parquet")
```

Mutating `state['v']` between calls changes which file the resolver returns, but the *function source* doesn't change and the cache won't notice unless the new file's mtime differs from the old one's. If you need a version-keyed cache, encode the version into the function arguments, not into a closure.

## Error handling

If the resolver raises any of `OSError`, `TypeError`, `ValueError`, `AttributeError`, `RuntimeError`, Cash:

1. Emits a `CashCacheIneffectiveWarning` exactly once per `(func, '')` pair (deduped via `_warn_once`).
2. Drops the failed resolver's contribution from the dynamic state hash — the cache key is built as if that resolver had returned nothing.
3. Continues processing any remaining resolvers in a list.
4. Lets the call complete and the result is cached against the (incomplete) key.

The warning text reads:

> `@cash.cache on {func_name}: dynamic_depends_on resolver raised {ErrorType} ({message}). Call will not include this dependency in the cache key — results may be stale if the underlying data changes.`

See `src/cash/core.py:1528-1537`. Catching the exception silently and continuing means a transiently failing resolver (e.g. a temporary `OSError`) does not break your pipeline — it just degrades to a broader cache hit while you fix it.

`f.explain()` uses a different variant — `_resolve_dynamic_dependencies_silent` at `src/cash/core.py:915-939` — which re-raises instead of warning, so introspection never emits warnings as a side effect. The resulting `CacheExplanation` carries `reason='key_uncomputable'` with the error type in `details` (`src/cash/core.py:785-794`).

## Performance

Two things to watch:

- **Resolver cost.** It runs on every call, so the overhead lands on cache hits too. `FileDataSource(path)` calls `os.path.getmtime` in its constructor (`src/cash/data_source.py:30`) — one stat per source, usually sub-millisecond. Custom subclasses that do anything heavier should cache internally.
- **`DataSource` reuse.** Returning a fresh `FileDataSource(path)` from the resolver every call means a new stat every call. That's fine for filesystem reads but if your custom `DataSource` is expensive to construct, consider memoising the resolver itself (a plain `functools.lru_cache` over `(path,)` is enough).

## Caveats

- **Non-`DataSource` returns are silent.** As noted above, anything that's not a `DataSource` instance is dropped from the key without a warning. If you suspect your resolver isn't being applied, check `f.explain()` — a key that doesn't reflect your dependency tells you the resolver returned something the gate at `src/cash/core.py:1523` rejected.
- **Resolver errors fail open, not closed.** A transient failure widens the cache. If correctness matters more than availability, validate the resolver's output yourself or call `f.cache_clear()` when you suspect drift.
- **`FileDataSource.__init__` snapshots mtime eagerly.** Each call constructs a fresh source, so the snapshot is the *current* mtime at the moment the resolver runs — exactly what you want for dynamic tracking. (This is the opposite of `file_depends_on=`, where the snapshot is taken once at decoration time. See [Custom File Sources](custom-file-sources.md).)
- **Closures over mutable state are a footgun.** See the example above — if a closure changes which `DataSource` you return without changing the function arguments, the cache may not notice. Encode anything that varies across calls into the arguments.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `dynamic_depends_on=callable` | `@cash.cache` kwarg | Calls *callable* with the function's args; expects a `DataSource` or list of them. Folded into the cache key as a sorted SHA-256 of each source's mtime (or `has_changed()` fallback). |
| `dynamic_depends_on=[callable1, callable2, ...]` | `@cash.cache` kwarg | Each resolver is called independently; results are pooled and hashed together. Equivalent to one resolver that concatenates the lists. |
| `cash.DataSource` | Public ABC | Subclass to track anything other than file mtime. Implement `get_id`, `has_changed`, `update_state`. |
| `cash.FileDataSource(path)` | Public class | mtime-based source for a single file. The canonical thing to return from a resolver. |
| `CashCacheIneffectiveWarning` | Warning | Fires once per function when a resolver raises one of the caught exception types. |
| `f.explain(*args).reason == 'key_uncomputable'` | Diagnostic | What `explain()` reports when the resolver itself raises (the silent variant re-raises and is caught upstream). |

## Related

- [Custom File Sources](custom-file-sources.md) — `file_depends_on=` for the static-path case, plus how to subclass `FileDataSource` for content hashing.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — `ttl=` and `cache_if=` for the other two invalidation axes.
- [Debugging and Monitoring](debugging-and-monitoring.md) — `f.explain()` shows the dynamic state hash and surfaces resolver errors.
- [Decorator (`@cash.cache`)](../../decorator.md) — full kwarg reference for `dynamic_depends_on=`, `depends_on=`, and `file_depends_on=`.
- [Caching Class Methods](caching-class-methods.md) — when the resolver wants to read fields off `self`.
