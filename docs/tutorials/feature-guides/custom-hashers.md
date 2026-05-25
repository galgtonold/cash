# Custom hashers — teaching Cash to fingerprint your types

Cash hashes every function argument to compute the cache key. The default uses `pickle.dumps(value)` followed by SHA-256 — fast and correct for built-in types and most data classes. For pandas, numpy, polars, PyArrow, modin, and dask, Cash recognises the type at the module level and substitutes a library-specific fingerprint. For your own custom types, register a hasher.

## Why this exists

There are three failure modes that custom hashers exist to solve, and you'll usually hit one of them by accident before you read about them here:

1. **"Unhashable argument" warning.** The argument is unpicklable (an open file handle, a thread lock, a C-extension object that doesn't survive `pickle.dumps`). Cash refuses to cache the call and emits `CashCacheIneffectiveWarning` naming the offending type. The warning message itself suggests `cash.register_hasher(YourType, ...)` (`src/cash/core.py:706-712`).
2. **Slow pickle of complex objects.** Some objects pickle correctly but slowly — a 200 MB embedding model, a giant dict, a nested graph. The pickle round-trip on every call dominates the cache lookup. A bespoke hasher that reads one or two structural fields is orders of magnitude faster.
3. **False misses from non-deterministic pickling.** Two equal instances may pickle to different byte strings — protocol-buffer messages with map fields, dicts with non-deterministic hash ordering on objects, anything that captures memory addresses. A hand-written hasher derived from the type's identity-bearing fields produces stable keys.

If none of these bite you, you don't need this page. The defaults — pickle-and-SHA for ordinary objects, library-specific shortcuts for dataframes — handle the common case.

## Quick start

```python
import hashlib
import cash

class MyModel:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights   # numpy array

def hash_model(model: MyModel) -> str:
    return hashlib.sha256(
        f"{model.name}:{model.weights.tobytes()}".encode()
    ).hexdigest()

cash.register_hasher(MyModel, hash_model)

@cash.cache
def evaluate(model: MyModel, data):
    return model.weights @ data
```

That's the entire workflow. `cash.register_hasher` is a top-level helper that proxies to the default `Cash` instance (`src/cash/__init__.py:307-308`); if you've constructed your own `Cash(...)` object call `c.register_hasher(...)` on it instead. Subsequent `@cash.cache` calls that receive a `MyModel` argument will route through `hash_model` instead of attempting to pickle the instance.

## How registration works

`register_hasher` lives at `src/cash/core.py:1936-1970`. The signature is straightforward:

<!-- test:skip reason="signature illustration only" -->
```python
def register_hasher(self, type_: type, hasher_fn: Callable[[Any], str]) -> None: ...
```

Two things happen on registration:

1. The hasher's *source* is hashed via `_hash_callable_source` (`src/cash/core.py:458-479`). Resolution order is `inspect.getsource(fn)` first, then `fn.__code__.co_code`, then `fn.__call__.__code__.co_code` for callable instances, then `type(fn).__qualname__` as a last resort. The result is a stable hex digest of the hasher's identity.
2. The pair `(hasher_fn, src_hash)` is stored in `self._type_hashers[type_]` — an ordinary dict keyed on the type object (`src/cash/core.py:1970`).

When a cached function runs, `_serialize_args` (`src/cash/core.py:1544-1565`) walks each argument:

1. **Lineage short-circuit.** If the value carries `_cash_lineage_hash` (the output of an upstream `@cash.cache` call), that hash is used directly. Registered hashers never run for lineage-tagged values.
2. **Type-hasher dispatch.** Cash iterates `self._type_hashers.items()` in insertion order and tests each registered type with `isinstance(arg, type_)`. The first match wins. `isinstance` makes the dispatch MRO-aware: register a hasher on `BaseModel` and every subclass uses it.
3. **Built-in fallback.** If no registered hasher matched, `_try_builtin_type_hash` (`src/cash/core.py:1660-1697`) tries the library-specific shortcuts for pandas / numpy / polars / PyArrow / modin / dask.
4. **Pickle fallback.** Anything that's left goes through `pickle.dumps`, and the resulting bytes feed the SHA-256 that yields the args hash.

The returned hash is *not* the raw `hasher_fn(arg)` output — it's `f"{src_hash}:{hasher_fn(arg)}"` (`src/cash/core.py:1554`). Editing the hasher body invalidates every cache entry that depended on it, even when the hasher's output coincidentally matches the old one. This is by design: a silent change of hashing behaviour without a key change would silently return stale results.

## What makes a good hasher

Four properties, in priority order:

- **Deterministic.** Same input, same hash, every time. No `id(obj)`, no `time.time()`, no `random.random()`, no `hash(str)` (PYTHONHASHSEED randomises that across processes). If your hasher returns a different string on two runs of the same Python script for the same input, every cache entry is a guaranteed miss.
- **Total.** Capture every field that affects the function's output. If `MyModel` has a `temperature` knob and your hasher only reads `weights`, two models with different temperatures share a cache entry — your function returns the wrong answer.
- **Cheap.** The hasher runs on every call. If it takes longer than the function itself, caching is a loss. For large arrays, hash the shape, dtype, and a fixed-size sample (this is exactly what `_try_hash_numpy` does at `src/cash/core.py:1586-1599` for arrays >10 MB).
- **Stable across processes.** Don't depend on the process-local hash seed, `id()`, memory addresses, or anything else that varies between Python invocations. The whole point of disk caching is sharing entries across runs.

A useful sanity check: call your hasher twice on freshly constructed-equal instances. If the two outputs match, you're on the right track. If they don't, you're hashing identity, not value.

## Common patterns

### Wrap an existing serializer

If your type already has a canonical bytes representation, use it:

<!-- test:skip reason="references undefined name MyPydanticModel" -->
```python
import hashlib, json
import cash

def hash_pydantic(model):
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()

cash.register_hasher(MyPydanticModel, hash_pydantic)
```

Pydantic's `model_dump_json` is deterministic, total, and reasonably cheap. Same trick works for protobuf (`SerializeToString` with `deterministic=True`), msgspec (`msgspec.json.encode`), and most ORM models with a `to_dict()`.

### Hash structural fields

When only a handful of fields determine the output, hash those:

<!-- test:skip reason="references undefined name DatasetConfig" -->
```python
def hash_dataset_config(cfg):
    parts = (cfg.path, cfg.split, cfg.preprocessing_version, tuple(cfg.features))
    return hashlib.sha256(repr(parts).encode()).hexdigest()

cash.register_hasher(DatasetConfig, hash_dataset_config)
```

This is faster than serialising the whole object and gives you explicit control over what counts as "the same" config.

### Hash a fingerprint, not the connection

For objects that hold a handle to an external resource — DB connections, S3 clients, HTTP sessions — hash the *target*, not the handle:

<!-- test:skip reason="references undefined name sqlalchemy" -->
```python
def hash_db_conn(conn):
    return hashlib.sha256(conn.url.encode()).hexdigest()

cash.register_hasher(sqlalchemy.engine.Engine, hash_db_conn)
```

A new `Engine` constructed for the same URL hashes the same way, so the cache survives `Engine` re-creation across notebook restarts. This is the right move when the connection is "just plumbing" and the URL identifies the data source.

## Built-in registrations Cash provides automatically

Cash ships with module-level fingerprinting for the dataframe ecosystem. These are *not* installed via `register_hasher` — they live in `_try_builtin_type_hash` (`src/cash/core.py:1660-1697`) and run *after* the user registry, so a user-registered hasher always wins.

| Type | Hash strategy | Source |
|---|---|---|
| `pandas.DataFrame`, `pandas.Series` | `pd.util.hash_pandas_object(value).values.tobytes()` then SHA-256 | `src/cash/core.py:1574-1584` |
| `numpy.ndarray` | Full `tobytes()` SHA-256 when <10 MB; shape+dtype+endpoint sample otherwise | `src/cash/core.py:1586-1599` |
| `polars.DataFrame`, `Series`, `LazyFrame` | `hash_rows()`, `hash()`, or `explain()` output then SHA-256 | `src/cash/core.py:1601-1618` |
| `pyarrow.Table`, `RecordBatch` | Schema + row count + (when small) values | `src/cash/core.py:1620-1635` |
| `modin.DataFrame`, `Series` | Convert to pandas, then `hash_pandas_object` | `src/cash/core.py:1637-1648` |
| `dask` objects | `str(value.__dask_keys__())` then SHA-256 | `src/cash/core.py:1650-1658` |
| Generators / iterators | Returns `None` — never hashable | `src/cash/core.py:1693-1695` |
| Everything else | Falls through to `pickle.dumps` + SHA-256 | `_serialize_args` at `src/cash/core.py:1564-1565` |

If pandas isn't installed and you pass a pandas DataFrame anyway, the import inside `_try_hash_pandas` raises `ImportError` and the value falls through to pickle. So Cash never *requires* the optional dataframe libraries — it just uses better hashing when they're available.

## Caveats

### Bound methods carry `self`

When you cache a method, `self` is one of the arguments and Cash hashes it. The default — pickle the whole object — is usually wrong: methods on stateful objects (database handles, open sockets, large model state) are exactly the case where `register_hasher` matters most. See [Caching Class Methods](caching-class-methods.md) for the full pattern.

### Changing a hasher invalidates old entries

By design (`src/cash/core.py:1554`). The source hash is embedded in the per-argument digest, which folds into the args hash, which folds into the cache key. Edit `hash_model`'s body and every cache entry it produced becomes unreachable — the next call computes a fresh key and re-runs the function. If you want to refactor the hasher *without* invalidating, keep the function body byte-identical (rename via alias, or use `mark_pure`-style attribute patching instead of editing).

### MRO dispatch — subclasses inherit

`isinstance(arg, type_)` matches subclasses too. Register a hasher on `BaseEstimator` and every sklearn estimator subclass routes through it. This is almost always what you want, but it means: if you register hashers on both `Animal` and `Dog`, dispatch order is *insertion order*, not specificity. Register the most specific type first if you need different behaviour per subclass.

### Lambdas have source, but that source is part of the cache key

`inspect.getsource` works on lambdas, and `_hash_callable_source` uses it. So `cash.register_hasher(MyType, lambda x: x.id)` is fine — the lambda's source becomes part of the hasher fingerprint, and editing it invalidates entries the same way a named function would. The footgun: lambdas defined in a REPL or `exec`'d string have no source, fall to the bytecode fallback, and lose stability across Python upgrades. For long-lived caches, prefer a module-level `def`.

### No `unregister_hasher`

There's no API to remove a registration. The `_type_hashers` dict is intentionally not part of the public surface — once registered, a hasher stays until process exit (or you `reset_session()` the global `Cash`). In practice this never matters: hashers are set up once at import time and forgotten.

## Debugging hashers

When a custom hasher misbehaves, two probes catch most bugs:

<!-- test:skip reason="snippet shown out of context (references undefined w1, model, data)" -->
```python
# 1. Call the hasher manually on representative inputs.
h1 = hash_model(MyModel("a", w1))
h2 = hash_model(MyModel("a", w1))     # fresh copy
assert h1 == h2, "hasher is not deterministic"

# 2. Use explain() to see the resulting cache key.
explanation = evaluate.explain(model, data)
print(explanation.cache_key)
```

`f.explain()` returns a `CacheExplanation` whose `cache_key` field is the SHA-256 that would be used on the next call (`src/cash/core.py:99-106`). Call it before and after a hasher edit, or with two "should be equal" model instances; if the keys differ, the hasher is hashing identity rather than value.

When Cash can't hash an argument at all, the warning names the type explicitly and suggests the exact `register_hasher` call to make (`src/cash/core.py:710-712`). The same hint appears in `explain().details['hint']` for the `key_uncomputable` reason (`src/cash/core.py:807-811`). See [Debugging and Monitoring](debugging-and-monitoring.md) for the wider toolkit.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `cash.register_hasher(type_, hasher_fn)` | Module-level helper | Proxies to the global `Cash` singleton's `register_hasher`. Signature: `(type_: type, hasher_fn: Callable[[Any], str]) -> None`. |
| `Cash.register_hasher(type_, hasher_fn)` | Instance method | Same, but on a user-constructed `Cash(...)`. Defined at `src/cash/core.py:1936`. |
| `_type_hashers` registry | Private dict | `type → (hasher_fn, src_hash)`. Iterated in insertion order during `isinstance` dispatch. |
| Built-in fallbacks | `_try_builtin_type_hash` | Auto-applies to pandas / numpy / polars / PyArrow / modin / dask without registration. User registrations take precedence. |
| Pickle fallback | `_serialize_args` | `pickle.dumps` + SHA-256 for anything that didn't match a registered or built-in hasher. |

There is no `unregister_hasher`, no public way to list registered hashers, and no way to clear the registry short of `reset_session()`.

## Related

- [Caching Class Methods](caching-class-methods.md) — the `self` argument is exactly where `register_hasher` is most useful.
- [Decorator (`@cash.cache`)](../../decorator.md) — full reference for the decorator that consumes the hasher's output.
- [Debugging and Monitoring](debugging-and-monitoring.md) — `f.explain()` exposes the cache key and the `key_uncomputable` hint.
- [Custom File Sources](custom-file-sources.md) — companion escape hatch for tracking on-disk inputs Cash can't see.
- [Purity Decorators](purity-decorators.md) — orthogonal concern (purity says *whether* to cache; hashers say *how* to identify the call).
