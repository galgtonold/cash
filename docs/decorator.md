# `@cash.cache` — decorator guide

This page is the cohesive walkthrough of `@cash.cache`: when to use it,
what every parameter does, the wrapper methods you can call on a
decorated function, and the gotchas that bite people in practice.

For the auto-generated, exhaustive signature reference see the
[API reference](api/cash.md). For the notebook-side
equivalent (`%cash_on`) see [Notebook caching](notebook_caching_api.md).

---

## When to use the decorator

Reach for `@cash.cache` when you have a slow Python function whose
result depends on its arguments and (optionally) some external state
(files, configuration, other functions). The classic candidates:

- Network calls (`requests.get`, LLM completions, database queries)
- Expensive ETL (`pd.read_csv` of a 5 GB file followed by joins)
- CPU-bound transforms (feature extraction, simulations)
- Anything where "I've computed this exact thing already" is true at runtime

Don't use it for:

- Functions whose return value depends on hidden runtime state you
  can't capture in arguments or `depends_on=` (the cache will go
  stale silently)
- Functions called sub-microsecond in a hot loop (cache key
  computation alone will dominate the runtime)

For methods on stateful objects (database handles, model wrappers,
etc.) see the dedicated [caching class methods](tutorials/feature-guides/caching-class-methods.md)
recipe.

---

## The minimum

```python
import cash

@cash.cache
def slow_square(n):
    return sum(i * i for i in range(n))

slow_square(10_000_000)   # ~1 second
slow_square(10_000_000)   # microseconds — restored from cache
```

That's it. The default `Cash()` singleton writes a tiered RAM + disk
cache under `./.cash/`. The next call with the same `n` (this run or
next month) returns the stored value.

If you want a custom configuration (different backend, custom
directory, debug logging), instantiate `Cash(...)` explicitly:

```python
from cash import Cash

c = Cash(cache_dir="/tmp/my_app_cache", debug=True)

@c.cache
def slow_square(n):
    return sum(i * i for i in range(n))
```

---

## Parameters

All of these are keyword-only and optional.

| Param | What it does |
|---|---|
| `depends_on=` | List of `Callable` or `DataSource` that contributes to the cache key |
| `dynamic_depends_on=` | Callable(s) that receive the function's args and return `DataSource`(s) — for deps that depend on the call |
| `file_depends_on=` | Shorthand: file path(s) tracked as `FileDataSource` |
| `ttl=` | Time-to-live in seconds; `None` (default) = never expires |
| `cache_if=` | Predicate `(result) -> bool`; falsy result → don't cache (still returns to caller) |
| `chunk_max_items=` / `chunk_max_bytes=` | For iterator returns, chunk thresholds (1M items / 1 GB default) |
| `strict=` | Raise `CashImpureFunctionError` at first call if purity analyzer finds issues |
| `assume_safe=` | Silence the purity warning; you've audited and know caching is safe |

Mutually exclusive: `strict` and `assume_safe` — pass both and the
decorator raises `ValueError` immediately.

### `ttl=` — expiration

```python
@cash.cache(ttl=300)   # five minutes
def stock_price(symbol):
    return requests.get(f"https://api.example.com/{symbol}").json()
```

After the TTL elapses, the next call recomputes. Expired entries are
not removed from the backend automatically — call `cash.cleanup()` to
reclaim space, or run `python -m cash clear` from the CLI.

### `file_depends_on=` — invalidate when a file changes

```python
@cash.cache(file_depends_on="config.yaml")
def parse_config():
    return yaml.safe_load(open("config.yaml"))
```

The file's `(mtime, size)` is folded into the cache key. Edit the
file → next call recomputes. Pass a list for multiple files. For more
complex dependency tracking (database tables, API endpoints,
remote URLs), implement your own `DataSource` subclass and pass via
`depends_on=`.

### `depends_on=` — explicit dependency graph

```python
@cash.cache
def load_users():
    return db.query("SELECT * FROM users")

@cash.cache(depends_on=[load_users])
def user_summary():
    users = load_users()
    return {"total": len(users), "active": sum(1 for u in users if u.active)}
```

When `load_users`'s source changes (you edit the function), the source
hash flows up through the dependency graph and invalidates
`user_summary` too. Without `depends_on=` we can usually still detect
this via static analysis, but listing it explicitly makes the link
explicit and lets us follow it across modules.

### `dynamic_depends_on=` — deps that depend on args

When the data source depends on the call's arguments:

```python
from cash import FileDataSource

@cash.cache(dynamic_depends_on=lambda user_id: FileDataSource(f"/data/users/{user_id}.json"))
def load_user(user_id):
    return json.load(open(f"/data/users/{user_id}.json"))
```

The resolver runs with the same `args/kwargs` as the function on every
call. If the resolver raises, you get a one-shot
`CashCacheIneffectiveWarning` and the call proceeds without the dep
in the key.

### `cache_if=` — skip caching by result

```python
@cash.cache(cache_if=lambda r: r is not None)
def lookup(key):
    return cache_backend.get_or_none(key)
```

The predicate runs after the function returns. Falsy → don't cache (the
caller still gets the result). Useful for "don't cache misses",
"don't cache empty results", etc.

If the predicate itself raises, a one-shot `CashCacheIneffectiveWarning`
fires and the result isn't cached. Don't use the predicate to assert
business invariants — its job is purely "should this be cached".

**Iterator returns + `cache_if`:** the predicate is honored when the
result fits in a single chunk. For multi-chunk results, the predicate
is bypassed (warning fires) — see the iterator section below.

### `strict=` and `assume_safe=` — purity gates

By default, `@cash.cache` runs a static analyzer on the function body
(and module-bounded helpers) on first call. If it finds known-impure
calls, scope mutations, dynamic dispatch, or discarded-return calls,
it emits a `CashImpurityWarning`. The function still gets cached.

```python
@cash.cache
def fetch_user(uid):
    return requests.get(f"https://api/{uid}").json()
# First call: CashImpurityWarning fires because requests.get is impure.
```

Three modes:

- **default** (warn) — visible warning, cached anyway
- **`strict=True`** — raise `CashImpureFunctionError` instead. Good for
  CI: fail the build if anyone introduces caching of side-effecting code.
- **`assume_safe=True`** — silence the warning. Use after you've
  audited and know caching is correct (e.g., a memoized API call where
  the side effect is idempotent / harmless on hit).

See [Purity tutorial](tutorials/feature-guides/purity-decorators.md) for the full story including
`@pure`, `@stateful`, and `mark_pure`/`mark_stateful` for third-party
callables.

### `chunk_max_items=` / `chunk_max_bytes=` — iterator chunking

When the decorated function returns an iterator (generator,
`map`/`filter` result, custom iterator), the result is materialized
and stored in chunks. Defaults are 1M items / 1 GB per chunk.

<!-- test:skip reason="opens huge.log which doesn't exist in test env" -->
```python
@cash.cache(chunk_max_items=10_000)
def read_lines(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

for line in read_lines("huge.log"):
    process(line)
# Second run: chunks are read lazily from disk; RAM bounded by chunk size.
```

The cached iterator supports `iter()`, `__next__`, `close()`. Generator
methods `.send()` and `.throw()` are not supported — call them and you
get an `AttributeError` reminding you the iterator is a replay.

---

## Async support

`async def` functions are first-class. The wrapper is `async def` too:

<!-- test:skip reason="async httpx.AsyncClient requires real HTTP client / network" -->
```python
@cash.cache(ttl=60)
async def fetch_user(user_id):
    async with httpx.AsyncClient() as client:
        return (await client.get(f"/users/{user_id}")).json()

users = await asyncio.gather(*(fetch_user(i) for i in range(100)))
```

Concurrent `asyncio.gather` is safe: each unique key computes once,
duplicates wait on the first computation (when `use_locking=True` is
set on the `Cash` instance — for async this currently logs a warning
and proceeds unlocked; see the [API reference](api/cash.md)).

Async generators (`async def gen(): yield ...`) are **not** cached
yet — they emit a `CashCacheIneffectiveWarning` and are returned
unwrapped.

---

## Wrapper methods

Every decorated function gets four extra attributes:

### `func.cache_info()`

```python
@cash.cache
def f(x): return x * 2

f(1); f(1); f(2)
f.cache_info()
# {'hits': 1, 'misses': 2, 'hit_rate': 0.333..., 'total_time_saved': 0.0,
#  'warnings': []}
```

Keys:

- **`hits`**, **`misses`**, **`hit_rate`** — counters since the wrapper
  was created.
- **`total_time_saved`** — sum of execution times avoided on hits.
- **`warnings`** — rolling log (last 20) of recent `CashWarning`
  emissions for this function. Lets you discover silent misbehavior
  after the fact even when `warnings.simplefilter` swallowed the
  stderr emission.

### `func.cache_clear()`

Wipe backend entries whose key starts with this function's name. Also
resets stats, drops the warnings log, and forgets the `_warn_once`
dedup marks (so the next misbehavior re-warns instead of being silent).

### `func.explain(*args, **kwargs)`

Pure introspection — returns a `CacheExplanation` describing whether
the next call with these args would hit or miss the cache, and why:

```python
f.explain(5)
# [MISS] __main__.f — no_entry
#   cache_key: __main__.f:9a3c...:...
#   hint: No matching cache entry. First call with these arguments, or...

f(5)  # compute
f.explain(5)
# [HIT] __main__.f — hit
#   cache_key: __main__.f:9a3c...:...
#   cached_at: 1779637032.79
#   cache_age_seconds: 0.05
#   execution_time_saved: 0.0008
```

`reason` is one of `hit`, `key_uncomputable` (unhashable arg),
`no_entry` (first call / cache cleared / source changed),
`ttl_expired`, `file_changed`. `details` carries reason-specific
extras — see [`CacheExplanation`](api/cash.md#cacheexplanation).

Does NOT call your function, mutate stats, or write to the backend.
Safe to call from sync code even on async-wrapped functions.

### `func.__wrapped__`

The original undecorated function. Useful for testing — call it to
bypass caching entirely.

---

## How a call decides hit vs miss

```mermaid
flowchart TD
    A[Call f(args)] --> B{Cache key computable?}
    B -->|No - unhashable arg| W1[Warning, recompute, don't store]
    B -->|Yes| C{Entry in backend?}
    C -->|No| D[Compute, store]
    C -->|Yes| E{TTL expired?}
    E -->|Yes| D
    E -->|No| F{File deps fresh?}
    F -->|No| D
    F -->|Yes| G[Return cached value]
```

The cache key is `f"{func_name}:{state_hash}:{dynamic_hash}:{args_hash}"`.

- `state_hash` folds in the function's own source hash + every
  `depends_on` source + transitive helper hashes (so editing a helper
  invalidates).
- `dynamic_hash` folds in `dynamic_depends_on` resolver outputs (when
  set).
- `args_hash` is a SHA-256 over the pickled args (with custom hashers
  via `cash.register_hasher` taking precedence for non-picklable types).

Anything that affects the result should be in one of those. If it
isn't, the cache will go stale silently — that's where `func.explain()`
helps diagnose mysteries.

---

## Common gotchas

### Unhashable arguments

```python
import threading

@cash.cache
def f(lock):
    return id(lock)

f(threading.Lock())
# CashCacheIneffectiveWarning: cannot pickle '_thread.lock' object.
# Register cash.register_hasher(_thread.lock, fn) — or pass by value.
```

`threading.Lock`, sockets, open file handles, etc. can't be pickled,
which means we can't build a cache key. The warning explains which
type is the culprit. Either:

1. Don't pass the object; pass something hashable that identifies it.
2. `cash.register_hasher(LockType, lambda lock: id(lock))` if you're
   sure same-identity caching is what you want.

### Instance methods — `self` participates in the key

<!-- test:skip reason="Loader instance is unhashable (no register_hasher); cache is ineffective and stats stay 0/0" -->
```python
class Loader:
    def __init__(self, path):
        self.path = path

    @cash.cache
    def load(self):
        return pd.read_csv(self.path)

Loader("a.csv").load()
Loader("a.csv").load()   # MISS — different self objects = different args_hash
```

Two `Loader` instances with the same `path` produce two separate cache
entries because `self` (a different object each time) is part of the
args. Fix via [`register_hasher`](tutorials/feature-guides/caching-class-methods.md):

<!-- test:skip reason="Loader class defined in skipped previous fence" -->
```python
cash.register_hasher(Loader, lambda l: hashlib.sha256(l.path.encode()).hexdigest())
```

Now both instances share the same args_hash and the second call hits.

### Caching code with side effects

The purity analyzer warns by default if your function calls
`requests.post`, mutates globals, writes files, etc. The cached
behavior is: the side effect runs on the **first** call only. Every
hit replays the return value without the side effect.

If that's what you want (memoizing an API call where the network
roundtrip is the "side effect"), `assume_safe=True` silences the
warning. If it isn't, refactor: separate the pure compute from the
side effect, and only cache the pure part.

### Helper changed but cache didn't invalidate (legacy behavior — fixed)

Editing a plain helper called from a cached function used to leave the
parent's cache stale. As of v0.5.0b2, the analyzer captures helper
source hashes and folds them into the cache key — both cross-process
edits and in-process redefinitions (notebook cell rerun, REPL) are
picked up automatically. Per-call overhead is ~5-30μs.

### `@cash.cache` on a generator

Generators are materialized into a list (or chunks) on first call so
the cache replay can give back fresh iterators. Don't decorate a
function returning an infinite generator (it will hang trying to
exhaust). For very large finite iterators, tune `chunk_max_items=` /
`chunk_max_bytes=`.

### `cache_clear()` clears more than you'd expect on iterators

For chunked iterator caches, `cache_clear()` removes the manifest entry
but the individual chunk entries (keyed
`f"{cache_key}:chunk_{i}"`) are also caught by the
`startswith(func_name)` sweep. No orphans.

---

## Where to go next

- [API reference — Cash class](api/cash.md) — exhaustive signatures
- [Purity tutorial](tutorials/feature-guides/purity-decorators.md) — `@pure`, `@stateful`,
  `mark_pure`, `mark_stateful`
- [Caching class methods](tutorials/feature-guides/caching-class-methods.md) — recipe for
  stateful receivers via `register_hasher`
- [Choosing a backend](tutorials/feature-guides/choosing-a-backend.md) —
  picking RAM / disk / Redis / S3 tiers for your workload
- [Custom hashers](tutorials/feature-guides/custom-hashers.md) —
  `register_hasher` for non-picklable / domain-specific argument types
- [Dynamic dependencies](tutorials/feature-guides/dynamic-dependencies.md) —
  deeper walkthrough of `dynamic_depends_on=` patterns
- [Async caching](tutorials/feature-guides/async-caching.md) —
  `async def` functions, concurrency, and gotchas with locking
- [Thread safety](tutorials/feature-guides/thread-safety.md) —
  `use_locking`, concurrent decorator hits, and shared backends
- [Production transition](tutorials/feature-guides/production-transition.md) —
  moving notebook caches to long-lived services
- [Configuration](getting-started/configuration.md) — picking a
  backend, tier stacks, TOML / env / programmatic resolution
- [Notebook caching](notebook_caching_api.md) — the `%cash_on`
  alternative for statement-level caching in Jupyter
