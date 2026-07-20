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
| `allow_random=` | Silence the unseeded-randomness warning; you know the result is frozen |

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

`depends_on=` also accepts **plain, non-decorated** functions — the dep's
source is snapshotted at registration and folded into the cache key, so
editing it invalidates the dependent:

```python
def score(user):           # not decorated
    return user.visits * 2

@cash.cache(depends_on=[score])
def leaderboard():
    return sorted(load_users(), key=score)
```

Edit `score` → `leaderboard` recomputes. (Previously this edge was inert:
a non-decorated callable contributed nothing to the key, so the declared
dependency was silently ignored.) If a dep's source can't be read at all,
you get a warning rather than a silently dead edge.

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

### `allow_random=` — unseeded randomness

At decoration time, `@cash.cache` scans the function's source for draws
from an unseeded RNG and emits a one-shot `CashRandomnessWarning`:

```python
@cash.cache
def sample():
    return np.random.randn()      # no seed anywhere
# CashRandomnessWarning: Unseeded randomness detected: numpy.random.randn()
```

The warning matters because the first call's value is cached and
replayed forever — later calls never consult the RNG again, so the
"random" number is frozen, and it won't survive a cleared cache either.

This is the same detector the notebook path uses, so both paths agree on
what counts as unseeded. Two ways to make it silent:

- **Seed the RNG** — `np.random.seed(0)`, `random.seed(0)`, or
  `np.random.default_rng(42)`. A seeded draw is reproducible, so no
  warning fires. This is the real fix.
- **`allow_random=True`** — acknowledge the freeze and move on.

```python
@cash.cache(allow_random=True)
def jitter():
    return np.random.randn()
```

The notebook's [`# @cash:allow-random`](annotations.md#cashallow-random-alias-allowrandom)
comment is also honoured inside a decorated function's body.

!!! note
    `allow_random` suppresses a *warning*. It does **not** stop the
    caching — the value is still frozen. Use
    [`cache_if=`](#cache_if--skip-caching-by-result) or drop the
    decorator if you want a fresh draw every call.

Detection is source-based and runs **once per function at decoration
time**, so cached calls pay nothing for it. Two consequences: a function
with no retrievable source (defined via `exec`, or in a bare REPL) is
not scanned, and randomness *inside* a compiled library call — an
unseeded `estimator.fit()`, for example — is invisible to it. Pass an
explicit `random_state=` to such estimators.

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

!!! warning "In a notebook, `cache_info()` reads 0 / 0 — use `explain()` instead"
    The counters live on the **wrapper object**, and they count only since
    that wrapper was created. In a notebook, cash may rebuild the cell that
    defines your function, which re-runs the decorator and produces a fresh
    wrapper with fresh counters. So `cache_info()` can report
    `{'hits': 0, 'misses': 0}` **forever**, even while caching is working
    perfectly and saving you minutes.

    It is not telling you caching is broken — it is telling you *this
    wrapper* has not served a call yet. To check whether caching is actually
    working in a notebook, use either:

    ```python
    f.explain(1)      # -> [HIT] ... execution_time_saved: 23.54
    ```

    ```python
    %cash_stats       # session-wide hits, misses and net time saved
    ```

    Both read through to the real cache rather than a per-wrapper counter.
    `cache_info()` is reliable in scripts and long-lived processes, where
    the wrapper is created once.

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
  invalidates) + the content of any **module global the function reads**
  (see below).
- `dynamic_hash` folds in `dynamic_depends_on` resolver outputs (when
  set).
- `args_hash` is a SHA-256 over the pickled args (with custom hashers
  via `cash.register_hasher` taking precedence for non-picklable types).
  Dicts are canonicalised to sorted-key order first, so two dicts that
  are equal but for insertion order share a key —
  `f({"a": 1, "b": 2})` and `f({"b": 2, "a": 1})` hit the same entry.

### Module globals a function reads

A cached function that reads a module-level global — a config constant, a
dispatch dict of callables — invalidates when that global changes:

```python
TAX_RATE = 0.2

@cash.cache
def net(amount):
    return amount * (1 - TAX_RATE)

net(100)          # 80.0
TAX_RATE = 0.5
net(100)          # 50.0 — recomputed, not the stale 80.0
```

Only globals the function **reads** participate. Globals it *writes*
(`global x; x = ...`) or mutates in place are excluded — those are
side-effect accumulators, and folding them in would invalidate the
function on its own output. A read global whose value can't be hashed
warns once rather than failing the call.

Globals read inside a nested scope count too. A generator expression,
comprehension, or `lambda` compiles to its own code object, so detection
recurses into them (CAS-128):

```python
THRESHOLD = 10

@cash.cache
def count_big(values):
    return sum(v > THRESHOLD for v in values)   # THRESHOLD is tracked

count_big([5, 20])   # 1
THRESHOLD = 1
count_big([5, 20])   # 2 — recomputed
```

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

### C-extension callables and builtins

Caching a callable with no readable Python source — a C-extension
function, a builtin, a NumPy ufunc, a dispatcher, or a
`functools.partial` wrapping one — works rather than crashing. The
source-hashing and AST-analysis steps have no source to read for these,
so they degrade to a stable identity-based fallback instead of raising:

```python
import functools, numpy as np

cached_sqrt = cash.cache(np.sqrt)              # ufunc — fine
cached_max = cash.cache(functools.partial(max, 0))   # partial over a builtin — fine
```

Because there is no source to hash, cash cannot notice a change *inside*
a C extension (upgrading the library, say). That's the same blind spot
any source-based invalidation has; pin the dependency if it matters.

### Caching code with side effects

The purity analyzer warns by default if your function calls
`requests.post`, mutates globals, writes files, etc. The cached
behavior is: the side effect runs on the **first** call only. Every
hit replays the return value without the side effect.

If that's what you want (memoizing an API call where the network
roundtrip is the "side effect"), `assume_safe=True` silences the
warning. If it isn't, refactor: separate the pure compute from the
side effect, and only cache the pure part.

### Editing a helper invalidates the caller's cache

Editing a plain helper called from a cached function invalidates that
function's cache. The analyzer captures helper source hashes and folds
them into the cache key, so both cross-process edits and in-process
redefinitions (notebook cell rerun, REPL) are picked up automatically.
Per-call overhead is ~5-30μs.

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
