# Thread safety — `Cash(use_locking=True)` and double-checked locking

By default, two threads calling the same cached function with the same arguments at the same time each compute the result and race to write the cache entry. That's wasteful but safe — the second writer just overwrites. When the computation is expensive and the concurrency is real, opt into double-checked locking with `Cash(use_locking=True)`.

## The race in detail

Without locking, the wrapper goes `get → miss → compute → set` with no coordination between threads. Two concurrent calls with the same args produce the same timeline:

- Thread A: `backend.get(key)` → miss → starts computing.
- Thread B: `backend.get(key)` → miss → also starts computing.
- A finishes first, writes the entry.
- B finishes second, overwrites with an identical value.

Net result: the computation ran twice instead of once. Both threads return the right answer — there's no data corruption — but you paid double for compute and double for the serialize-and-store step. For a 50 ms function this is invisible; for a 30 s training run or a paid API call, it matters.

## The fix: double-checked locking

When `use_locking=True`, the miss path routes through `_compute_with_lock` (`src/cash/core.py:1699-1726`) instead of calling the compute closure directly (`src/cash/core.py:1204-1206`). The helper does three things:

1. **Acquire `self.backend.lock(cache_key)`** as a context manager (`src/cash/core.py:1710`).
2. **Re-read the cache inside the lock** (`src/cash/core.py:1711`). If the entry now exists (because a concurrent caller wrote it while the current thread was waiting on the lock), return the cached value — `_validate_ttl` runs, then `_wrap_iterator_hit` reconstructs iterator chunks if needed (`src/cash/core.py:1716-1718`).
3. **Otherwise compute and store inside the lock** (`src/cash/core.py:1723`).

The "double check" is the read on step 2 — the first check was the lock-free read on the wrapper's hot path before `_compute_with_lock` was called. The second read is needed because the first one is racy with concurrent writers.

If lock acquisition itself raises `OSError`, `RuntimeError`, or `TimeoutError`, the helper emits a `CashCacheIneffectiveWarning` via `_warn_lock_failed` and falls through to an unlocked compute (`src/cash/core.py:1724-1726`). The user's function still runs and the result is still cached — the redundancy guarantee is best-effort, not absolute.

## Enabling

```python
from cash import Cash
c = Cash(use_locking=True)
c.register_magic()

@c.cache
def expensive(x):
    ...
```

The flag is a `Cash`-instance option (`src/cash/core.py:343,379`), not a per-decorator one. All functions registered through this instance go through the lock path on misses; switch instances if you want a mix.

Lock acquisition uses **the cache backend itself** — `self.backend.lock(cache_key)` returns a context manager whose semantics are defined by the backend subclass. See the next section for the reality of what each backend implements.

## Which backends actually implement locking

This is the load-bearing detail of the feature. The base class default at `src/cash/backends/_base.py:318-324` is a `nullcontext()` — i.e. **no lock at all**:

```python
def lock(self, key: str) -> contextlib.AbstractContextManager:
    """Return a context manager that acquires a lock for the given key.

    Default implementation does nothing (no-op context manager).
    """
    from contextlib import nullcontext
    return nullcontext()
```

Backends inherit this unless they override. The override status across the shipped backends:

| Backend | Overrides `lock()`? | What it actually does |
|---|---|---|
| `RedisBackend` | Yes (`src/cash/backends/redis_backend.py:199-209`) | Distributed lock via `client.lock(lock_name, timeout=60, blocking_timeout=10)` — Redis's standard `SET NX PX` recipe. Lock name is `{prefix}lock:{cache_key}`. |
| `SQLiteBackend` | Yes (`src/cash/backends/sqlite_backend.py:307-309`) | **No-op** (`contextlib.nullcontext()`). The override exists to document intent — SQLite handles its own transaction-level locking through the backend's internal `RLock`, and the docstring says so explicitly. |
| `FileBackend` | No | Inherits the no-op default. |
| `InMemoryBackend` | No | Inherits the no-op default. |
| `S3Backend` | No | Inherits the no-op default. S3 has no native primitive for this anyway. |
| `CascadingBackend` | No | Inherits the no-op default. |
| `TieredBackend` | No | Inherits the no-op default. **This is the default `Cash()` backend.** |

**Practical consequence:** with the default `Cash(use_locking=True)` and no explicit backend, the lock is a no-op — the double check still happens (re-read inside the lock-free context), but two truly concurrent threads can still both miss the second read and both compute. The feature delivers the redundancy-avoidance guarantee only against `RedisBackend`.

The wrapper code is correct — the `with self.backend.lock(...)` block is in place. The contention story is purely a property of the backend you wire up.

## When to use

- **Multi-process workers sharing a Redis cache.** Two Airflow workers, two Lambda functions, two CLI scripts on different machines pointing at the same Redis instance — the distributed lock is real and the redundancy guarantee holds across processes and across hosts.
- **Multi-threaded sync code where the function is expensive enough that the lock latency is dwarfed by the saved compute.** Wire up a custom backend whose `lock()` returns a per-key `threading.Lock()` from a dict; the override is small and the wrapper already routes through it.
- **Cases where redundant computes have side effects you can't afford.** Paid API calls, GPU jobs you bill against quota, anything that increments a remote counter. The "second writer wins" semantic is benign for pure functions; it isn't for impure ones (and Cash will already have warned you about purity — see [Purity Decorators](purity-decorators.md)).

## When NOT to use

- **Single-threaded code.** There are no concurrent callers — the lock adds overhead with nothing to defend against.
- **Default backend, no custom override.** As established above, the shipped non-Redis backends don't implement `lock()` — turning the flag on costs you the extra `backend.get()` round-trip on every miss without buying you anything.
- **Async-only code.** The async wrapper does not dispatch through `_compute_with_lock`. It emits a one-shot `CashCacheIneffectiveWarning` and proceeds without lock (`src/cash/core.py:1258-1267`). See [Async Caching](async-caching.md) for the rationale — the lock helper takes a sync `compute_and_store` closure and the async wrapper would need to await inside the lock block.
- **Cheap computations.** If your function runs in under a millisecond, the redundant work from a lock-free miss is cheaper than the lock acquisition itself.

## Lock behaviour details

- **Per cache key, not per function.** The key passed to `backend.lock()` is the full `func_name:state_hash:dynamic_hash:args_hash` cache key (`src/cash/core.py:1710,1728-1729`), so two different arg-tuples for the same function don't serialize on each other.
- **Times out gracefully.** The Redis backend passes `timeout=60` (lock TTL — auto-released after 60 s in case the holder crashed) and `blocking_timeout=10` (max 10 s wait to acquire). A failed acquisition raises into the `except` arm of `_compute_with_lock`, which surfaces a warning and falls through to an unlocked compute (`src/cash/core.py:1724-1726`). The user's call never hangs indefinitely.
- **Lock acquisition errors don't crash.** `OSError`, `RuntimeError`, and `TimeoutError` are caught and converted to `CashCacheIneffectiveWarning` via `_warn_lock_failed` at `src/cash/core.py:1852-1871`. Other exception types propagate — if the backend raises something unusual, you'll see the original traceback.
- **The lock is held across the user's compute.** This is intentional — it's what makes the redundancy guarantee work — but it means a slow function holds the lock for its full duration. For minutes-long computes against a shared Redis, tune the Redis `timeout=` if you need a longer ceiling, or rethink whether the lock is the right tool (single-flight at the call site may be simpler).
- **No deadlock detection.** Two cached functions that call each other under `use_locking=True` against a real backend lock can deadlock on cyclic key acquisition. Cash does not detect this; the Redis lock will eventually time out on `blocking_timeout=10`, but you'll see two warnings and two redundant computes.

## Debugging concurrency

The standard introspection surface works:

- `f.cache_info()` reports hit/miss counts. Under contention without locking, you'll see misses that "should" have been hits — count them across runs to estimate redundancy. Lock failures land in `cache_info()['warnings']` per the `_warn_once` rolling log (`src/cash/core.py:1907-1912`).
- `f.explain()` shows the cause of the next call's miss or hit. It doesn't currently surface "this call was about to be a miss but the lock turned it into a hit" — but the absence of a redundant compute is itself the evidence.
- Set `Cash(debug=True)` to log decorator calls with their hit/miss outcome — under load you can grep for repeated misses on the same key as a sanity check.

## Caveats

- **Latency on every cache miss.** The double-checked-locking path adds two operations per miss (acquire + re-read) on top of the compute. For Redis that's two network round-trips; for the no-op backends it's effectively free.
- **Lock failure reverts to no-lock.** The redundancy guarantee is best-effort. A flaky Redis connection or a corrupt file lockfile produces the same outcome as `use_locking=False` for the duration of the outage, plus a warning.
- **Not supported for async.** See [Async Caching](async-caching.md). The async wrapper does its own warning emit and skips the lock helper entirely.
- **Lock TTL is fixed at the backend.** The Redis backend's `timeout=60` is hardcoded in `src/cash/backends/redis_backend.py:209`; there is no `Cash`-level option to raise it. If you need longer holds, subclass `RedisBackend` and override `lock()`.
- **No per-call disable.** The `use_locking` flag is on the `Cash` instance — there's no `@cash.cache(use_locking=False)` to override it for a single hot function. Use a second `Cash` instance if you need a mix.
- **The flag costs you nothing if the backend is a no-op.** On the default `TieredBackend`, `Cash(use_locking=True)` adds the double-checked path but the lock itself does nothing, so the only real overhead is the extra `backend.get()` inside the (no-op) context manager. Cheap; also useless.

## API reference

| Surface | Effect |
|---|---|
| `Cash(use_locking=True)` | Route the miss path through `_compute_with_lock` for every function registered against this instance (`src/cash/core.py:343,379,1204-1206`). |
| `Cash(use_locking=False)` (default) | Skip the lock helper; compute closure runs directly. |
| `RedisBackend.lock(key)` | Returns the result of `client.lock(f"{prefix}lock:{key}", timeout=60, blocking_timeout=10)` — a real distributed lock (`src/cash/backends/redis_backend.py:199-209`). |
| `SQLiteBackend.lock(key)` | Returns `contextlib.nullcontext()`. SQLite serializes via its own connection-level `RLock`; the override is documentary. |
| `FileBackend.lock(key)` | Inherits the base no-op. |
| `InMemoryBackend.lock(key)` | Inherits the base no-op. |
| `S3Backend.lock(key)` | Inherits the base no-op. S3 has no equivalent primitive. |
| `CascadingBackend.lock(key)`, `TieredBackend.lock(key)` | Inherit the base no-op. The default `Cash()` backend is `TieredBackend`. |
| `CacheBackend.lock(key)` (base) | `contextlib.nullcontext()` (`src/cash/backends/_base.py:318-324`). Subclasses override as needed. |
| `CashCacheIneffectiveWarning` (lock failed) | Emitted once per `(func_name, "lock_failed")` when lock acquisition raises `OSError`, `RuntimeError`, or `TimeoutError` (`src/cash/core.py:1852-1871`). The call proceeds without the lock. |
| `CashCacheIneffectiveWarning` (async + locking) | Emitted once per func when the async wrapper sees `use_locking=True` (`src/cash/core.py:1258-1267`). The await proceeds without lock. |

## Related

- [Async Caching](async-caching.md) — the async wrapper skips `_compute_with_lock` and emits a warning; this page is its sync counterpart.
- [Choosing a Backend](choosing-a-backend.md) — the lock-implementation reality depends entirely on which backend you pick. Redis is the only shipped backend that holds a real lock.
- [Debugging and Monitoring](debugging-and-monitoring.md) — how to read `cache_info()['warnings']` and surface lock-acquisition failures after the fact.
- [Decorator (`@cash.cache`)](../../decorator.md) — full decorator-kwarg reference; `use_locking` lives on the `Cash` constructor, not on individual decorations.
