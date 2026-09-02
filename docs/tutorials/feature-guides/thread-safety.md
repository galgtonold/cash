# Thread safety — `Cash(use_locking=True)` and double-checked locking

By default, two threads calling the same cached function with the same arguments at the same time each compute the result and race to write the cache entry. That's wasteful but safe — the second writer just overwrites. When the computation is expensive and the concurrency is real, opt into double-checked locking with `Cash(use_locking=True)`.

**The short version:** `use_locking=True` gives you **in-process single-flight on every shipped backend** — the default `Cash()` included, and on the async path too. You only need `RedisBackend` when the concurrent callers live in **different processes or on different hosts**.

## The race in detail

Without locking, the wrapper goes `get → miss → compute → set` with no coordination between threads. Two concurrent calls with the same args produce the same timeline:

- Thread A: `backend.get(key)` → miss → starts computing.
- Thread B: `backend.get(key)` → miss → also starts computing.
- A finishes first, writes the entry.
- B finishes second, overwrites with an identical value.

Net result: the computation ran twice instead of once. Both threads return the right answer — there's no data corruption — but you paid double for compute and double for the serialize-and-store step. For a 50 ms function this is invisible; for a 30 s training run or a paid API call, it matters.

## The fix: double-checked locking

<!-- claim: cash/core.py:Cash._compute_with_lock @4cc0528e, cash/core.py:Cash._warn_lock_failed @28a3d6e3 -->
When `use_locking=True`, the miss path routes through `Cash._compute_with_lock` instead of calling the compute closure directly. The helper does three things:

1. **Acquire `self.backend.lock(cache_key)`** as a context manager.
2. **Re-read the cache inside the lock.** If the entry now exists (because a concurrent caller wrote it while the current thread was waiting on the lock), return the cached value — `_validate_ttl` runs, then `_wrap_iterator_hit` reconstructs iterator chunks if needed.
3. **Otherwise compute and store inside the lock.**

The "double check" is the read on step 2 — the first check was the lock-free read on the wrapper's hot path before `_compute_with_lock` was called. The second read is needed because the first one is racy with concurrent writers.

Acquisition, compute, and release are deliberately separated. If **acquisition** raises anything at all, the helper emits a `CashCacheIneffectiveWarning` via `_warn_lock_failed` and falls through to an unlocked compute — the user's function still runs and the result is still cached, so the redundancy guarantee is best-effort, not absolute. If **release** fails, it is logged at debug and swallowed so a release error can't re-run the compute. A **compute** exception propagates normally and is never mistaken for a lock failure.

## Enabling

```python
from cash import Cash
c = Cash(use_locking=True)
c.register_magic()

@c.cache
def expensive(x):
    ...
```

<!-- claim: cash/core.py:Cash.__init__ @158d0cb2 -->
The flag is a `Cash`-instance option, not a per-decorator one. All functions registered through this instance go through the lock path on misses; switch instances if you want a mix.

Lock acquisition uses **the cache backend itself** — `self.backend.lock(cache_key)` returns a context manager whose semantics are defined by the backend subclass. See the next section for what each backend implements.

## Which backends implement locking

<!-- claim: cash/backends/_base.py:CacheBackend.lock @03560a4b, cash/backends/_base.py:CacheBackend._inprocess_key_lock @f545fb46, cash/backends/_base.py:_KEY_LOCK_BOOTSTRAP @43b21da4 -->
All of them. The base-class implementation, `CacheBackend.lock`, is a **real per-key `threading.RLock`**, drawn from a process-local registry (`CacheBackend._inprocess_key_lock`) that is created lazily and guarded by a per-instance meta-lock, with a module-global bootstrap lock (`_KEY_LOCK_BOOTSTRAP`) serializing the first-use race. Every backend that doesn't override `lock()` inherits this — so `use_locking=True` honors its promise on the default `Cash()`.

It's an `RLock`, not a `Lock`, so a compute that re-enters the same key on the same thread (memoized recursion, a function that calls itself) re-acquires instead of deadlocking. Other threads still block until the leader fully releases.

There are exactly **two** `lock()` definitions in the codebase:

<!-- claim: cash/backends/redis_backend.py:RedisBackend.lock @cfdf2e01 -->
| Backend | `lock()` | What it actually does |
|---|---|---|
| `CacheBackend` (base) | **Defines** | Per-key `threading.RLock` from a process-local registry. **In-process** single-flight. Inherited by everything below except Redis. |
| `RedisBackend` | **Overrides** | Distributed lock via `client.lock(lock_name, timeout=60, blocking_timeout=10)` — Redis's standard `SET NX PX` recipe. Lock name is `{prefix}lock:{cache_key}`. **Cross-process, cross-host.** |
| `SQLiteBackend` | Inherits — **deliberately** | Gets the base in-process `RLock`. A `NOTE` in `sqlite_backend.py` explains why there is no override: the backend's internal `self._lock` guards *storage integrity* (concurrent DB writes), which is a distinct concern from *compute single-flight*. |
| `FileBackend` | Inherits | Base in-process `RLock`. |
| `InMemoryBackend` | Inherits | Base in-process `RLock`. |
| `S3Backend` | Inherits | Base in-process `RLock`. S3 has no native primitive for cross-process locking. |
| `CascadingBackend` | Inherits | Base in-process `RLock`. |
| `TieredBackend` | Inherits | Base in-process `RLock`. **This is the default `Cash()` backend.** |

**Practical consequence:** with the default `Cash(use_locking=True)` and no explicit backend, two concurrent threads in the same process single-flight correctly — one computes, the other blocks and then observes the stored result on the double check. What the in-process lock cannot do is coordinate **separate processes**: two `multiprocessing` workers, two Airflow tasks, two containers each hold their own registry and will each compute once. That, and only that, is what `RedisBackend` buys you.

## When to use

- **Multi-threaded sync code where the function is expensive enough that the lock latency is dwarfed by the saved compute.** This works out of the box now — no custom backend required.
- **Concurrent `asyncio` awaits of the same key.** Supported; see [Async Caching](async-caching.md) and the section below.
- **Multi-process / multi-host workers sharing a Redis cache.** Two Airflow workers, two Lambda functions, two CLI scripts on different machines pointing at the same Redis instance — the distributed lock is real and the guarantee holds across processes and across hosts. **This is the case Redis is required for.**
- **Cases where redundant computes have side effects you can't afford.** Paid API calls, GPU jobs you bill against quota, anything that increments a remote counter. The "second writer wins" semantic is benign for pure functions; it isn't for impure ones (and Cash will already have warned you about purity — see [Purity Decorators](purity-decorators.md)).

## When NOT to use

- **Single-threaded, single-process, non-async code.** There are no concurrent callers — the lock adds overhead with nothing to defend against.
- **Cheap computations.** If your function runs in under a millisecond, the redundant work from a lock-free miss is cheaper than the lock acquisition and the extra `backend.get()` on every miss.
- **Cross-process coordination without Redis.** The in-process `RLock` is per-process by construction. If your concurrency is `multiprocessing`/`joblib`/separate services, `use_locking=True` on a file or tiered backend will not serialize them — reach for `RedisBackend`.

## Async

<!-- claim: cash/core.py:Cash._make_async_wrapper @1f6d895c -->
`use_locking=True` **is supported on the async path**, via in-process single-flight rather than `_compute_with_lock`. Concurrent awaits of the same cache key coalesce: the first awaiter (the *leader*) registers an `asyncio.Event` in `self._async_inflight`, computes, and stores; other awaiters of the same key (the *followers*) `await` the event and then read the stored result. If the leader stored nothing — `cache_if` rejected the value, or the compute raised — followers fall through and compute themselves, so correctness is never traded for the optimization.

The coalescing is keyed on the running event loop, so it dedupes an `asyncio.gather` within one process, not across processes. For cross-process async, you still want Redis. Test reference: `tests/test_core/test_async_single_flight.py`.

## Lock behaviour details

- **Per cache key, not per function.** The key passed to `backend.lock()` is the full `func_name:state_hash:dynamic_hash:args_hash` cache key, so two different arg-tuples for the same function don't serialize on each other.
- **Any acquisition failure degrades to an unlocked compute.** `_compute_with_lock` catches `Exception` broadly on `__enter__` — a Redis `LockError` on contention/timeout, a dropped connection, an `OSError` on a file lock — surfaces a `CashCacheIneffectiveWarning` and proceeds without the lock. The user's call never hangs indefinitely and never crashes on a lock problem.
- **Redis times out gracefully.** The Redis backend passes `timeout=60` (lock TTL — auto-released after 60 s in case the holder crashed) and `blocking_timeout=10` (max 10 s wait to acquire). A failed acquisition lands in the degrade path above.
- **The lock is held across the user's compute.** This is intentional — it's what makes the redundancy guarantee work — but it means a slow function holds the lock for its full duration. For minutes-long computes against a shared Redis, tune the Redis `timeout=` if you need a longer ceiling.
- **Re-entrant on the same thread.** The base lock is an `RLock`, so same-thread recursion into the same key is safe. Cyclic acquisition *across* threads or against a Redis lock can still deadlock; Cash does not detect this. The Redis lock will eventually time out on `blocking_timeout=10`, but you'll see warnings and redundant computes.

## Debugging concurrency

The standard introspection surface works:

- `f.cache_info()` reports hit/miss counts. Under contention without locking, you'll see misses that "should" have been hits — count them across runs to estimate redundancy. Lock failures land in `cache_info()['warnings']` per the `_warn_once` rolling log.
- `f.explain()` shows the cause of the next call's miss or hit. It doesn't currently surface "this call was about to be a miss but the lock turned it into a hit" — but the absence of a redundant compute is itself the evidence.
- Set `Cash(debug=True)` to log decorator calls with their hit/miss outcome — under load you can grep for repeated misses on the same key as a sanity check.

## Caveats

- **Latency on every cache miss.** The double-checked-locking path adds two operations per miss (acquire + re-read) on top of the compute. For Redis that's two network round-trips; for the in-process lock it's an uncontended `RLock` acquire — nanoseconds.
- **Lock failure reverts to no-lock.** The redundancy guarantee is best-effort. A flaky Redis connection produces the same outcome as `use_locking=False` for the duration of the outage, plus a warning.
- **In-process only, unless Redis.** Worth repeating because it's the one thing the flag cannot do on its own: the default lock does not span processes.
- **Lock TTL is fixed at the backend.** The Redis backend's `timeout=60` is hardcoded in `RedisBackend.lock`; there is no `Cash`-level option to raise it. If you need longer holds, subclass `RedisBackend` and override `lock()`.
- **No per-call disable.** The `use_locking` flag is on the `Cash` instance — there's no `@cash.cache(use_locking=False)` to override it for a single hot function. Use a second `Cash` instance if you need a mix.

## API reference

| Surface | Effect |
|---|---|
| `Cash(use_locking=True)` | Route the miss path through `Cash._compute_with_lock` for every function registered against this instance. On the async path, enables in-process single-flight instead. |
| `Cash(use_locking=False)` (default) | Skip the lock helper; compute closure runs directly. |
| `CacheBackend.lock(key)` (base) | Per-key `threading.RLock` from a process-local registry — **in-process single-flight**. Inherited by every backend except Redis. |
| `RedisBackend.lock(key)` | Returns `client.lock(f"{prefix}lock:{key}", timeout=60, blocking_timeout=10)` — a real **distributed** lock. The only backend that coordinates across processes. |
| `SQLiteBackend.lock(key)`, `FileBackend.lock(key)`, `InMemoryBackend.lock(key)`, `S3Backend.lock(key)`, `CascadingBackend.lock(key)`, `TieredBackend.lock(key)` | Inherit the base in-process `RLock`. The default `Cash()` backend is `TieredBackend`. |
| `CashCacheIneffectiveWarning` (lock failed) | Emitted once per `(func_name, "lock_failed")` when lock acquisition raises (`Cash._warn_lock_failed`). The call proceeds without the lock. |

## Related

- [Async Caching](async-caching.md) — `use_locking=True` is supported there too, via in-process single-flight; this page is its sync counterpart.
- [Choosing a Backend](choosing-a-backend.md) — every backend single-flights in-process; pick Redis when you need it across processes.
- [Debugging and Monitoring](debugging-and-monitoring.md) — how to read `cache_info()['warnings']` and surface lock-acquisition failures after the fact.
- [Decorator (`@cash.cache`)](../../decorator.md) — full decorator-kwarg reference; `use_locking` lives on the `Cash` constructor, not on individual decorations.
