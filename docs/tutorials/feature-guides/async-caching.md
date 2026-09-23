# Async functions

!!! info "Applies to: decorator"
    Code that puts `@cash.cache` on `async def` functions.

`@cash.cache` works on `async def` functions with no extra option. Cash stores
the awaited result, not the coroutine, and on a hit your `await` returns the
stored value without running the body.

```python
import asyncio
import cash

n = {"calls": 0}

@cash.cache
async def f(x):
    n["calls"] += 1                # @cash:assume-safe (a counter for this demo)
    await asyncio.sleep(0.1)       # stands in for a request
    return x * x

async def main():
    print(await f(3))              # first call: runs the body
    print(await f(3))              # cache hit: no sleep
    print(n["calls"])              # 1

asyncio.run(main())
```

<!-- claim: cash/core.py:Cash._make_wrapper @8d59b655 -->
Everything else works as on a sync function, because the code around the body
is shared: `ttl=`, `depends_on=`, `dynamic_depends_on=`, `file_depends_on=`,
file reads inside the body, `cache_if=`, the side-effect checks with `strict=`
and `assume_safe=`, and iterator results. `f.cache_info()`, `f.cache_clear()`
and `f.explain()` are plain synchronous calls on an async function too.

## Concurrent awaits

<!-- claim: cash/decorator/runtime.py:RuntimeMixin._single_flight @790b5e9b -->
By default, two tasks that await the same uncached key at once both run the
body. With `Cash(use_locking=True)` they share one computation: the first runs
the body, the others wait and read its result. This covers every event loop in
the process, not other processes. See
[Threads and processes](thread-safety.md).

```python
from cash import Cash

app = Cash(use_locking=True)

@app.cache
async def fetch(uid):
    await asyncio.sleep(0.1)       # stands in for a request
    return {"id": uid}

async def load_all():
    return await asyncio.gather(fetch(1), fetch(2), fetch(2), fetch(3))   # fetch(2) runs once

asyncio.run(load_all())
```

## Expiry

A `ttl=` works as on a sync function. After it passes, the next await runs the
body again:

```python
@cash.cache(ttl=3600)
async def stock_quote(symbol):
    return await price_api.get(symbol)

async def demo_ttl():
    await stock_quote("ACME")      # first call: cache miss, calls the API
    await stock_quote("ACME")      # second call: cache hit
    # test:inject: import time as _t; _saved_time = _t.time; _t.time = lambda: _saved_time() + 7200
    await stock_quote("ACME")      # cache miss: the ttl has passed
    # test:inject: _t.time = _saved_time

asyncio.run(demo_ttl())
```

## Cache the leaves or the batch

```python
# Each request cached on its own: adding one id computes only that one.
@cash.cache
async def fetch_one(uid):
    return await api.get(uid)

async def fetch_many(uids):
    return await asyncio.gather(*(fetch_one(u) for u in uids))

# The whole batch as one entry: one read and one write per batch.
@cash.cache
async def fetch_batch(uids):
    return await asyncio.gather(*(api.get(u) for u in uids))
```

Cache the leaves when the set of inputs keeps changing; cache the batch when it
is stable and many small entries would cost more than one big one.

A call to an HTTP or LLM client inside the body warns once, on the first call,
that it opened a network connection
([`IMPURE-OBSERVED-EFFECTS`](../../warnings.md#impure-observed-effects)). See
[LLM API calls](../use-cases/llm-api-calls.md) for how to
accept it.

## Limits

<!-- claim: cash/core.py:Cash.cache @75e545d3 -->
- **Async generators are not cached.** An `async def` that uses `yield` is
  returned undecorated, with a warning
  ([`CACHE-ASYNC-GENERATOR`](../../warnings.md#cache-async-generator)). To cache
  a stream, have the coroutine **return** a sync generator or a list instead;
  that result is cached as an iterator.
- **Don't cache live objects.** A session, a connection or a cursor belongs to
  the event loop that made it. Cache the data you read through it.
- **A hit skips the body, locks included.** An `asyncio.Lock` or rate limiter
  inside the body is not taken on a hit.
- **`cache_if=` must be a plain function.** It gets the awaited value. An
  `async def` predicate returns a coroutine, which counts as true.
- **Storing runs on the event loop.** On a miss, serializing the result
  happens before your `await` returns, so a very large result blocks the loop
  briefly.
- **A cancelled call is not counted** in `cache_info()`, and nothing is stored.

## Related

- [Threads and processes](thread-safety.md): `use_locking=True`.
- [Iterators](iterator-caching.md): what happens to a returned generator.
- [LLM API calls](../use-cases/llm-api-calls.md)
