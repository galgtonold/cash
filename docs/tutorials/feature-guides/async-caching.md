# Async caching — `@cash.cache` on coroutine functions

Cash caches `async def` functions with the same TTL, file-dependency tracking, purity analysis, and iterator-chunking it gives sync functions. This page covers the differences between the async and sync wrappers and the one limitation the async path doesn't share with the sync one: async generators.

## Why this exists

The natural cached unit for `async def` is the *awaited result*, not the coroutine object. A naive `functools.cache` on a coroutine function would return the same exhausted coroutine on every hit — awaitable exactly once. Cash unwraps the await inside the wrapper, stores the awaited value under the same cache-key scheme used for sync functions, and on a hit returns the value directly so the caller's `await` resolves immediately without re-running the coroutine body.

<!-- claim: cash/core.py:Cash._make_async_wrapper @93cbbcc0, cash/core.py:Cash._make_wrapper @ceedc60c -->
The dispatch happens at decoration time: `inspect.iscoroutinefunction(func)` selects `_make_async_wrapper`, everything else falls through to `_make_wrapper`. Both wrappers share the helpers (`_resolve_cache_key`, `_try_get_cached`, `_store_in_cache`, `_wrap_iterator_hit`, `_write_chunks`), so the storage layout and metadata shape are identical.

## Quick start

```python
import asyncio
import cash
import aiohttp

@cash.cache
async def fetch_user(uid):
    async with aiohttp.ClientSession() as s:
        async with s.get(f"https://api.example.com/users/{uid}") as r:
            return await r.json()

async def demo_cached_await():
    user = await fetch_user(42)   # First call: cache miss — runs the request
    user = await fetch_user(42)   # Second call: cache hit — returns the cached dict

asyncio.run(demo_cached_await())
```

No async-specific decorator flag. `@cash.cache` recognises the coroutine function and wires up the async wrapper automatically.

### Verifying it's actually caching

A quick sanity check that the second await is served from cache, not the wire:

<!-- test:expect-warning reason="the call counter mutates an enclosing dict, which the purity analyzer correctly flags — it is the teaching device here, since counting the computes is exactly how you show the second await didn't run the body" -->
```python
import asyncio
import cash

c = cash.Cash(cache_dir="./cache_async_demo", register_magic=False)
n = {"calls": 0}

@c.cache
async def f(x):
    n["calls"] += 1
    await asyncio.sleep(0)        # actually async-suspends the coroutine
    return x * 11

async def demo_compute_count():
    print(await f(3), await f(3))  # 33 33
    print("compute count:", n["calls"])   # 1
    print(f.cache_info())                  # {'hits': 1, 'misses': 1, ...}

asyncio.run(demo_compute_count())
```

The pattern matches `test_async_function_caches` and `test_async_cache_info` in `tests/test_core/test_async_basic.py`.

## What works on async wrappers

<!-- claim: cash/core.py:Cash._make_async_wrapper @93cbbcc0 broad="the parity list is a claim about the whole async wrapper body" -->
The async wrapper mirrors the sync wrapper feature-for-feature with the following confirmed parity:

- **TTL and freshness.** `_validate_ttl` on the hit path is shared between wrappers; `ttl=` works identically.
- **Static and dynamic dependencies.** `depends_on=` and `dynamic_depends_on=` go through `_resolve_cache_key`, which is the same call the sync wrapper uses.
- **File dependency auto-tracking.** The `FileAccessTracker` block wraps the `await func(*args, **kwargs)` call, so `pandas.read_*`, `numpy.load`, `joblib.load`, and bare `open()` calls inside the coroutine body are auto-tracked the same way they would be in a sync function. Test reference: `test_async_auto_track_open` in `tests/test_core/test_async_file_tracking.py`.
- **Purity analysis.** `_analyze_dependencies` runs on the first call regardless of sync/async; the AST-level analyzer doesn't distinguish coroutine functions from regular ones, so impurity warnings, `@cash.pure`, `assume_safe`, and `strict` apply unchanged.
- **`cache_if=` predicate.** Applied identically on the non-iterator path and on the single-chunk path.
- **Iterator chunking for `async def` returning a sync iterator.** When a coroutine body executes `return (i for i in range(n))` or similar, the await produces a regular generator object. `_is_one_shot_iterator(res)` catches it and dispatches through the same `_write_chunks` / `_store_chunked_manifest` path the sync wrapper uses. The single-chunk fast path applies; the multi-chunk path returns `_ChunkedCachedIterator`. Test reference: `test_async_function_returning_iterator` in `tests/test_core/test_iterator_caching.py`:

    ```python
    @cash.cache
    async def make_iter(stop):
        await asyncio.sleep(0)
        return (i for i in range(stop))   # sync generator on the return

    list(await make_iter(4))   # [0, 1, 2, 3] — first call materialises + chunks
    list(await make_iter(4))   # [0, 1, 2, 3] — second call replays from chunks
    ```
- **Introspection API.** `cache_info()`, `cache_clear()`, and `explain()` are attached by `_wrap_with_stats` for both wrapper kinds. `explain()` is synchronous even on async-wrapped functions — it inspects the next call's outcome without awaiting anything.
- **Lineage attachment.** `_attach_lineage(res, cache_key)` runs after the await on non-iterator returns, so a cached pandas DataFrame returned from an `async def` gets its `_cash_lineage_hash` attribute and can short-circuit downstream args-hashing the same as a sync return.

## What is NOT supported

One path is explicitly opted out on the async side:

<!-- claim: cash/core.py:Cash.cache @91b24c1a -->
- **Async generators (`async def` with `yield`).** Detected at `src/cash/core.py` *before* the async/sync wrapper split. The decorator emits a one-shot `CashCacheIneffectiveWarning` ("async generators are not cached in this release. The function is returned unwrapped.") and returns the bare async generator function. The user can still iterate it; nothing is cached. Test reference: `test_async_generator_emits_warning_and_returns_unwrapped` in `tests/test_core/test_async_generator_warns.py`. The escape hatch when you need the chunked-cache treatment is to write `async def f(): return (... for ... in ...)` (return a sync generator from the coroutine) — that path *is* supported, see the iterator bullet above.

**Single-flight coalescing is supported on the async side**, enabled by constructing your Cash instance with `Cash(use_locking=True)` — it is a *constructor* option, **not** a decorator keyword (`@cash.cache(use_locking=True)` raises `TypeError`). Concurrent awaits of the same key then coalesce so the function computes once. The first awaiter (the *leader*) registers an `asyncio.Event`, computes, and stores; other awaiters of the same key (the *followers*) wait on the event and then read the stored result. If the leader stored nothing (e.g. `cache_if` rejected the value), followers fall through and compute themselves, so correctness is never sacrificed for the optimization. This is *in-process* coalescing keyed on the running event loop — it dedupes an `asyncio.gather` within one process, not across processes (use a distributed lock for that). Test reference: `tests/test_core/test_async_single_flight.py`.

## Notebook cells with top-level `await`

Everything above is about `@cash.cache` on an `async def`. The notebook path has
its own async entry point, and it needs no opt-in: a cell using Jupyter's
top-level `await` is cached like any other cell.

<!-- test:skip reason="illustrative — top-level await requires a live IPython kernel" -->
```python
%%cash
rows = await db.fetch("SELECT * FROM events")   # cached like any other cell
```

<!-- claim: cash/notebook/ipython/cell_executor.py:CellExecutor.execute_cell_async @d4521d68, cash/notebook/statement/processor.py:StatementProcessor.process_statement_async @db78635d -->
ipykernel dispatches such a cell through `shell.run_cell_async` rather than the
`pre_run_cell` hook that `%cash_on` patches, so cash intercepts that entry point
too and routes the cell into `CellExecutor.execute_cell_async` →
`StatementProcessor.process_statement_async`. That pipeline is the line-for-line
twin of the sync one, so awaited cells get lineage tracking, upstream reset, and
result caching alike.

The cache-hit check runs *before* the coroutine is built, so an unchanged re-run
skips the `await` entirely rather than re-issuing the request — which is the
whole point when the awaited call is a paid API.

## Common patterns

### LLM and HTTP API calls

The original motivation. API responses are deterministic in their input prompt and idempotent for cache purposes; the latency and dollar cost of a re-call dwarf the cache lookup. Wrap the outermost function whose inputs are stable across iteration:

```python
@cash.cache(ttl=3600)
async def llm_complete(prompt: str, model: str = "claude-opus-4-7"):
    async with anthropic.AsyncAnthropic() as client:
        resp = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return resp.content[0].text
```

See [LLM API Calls](../use-cases/llm-api-calls.md) for the full prompt-iteration workflow and how `ttl=` interacts with prompt edits.

A typical TTL pattern for "data is fine if it's under an hour old":

```python
@cash.cache(ttl=3600)
async def stock_quote(symbol):
    return await price_api.get(symbol)

async def demo_ttl_expiry():
    q1 = await stock_quote("ACME")           # First call: cache miss — hits the API
    q2 = await stock_quote("ACME")           # Second call: cache hit — cached value
    # test:inject: import time as _t; _saved_time = _t.time; _t.time = lambda: _saved_time() + 7200
    q3 = await stock_quote("ACME")           # cache miss — TTL expired, refetches
    # test:inject: _t.time = _saved_time

asyncio.run(demo_ttl_expiry())
```

The TTL check happens in `_validate_ttl` on the sync hit path before the wrapper returns; expired entries fall through to the recompute branch and the await runs again. Test reference: `test_async_ttl_expires` in `tests/test_core/test_async_ttl.py`.

### Parallel scatter-gather

`asyncio.gather` over a cached async function gives you parallel hits on the same backend with no contention on the read path — each `get` is a sync call that returns immediately on hit:

<!-- test:skip reason="top-level await re-enters an event loop already running asyncio.run(...) earlier on the page" -->
```python
@cash.cache
async def fetch(uid):
    ...

results = await asyncio.gather(*[fetch(uid) for uid in uids])
```

If `uids` contains the same id twice and both calls miss, the default computes both. Construct your instance as `Cash(use_locking=True)` to coalesce them via in-process single-flight (one computes, the other reads the stored result), or pre-deduplicate the input list.

### Cache the orchestrator vs. the leaves

Both shapes work; pick the one whose granularity matches your re-run unit:

```python
# Leaf-level caching: each fetch is cached independently.
@cash.cache
async def fetch_one(uid):
    return await api.get(uid)

async def fetch_many(uids):
    return await asyncio.gather(*[fetch_one(u) for u in uids])

# Orchestrator-level caching: the whole batch is cached as one blob.
@cash.cache
async def fetch_many(uids):
    return await asyncio.gather(*[api.get(u) for u in uids])
```

Leaf-level wins when partial reruns are common (one new `uid` added — only that fetch misses). Orchestrator-level wins when the batch shape is stable and the overhead of N small backend writes is worse than one big one.

## Mixing sync and async caches

<!-- claim: cash/core.py:Cash._compute_cache_key @a3272962 -->
Cache keys are built from `func_name:state_hash:dynamic_hash:args_hash`, and `func_name` is derived from the function's qualified name. A sync `process(x)` and an async `process(x)` defined in the same module have different `__qualname__`s only if you give them different names; if you accidentally name them the same, the second decoration replaces the first in `self.functions[func_name]`. Use distinct names (`process_sync` / `process_async`) when you keep both — there is no automatic disambiguation by sync-vs-async dispatch.

When the function names are distinct (the normal case), sync and async caches coexist with zero risk of cross-collision: the func-name prefix segregates their key spaces entirely.

## Performance notes

- **Cache lookup is sync.** `self.backend.get(cache_key)` runs before any await. A hit returns without touching the event loop beyond the coroutine's normal entry/exit overhead — the wrapper's `await` is satisfied by an already-resolved value.
- **On a miss, the only added cost is serialization.** The underlying `await func(*args, **kwargs)` runs at full speed inside the `FileAccessTracker` block; after the await completes, `_store_in_cache` does a sync write to the backend. This write is synchronous — it blocks the event loop for the duration of the serialize-and-write step. For multi-megabyte payloads on a heavily loaded loop, consider lowering the backend's tier ceilings so the heavy write lands on a background path (see [Smart Persistence](smart-persistence.md)) or moving the write off-loop via `asyncio.to_thread(...)` at the call site.
- **The `_resolve_cache_key` fast path is unchanged from sync.** If args are unhashable, the wrapper still ends up calling `func(*args, **kwargs)` from `_resolve_cache_key`; in the async case the wrapper detects that the returned value is a coroutine and awaits it before returning. This means an unhashable-arg miss is fully transparent — your `await` resolves to the actual return value, not to a coroutine.

## Caveats

- **Don't cache async functions that return live connections.** Anything whose validity is tied to the *current* event loop — open sockets, `aiohttp.ClientSession`, database cursors, file handles — will be useless on the next call because the loop that owned it is gone. Cache the *data* extracted from those objects, not the objects themselves.
- **`asyncio.Lock()` held inside the function body is skipped on hits.** The lock acquisition lives inside `func(*args, **kwargs)`; on a cache hit the wrapper never enters the function, so any side-effecting "I hold this lock" semantics don't fire. This is the same behaviour as the sync wrapper with `threading.Lock()`, but it's worth flagging because async-lock patterns (e.g. fair-queue rate limiters) are more common.
- **In-process await dedup needs `Cash(use_locking=True)`.** By default two `asyncio.gather`d calls of the same key both compute on a cold cache. Construct the instance with `Cash(use_locking=True)` for built-in single-flight coalescing (one computes, the rest read the stored result) — it is a constructor option, not a decorator keyword. It is *in-process* only — keyed on the running event loop, it does not dedup across processes. If you need a hand-rolled equivalent (or cross-process semantics), gate at the call site:

    ```python
    _inflight: dict[str, asyncio.Task] = {}
    async def fetch_dedup(uid):
        if uid not in _inflight:
            _inflight[uid] = asyncio.create_task(fetch_user(uid))
        try:
            return await _inflight[uid]
        finally:
            _inflight.pop(uid, None)
    ```

- **Async generators are not cached, period.** Even with `# @cash:persist` or `strict=True`, the warning fires and the bare async generator is returned. If you have an async generator and want chunked caching, refactor it to `async def f(): return (collect_into_sync_gen(...))` — the await materialises whatever you build inside, then Cash's iterator-chunking sees a regular generator on the return value.
- **The `cache_if=` predicate runs synchronously after the await.** It receives the awaited value, not a coroutine. Don't write an `async def` predicate — Cash will coerce its return to bool and you'll cache the coroutine-truthiness (`True`), not your intended check.
- **Stats wrapper drains after the await.** `cache_info()` is updated by `_drain_stats` which runs after the inner wrapper resolves. A cancelled coroutine never reaches the drain, so its hit/miss is not counted — match the same caveat that exists for any cancelled-task pattern.

## API reference

The decorator surface is unchanged between sync and async — the same kwargs work, with the limitations called out above.

| Surface | Effect on async wrappers |
|---|---|
| `@cash.cache` (no args) | Works on any `async def` that doesn't `yield`. Returns the original async generator unwrapped if it does. |
| `ttl=N` | Honored. `_validate_ttl` is shared between wrappers. |
| `depends_on=[...]` | Honored. Static dependency graph is wrapper-agnostic. |
| `dynamic_depends_on=...` | Honored. Resolved by the same sync helper before the await. |
| `file_depends_on=...` | Honored. Augments the `FileDataSource` set the wrapper consults at key-build time. |
| `cache_if=fn` | Honored. Sync predicate; runs on the awaited value. |
| `chunk_max_items`, `chunk_max_bytes` | Honored when the awaited value is a one-shot iterator. |
| `Cash(use_locking=True)` — **constructor, not a decorator kwarg** | **Supported** via in-process single-flight: concurrent same-key awaits coalesce (leader computes, followers read the stored result). In-process only, keyed on the running event loop. Passing it to the decorator (`@cash.cache(use_locking=True)`) raises `TypeError`. See `tests/test_core/test_async_single_flight.py`. |
| `strict=True`, `assume_safe=True` | Honored. Same purity-mode wiring as sync. |
| `cache_info()`, `cache_clear()`, `explain()` | Attached by `_wrap_with_stats`; all three are synchronous even on async wrappers. |
| Async generator (`async def` with `yield`) | **Not supported.** Warns once at decoration time, returns the function unwrapped. |

## Related

- [LLM API Calls](../use-cases/llm-api-calls.md) — the use case that motivates most async-caching work: prompt iteration, TTL tuning, and structured-output validation.
- [Iterator Caching](iterator-caching.md) — the chunking path that activates when an `async def` returns a sync generator.
- [Thread Safety](thread-safety.md) — the sync side of the locking story; useful background for the `use_locking` single-flight described above.
- [Decorator (`@cash.cache`)](../../decorator.md) — full kwarg reference for the decorator surface listed in the table above.
