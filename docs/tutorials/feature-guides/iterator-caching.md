# Iterator caching — chunked storage for generators and one-shot iterables

Cash can cache functions that return generators, `map()` / `filter()` objects, or any single-pass iterator. It materializes the iterator into chunks on disk, then hands out cached chunks on subsequent calls — bounded memory, full replay semantics.

**On a miss the items stream through as they are produced.** Caching a
generator does not change when the caller sees things: cash tees each item to
you and into the chunk buffer at the same time, and commits once the producer
is exhausted. Measured on a token stream, first item at 515 ms cached against
500 ms uncached; before this it was 2444 ms — the whole completion arriving at
once after the full latency.

The cost is at the other end: **a generator you abandon caches nothing.** Cash
only produces what you consume, so stopping after three of twenty-five items
leaves no complete result to store, and the next call recomputes. Storing the
three under the full result's key would be a wrong answer rather than a slow
one.

## Why generators are tricky

A generator is one-shot. The object you'd cache after the first call is already exhausted — handing the same object back on call two would yield nothing. You have two paths out of this:

1. **Materialize to a list.** Cheap to retrieve, but the entire result has to fit in RAM both at cache-write time and at every cache-read. A 50 GB log scanner that streams comfortably becomes a 50 GB allocation.
2. **Materialize to chunks.** Stream into bounded buffers, write each one as a separate cache entry, and replay them lazily. Memory stays bounded by chunk size on both ends. Cash takes this path.

<!-- claim: cash/core.py:_is_one_shot_iterator @d99fbe0d -->
Detection happens in `_is_one_shot_iterator`, which returns True when `iter(value) is value`. Generators, generator expressions, `map`/`filter`/`zip` results, and any custom class whose `__iter__` returns `self` all qualify. Plain collections (`list`, `dict`, `set`, `tuple`, `range`, `str`) fail that test and are cached as ordinary blobs — they are already reusable.

## Quick start

```python
# test:inject: def parse(line): return line.strip()
# test:inject: open("data.txt", "w").write("foo\nbar\nbaz\n")
import cash

@cash.cache
def stream_records(path):
    with open(path) as f:
        for line in f:
            yield parse(line)

for rec in stream_records("data.txt"):   # First call: streams + caches in chunks
    ...
for rec in stream_records("data.txt"):   # Second call: replays from cached chunks
    ...
```

No decorator option to enable — iterator detection is automatic. Pre-existing `@cash.cache` decorations on generator functions pick up chunked storage without code changes.

## How chunking works

<!-- claim: cash/core.py:Cash._stream_and_store @08822547 broad="the loop, the tracker scope and the commit rule are one mechanism", cash/notebook/object_hashing.py:estimate_object_size @9aeb760a -->
The write path lives in `Cash._stream_and_store`. The loop is:

1. Pull one item from the user's iterator, with the `FileAccessTracker` live — so a file the generator reads lazily is recorded as a dependency. It is entered once and suspended around each yield, so the caller's own reads in its loop body are not attributed to the generator.
2. Track running byte size via `estimate_object_size` from `cash.notebook.object_hashing`.
3. When `len(buffer) >= chunk_max_items` **or** `buffer_bytes >= chunk_max_bytes`, flush the buffer to the backend under key `f"{cache_key}:chunk_{i}"`, increment `i`, and reset the buffer.
4. **Yield the item to the caller**, then go back to 1.
5. When the source iterator stops, flush any tail buffer, then commit the manifest.

Time is accumulated only across step 1, never wall-clock, so a slow consumer cannot inflate the number the persistence decision reads.

If the caller abandons the iterator, or the producer raises, step 5 never runs: the chunks written so far are unreferenced and are dropped on the way out, and nothing is stored.

On exhaust, Cash writes a **manifest entry** at the canonical `cache_key` carrying `iterator_storage="chunked"`, `n_chunks`, and `total_items`. The manifest is what the hit path reads first.

<!-- claim: cash/core.py:Cash.cache @8a01c3c8 broad="the defaults are keyword arguments of the decorator itself" -->
Defaults:

- `chunk_max_items = 1_000_000`
- `chunk_max_bytes = 1_000_000_000` (1 GB)

For most workloads both ceilings are high enough that the entire result lands in a single chunk; storage is then indistinguishable from caching a list.

For a function `stream(n)` with cache key `K = "stream:abc123…"` yielding 25 items at `chunk_max_items=10`, the backend ends up holding four entries:

| Key | Value | Metadata excerpt |
|---|---|---|
| `stream:abc123…` | `{"n_chunks": 3, "total_items": 25}` | `iterator_storage='chunked', ttl, args_hash, state_hash` |
| `stream:abc123…:chunk_0` | `[item_0, …, item_9]` | `ttl, timestamp` |
| `stream:abc123…:chunk_1` | `[item_10, …, item_19]` | `ttl, timestamp` |
| `stream:abc123…:chunk_2` | `[item_20, …, item_24]` | `ttl, timestamp` |

The manifest is always written *last*, so a process killed mid-write cannot produce a manifest pointing at incomplete chunks — the next call sees a clean miss.

### Single-chunk fast path

`_stream_and_store` recognises the common case where the iterator exhausts before any threshold is crossed. Nothing is written while the buffer is still filling, so at exhaustion the whole result is in hand and `cache_if` can be applied to it before anything is committed. It then either writes one chunk plus the manifest, or skips storage entirely if the predicate rejected the result. The caller has already received every item either way — the predicate gates *storage*, never what was yielded.

### Multi-chunk path

When a threshold trips, chunks are written as they fill. A subsequent `cache_if=` predicate cannot be applied — running it would require materializing all chunks back into memory, undoing the bound. Cash emits a one-shot `CashCacheIneffectiveWarning` at the moment of the second-chunk transition (and again at tail-flush) and stores the result without consulting the predicate.

The warning text reads:

> `[CACHE-IF-BYPASSED] @cash.cache on {func_name}: the result exceeded a single chunk (chunk_max_items=…, chunk_max_bytes=…), so it was cached without cache_if ever being consulted.`
> `  Fix: raise chunk_max_items / chunk_max_bytes above the size this result reaches, or return a list instead of an iterator, so the whole result arrives in one piece for the predicate to see.`

The warning is keyed per function via `_warn_once` and fires once per process — repeated multi-chunk calls don't spam the log.

## What types are supported

Anything that satisfies `iter(value) is value`:

- Generator functions (`def f(): yield ...`)
- Generator expressions (`(x*2 for x in xs)`)
- `map()`, `filter()`, `zip()`, `enumerate()` return values
- Custom classes whose `__iter__` returns `self` (the standard iterator-class shape)
- `itertools` chain/islice/dropwhile/... — anything from `itertools` that yields a one-shot view

An empty iterator — a generator that returns before its first yield — is handled too: Cash writes a zero-chunk manifest, and a subsequent call returns an empty iterator without recomputing. Test reference: `test_empty_generator_chunked` in `tests/test_core/test_iterator_caching.py`.

What is **not** supported:

- **Async generators** (`async def gen(): yield ...`). The decorator detects them and returns the function unwrapped with a `CashCacheIneffectiveWarning`. See [Async Caching](async-caching.md) for the patterns that work with async code. An ordinary `async def` function that **returns** (rather than `yield`s) a sync iterator does chunk-store normally — the async wrapper streams it through the same `_stream_and_store` path.
- **Infinite iterators** (`itertools.count()`, `while True: yield ...`). These now *iterate* fine — items stream through as they are produced — but they never finish, so nothing is ever stored and every call recomputes. If you have a streaming computation that's morally infinite but bounded by the caller (`itertools.islice` at the call site), apply the bound *inside* the cached function so the cached iterator is finite and can actually be cached.
- **Iterators whose items don't pickle.** A chunk write that hits `pickle.PicklingError` raises a `CashCacheStoreFailedWarning` and the partially-written chunks are left orphaned (the manifest is never committed, so the next call sees a miss; orphans are reclaimed by `cleanup()`). Items that pickle slowly (large numpy arrays, custom classes without `__reduce__`) work but inflate write time linearly with chunk size.

## Replay semantics

<!-- claim: cash/core.py:Cash._wrap_iterator_hit @82a2a1ae, cash/core.py:_StreamingCachedIterator @435e32b0 broad="the claim is about the whole replay wrapper", cash/core.py:_ChunkedCachedIterator @d808794a broad="the claim is about the whole replay wrapper" -->
On a cache hit, the dispatch at `Cash._wrap_iterator_hit` reads `metadata['iterator_storage']` and returns a **fresh** `_ChunkedCachedIterator(cash, cache_key, n_chunks)` — a lazy iterator that fetches one chunk at a time. That is *every* iterator hit, single-chunk included: a one-chunk result is still stored as a manifest plus one chunk entry, so it replays through the same path.

`_StreamingCachedIterator` is the other half, and it belongs to the **first** call rather than to a hit. It wraps `_stream_and_store`, so a miss hands you the producer's own items at the producer's own pace while the chunks fill behind you — there is nothing to read back out of the backend, because the result does not exist yet:

| | first call (miss) | any later call (hit) |
|---|---|---|
| stayed within one chunk | `_StreamingCachedIterator` | `_ChunkedCachedIterator` |
| crossed a threshold | `_StreamingCachedIterator` | `_ChunkedCachedIterator` |

Two consequences:

- **Each call gets its own iterator.** Calling `f(x)` twice produces two independent iterators that share the cached chunks but advance independently. Test reference: `test_generator_returns_independent_iterators` in `tests/test_core/test_iterator_caching.py`.
- **Chunks are loaded lazily.** `_ChunkedCachedIterator.__next__` only calls `self._cash.backend.get(chunk_key)` when iteration crosses into a new chunk. If the caller breaks out early, unreached chunks are never read from disk.

A short demonstration of both properties:

```python
import cash

@cash.cache(chunk_max_items=10)
def stream():
    yield from range(100)         # 10 chunks of 10 items each

list(stream())                    # first call: computed, and now cached

it_a = stream()                   # cached — a replay iterator
it_b = stream()                   # cached, and independent of it_a
next(it_a)                        # 0 — reads chunk_0 from backend
next(it_a)                        # 1 — chunk_0 still in memory
next(it_b)                        # 0 — re-reads chunk_0 (independent iterator)

for _ in range(8): next(it_a)     # finishes chunk_0
next(it_a)                        # 10 — boundary: reads chunk_1

# Stop here. chunk_2..chunk_9 were never fetched from disk.
```

The first line matters: replay iterators only exist once there is something to
replay. Drop it and all three calls are misses, each producing its own stream
and caching nothing, because none of them is ever exhausted.

The returned object satisfies the iterator protocol — `iter(x) is x`, `__next__`, `close()`. Generator-specific methods (`.send`, `.throw`) raise `AttributeError` with a message pointing at caching as the cause: the cached iterator replays a list, it is not a coroutine.

## Chunk eviction

<!-- claim: cash/core.py:_ChunkedCachedIterator.__next__ @01f2db53 -->
`_ChunkedCachedIterator` is robust to mid-iteration chunk loss. If `backend.get(chunk_key)` returns `(None, None)` — e.g. an L1-only backend evicted that chunk under memory pressure, or `cleanup()` ran between iterations — `__next__` raises `StopIteration` instead of propagating an error. Iteration terminates at the last contiguous run of available data.

The next call to the decorated function will see a miss on the manifest key (manifests live in the same backend tier as chunks, so they're evicted together in typical configurations) and recompute. Test reference: `test_chunked_iterator_missing_chunk_terminates_safely` in `tests/test_core/test_iterator_caching.py`.

<!-- claim: cash/core.py:Cash._write_one_chunk @b1966607 -->
TTL is honored uniformly: each chunk inherits the manifest's TTL (`Cash._write_one_chunk` propagates it), so `Cash.cleanup()` reclaims expired chunks alongside the expired manifest. Test reference: `test_chunked_chunks_inherit_manifest_ttl` in `tests/test_core/test_iterator_caching.py`.

## Persistence and backend tiers

Chunks are stored under derived keys (`{cache_key}:chunk_0`, `:chunk_1`, …) but they are *ordinary backend entries*. Anything the backend does — gzip compression, file-tier persistence, the L1/L2 split in a `TieredBackend` — applies to each chunk independently.

Two practical implications:

- **Across-process replay works out of the box.** Pointing a fresh `Cash` instance at the same `cache_dir` recovers iterators just like scalar returns. Test reference: `test_chunked_persists_across_instances` in `tests/test_core/test_iterator_caching.py`.
- **L1-only entries can be partial after eviction.** A `TieredBackend` with a small RAM ceiling may evict chunk_5 of a 20-chunk result while keeping chunks 0–4 and 6–19 plus the manifest. The next iteration terminates at chunk_4 (the missing-chunk path described above), and the next call recomputes from scratch. If you want stable cross-process iterators, configure a file-tier backend so the chunks land on disk.

The hit path's metadata flag `iterator_storage='chunked'` is what distinguishes a chunked manifest from a scalar list return that happens to live under a similar key. Anything without that flag is returned as-is, which is the right thing for non-iterator return types.

## Configuration

Two keyword-only decorator parameters control chunk boundaries:

```python
import cash

@cash.cache(chunk_max_items=10_000, chunk_max_bytes=100_000_000)
def stream_records(path):
    # Yield records one-by-one; Cash chunks them automatically.
    return iter(range(100))

list(stream_records("events.jsonl"))   # first call — chunked as configured
list(stream_records("events.jsonl"))   # cache hit — replayed from the chunks
```

| Knob | Default | Effect |
|---|---|---|
| `chunk_max_items` | `1_000_000` | Close the current chunk after this many items. |
| `chunk_max_bytes` | `1_000_000_000` | Close the current chunk after this many bytes (estimated via `estimate_object_size`). |

A chunk closes when **either** threshold is hit, whichever comes first. There is no global toggle for the chunking system — it is the only path for one-shot iterators. To opt out, return a list (`return list(gen())`) and Cash will cache the list as an ordinary blob.

Tuning notes:

- **Lower `chunk_max_items`** for fine-grained partial-consumption skips. If your callers frequently break out after the first dozen items, a smaller chunk means less I/O on the hit path. A chunk of 100 items is roughly free to read; a chunk of a million items requires deserializing all million before yielding the first one.
- **Lower `chunk_max_bytes`** when individual items are large (DataFrame rows, embedding tensors) and the default 1 GB ceiling would buffer too much before the first flush. A common pattern for 1–10 MB items is `chunk_max_bytes=100_000_000` — ten items per chunk gives the file backend something to deduplicate around while keeping in-memory buffers manageable.
- **Single items above `chunk_max_bytes`** are not split — a 2 GB tensor yielded with `chunk_max_bytes=1_000_000_000` is stored as its own chunk. The threshold is checked *after* an item is appended, so it acts as a "close-after-this-item" trigger rather than a hard cap on serialized size. Test reference: `test_item_larger_than_chunk_max_bytes` in `tests/test_core/test_iterator_caching.py`.
- **Raise both** if your iterator is short and you want the single-chunk fast path. The fast path skips the chunk-key indirection on read; it's the natural choice when the result reliably fits in memory and the bytes budget is comfortable.

## Caveats

- **Partial consumption on a miss caches nothing.** The miss path produces only what the caller consumes, so stopping after ten of a thousand items leaves no complete result to store and the next call recomputes. Storing the ten under the full result's key would be a wrong answer rather than a slow one. On a *hit* partial consumption is free — only the chunks the caller reaches are loaded. Test reference: `test_chunked_iterator_partial_consumption_caches_nothing` in `tests/test_core/test_iterator_caching.py`.
- **`cache_if` is bypassed on multi-chunk results.** As described above, the predicate cannot run without re-materializing chunks. The bypass warning is keyed per-function and fires once per process. To keep `cache_if` gating in effect, **raise** the thresholds until the whole result fits one chunk, so the single-chunk path stays in play — lowering them produces *more* chunks and so guarantees the bypass.
<!-- claim: cash/core.py:Cash._attach_lineage @8145e913 -->
- **No `_cash_lineage_hash` on iterator returns.** Cash's lineage-tracking optimization attaches a `_cash_lineage_hash` attribute to non-iterator return values so downstream `@cash.cache` calls can short-circuit the args hash (via `Cash._attach_lineage`). Iterator wrappers don't get this attribute — passing a cached iterator to another `@cash.cache` function will re-hash its contents the normal way. Materialize to a list if you want the lineage short-circuit.
- **Purity analysis treats generator bodies like any function.** A generator that calls `time.time()`, mutates module-level state, or reads `os.environ` inside the loop still triggers `CashImpurityWarning`. Apply `@cash.pure` / `assume_safe` / `strict` exactly as you would for a non-generator function. See [Purity Decorators](purity-decorators.md).
- **Backend store failures are per-chunk.** If a chunk write raises, the wrapper emits `CashCacheStoreFailedWarning` and continues to the next chunk; the manifest is still written at the end. A later read that hits the missing chunk terminates iteration early (the same path as eviction). To detect this, watch for the warning rather than relying on the iterator length matching `total_items`.
- **`use_locking=True` is supported.** The locked hit path dispatches through `_wrap_iterator_hit` like the unlocked path, so a re-read after lock acquisition correctly returns a fresh iterator wrapper rather than the raw manifest dict. Test reference: `test_use_locking_dispatches_chunked_on_locked_hit` in `tests/test_core/test_iterator_caching.py`.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `chunk_max_items=N` | `@cash.cache` kwarg | Close current chunk after `N` items. Default `1_000_000`. |
| `chunk_max_bytes=N` | `@cash.cache` kwarg | Close current chunk after `N` bytes (estimated). Default `1_000_000_000`. |
| `_StreamingCachedIterator` | Internal | **First-call** wrapper. Passes the producer's items through as they arrive and fills the chunk buffers behind them; commits only on exhaustion; iter-self; supports `close()`, which abandons the entry. |
| `_ChunkedCachedIterator` | Internal | Lazy replay, and the wrapper for **every** cache hit. Fetches one chunk at a time; iter-self; terminates cleanly on missing chunks; supports `close()`. |
| `metadata['iterator_storage'] = 'chunked'` | Backend metadata | Flag the hit path reads to choose `_ChunkedCachedIterator` over a single-blob return. |
| `f"{cache_key}:chunk_{i}"` | Backend key format | Chunk keys are derived from the manifest key by suffix. `cleanup()` reclaims them alongside the manifest. |
| `CashCacheIneffectiveWarning` | Warning | Fires once on the first multi-chunk transition when `cache_if=` is set; also fires once if an async-generator function is decorated. |
| `CashCacheStoreFailedWarning` | Warning | Fires when a backend write raises during chunk or manifest storage. |

## Related

- [Async Caching](async-caching.md) — patterns for async functions; async generators are unsupported, but async functions that *return* a sync iterator chunk-store normally.
- [Caching Class Methods](caching-class-methods.md) — bound methods that return iterators inherit chunked storage with no extra work; the page covers `register_hasher` for the `self` argument.
- [Smart Persistence](smart-persistence.md) — how the backend tiers decide where chunks live (RAM vs disk) and what eviction means in practice.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — `cache_if=`, `ttl=`, and the interaction between predicates and the chunked path.
- [Decorator (`@cash.cache`)](../../decorator.md) — full kwarg reference for `chunk_max_items`, `chunk_max_bytes`, and every other knob.
