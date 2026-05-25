# Iterator caching — chunked storage for generators and one-shot iterables

Cash can cache functions that return generators, `map()` / `filter()` objects, or any single-pass iterator. It materializes the iterator into chunks on disk, then hands out cached chunks on subsequent calls — bounded memory, full replay semantics.

## Why generators are tricky

A generator is one-shot. The object you'd cache after the first call is already exhausted — handing the same object back on call two would yield nothing. You have two paths out of this:

1. **Materialize to a list.** Cheap to retrieve, but the entire result has to fit in RAM both at cache-write time and at every cache-read. A 50 GB log scanner that streams comfortably becomes a 50 GB allocation.
2. **Materialize to chunks.** Stream into bounded buffers, write each one as a separate cache entry, and replay them lazily. Memory stays bounded by chunk size on both ends. Cash takes this path.

Detection happens at `src/cash/core.py:283-295` — `_is_one_shot_iterator` returns True when `iter(value) is value`. Generators, generator expressions, `map`/`filter`/`zip` results, and any custom class whose `__iter__` returns `self` all qualify. Plain collections (`list`, `dict`, `set`, `tuple`, `range`, `str`) fail that test and are cached as ordinary blobs — they are already reusable.

## Quick start

```python
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

The write path lives in `_write_chunks` at `src/cash/core.py:2026-2142`. The loop is:

1. Pull items from the user's iterator into an in-memory buffer.
2. Track running byte size via `estimate_object_size` from `cash.notebook.object_hashing`.
3. When `len(buffer) >= chunk_max_items` **or** `buffer_bytes >= chunk_max_bytes`, flush the buffer to the backend under key `f"{cache_key}:chunk_{i}"`, increment `i`, and reset the buffer (`src/cash/core.py:2077-2098`).
4. When the source iterator stops, flush any tail buffer the same way.

On exhaust, Cash writes a **manifest entry** at the canonical `cache_key` carrying `iterator_storage="chunked"`, `n_chunks`, and `total_items` (`src/cash/core.py:2183-2218`). The manifest is what the hit path reads first.

Defaults (`src/cash/core.py:524-525`):

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

The manifest is always written *last* (`src/cash/core.py:1167-1171` for the multi-chunk path), so a process killed mid-write cannot produce a manifest pointing at incomplete chunks — the next call sees a clean miss.

### Single-chunk fast path

`_write_chunks` recognises the common case where the iterator exhausts before any threshold is crossed. It returns the buffered list back to the caller (the `single_chunk_buffer` second element of its tuple, `src/cash/core.py:2104-2110`) without writing anything yet — that lets the caller apply `cache_if` to the materialized result before committing the single chunk. The wrapper at `src/cash/core.py:1131-1163` then either writes one chunk plus the manifest, or skips storage entirely if the predicate rejected the result.

### Multi-chunk path

When a threshold trips, chunks are written as they fill. A subsequent `cache_if=` predicate cannot be applied — running it would require materializing all chunks back into memory, undoing the bound. Cash emits a one-shot `CashCacheIneffectiveWarning` at the moment of the second-chunk transition (`src/cash/core.py:2081-2094` and again at tail-flush `src/cash/core.py:2121-2134`) and stores the result without consulting the predicate.

The warning text reads:

> `@cash.cache on {func_name}: cache_if was bypassed because the result exceeded a single chunk (chunk_max_items=…, chunk_max_bytes=…). The result is cached without consulting the predicate. To keep cache_if gating in effect, lower the chunk thresholds or materialize the iterator manually.`

The warning is keyed per function via `_warn_once` and fires once per process — repeated multi-chunk calls don't spam the log.

## What types are supported

Anything that satisfies `iter(value) is value`:

- Generator functions (`def f(): yield ...`)
- Generator expressions (`(x*2 for x in xs)`)
- `map()`, `filter()`, `zip()`, `enumerate()` return values
- Custom classes whose `__iter__` returns `self` (the standard iterator-class shape)
- `itertools` chain/islice/dropwhile/... — anything from `itertools` that yields a one-shot view

An empty iterator — a generator that returns before its first yield — is handled too: Cash writes a zero-chunk manifest, and a subsequent call returns an empty iterator without recomputing. Test reference: `test_empty_generator_chunked` at `tests/test_core/test_iterator_caching.py:756`.

What is **not** supported:

- **Async generators** (`async def gen(): yield ...`). The decorator detects them at `src/cash/core.py:627-636` and returns the function unwrapped with a `CashCacheIneffectiveWarning`. See [Async Caching](async-caching.md) for the patterns that work with async code. An ordinary `async def` function that **returns** (rather than `yield`s) a sync iterator does chunk-store normally — the async wrapper at `src/cash/core.py:1278-1329` dispatches through the same `_write_chunks` path.
- **Infinite iterators** (`itertools.count()`, `while True: yield ...`). Caching exhausts the iterator on the first call before returning anything. The first call never returns. If you have a streaming computation that's morally infinite but bounded by the caller (`itertools.islice` at the call site), apply the bound *inside* the cached function so the cached iterator is finite.
- **Iterators whose items don't pickle.** A chunk write that hits `pickle.PicklingError` raises a `CashCacheStoreFailedWarning` and the partially-written chunks are left orphaned (the manifest is never committed, so the next call sees a miss; orphans are reclaimed by `cleanup()`). Items that pickle slowly (large numpy arrays, custom classes without `__reduce__`) work but inflate write time linearly with chunk size.

## Replay semantics

On a cache hit, the dispatch at `_wrap_iterator_hit` (`src/cash/core.py:1051-1077`) reads `metadata['iterator_storage']` and returns a **fresh** iterator instance:

- Single-chunk hits return `_ListCachedIterator(items)` — a thin wrapper over `iter(list)` that streams the cached list (`src/cash/core.py:120-162`).
- Multi-chunk hits return `_ChunkedCachedIterator(cash, cache_key, n_chunks)` — a lazy iterator that fetches one chunk at a time (`src/cash/core.py:171-250`).

Two consequences:

- **Each call gets its own iterator.** Calling `f(x)` twice produces two independent iterators that share the cached chunks but advance independently. Test reference: `test_generator_returns_independent_iterators` at `tests/test_core/test_iterator_caching.py:42`.
- **Chunks are loaded lazily.** `_ChunkedCachedIterator.__next__` only calls `self._cash.backend.get(chunk_key)` when iteration crosses into a new chunk (`src/cash/core.py:210-230`). If the caller breaks out early, unreached chunks are never read from disk.

A short demonstration of both properties:

```python
@cash.cache(chunk_max_items=10)
def stream():
    yield from range(100)         # 10 chunks of 10 items each

it_a = stream()
it_b = stream()                   # independent — does not share state with it_a
next(it_a)                        # 0 — reads chunk_0 from backend
next(it_a)                        # 1 — chunk_0 still in memory
next(it_b)                        # 0 — re-reads chunk_0 (independent iterator)

for _ in range(8): next(it_a)     # finishes chunk_0
next(it_a)                        # 10 — boundary: reads chunk_1

# Stop here. chunk_2..chunk_9 were never fetched from disk.
```

The returned object satisfies the iterator protocol — `iter(x) is x`, `__next__`, `close()`. Generator-specific methods (`.send`, `.throw`) raise `AttributeError` with a message pointing at caching as the cause: the cached iterator replays a list, it is not a coroutine.

## Chunk eviction

`_ChunkedCachedIterator` is robust to mid-iteration chunk loss. If `backend.get(chunk_key)` returns `(None, None)` — e.g. an L1-only backend evicted that chunk under memory pressure, or `cleanup()` ran between iterations — `__next__` raises `StopIteration` instead of propagating an error (`src/cash/core.py:225-229`). Iteration terminates at the last contiguous run of available data.

The next call to the decorated function will see a miss on the manifest key (manifests live in the same backend tier as chunks, so they're evicted together in typical configurations) and recompute. Test reference: `test_chunked_iterator_missing_chunk_terminates_safely` at `tests/test_core/test_iterator_caching.py:380`.

TTL is honored uniformly: each chunk inherits the manifest's TTL (`src/cash/core.py:2095, 2135, 2167`), so `Cash.cleanup()` reclaims expired chunks alongside the expired manifest. Test reference: `test_chunked_chunks_inherit_manifest_ttl` at `tests/test_core/test_iterator_caching.py:973`.

## Persistence and backend tiers

Chunks are stored under derived keys (`{cache_key}:chunk_0`, `:chunk_1`, …) but they are *ordinary backend entries*. Anything the backend does — gzip compression, file-tier persistence, the L1/L2 split in a `TieredBackend` — applies to each chunk independently.

Two practical implications:

- **Across-process replay works out of the box.** Pointing a fresh `Cash` instance at the same `cache_dir` recovers iterators just like scalar returns. Test reference: `test_chunked_persists_across_instances` at `tests/test_core/test_iterator_caching.py:835`.
- **L1-only entries can be partial after eviction.** A `TieredBackend` with a small RAM ceiling may evict chunk_5 of a 20-chunk result while keeping chunks 0–4 and 6–19 plus the manifest. The next iteration terminates at chunk_4 (the missing-chunk path described above), and the next call recomputes from scratch. If you want stable cross-process iterators, configure a file-tier backend so the chunks land on disk.

The hit path's metadata flag `iterator_storage='chunked'` is what distinguishes a chunked manifest from a scalar list return that happens to live under a similar key (`src/cash/core.py:1070-1077`). A pre-0.5 entry without that flag is read through the legacy `materialized_iterator=True` path, which loads the full list and wraps it in `_ListCachedIterator` — same external behavior, different on-disk shape.

## Configuration

Two keyword-only decorator parameters control chunk boundaries:

```python
@cash.cache(chunk_max_items=10_000, chunk_max_bytes=100_000_000)
def stream_records(path):
    ...
```

| Knob | Default | Effect |
|---|---|---|
| `chunk_max_items` | `1_000_000` | Close the current chunk after this many items. |
| `chunk_max_bytes` | `1_000_000_000` | Close the current chunk after this many bytes (estimated via `estimate_object_size`). |

A chunk closes when **either** threshold is hit, whichever comes first. There is no global toggle for the chunking system — it is the only path for one-shot iterators. To opt out, return a list (`return list(gen())`) and Cash will cache the list as an ordinary blob.

Tuning notes:

- **Lower `chunk_max_items`** for fine-grained partial-consumption skips. If your callers frequently break out after the first dozen items, a smaller chunk means less I/O on the hit path. A chunk of 100 items is roughly free to read; a chunk of a million items requires deserializing all million before yielding the first one.
- **Lower `chunk_max_bytes`** when individual items are large (DataFrame rows, embedding tensors) and the default 1 GB ceiling would buffer too much before the first flush. A common pattern for 1–10 MB items is `chunk_max_bytes=100_000_000` — ten items per chunk gives the file backend something to deduplicate around while keeping in-memory buffers manageable.
- **Single items above `chunk_max_bytes`** are not split — a 2 GB tensor yielded with `chunk_max_bytes=1_000_000_000` is stored as its own chunk. The threshold is checked *after* an item is appended, so it acts as a "close-after-this-item" trigger rather than a hard cap on serialized size. Test reference: `test_item_larger_than_chunk_max_bytes` at `tests/test_core/test_iterator_caching.py:785`.
- **Raise both** if your iterator is short and you want the single-chunk fast path. The fast path skips the chunk-key indirection on read; it's the natural choice when the result reliably fits in memory and the bytes budget is comfortable.

## Caveats

- **Materialization on the first call.** The cache-miss path drains the user's iterator to completion before returning the replay wrapper (`src/cash/core.py:1124-1128`). If the caller plans to consume only the first ten items, the first call still does the full work. Subsequent calls benefit from the cache; the first one does not. Partial consumption is fully supported on cache hits — only chunks the caller reaches are loaded. Test reference: `test_chunked_iterator_partial_consumption_then_hit` at `tests/test_core/test_iterator_caching.py:806`.
- **`cache_if` is bypassed on multi-chunk results.** As described above, the predicate cannot run without re-materializing chunks. The bypass warning is keyed per-function and fires once per process. To keep `cache_if` gating in effect on large iterators, lower the chunk thresholds enough that the single-chunk path stays in play.
- **No `_cash_lineage_hash` on iterator returns.** Cash's lineage-tracking optimization attaches a `_cash_lineage_hash` attribute to non-iterator return values so downstream `@cash.cache` calls can short-circuit the args hash (`src/cash/core.py:1180, 1332`). Iterator wrappers don't get this attribute — passing a cached iterator to another `@cash.cache` function will re-hash its contents the normal way. Materialize to a list if you want the lineage short-circuit.
- **Purity analysis treats generator bodies like any function.** A generator that calls `time.time()`, mutates module-level state, or reads `os.environ` inside the loop still triggers `CashImpurityWarning`. Apply `@cash.pure` / `assume_safe` / `strict` exactly as you would for a non-generator function. See [Purity Decorators](purity-decorators.md).
- **Backend store failures are per-chunk.** If a chunk write raises, the wrapper emits `CashCacheStoreFailedWarning` and continues to the next chunk; the manifest is still written at the end. A later read that hits the missing chunk terminates iteration early (the same path as eviction). To detect this, watch for the warning rather than relying on the iterator length matching `total_items`.
- **`use_locking=True` is supported.** The locked hit path dispatches through `_wrap_iterator_hit` like the unlocked path, so a re-read after lock acquisition correctly returns a fresh iterator wrapper rather than the raw manifest dict. Test reference: `test_use_locking_dispatches_chunked_on_locked_hit` at `tests/test_core/test_iterator_caching.py:910`.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `chunk_max_items=N` | `@cash.cache` kwarg | Close current chunk after `N` items. Default `1_000_000`. |
| `chunk_max_bytes=N` | `@cash.cache` kwarg | Close current chunk after `N` bytes (estimated). Default `1_000_000_000`. |
| `_ListCachedIterator` | Internal | Single-chunk replay wrapper. Streams a materialized list; iter-self; supports `close()`. (`src/cash/core.py:120`) |
| `_ChunkedCachedIterator` | Internal | Multi-chunk lazy replay. Fetches one chunk at a time; iter-self; terminates cleanly on missing chunks; supports `close()`. (`src/cash/core.py:171`) |
| `metadata['iterator_storage'] = 'chunked'` | Backend metadata | Flag the hit path reads to choose `_ChunkedCachedIterator` over a single-blob return (`src/cash/core.py:1070-1077`). |
| `f"{cache_key}:chunk_{i}"` | Backend key format | Chunk keys are derived from the manifest key by suffix. `cleanup()` reclaims them alongside the manifest. |
| `CashCacheIneffectiveWarning` | Warning | Fires once on the first multi-chunk transition when `cache_if=` is set; also fires once if an async-generator function is decorated. |
| `CashCacheStoreFailedWarning` | Warning | Fires when a backend write raises during chunk or manifest storage. |

## Related

- [Async Caching](async-caching.md) — patterns for async functions; async generators are unsupported, but async functions that *return* a sync iterator chunk-store normally.
- [Caching Class Methods](caching-class-methods.md) — bound methods that return iterators inherit chunked storage with no extra work; the page covers `register_hasher` for the `self` argument.
- [Smart Persistence](smart-persistence.md) — how the backend tiers decide where chunks live (RAM vs disk) and what eviction means in practice.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — `cache_if=`, `ttl=`, and the interaction between predicates and the chunked path.
- [Decorator (`@cash.cache`)](../../decorator.md) — full kwarg reference for `chunk_max_items`, `chunk_max_bytes`, and every other knob.
