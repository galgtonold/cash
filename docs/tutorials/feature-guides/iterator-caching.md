# Iterators

!!! info "Applies to: decorator"
    Code that caches functions returning generators, `map`/`filter` objects or
    other one-pass iterators.

A generator can be read only once, so storing the object itself would hand the
next caller an empty iterator. cash stores the **items** instead, in chunks, and
gives each later call a fresh iterator over them. Nothing needs switching on.

```python
# test:inject: open("data.txt", "w").write("foo\nbar\nbaz\n")
import cash

@cash.cache
def records(path):
    with open(path) as f:
        for line in f:
            yield line.strip()

list(records("data.txt"))   # first call: reads, stores the items
list(records("data.txt"))   # cache hit: replays the stored items
```

<!-- claim: cash/decorator/iterators.py:is_one_shot_iterator @0ce938f7 -->
cash treats a result as an iterator when `iter(result) is result`: generators,
generator expressions, `map`, `filter`, `zip`, most of `itertools`, and your own
iterator classes. Lists, tuples, dicts and ranges can be read many times and
are cached as ordinary values.

## The first call streams

<!-- claim: cash/decorator/store.py:ResultStore.stream_and_store @0e824f20 broad="the loop, the tracker scope and the commit rule are one mechanism" -->
On a miss you get each item as the function produces it, so caching does not
delay the first item. cash copies the items into chunks as they pass and
stores the result once the generator is exhausted. Files the generator reads
while it runs are tracked, as in any cached function.

**A generator you don't finish caches nothing.** If you stop early, or the
generator raises, there is no complete result, so nothing is stored and the
next call runs the function again. An infinite generator never finishes, so it
never caches: put the limit (`itertools.islice`) inside the cached function.

**Two streams of one call store separately.** When two callers miss the same
call before either finishes (two threads, two requests), each stream writes
its own chunks. The one that finishes last is what later calls replay, whole;
never a mix of the two runs. A stream that stops early removes only its own
chunks.

## Later calls replay

<!-- claim: cash/decorator/runtime.py:CallRunner._wrap_iterator_hit @ea103db2, cash/decorator/iterators.py:ChunkedCachedIterator @9273f8c8 broad="the claim is about the whole replay wrapper" -->
Each hit returns a new, independent iterator over the stored chunks. Chunks are
loaded one at a time as you reach them, so memory stays bounded by the chunk
size, and stopping early never reads the rest:

```python
import cash

@cash.cache(chunk_max_items=10)
def stream():
    yield from range(100)          # 10 chunks of 10 items

list(stream())                     # first call: computed and stored

a = stream()                       # cached: a replay iterator
b = stream()                       # cached, independent of a
next(a), next(a), next(b)          # (0, 1, 0): only chunk 0 read
```

The replay supports `iter()`, `next()` and `close()`. Generator methods
`.send()` and `.throw()` raise `AttributeError`, because a replay is not a
running generator.

<!-- claim: cash/decorator/iterators.py:ChunkedCachedIterator.__next__ @64b46718, cash/decorator/runtime.py:CallRunner._chunks_are_intact @898490ec -->
**A missing chunk is recomputed, never skipped.** A stored result is only
served when all its chunks are present; otherwise the call runs the function
again.

If a chunk disappears while you are reading (another process cleared
the cache, the RAM tier evicted it), the rest of the run comes from the
function, starting where the replay stopped. Where cash cannot recompute (a hit
inside an async function), the loss raises. You never get a shortened result.

## Chunk size

<!-- claim: cash/decorator/cached_function.py:CHUNK_MAX_ITEMS == 1_000_000, cash/decorator/cached_function.py:CHUNK_MAX_BYTES == 1_000_000_000 -->
| Parameter | Default | A chunk closes after |
|---|---|---|
| `chunk_max_items=` | `1_000_000` | this many items |
| `chunk_max_bytes=` | `1_000_000_000` | this many bytes (estimated) |

Whichever limit is reached first closes the chunk. With the defaults most
results fit in one chunk.

Lower `chunk_max_items` if callers often stop after a
few items, so a hit reads less. Lower `chunk_max_bytes` for large items such as
arrays or frames. One item bigger than `chunk_max_bytes` becomes a chunk of its
own; it is never split. To opt out of chunking, return a list.

## Limits

- **`cache_if=` sees the result only if it fits one chunk.** For a larger
  result the predicate cannot run without loading every chunk back, so the
  result is stored anyway, with a warning
  ([`CACHE-IF-BYPASSED`](../../warnings.md#cache-if-bypassed)). Raise the
  chunk limits if you need the predicate.
- <!-- claim: cash/decorator/store.py:ResultStore._write_one_chunk @744aeb1e -->
  **A chunk that fails to store** warns
  ([`STORE-CHUNK-FAILED`](../../warnings.md#store-chunk-failed)). The result is
  then incomplete, so the next call recomputes it rather than serving part of
  it. Items that can't be pickled cause this.
- **Chunks share the entry's `ttl`** and are removed with it by
  `cash clear --expired`, `cash.cleanup()` and `f.cache_clear()`.
- **Keep chunks on disk for other processes.** The default stack writes every
  chunk to disk, however quickly the generator ran, as it does any decorated
  result, so the next process replays it. With a RAM-only backend, an evicted
  chunk means a recompute.
- **`frozen=True` has no effect on an iterator result.** A cached function that
  receives one hashes its contents. Return a list if you need `frozen=`.
- **Side-effect checks apply as usual.** A generator that reads the clock
  (`time.time()`, `datetime.now()`) inside the loop gets the same warning as any
  cached function, and `assume_safe=`, `with cash.assume_safe():` and
  `strict=` work unchanged.
- **Async generators are not cached**; see [Async functions](async-caching.md).
  A coroutine that *returns* a sync generator is cached like any iterator.

## Related

- [The `@cash.cache` guide](../../decorator.md): parameters, including
  `chunk_max_items=` and `chunk_max_bytes=`.
- [Async functions](async-caching.md): async generators are not cached.
- [Class methods](caching-class-methods.md): a method can return an iterator
  too.
