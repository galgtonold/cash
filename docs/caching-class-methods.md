# Caching class methods

`@cash.cache` on a bound method puts `self` into the cache key like
any other argument. Two failure modes follow:

1. **`self` is unpicklable.** Instances holding file handles, database
   connections, threads, or locks fail the pickle path silently — cash
   can't build a key and your method recomputes on every call. Since
   v0.5.0b2 you'll see a `CashCacheIneffectiveWarning` on the first
   such call telling you which type was unpicklable.
2. **`self` is picklable but holds incidental state.** Lazy attributes,
   a `__cache__` dict, or a heavy `self.df` change the pickle bytes
   even when the *logical* identity is unchanged. Two equivalent
   instances miss each other; pickling 200 MB of `self.df` costs you
   on every call.

The fix is the same in both cases: tell cash how to summarise your
type into a cache key.

## Register a type-level hasher

```python
import cash
import hashlib

c = cash.Cash()

class Loader:
    def __init__(self, dataset_id, db_conn):
        self.dataset_id = dataset_id
        self._db = db_conn  # unpicklable; not part of identity

    @c.cache
    def load(self, version):
        return self._db.query(self.dataset_id, version)

c.register_hasher(
    Loader,
    lambda self: hashlib.sha256(self.dataset_id.encode()).hexdigest(),
)
```

Order matters: register the hasher after `Loader` is defined but before
you call any `@c.cache` method on an instance.

`register_hasher` is set once per type. Every `@c.cache` method that
takes a `Loader` instance — whether as `self` or as a regular
argument — automatically uses the hasher.

## Singleton-service shortcut

For a stateless service object — a wrapper around an external
resource with no logical identity of its own — use a constant hasher:

```python
c.register_hasher(MyService, lambda _: "singleton")
```

This effectively drops `self` from the cache key. Use with care: two
different `MyService()` instances now share cache entries, which is
only correct when the instances are truly interchangeable.

## What `__hash__` won't do

Defining `__hash__` on your class doesn't help cash. Cash uses a
256-bit composite cache key (SHA-256). Python's built-in `hash()` is
a 64-bit value designed for hash-table bucketing, not for collision
resistance at cache-key scale. Mixing the two would risk silent
wrong-result bugs when collisions occur. `register_hasher` returns
a SHA-256 hex digest by convention and is the supported path for
custom key derivation.

## Composing with other args

The hasher is applied to *any* arg of the registered type — `self`
included. Other args go through the standard path (built-in pickle
or another `register_hasher` entry). So a method on a `Loader`
taking a pandas `DataFrame` benefits from both your `Loader` hasher
and cash's built-in pandas hasher with no extra work.

## Caching iterator-returning functions

`@c.cache` on a function that returns a one-shot iterator (a generator,
`map`/`filter` result, or custom iterator) stores the iterator's values
in chunks. Small iterators land in a single chunk and behave like a
materialized list; large iterators split across multiple backend keys.
On a cache hit, chunks are fetched lazily as the user iterates — RAM is
bounded by chunk size, not by the total result size.

```python
@c.cache
def stream_rows():
    for row in db.query("SELECT * FROM events"):
        yield row

for row in stream_rows():
    process(row)
```

Two keyword-only parameters control the chunk boundaries:

- `chunk_max_items` (default `1_000_000`) — close a chunk after this
  many items.
- `chunk_max_bytes` (default `1_000_000_000`) — close a chunk after this
  many bytes (estimated via the cash size estimator).

A chunk closes when either threshold is reached. For most workloads
the defaults are large enough that small iterators stay in a single
chunk; large iterators auto-chunk without any opt-in.

```python
@c.cache(chunk_max_items=10_000)
def smaller_chunks():
    yield from db.query(...)
```

### Iterator protocol

The object returned to your code preserves the iterator protocol
(`iter(x) is x`, `__next__`, `close`). Generator-specific methods
(`.send`, `.throw`) are not supported on the cached wrapper — a
cached iterator replays stored values; it is not a coroutine. Calling
those methods raises `AttributeError`.

### `cache_if` with chunked storage

When `cache_if` is provided alongside a chunked-iterator return:

- **Single-chunk result:** the predicate runs on the materialized
  chunk and decides whether to cache, exactly as for non-iterator
  return values.
- **Multi-chunk result:** the predicate cannot run without
  materializing the full result (defeating chunking). At the moment
  the second chunk is created, cash emits a one-shot
  `CashCacheIneffectiveWarning` and proceeds to cache the result
  without consulting the predicate. The warning is keyed per function
  and fires once per process.

To keep `cache_if` gating in effect on large iterators, either lower
`chunk_max_items` / `chunk_max_bytes` so single-chunk is the common
case, or materialize the iterator manually (`return list(generator)`)
before returning.

### Async generators are not supported

A function defined as `async def gen(): yield ...` (Python's
async-generator protocol) is returned unwrapped and emits a
`CashCacheIneffectiveWarning`. The chunked-storage approach above
applies to sync iterators and to async functions that *return*
(rather than `yield`) a sync iterator.

### Trade-offs

The chunked path eagerly exhausts the underlying generator on the
first call, even if the user only consumes the first few items. If
your generator is infinite (`while True: yield ...`) or
lazy-by-design (a search that the caller stops at the first match),
do not decorate it with `@c.cache` — your process will not return
from the first call. Per-call resume semantics are a planned future
extension.

## Related

- [`@cash.cache` API reference](api_reference.md)
- [Production transition tutorial](tutorials/production_transition.md)
