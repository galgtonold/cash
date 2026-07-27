# Caching class methods

`@cash.cache` on a bound method puts `self` into the cache key like
any other argument. Two failure modes follow:

1. **`self` is unpicklable.** Instances holding file handles, database
   connections, threads, or locks fail the pickle path silently — cash
   can't build a key and your method recomputes on every call. Cash
   emits a `CashCacheIneffectiveWarning` on the first such call, naming
   the type that could not be pickled.
2. **`self` is picklable but holds incidental state.** Lazy attributes,
   a `__cache__` dict, or a heavy `self.df` change the pickle bytes
   even when the *logical* identity is unchanged. Two equivalent
   instances miss each other; pickling 200 MB of `self.df` costs you
   on every call.

The fix is the same in both cases: tell cash how to summarise your
type into a cache key.

<!-- claim: cash/core.py:Cash.register_hasher @2cc59e2a, cash/core.py:Cash._hash_arg_payload @a311dda3 -->
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

<!-- claim: cash/core.py:Cash._hash_arg_payload @a311dda3 -->
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

## Methods that return iterators

A bound method that yields values or returns a `map`/`filter`/generator
expression is cached using Cash's chunked-iterator storage — same
mechanics as for any plain function, with `self` simply contributing
to the cache key through the registered hasher (or pickle fallback).

```python
class Loader:
    @c.cache
    def stream_rows(self, table):
        for row in self._db.query(f"SELECT * FROM {table}"):
            yield row
```

Two class-method-specific notes:

- **`register_hasher(Loader, ...)` applies to the iterator-method path
  exactly as it does to a scalar-returning method.** The hasher's
  output digests `self`; chunk keys are then derived from the same
  cache key as any other call.
- **The hasher must be cheap on every call.** Cash exhausts the
  underlying generator on the first call to populate chunks, but the
  *hasher* runs on every subsequent hit as part of cache-key
  resolution. If your hasher reads expensive fields off `self`,
  iterator hits inherit that cost. See [Custom Hashers](custom-hashers.md)
  for the cost model.

For the storage mechanics — chunk size knobs, `cache_if` bypass on
multi-chunk results, replay semantics, eviction behavior — see
[Iterator Caching](iterator-caching.md).

## Related

- [Custom Hashers](custom-hashers.md) — register hashers for any type, not just bound methods.
- [Purity Decorators](purity-decorators.md) — control what Cash treats as deterministic.
- [Iterator Caching](iterator-caching.md) — the chunked storage system that handles generator returns.
- [Decorator (`@cash.cache`)](../../decorator.md) — the underlying decorator API.
