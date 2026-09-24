# Class methods

!!! info "Applies to: decorator"
    Code that puts `@cash.cache` on methods.

`@cash.cache` works on methods. `self` is part of the key like any other
argument, hashed by its state, so two instances with equal attributes share
entries. That default goes wrong in two ways:

- **`self` can't be pickled.** An instance holding a connection, a thread or a
  lock gives cash nothing to hash. Every call runs uncached, with a warning
  ([`KEY-UNHASHABLE-ARG`](../../warnings.md#key-unhashable-arg)).
- **`self` carries state that doesn't matter.** A lazy attribute, a memo dict or
  a large `self.df` changes the key, or makes it slow to build, while the
  instance means the same thing.

The fix for both is to tell cash what identifies an instance.

## Register a hasher for the class

<!-- claim: cash/core.py:Cash.register_hasher @f48a324b, cash/decorator/arg_hashing.py:ArgHasher.hash_payload @c4f48efb -->
```python
import hashlib
from cash import Cash

app = Cash()

class Loader:
    def __init__(self, dataset_id, db):
        self.dataset_id = dataset_id
        self.db = db                    # a connection: not part of the identity

    @app.cache
    def load(self, version):
        return self.db.query(self.dataset_id, version)

app.register_hasher(
    Loader,
    lambda loader: hashlib.sha256(loader.dataset_id.encode()).hexdigest(),
)
```

Register it before the first call. The hasher then applies wherever a `Loader`
is an argument of a cached function, as `self` or not. The method's other
arguments are hashed as usual.

The hasher must name **everything** that changes the result. Here two loaders
with the same `dataset_id` but different databases share entries. That is
right only if both databases hold the same data. See
[Custom hashers](custom-hashers.md#what-makes-a-good-hasher).

For a service object with no identity of its own, a constant hasher drops
`self` from the key:

```python
app.register_hasher(MyService, lambda _: "singleton")
```

Every instance then shares entries, which is correct only when instances are
interchangeable.

`__hash__` doesn't help: Python's `hash()` is 64 bits and meant for dict
buckets, too weak for a cache key. Use `register_hasher`.

## Methods that return iterators

A method that yields is cached like any iterator; `self` goes into the key
through the same hasher. See [Iterators](iterator-caching.md).

## Related

- [Custom hashers](custom-hashers.md)
- [The `@cash.cache` guide](../../decorator.md#methods-and-self)
