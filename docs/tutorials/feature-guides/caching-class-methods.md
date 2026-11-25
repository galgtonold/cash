# Class methods

!!! info "Applies to: decorator"
    Code that puts `@cash.cache` on methods.

`@cash.cache` works on methods. `self` is part of the key like any other
argument, hashed by its state, so two instances with equal attributes share
entries. That default goes wrong in three ways:

- **`self` holds a lot of data.** An instance holding several large frames is
  read in full to build the key on every call, even for a method that returns
  a row count. That can take seconds per call.
- **`self` can't be pickled.** An instance holding a connection, a thread or a
  lock gives cash nothing to hash. Every call runs uncached, with a warning
  that names the attribute
  ([`KEY-UNHASHABLE-ARG`](../../warnings.md#key-unhashable-arg)).
- **`self` carries state that doesn't matter.** A lazy attribute or a memo
  dict changes the key while the instance means the same thing.

The fix for all three is to tell cash what identifies an instance.

## Give the class a `__cash_key__`

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.cash_key_hash @27ffc2c9 -->
```python
from cash import Cash

app = Cash()

class FakeDB:
    def query(self, dataset_id, version):
        return f"{dataset_id}@{version}"

class Loader:
    def __init__(self, dataset_id, db):
        self.dataset_id = dataset_id
        # a connection: not part of the identity
        self.db = db

    def __cash_key__(self):
        return self.dataset_id

    @app.cache
    def load(self, version):
        return self.db.query(self.dataset_id, version)

Loader("sales", FakeDB()).load(1)   # runs
Loader("sales", FakeDB()).load(1)   # a hit: same dataset_id
# test:inject: load = Loader.load  # lets the harness read cache_info
```

`self` is then keyed by what `__cash_key__` returns, and none of the data it
holds is read. The code it holds still counts: editing `FakeDB.query`
recomputes. The method's other arguments are hashed as usual.

The key must name **everything** that changes the result. Here two loaders
with the same `dataset_id` but different databases share entries. That is
right only if both databases hold the same data. When the data behind an id
can change, put a version in the key. cash checks a key against the content it
stands for once per object and warns
[`KEY-STALE-CASH-KEY`](../../warnings.md#key-stale-cash-key) when one key
stands for two contents. See
[Custom hashers](custom-hashers.md#cash-key) for the rules.

For a service object with no identity of its own, a constant key drops `self`
from the key:

```python
class Service:
    def __cash_key__(self):
        return "singleton"
```

Every instance then shares entries, which is correct only when instances are
interchangeable.

`__hash__` doesn't help: Python's `hash()` is 64 bits and meant for dict
buckets, too weak for a cache key, and it is often identity-based.

## A class you don't own

<!-- claim: cash/core.py:Cash.register_hasher @0fc93c81 -->
For a class you can't add a method to, register a hasher for it before the
first call. It applies wherever an instance is an argument of a cached
function, as `self` or not:

<!-- test:skip reason="fragment: names a type the page does not define" -->
```python
app.register_hasher(ThirdPartyClient, lambda c: c.base_url)
```

See [Custom hashers](custom-hashers.md#registering-a-hasher).

## Methods that set attributes on `self`

<!-- claim: cash/decorator/attribute_replay.py:AttrVerdict.of @833f9c5b, cash/rewrite_in_place.py:attribute_refusal @0ad90168 -->
A method that only sets attributes on `self`, as a model's `fit` does, is
cached too. The entry keeps the attributes the call set, and a hit sets them
again on the instance you call it on:

```python
import cash

class Model:
    def __init__(self, k):
        self.k = k

    @cash.cache
    def fit(self, xs):
        self.coef_ = sum(xs) * self.k
        return self

m = Model(2).fit([1, 2, 3])       # runs
m2 = Model(2)
m2.fit([1, 2, 3])                 # a hit: sets m2.coef_
print(m2.coef_)                   # 12
# test:inject: fit = Model.fit  # lets the harness read cache_info
```

An object `self` already held as an attribute (a kernel, a config object
of a plain class) may have attributes set too, one level down: a hit sets
them on the object your instance holds.

This works when `self` is keyed by all it holds, so not with a `__cash_key__`
or a registered hasher, and when the call changes nothing in place: a method
that appends to a list `self` already holds (`self.history.append(x)`) runs
every time, and the warning names the line. The full conditions are in
[Results cash refuses to store](../../decorator-limitations.md#results-cash-refuses-to-store).

## Methods that return iterators

A method that yields is cached like any iterator, and `self` is keyed the
same way. See [Iterators](iterator-caching.md).

## Related

- [Custom hashers](custom-hashers.md): control how `self` and other arguments
  are identified.
- [Methods and `self`](../../decorator-limitations.md#methods-and-self): the
  limits of hashing `self`.
