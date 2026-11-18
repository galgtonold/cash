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
  ([`KEY-UNHASHABLE-ARG`](../../warnings.md#key-unhashable-arg)).
- **`self` carries state that doesn't matter.** A lazy attribute or a memo
  dict changes the key while the instance means the same thing.

The fix for all three is to tell cash what identifies an instance.

## Give the class a `__cash_key__`

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.cash_key_hash @839a8823 -->
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

<!-- claim: cash/core.py:Cash.register_hasher @f8a61573 -->
For a class you can't add a method to, register a hasher for it before the
first call. It applies wherever an instance is an argument of a cached
function, as `self` or not:

<!-- test:skip reason="fragment: names a type the page does not define" -->
```python
app.register_hasher(ThirdPartyClient, lambda c: c.base_url)
```

See [Custom hashers](custom-hashers.md#registering-a-hasher).

## Methods that set attributes on `self`

<!-- claim: cash/decorator/attribute_replay.py:AttrVerdict.of @5152391a, cash/rewrite_in_place.py:AttrState.replay @0e4c9439 -->
A method that only binds attributes on `self` and returns it, as an
estimator's `fit` does, is cached: the entry keeps the attributes the call
bound, and a hit binds them again on the instance you call it on.

```python
import cash

class Scaler:
    def __init__(self, factor):
        self.factor = factor

    @cash.cache
    def fit(self, values):
        self.peak_ = max(values) * self.factor
        return self

s = Scaler(2).fit([1, 5, 3])   # first call: runs
t = Scaler(2).fit([1, 5, 3])   # cache hit: t.peak_ is set too
# test:inject: assert t.peak_ == 10
# test:inject: fit = Scaler.fit  # lets the harness read cache_info
```

This holds only while the key holds `self` by all it holds: a class with a
`__cash_key__` or a registered hasher is not replayed this way, and neither
is a method that writes into a list or dict `self` already holds
(`self.history.append(x)`). Such a call runs every time, and the warning
names the line that stopped it. The full rules are under
[Results cash refuses to store](../../decorator-limitations.md#results-cash-refuses-to-store).

## Methods that return iterators

A method that yields is cached like any iterator, and `self` is keyed the
same way. See [Iterators](iterator-caching.md).

## Related

- [Custom hashers](custom-hashers.md): control how `self` and other arguments
  are identified.
- [Methods and `self`](../../decorator-limitations.md#methods-and-self): the
  limits of hashing `self`.
