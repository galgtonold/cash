# Custom hashers

!!! info "Applies to: decorator"
    Code that passes its own types to cached functions and needs to control how
    cash identifies them.

cash hashes every argument of a cached call to build the key. Built-in values
and most plain classes are pickled and hashed. pandas, numpy, polars, PyArrow,
modin, dask and scipy.sparse values get content hashers of their own. Tell
cash what identifies a value instead when:

- **Hashing it is slow**, and a few fields identify it: an object holding
  several large frames, where every call reads all of them to build the key.
- **It can't be pickled** (it holds a lock, a socket, a C handle). cash warns
  [`KEY-UNHASHABLE-ARG`](../../warnings.md#key-unhashable-arg) and runs the call
  uncached; see
  [Arguments cash cannot hash](../../decorator-limitations.md#arguments-cash-cannot-hash).
- **Equal instances pickle differently**, so every call misses.

For your own class, give it a `__cash_key__` method. For a type you don't own,
register a hasher.

## Your own class: `__cash_key__` {#cash-key}

```python
import pandas as pd
import cash

class Dataset:
    def __init__(self, name, version):
        self.name = name
        self.version = version
        # stands in for several large frames loaded from disk
        self.frames = {"sales": pd.DataFrame({"amount": range(9)})}

    def __cash_key__(self):
        return (self.name, self.version)

    @cash.cache
    def revenue(self):
        return int(self.frames["sales"]["amount"].sum())

sales = Dataset("sales-2026", version=3)
sales.revenue()   # runs
sales.revenue()   # a hit: the frame is not read to build the key
# test:inject: revenue = Dataset.revenue  # lets the harness read cache_info
```

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.cash_key_hash @579078d7, cash/decorator/cash_key.py:cash_key_method @affd7858 -->
`__cash_key__` returns what identifies the object, and the key uses that
instead of reading the frames. Building the key costs microseconds however
much data the object holds, and a stored result is found again after a
restart.

- **It applies wherever the object goes**: as `self` in a cached method, as an
  argument of any cached function, and inside a list, tuple, set or dict
  argument.
- **Return anything cash can key**: a string, a number, a tuple of them, a
  frame, or another object with its own `__cash_key__`. The class's name is
  part of the key too, so two classes returning the same value don't share
  entries.
- **Its code is part of the key.** Editing `__cash_key__` invalidates what it
  keyed. The code of the class and its methods still counts, as for any
  class of yours.
- **Subclasses inherit it.** Set `__cash_key__ = None` on a subclass to key
  its instances by content again.
- **A hasher registered for the type wins** over `__cash_key__`.

The key must change whenever the data does. If it doesn't, cash serves
results computed from the old data. Return a version you bump, a content id,
or the source file's path with its modification time. Here, bump
`version` whenever the data behind `sales-2026` changes.

<!-- claim: cash/decorator/cash_key.py:KeyCheck._check @3770f723, cash/config/schema.py:CashConfig.check_cash_keys == True -->
cash checks this for you. The first time an object is keyed by its
`__cash_key__` in a process, cash reads its content on a background thread
and compares it with what the same key stood for before, in this run or an
earlier one. Different content behind one key warns
[`KEY-STALE-CASH-KEY`](../../warnings.md#key-stale-cash-key). The call never
waits for the check, and each object is checked once, so an object changed in
place later in the same run is not caught. Turn the check off with
`check_cash_keys=False`.

## Registering a hasher

For a type you don't own, or to keep the identity outside the class, register
a hasher for the type:

```python
import hashlib
import cash

class MyModel:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights   # a numpy array

def hash_model(model):
    return hashlib.sha256(
        model.name.encode() + model.weights.tobytes()
    ).hexdigest()

cash.register_hasher(MyModel, hash_model)

@cash.cache
def evaluate(model, data):
    return model.weights @ data
```

<!-- claim: cash/core.py:Cash.register_hasher @f8a61573, cash/source_norm.py:callable_identity @4e3b5359 -->
From now on, every `MyModel` argument is identified by `hash_model(model)`,
and so is a `MyModel` inside a list, tuple, set or dict argument. On your own
`Cash(...)` instance, call `app.register_hasher(...)` instead.

- **Subclasses match.** Dispatch uses `isinstance`, so a hasher for a base class
  covers its subclasses. When several registrations match, the first one
  registered wins, so register the most specific type first.
- **The hasher's code is part of the key.** Editing `hash_model` invalidates
  the entries it produced, even if its output stays the same.
- **A registration lasts for the process.** There is no way to unregister one;
  register hashers once, at import time.
- **Don't register for functions.** A hasher for `types.FunctionType`,
  `types.MethodType` or `functools.partial` warns
  ([`KEY-CALLABLE-HASHER`](../../warnings.md#key-callable-hasher)): it would
  cover every function passed to any cached call. Pass what a closure captures
  as a plain argument instead.

<!-- claim: cash/decorator/code_args.py:CodeArgs.iter_code_carriers @5789f237, cash/decorator/arg_hashing.py:ArgHasher.keys_by_registration @6cecb0bf -->
The hasher is the value's whole identity. The code of the value's class still
counts when the class is yours, but cash does not search the value for code it
holds: a function stored on an instance, or the handlers and streams a
registered `logging.Logger` reaches. If the result depends on such code, make
the hasher return something that changes with it, or name it with
`@cash.cache(depends_on=[...])`.

## What makes a good hasher

- **Deterministic across processes.** Same value, same string, in every run.
  Never use `id()`, `hash()` of a string (randomised per process), the time or
  a memory address. A hasher that changes between runs makes every call miss.
- **Complete.** Include every field that affects the result. A hasher that
  reads `weights` but not a `temperature` field gives two different models one
  entry, and the second call returns the first one's answer.
- **Cheap, but never by sampling.** The hasher runs on every call. Reading only
  part of the data collides different values into a wrong hit. Hash a smaller
  identity only if it really identifies the value, such as a version you
  control.

<!-- claim: cash/object_hashing.py:hash_numpy @f6df9c37 -->
To check a hasher, call it on two equal but separately built instances. The
strings must match. `evaluate.explain(model, data).cache_key` shows the key a
call would use.

## Common patterns

Use a canonical serializer the type already has:

```python
def hash_pydantic(model):
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()

cash.register_hasher(MyPydanticModel, hash_pydantic)
```

The same works for protobuf (`SerializeToString(deterministic=True)`) and
msgspec (`msgspec.json.encode`).

Hash the fields that decide the result:

```python
def hash_dataset_config(cfg):
    parts = (
        cfg.path,
        cfg.split,
        cfg.preprocessing_version,
        tuple(cfg.features),
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()

cash.register_hasher(DatasetConfig, hash_dataset_config)
```

For a handle to outside data (a database engine, a client), hash what it points
at, not the handle:

```python
def hash_engine(engine):
    return hashlib.sha256(str(engine.url).encode()).hexdigest()

cash.register_hasher(sqlalchemy.engine.Engine, hash_engine)
```

A new engine for the same URL then gets the same key. The key says nothing
about the data behind the URL, so add `ttl=` or a `depends_on=` source if that
data changes.

## Overriding a built-in content hasher

<!-- claim: cash/object_hashing.py:builtin_hash @dd82c01b broad="the list enumerates every type the builtin dispatcher recognises", cash/object_hashing.py:builtin_hash_family @b0c04a56 -->
cash hashes these types by their full content, before it looks at your
registrations:

| Type | Hashed from |
|---|---|
| pandas `DataFrame`, `Series` | schema (names, dtypes with their categories, index and its `freq`), `attrs` and every value |
| numpy `ndarray` | shape, dtype, memory order and the full buffer |
| polars `DataFrame`, `Series`, `LazyFrame` | schema and row hashes, an `Object` column by its values' content; a `LazyFrame` by its serialized plan |
| PyArrow `Table`, `RecordBatch` | schema and every row, dictionaries included |
| modin `DataFrame`, `Series` | converted to pandas, then as pandas |
| dask collections | the task keys and the schema |
| scipy.sparse matrices and arrays | type, shape, dtype, format and the arrays the format stores |

A plain registration for one of these types could never run, so it raises
`ValueError`. Pass `override=True` to replace the built-in:

<!-- test:skip reason="illustrative: frames carry no dataset_version attribute here" -->
```python
import pandas as pd

cash.register_hasher(
    pd.DataFrame,
    lambda df: df.attrs["dataset_version"],   # set by your loader
    override=True,
)
```

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.hash_payload @99cd3791 -->
Your hasher then becomes the value's whole identity: two frames it hashes alike
share one entry, and the second call gets the first one's result.

Override only
when you hold an identity the value itself does not show, such as a dataset
version or a content id, and when content hashing is slow **relative to the
work**. An 800 MB array costs about 0.3 s to hash; that matters in a loop of
fast calls, not before a six-second fit.

Your own subclass of one of these types is not covered by the built-ins; a
plain registration for it works.

## Related

- [Class methods](caching-class-methods.md): keying `self`.
- [Arguments cash cannot hash](../../decorator-limitations.md#arguments-cash-cannot-hash):
  what to do when a type cannot be pickled at all.
- [An argument that does not change the result](../../decorator-limitations.md#an-argument-that-does-not-change-the-result):
  a hasher that returns a constant, for a logger.
- [File dependencies](custom-file-sources.md): how file arguments and reads
  are tracked.
