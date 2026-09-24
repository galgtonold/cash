# Custom hashers

!!! info "Applies to: decorator"
    Code that passes its own types to cached functions and needs to control how
    cash identifies them.

Cash hashes every argument of a cached call to build the key. Built-in values
and most plain classes are pickled and hashed. pandas, numpy, polars, PyArrow,
modin and dask values get content hashers of their own. Register a hasher for
your own type when:

- **It can't be pickled** (it holds a lock, a socket, a C handle). Cash warns
  [`KEY-UNHASHABLE-ARG`](../../warnings.md#key-unhashable-arg) and runs the call
  uncached; see
  [Arguments cash cannot hash](../../decorator.md#arguments-cash-cannot-hash).
- **Pickling it is slow**, and a few fields identify it.
- **Equal instances pickle differently**, so every call misses.

## Registering a hasher

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

<!-- claim: cash/core.py:Cash.register_hasher @2ae870d0, cash/source_norm.py:callable_identity @f9ec85f7 -->
From now on, every `MyModel` argument is identified by `hash_model(model)`. On
your own `Cash(...)` instance, call `app.register_hasher(...)` instead.

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

<!-- claim: cash/object_hashing.py:hash_numpy @8f6f3203 -->
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
    parts = (cfg.path, cfg.split, cfg.preprocessing_version, tuple(cfg.features))
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

<!-- claim: cash/object_hashing.py:builtin_hash @bd4210c7 broad="the list enumerates every type the builtin dispatcher recognises", cash/object_hashing.py:builtin_hash_family @ac9cffe9 -->
Cash hashes these types by their full content, before it looks at your
registrations:

| Type | Hashed from |
|---|---|
| pandas `DataFrame`, `Series` | schema (names, dtypes, index) and every value |
| numpy `ndarray` | shape, dtype, memory order and the full buffer |
| polars `DataFrame`, `Series`, `LazyFrame` | schema and row hashes; a `LazyFrame` by its serialized plan |
| PyArrow `Table`, `RecordBatch` | schema, row count and every column buffer |
| modin `DataFrame`, `Series` | converted to pandas, then as pandas |
| dask collections | the task keys and the schema |

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

<!-- claim: cash/decorator/arg_hashing.py:ArgHashingMixin._hash_arg_payload @7bc7e4ca -->
Your hasher then becomes the value's whole identity: two frames it hashes alike
share one entry, and the second call gets the first one's result. Override only
when you hold an identity the value itself does not show, such as a dataset
version or a content id, and when content hashing is slow **relative to the
work**. An 800 MB array costs about 0.3 s to hash; that matters in a loop of
fast calls, not before a six-second fit.

Your own subclass of one of these types is not covered by the built-ins; a
plain registration for it works.

## Related

- [Class methods](caching-class-methods.md): hashing `self`.
- [The `@cash.cache` guide](../../decorator.md#arguments-cash-cannot-hash)
- [File dependencies](custom-file-sources.md)
