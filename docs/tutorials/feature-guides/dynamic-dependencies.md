# Dynamic dependencies

!!! info "Applies to: decorator"
    Code whose cached functions depend on outside data that cash cannot see,
    where which data depends on the arguments.

You rarely need `dynamic_depends_on=`. A file the body reads through a tracked
reader is recorded automatically, whatever path the arguments produce, so
`pd.read_parquet(f"data/{name}.parquet")` needs nothing extra. See
[File dependencies](custom-file-sources.md).

Use `dynamic_depends_on=` when the data is invisible to cash **and** the
arguments decide which data it is: a dataset version in a catalog service, a
table in a database, an HDF5 file read through `h5py`.

## A resolver

A resolver takes the same arguments as the cached function and returns a
`DataSource`. Cash calls it on every lookup and puts the source's
`state_token()` into the key, so the entry changes when the token does:

```python
import cash
from cash import DataSource

CATALOG = {"features": 3, "labels": 1}   # stands in for a catalog service

class DatasetVersion(DataSource):
    def __init__(self, name):
        self.name = name

    def get_id(self):
        return f"dataset:{self.name}"

    def state_token(self):
        return str(CATALOG[self.name])   # the current version

def version_of(name):
    return DatasetVersion(name)

@cash.cache(dynamic_depends_on=version_of)
def load(name):
    return {"name": name, "rows": 1000}

load("features")   # first call: computes
load("features")   # cache hit
load("labels")     # cache miss: different arguments
# test:inject: CATALOG["features"] = 4
load("features")   # cache miss: the version moved
```

<!-- claim: cash/decorator/registry.py:RegistryMixin._resolve_dynamic_dependencies @1d703750, cash/data_source.py:DataSource.state_token @89498b3e -->
A resolver may return one `DataSource`, a list of them, or `None` (no dynamic
dependency for this call). You can also pass a list of resolvers; their sources
are pooled. The order of the sources does not matter.

## Writing the token

`state_token()` must return a value that **changes when the data changes**: a
version, an ETag, a row count with a last-modified time, a digest.

<!-- claim: cash/data_source.py:state_token_of @01fb9637 -->
- **Return a value, not a `bool`.** A flag like "is it fresh?" has two states
  and cannot tell one version from the next. Cash warns if it sees a `bool`.
- **Keep it cheap.** The resolver and the token run on every lookup, hits
  included, so their cost lands on every call.

The same class works in `depends_on=[...]` when the source is fixed; see
[Custom data sources](../../api/data_sources.md#custom-data-sources).

<!-- claim: cash/data_source.py:FileDataSource @a09d1326 broad="the mtime contract is a property of the whole class" -->
`FileDataSource(path)` is the built-in source for a file. Its token is the
file's **modification time**, so a `touch` recomputes and a quick same-size edit
on a file system with one-second timestamps can be missed. For a file cash can
see, rely on automatic tracking or `file_depends_on=`, which check content.

## When the resolver fails

If the resolver raises, or returns something that is not a `DataSource` (a
string, a number), cash cannot key the call. The call runs **uncached**, and
cash warns once per function
([`KEY-DYNAMIC-DEP-FAILED`](../../warnings.md#key-dynamic-dep-failed)). A raw
value is never folded in as if there were no dependency. Wrap the value in a
`DataSource`, as above.

<!-- claim: cash/decorator/explain.py:ExplainMixin._explain_call @bd141dbf -->
`f.explain(...)` reports such a call as `key_uncomputable`, with the reason.

## Combining with other options

`dynamic_depends_on=` adds to everything else in the key. Changed arguments,
code, `depends_on=` sources and files read or named with `file_depends_on=`
still invalidate, and `ttl=` still expires the entry.

## Related

- [File dependencies](custom-file-sources.md): files cash tracks for you.
- [Custom data sources](../../api/data_sources.md#custom-data-sources)
- [The `@cash.cache` guide](../../decorator.md#file_depends_on-and-depends_on)
