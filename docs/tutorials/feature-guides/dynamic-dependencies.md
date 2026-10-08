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
`DataSource`. cash calls it on every lookup and puts the source's
`state_token()` into the key, so the entry changes when the token does:

```python
import cash
from cash import DataSource

# stands in for a catalog service
CATALOG = {"features": 3, "labels": 1}

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

<!-- claim: cash/decorator/registry.py:resolve_dynamic_dependencies @379962b9, cash/data_source.py:DataSource.state_token @89498b3e -->
A resolver may return one `DataSource`, a list of them, or `None` (no dynamic
dependency for this call). You can also pass a list of resolvers; their sources
are pooled. The key takes the sources in the order they come back, so return
them in a stable order: the same sources in another order miss once.

## Writing the token

`state_token()` must return a value that **changes when the data changes**: a
version, an ETag, a row count with a last-modified time, a digest.

<!-- claim: cash/data_source.py:state_token_of @60e6073c -->
- **Return a value, not a `bool`.** A flag like "is it fresh?" has two states
  and cannot tell one version from the next. cash warns if it sees a `bool`.
- **Keep it cheap.** The resolver and the token run on every lookup, hits
  included, so their cost lands on every call.

The same class works in `depends_on=[...]` when the source is fixed; see
[Custom data sources](../../api/data_sources.md#custom-data-sources).

<!-- claim: cash/file_source.py:FileDataSource @ec702e4c broad="the content-digest contract is a property of the whole class" -->
`FileDataSource(path)` is the built-in source for a file. Its token is the
file's **content digest**, the same check an automatically tracked read gets:
a `touch` that leaves the bytes alone keeps the entry, and an edit recomputes
even when it leaves the timestamp where it was.

The digest is remembered per
file stat, so an unchanged file costs one `stat` per lookup.

## Cached functions that call it

<!-- claim: cash/decorator/file_deps.py:pass_dynamic_sources_up @9f602771, cash/decorator/dynamic_sources.py:dynamic_sources_fresh @d071cce5, cash/decorator/dynamic_sources.py:held_sources @a009e761 -->
A cached function that calls `load` depends on the same sources, though its
own key never sees them. cash keeps them in its entry instead:

- a `FileDataSource` or `RemoteFileDataSource` becomes a file the caller's
  entry checks, as for a file the caller read itself;
- any other source is stored in the caller's entry, pickled, with the token it
  gave when `load` was called. Every lookup of the caller asks the source for
  its token again: the same token serves the entry, another one recomputes
  it, and so does a `state_token()` that raises.
  A source that pickles to more than 4 kB (one that carries data) is
  stored once, beside the entries, and each caller's entry names it.

<!-- claim: cash/decorator/dynamic_sources.py:Resolution @9cd04e3d broad="the kept resolver call, its pickling and its asking again are the class as a whole", cash/decorator/dynamic_sources.py:entry_resolutions @725d4d04 -->
The resolver is asked again too, with the arguments `load` was called with,
on every lookup of the caller and in every process: a resolver that hands
out a new source object after a catalog refresh, or names another file,
recomputes the caller even when the old object still gives its old token.
The call is kept in the caller's entry, pickled (a resolver that does not
pickle, such as a `lambda`, by the name of the cached function it is
declared on), and comes back from a process pool's workers with their
sources. A call whose arguments pickle to more than 4 kB (an array, a frame)
is not copied into the cache: the process that made it keeps it, up to
64 MB of such arguments, the most recent first; past that, or in another
process, its caller recomputes.
In the process that wrote the entry, the source object itself is asked; a
later process asks the copy pickled with the entry, so **`state_token()` must
read the version from where it lives** (the catalog, the database, the
server), not from an attribute the object set when it was made, as
`DatasetVersion` above does.

<!-- claim: cash/decorator/store.py:ResultStore._unpicklable_source_refusal @176a0ddd, cash/backends/persistence_policy.py:PersistencePolicy.decide @9833f8dc -->
A source that cannot be pickled (one holding an open connection or a lock)
can only be asked by the process that has it. The caller's entry then stays
in RAM for this process, and is not written to disk; with a backend that has
no RAM tier it is not stored at all. Either way cash warns once with
[`STORE-UNTRACKED-SOURCE`](../../warnings.md#store-untracked-source). Open
the connection inside `state_token()` and the source pickles.

## When the resolver fails

If the resolver raises, or returns something that is not a `DataSource` (a
string, a number), cash cannot key the call. The call runs **uncached**, and
cash warns once per function
([`KEY-DYNAMIC-DEP-FAILED`](../../warnings.md#key-dynamic-dep-failed)). A cached function
that called it is not stored either: nothing recorded what it depends on. A raw
value is never folded in as if there were no dependency. Wrap the value in a
`DataSource`, as above.

<!-- claim: cash/decorator/explain.py:Explainer.explain @e18309b7 -->
`f.explain(...)` reports such a call as `key_uncomputable`, with the reason.

## Combining with other options

`dynamic_depends_on=` adds to everything else in the key. Changed arguments,
code, `depends_on=` sources and files read or named with `file_depends_on=`
still invalidate, and `ttl=` still expires the entry.

## Related

- [File dependencies](custom-file-sources.md): files cash tracks for you.
- [Custom data sources](../../api/data_sources.md#custom-data-sources): the
  `DataSource` reference.
- [`depends_on=`](../../decorator.md#file_depends_on-and-depends_on): a fixed
  dependency, named on the decorator.
