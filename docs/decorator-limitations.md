# Known limitations of `@cash.cache`

!!! info "Applies to: decorator"
    Scripts, services and libraries that use `@cash.cache`.

Most functions cache correctly with a bare `@cash.cache`. This page lists the
cases that need a change on your side, and what that change is.

## Arguments cash cannot hash

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.hash_payload @cba92fc6 -->
An argument that cannot be pickled (a lock, an open file, a live connection, a
closure) cannot be keyed. The call runs uncached and warns
[`KEY-UNHASHABLE-ARG`](warnings.md#key-unhashable-arg); any other failure while
building the key does the same. Pass a plain value that identifies the object
instead. For a class of yours, give it a
[`__cash_key__`](tutorials/feature-guides/custom-hashers.md#cash-key)
method that returns one. For a type you don't own, register a hasher that
returns a **stable** identifying value, such as a database URL:

```python
import hashlib
import cash

class Store:
    def __init__(self, url):
        self.url = url

def hash_store(store):
    return hashlib.sha256(store.url.encode()).hexdigest()

cash.register_hasher(Store, hash_store)
```

Never hash by `id()`: ids repeat across processes, so a later run could get
another object's entry. See [Custom hashers](tutorials/feature-guides/custom-hashers.md).

## A table changed past pandas

<!-- claim: cash/decorator/arg_hashing.py:watch_array_handles @9c632881, cash/decorator/arg_hashing.py:_watch_callback_handouts @9312b86e -->
Under pandas copy-on-write, a table passed to a cached function again is not
hashed again while pandas has given it no new data: pandas copies a column
before it writes it. A write that goes past pandas is seen as long as it goes
through memory pandas handed out: `df["x"].array`, a view from `.values`,
`.to_numpy()` or `np.asarray` that you made writable again, and the arrays
pandas passes to your function from `df.apply(f, raw=True)`,
`rolling(...).apply(f, raw=True)`, `expanding().apply(f, raw=True)`, `pipe`
and a groupby's `apply`, `agg` and `transform`. A NumPy view pandas made
somewhere else and you made writable is not seen, and the cached function
returns its old result for the table: write through pandas
(`df.loc[...] = ...`) instead.

## An argument that does not change the result

<!-- claim: cash/core.py:Cash.cache @470582df -->
Every argument is part of the key, so a logger, a progress callback or a
`verbose=` flag splits the cache: `fit(data, verbose=True)` misses after
`fit(data)` ran. Leave such a parameter out with `ignore=["verbose"]` or a
`cash.Ignore[bool]` annotation, or say what decides the result with `key=`;
see [Leaving arguments out of the key](decorator.md#leaving-arguments-out-of-the-key).

When the extra argument's work should happen on every call, hits included,
keep it out of the cached function instead. A thin wrapper takes it and calls
a cached core with only the arguments that change the result:

```python
import logging
import cash

@cash.cache
def _fit(data):
    return sum(data) / len(data)

def fit(data, log=None, verbose=False):
    if verbose and log:
        log.info("fitting %d rows", len(data))
    return _fit(data)

fit([1, 2, 3], log=logging.getLogger("a"), verbose=True)
fit([1, 2, 3])   # _fit hits: log and verbose are not in its key
# test:inject: assert _fit.cache_info()["hits"] == 1; _fit.cache_clear()
# test:inject: _fit([1, 2, 3])  # first call after the clear
```

The wrapper's own work runs on every call, so the log line appears on a hit
too, which a logger inside the cached body would not do.

<!-- claim: cash/core.py:Cash.register_hasher @f8a61573 -->
For an argument **type** that never affects a result, a hasher that returns a
constant does the same without a wrapper. Every logger then counts as the same
value:

```python
import logging
import cash

cash.register_hasher(logging.Logger, lambda log: "any-logger")

@cash.cache
def fit_logged(data, log):
    log.info("fitting %d rows", len(data))
    return sum(data) / len(data)

fit_logged([1, 2, 3], logging.getLogger("a"))   # first call: runs
fit_logged([1, 2, 3], logging.getLogger("b"))   # cache hit
```

The hasher applies to every cached function in the process, so use it only
for a type that is never input data. Don't do this for `bool` or `int`: leave
a `verbose=` flag out with `ignore=`.

## Methods and `self`

`self` is an argument like any other, hashed by its state: two instances with
equal attributes share entries. An unpicklable attribute makes every call
uncached, and a large one (a frame, a model) is read on every call. A
`__cash_key__` method on the class fixes both. See
[Class methods](tutorials/feature-guides/caching-class-methods.md).

## Code you pass as an argument

A class or function of yours passed as an argument is keyed by its code and by
what that code reads, so editing it recomputes. Three cases are not covered:

- **Library classes and functions** are keyed by name, not code. See
  [Code in installed packages](#code-in-installed-packages).
- **A closure or `lambda`** can't be pickled, so the call runs uncached. Pass a
  module-level function and give it the captured value as an argument.
- **An implementation picked at run time** (`HANDLERS[name]()` on a dict built
  inside the body, a plugin registry) is not reached. Name the candidates:
  `@cash.cache(depends_on=[FastPath, ExactPath])`.

A marker class you pass but whose code never affects the result can be excluded
with `@cash.opaque`; see [Purity markers](tutorials/feature-guides/purity-decorators.md#cashopaque-leave-a-class-out-of-the-key).

## Code in installed packages

<!-- claim: cash/install_paths.py:in_own_package @168c5d1e, cash/install_paths.py:is_user_path @d210933e, cash/dependency_state.py:DependencyStateHasher.compute @8e272f43 -->
cash follows the code a cached function calls, and puts it in the key, when
that code is yours. Yours means:

- a file outside the Python installation: your project, or a package installed
  with `pip install -e` (an editable install runs from your source folder);
- the package the cached function itself is in, wherever it is installed. A
  cached function in `mytool.jobs` is keyed by the code it calls anywhere in
  `mytool`, even when `mytool` is installed with a plain `pip install`.

Any other code in `site-packages` is a library, and that includes your team's
internal package when it is installed with a plain `pip install`. cash keys a
call into it by name and does not look inside, so after an upgrade the old
results are still served.

Pinning versions does not change this: the version is not part of the key. A
pin keeps everyone on the same code; it does not make an upgrade recompute.
After an upgrade that changes results, clear the affected entries
(`f.cache_clear()` or [`cash clear --function`](cli.md#cash-clear-path-all)),
or tell cash what to watch:

- **While you work on the package**, install it with `pip install -e`. Its
  code is then tracked like your project's.
- **`depends_on=[teamlib.score]`** puts the source of that function, as
  installed, in the key, so an upgrade that changes `score` recomputes. Only
  that function's own code counts, not the functions it calls: an upgrade that
  changes only a helper inside `teamlib` keeps the old results. Name each
  function whose change matters.
- **To recompute on every release**, depend on the installed version with a
  `DataSource`. A new version number recomputes; changed code reinstalled
  under the same version number does not.

<!-- test:skip reason="needs an installed package named teamlib" -->
```python
import importlib.metadata

import cash
import teamlib


class InstalledVersion(cash.DataSource):
    def __init__(self, dist):
        self.dist = dist

    def get_id(self):
        return f"installed:{self.dist}"

    def state_token(self):
        return importlib.metadata.version(self.dist)


@cash.cache(depends_on=[InstalledVersion("teamlib")])
def forecast(region):
    return teamlib.score(region)
```

## Reads cash cannot see

<!-- claim: cash/tracking/reader_patches.py:_patch_multiprocessing_pool @a7a12595, cash/tracking/reader_patches.py:_patch_process_pool_submit @1f70759c -->
cash does not record a file opened by a C extension, `os.open`, a subprocess,
or a `threading.Thread` you start. Name such a file with `file_depends_on=`.

Reads in work the function hands to a pool are recorded: a
`ThreadPoolExecutor`, `ProcessPoolExecutor`, `multiprocessing.Pool` (or
`ThreadPool`), or joblib's default backend (`Parallel(n_jobs=...)`, a
scikit-learn `n_jobs=`). Each task then runs under a tracker of its own in the
worker: about 20 µs a task, which only shows for tiny tasks sent one at a time
(`pool.imap` with the default `chunksize=1`).

A polars `LazyFrame` argument from `scan_csv` is keyed by its path, not the
file's content; collect it first.

## An edit that keeps the size and timestamps

<!-- claim: cash/tracking/file_dep_snapshot.py:_unchanged_since_hashed @809a68f2 -->
cash checks a file's size and timestamps first and reads its content only when
one of them moved. On Linux and macOS every write moves the inode change time,
so this never misses. On **Windows**, two kinds of edit move nothing: a write
whose modification time is put back afterwards (`os.utime`, `shutil.copystat`,
`robocopy /COPY:T`), and a write through `np.memmap(path, mode="r+")`.

After such a write, touch the file (`Path(path).touch()`). cash then reads its
content, and if it changed, the calls that read it recompute.

## Relative paths and working directories

`open("data.csv")` from two working directories reads two different files under
one key, and so does a relative `file_depends_on="data.csv"`. Each switch
recomputes and replaces the other directory's entry. Build
paths from the project root or pass absolute paths.

## Results cash refuses to store

- A matplotlib `Figure` or `Axes` is never cached
  ([`CACHE-IDENTITY-COUPLED`](warnings.md#cache-identity-coupled)): a restored
  copy would become pyplot's "current figure" and `plt.savefig()` would save a
  blank image. Cache the data and draw from it.
- A result that can't be pickled is not stored
  ([`STORE-FAILED`](warnings.md#store-failed)), unless the function is
  `frozen=True`.
- A call that changed its argument in place is not stored, so it runs every
  time. Return a modified copy instead (`rows = sorted(rows)`).

## Generators and async functions

A generator result is stored in chunks as you consume it, and nothing is stored
if you stop early; an infinite generator never finishes, so it never caches. See
[Iterators](tutorials/feature-guides/iterator-caching.md). `async def` functions
cache like sync ones; async generators are returned undecorated with a warning.
See [Async functions](tutorials/feature-guides/async-caching.md).

## Edits inside a running process

cash reads a helper's source once per process, so an edit between two calls
of one long-running process is seen only by the next process. If a source file
changes on disk after the process started (a deploy), cash keys by the code
that is running and warns [`KEY-SOURCE-CHANGED`](warnings.md#key-source-changed).

## Using a decorated function in a notebook

Decorated functions work in a notebook, but re-running the cell that defines
one creates a new wrapper with new counters, so `cache_info()` may read zero.
Use `explain()` there.

## Related

- [The `@cash.cache` guide](decorator.md): where results go, what invalidates
  an entry, and the parameters.
- [Custom hashers](tutorials/feature-guides/custom-hashers.md): identify your
  own types in the key.
- [File dependencies](tutorials/feature-guides/custom-file-sources.md): which
  file reads are tracked, and how to name the others.
- [Seeing what cash did](decorator.md#seeing-what-cash-did): find out why a
  call missed.
