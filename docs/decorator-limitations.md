# Known limitations of `@cash.cache`

!!! info "Applies to: decorator"
    Scripts, services and libraries that use `@cash.cache`.

Most functions cache correctly with a bare `@cash.cache`. This page lists the
cases that need a change on your side, and what that change is.

## Arguments cash cannot hash

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.hash_payload @34fca9bf -->
An argument that cannot be pickled (a lock, an open file, a live connection, a
closure) cannot be keyed. The call runs uncached and warns
[`KEY-UNHASHABLE-ARG`](warnings.md#key-unhashable-arg); any other failure while
building the key does the same. Pass a plain value that identifies the object
instead, or register a hasher that returns a **stable** identifying value, such
as a database URL:

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

## Methods and `self`

`self` is an argument like any other, hashed by its state: two instances with
equal attributes share entries. An unpicklable attribute makes every call
uncached. See [Class methods](tutorials/feature-guides/caching-class-methods.md).

## Code you pass as an argument

A class or function of yours passed as an argument is keyed by its code and by
what that code reads, so editing it recomputes. Three cases are not covered:

- **Library classes and functions** are keyed by name, not code. Pin versions.
- **A closure or `lambda`** can't be pickled, so the call runs uncached. Pass a
  module-level function and give it the captured value as an argument.
- **An implementation picked at run time** (`HANDLERS[name]()` on a dict built
  inside the body, a plugin registry) is not reached. Name the candidates:
  `@cash.cache(depends_on=[FastPath, ExactPath])`.

A marker class you pass but whose code never affects the result can be excluded
with `@cash.opaque`; see [Purity markers](tutorials/feature-guides/purity-decorators.md#cashopaque-leave-a-class-out-of-the-key).

## Reads cash cannot see

<!-- claim: cash/tracking/reader_patches.py:_patch_multiprocessing_pool @a7a12595, cash/tracking/reader_patches.py:_patch_process_pool_submit @56972fff -->
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
one key. Each switch recomputes and replaces the other directory's entry. Build
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
