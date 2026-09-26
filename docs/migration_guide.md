# Coming from other caches

!!! info "Applies to: both paths"
    If you already cache with another tool: the same code before and after.
    How the tools compare is on [Why cash?](why-cash.md#compared-with-other-tools).

For each cache you may use now, this page shows your code before and after
the switch, and what the tool's options become in cash.

## `functools.lru_cache`

<div class="grid" markdown>

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { title="Before: lru_cache" }
from functools import lru_cache

@lru_cache(maxsize=128)
def load_prices(ticker):
    ...
```

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { title="After: cash" }
import cash

@cash.cache
def load_prices(ticker):
    ...
```

</div>

Results are stored on disk, so the next process gets them too, and arguments
need not be hashable: lists, dicts, DataFrames and arrays are keyed by content.

| `lru_cache` | cash |
|---|---|
| `maxsize=128` | No per-function limit. The whole cache has a size cap: `max_cache_size` for the disk, `max_memory_entries` for the RAM tier. See [Cache size limit](decorator.md#cache-size-limit) |
| `typed=True` | Always on: arguments are keyed by content and type |
| `f.cache_info()` | Same name. Returns a dict with `hits`, `misses` and why calls missed |
| `f.cache_clear()` | Same name. Also deletes the function's entries on disk |

## `joblib.Memory`

<div class="grid" markdown>

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { title="Before: joblib.Memory" }
from joblib import Memory

memory = Memory("/tmp/joblib-cache", verbose=0)

@memory.cache(ignore=["verbose"])
def transform(data, verbose=False):
    return data.apply(complex_transform)
```

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { title="After: cash" }
import cash

@cash.cache
def _transform(data):
    return data.apply(complex_transform)

def transform(data, verbose=False):
    return _transform(data)
```

</div>

joblib hashes the decorated function's own source. cash also follows the
helpers it calls (`complex_transform` here) and the files it reads.

| `joblib.Memory` | cash |
|---|---|
| `Memory("/tmp/joblib-cache")` | `.cash` at your project root by default. Choose another with `CASH_CACHE_DIR` or `Cash(cache_dir=...)`; see [Cache folder](decorator.md#cache-folder) |
| `ignore=["verbose"]` | No `ignore=`: every argument is in the key. Use a wrapper, as above; see [An argument that does not change the result](decorator-limitations.md#an-argument-that-does-not-change-the-result) |
| `verbose=` | `CASH_VERBOSE=1` logs one line per call |
| `compress=True` | The `compress` setting ([Configuration](getting-started/configuration.md#all-settings)) |
| `memory.reduce_size(bytes_limit=...)` | `max_cache_size`, enforced on every write |
| `cache_validation_callback=expires_after(...)` | `ttl=` |
| `transform.clear()` | `transform.cache_clear()` |
| `memory.clear()` | `cash clear --all` |

## `diskcache`

<div class="grid" markdown>

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { title="Before: diskcache" }
from diskcache import Cache

cache = Cache("/tmp/diskcache")

@cache.memoize(expire=3600)
def transform(data):
    return data.apply(complex_transform)
```

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { title="After: cash" }
import cash

@cash.cache(ttl=3600)
def transform(data):
    return data.apply(complex_transform)
```

</div>

`memoize` keys on the function's name and arguments, so editing the function
keeps serving the old result. cash recomputes when the function, a helper it
calls or a file it reads changes.

| `diskcache` | cash |
|---|---|
| `Cache("/tmp/diskcache")` | `.cash` at your project root, or `CASH_CACHE_DIR`; see [Cache folder](decorator.md#cache-folder) |
| `expire=3600` | `ttl=3600` (seconds, or a `datetime.timedelta`) |
| `typed=True` | Always on |
| `ignore=` | No `ignore=`; see [An argument that does not change the result](decorator-limitations.md#an-argument-that-does-not-change-the-result) |
| `size_limit=` | `max_cache_size` |
| `cache.clear()` | `cash clear --all`, or `transform.cache_clear()` for one function |

## Hand-written pickle files

=== "Decorator"

    <div class="grid" markdown>

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { title="Before: a pickle file" }
    import os
    import pickle

    if os.path.exists("result.pkl"):
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)
    else:
        result = expensive_computation()
        with open("result.pkl", "wb") as f:
            pickle.dump(result, f)
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { title="After: cash" }
    import cash

    @cash.cache
    def expensive_computation():
        ...

    result = expensive_computation()
    ```

    </div>

=== "Notebook"

    <div class="grid" markdown>

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell title="Before: a pickle file" }
    import os
    import pickle

    if os.path.exists("result.pkl"):
        with open("result.pkl", "rb") as f:
            result = pickle.load(f)
    else:
        result = expensive_computation()
        with open("result.pkl", "wb") as f:
            pickle.dump(result, f)
    ```

    <!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
    ```python { .nb-cell title="After: cash, below the %cash_on cell" }
    result = expensive_computation()
    ```

    </div>

A pickle file goes stale silently when the code or the data behind it
changes. cash recomputes then.

## `%store` (notebook)

<div class="grid" markdown>

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { .nb-cell title="Before: %store" }
%store -r df
if "df" not in dir():
    df = pd.read_csv("large_file.csv")
    %store df
```

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { .nb-cell title="After: cash, in the first cell" }
import cash
%cash_on
```

</div>

<!-- test:skip reason="illustrative: before/after code; the other tools and data are not set up" -->
```python { .nb-cell title="Any cell below it" }
df = pd.read_csv("large_file.csv")
```

`%store` never invalidates anything. cash reloads `df` after a restart and
reads the file again when `large_file.csv` changes.

## `jupyter-cache` (notebook)

Nothing to replace: `jupyter-cache` skips re-executing unchanged notebooks in
a book build, and cash caches statements while you work. Use both if you need
both.

## Related

- [Why cash?](why-cash.md#compared-with-other-tools): what each tool does and
  does not track, side by side.
- [Decorator quick start](getting-started/quickstart-script.md): the first ten
  minutes with `@cash.cache`.
- [Notebook quick start](getting-started/quickstart-notebook.md): the first
  ten minutes with `%cash_on`.
- [Configuration](getting-started/configuration.md): every setting named on
  this page.
