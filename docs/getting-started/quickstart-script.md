# Quick start: scripts


Outside notebooks, wrap any function with `@cash.cache`. It caches by the function's
**arguments** and its own **source code**, so an identical call returns instantly and
editing the body invalidates the entry:

<!-- test:skip reason="illustrative — uses a 5s sleep to stand in for real work" -->
```python
import cash

@cash.cache
def expensive_function(x, y):
    import time
    time.sleep(5)          # stand-in for real work
    return x + y

expensive_function(1, 2)   # first call: ~5 seconds
expensive_function(1, 2)   # second call: instant
```

<!-- claim: cash/backends/persistence_policy.py:COMPUTE_FLOOR_S == 0.1 -->
Results that cost more than ~0.1s persist to disk across processes and restarts;
cheaper ones are kept in memory for the session (see the [cost model](../cost-model.md)).

<!-- claim: cash/decorator/arg_hashing.py:ArgHashingMixin._hash_arg_payload @7bc7e4ca, cash/object_hashing.py:builtin_hash @bd4210c7 -->
### Pass DataFrames and arrays — they just work

Arguments don't need to be hashable. Cash hashes by **content**, so a DataFrame or
array is fine — and two *content-equal* frames hit the same entry, even if they're
different objects:

<!-- test:skip reason="illustrative — references undefined df/weights" -->
```python
@cash.cache
def summarize(df, weights):        # a pandas DataFrame and a numpy array
    return (df * weights).sum()

summarize(df, weights)             # runs
summarize(df.copy(), weights.copy())   # instant — same content, different objects
```

Built-in hashers cover pandas, numpy, polars, PyArrow, modin, and dask; register your
own for custom types with [`register_hasher`](../decorator.md).

### Edit a helper — the cache notices

Cash folds the source of the functions you *call* into the key, **transitively**.
`pipeline` below never mentions `clean` — but editing `clean` still invalidates it:

```python
import cash

def clean(text):
    return text.strip().lower()

def features(text):
    return len(set(clean(text).split()))

@cash.cache
def pipeline(text):
    return features(text)

sample = "  The cat  the HAT  "
print(pipeline(sample))          # 3  -> {'the', 'cat', 'hat'}
print(pipeline(sample))          # 3  -> cache hit; the body never ran
print(pipeline.cache_info())     # {'hits': 1, 'misses': 1, ...}
```

Now edit `clean`, two calls below `pipeline` — stop lower-casing:

<!-- test:skip reason="the docs harness infers expected hits/misses from call sites, which cannot model an invalidation between two identical calls; this fence is executed and its output asserted by tests/docs/test_quickstart_helper_invalidation.py instead" -->
```python
def clean(text):
    return text.strip()

print(pipeline(sample))          # 4  -> {'The', 'cat', 'the', 'HAT'}
print(pipeline.cache_info())     # {'hits': 1, 'misses': 2, ...}
```

The answer changed *and* `misses` went up: cash threw the cached result away
because a function `pipeline` calls indirectly changed. You don't have to
remember which entries a helper edit reaches.

<!-- claim: cash/decorator/code_identity.py:CodeIdentityMixin._is_user_module @998fb383 -->
That holds across files: a helper imported from another module of your project
is followed the same way, with nothing to declare. What cash stops at is
installed code — `site-packages` and the standard library. If a third-party
function's identity matters to a result, name it with `depends_on=`.

### File reads are tracked here too

The automatic file tracking from notebook mode applies in scripts as well — no
annotation required:

<!-- test:skip reason="illustrative — references missing data.csv" -->
```python
@cash.cache
def load():
    return pd.read_csv("data.csv")   # change data.csv on disk → recomputes
```

For explicit control there's `@cash.cache(file_depends_on="data.csv")` and `depends_on=`.

And the two paths meet: **call a `@cash.cache` function inside a notebook and its
hits show up on that cell's badge** — the same engine, either way.

The full decorator surface — `ttl`, `cache_if`, async support, iterator caching, the
purity analyzer, and `explain()` — is in the [decorator guide](../decorator.md).

## Configuration (both paths)

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_on @8cbb17b4 -->
`%cash_on` takes only an optional `ttl=N`; to pick a different backend or cache
directory, call `cash.configure(...)` first — `cash.configure(cache_dir="./my_cache")`
before `%cash_on` in a notebook, or before the first `@cash.cache` call in a script.
Both paths share that one default instance. Optional backends — SQLite,
Redis, S3 — install via extras (`pip install "cash-lib[redis]"`, `[s3]`, `[all]`) and
work the same for either path. See [Configuration](configuration.md).

## What's next?

- **[Why Cash?](../why-cash.md)** — when Cash helps, and how it compares to other caches.
- **[Notebook Caching](../notebook_caching_api.md)** & **[Decorator guide](../decorator.md)** — the full reference for each path.
- **[Annotations](../annotations.md)** — per-statement control (`# @cash:no-cache`, `ttl`, `persist`, `allow-random`).
- **[Magic commands](../magics.md)** — `%cash_stats`, `%cash_help`, `%cash_debug`, and friends. Run `%cash_help` in any notebook for the list.
- **[For coding agents](../for-coding-agents.md)** — using cash from Claude Code, Copilot, Cursor and friends (text badges, what an agent should read).
- **[Benchmarks](../benchmarks.md)** — measured speedups, and how to reproduce them on your own workload.
- **Tutorials:** [Data science](../tutorials/use-cases/data-science.md) · [LLM API calls](../tutorials/use-cases/llm-api-calls.md) · [Caching class methods](../tutorials/feature-guides/caching-class-methods.md) · [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md).
