# Quick Start

Cash works two ways. Pick the one that matches how you write code — each track
below is self-contained.

<div class="cash-doc-grid" markdown="0">
  <a class="cash-cta-card primary" href="#in-a-notebook">
    <h3>In a notebook →</h3>
    <p><code>%cash_on</code> and every cell caches automatically — statement-level,
    with a badge above each cell. For interactive work in Jupyter, Colab, or VS Code.</p>
  </a>
  <a class="cash-cta-card" href="#in-a-script">
    <h3>In a script →</h3>
    <p><code>@cash.cache</code> on a function — caches by its arguments and its own
    source code. For modules, pipelines, and batch jobs.</p>
  </a>
</div>

Most people start in a notebook, but the two share one engine — the same lineage,
hashing, and backends underneath — so nothing is second-class.

---

## In a notebook

!!! tip "Prefer to just try it?"
    Run cash in your browser with no install —
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb) — then come back here to wire it into your own notebooks.

### Step 1: Enable Cash

```python { .nb-cell }
import cash

# Enable auto-caching for all subsequent cells
%cash_on
```

That's the whole setup — no decorators, no config file. (`import cash` auto-registers
the magics, so `%load_ext cash` is not required.)

!!! tip "Skip the boilerplate"
    Run `cash autoload on` once and every new IPython/Jupyter kernel starts with
    `import cash` plus `%cash_on` already executed. See the [CLI reference](../cli.md).

### Step 2: Write code normally

<!-- test:skip reason="illustrative — references missing large_dataset.csv" -->
```python { .nb-cell }
import pandas as pd

df = pd.read_csv('large_dataset.csv')   # cached after the first run
print(f"Loaded {len(df)} rows")
```

### Step 3: Re-run — it's instant

Run the cell again and the `read_csv` result loads from cache. The badge above the
cell flips from `EXECUTED` (ochre) to `RESTORED` (green):

**First run** — Cash ran the statement and cached the result:

<iframe class="cash-badge" src="/_badges/quickstart_first_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Second run** — Cash restored the value instead of recomputing:

<iframe class="cash-badge" src="/_badges/quickstart_second_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

See [Reading the Cash badge](../badges.md) for the full anatomy.

### Change one thing upstream — only what depends on it re-runs

This is the part a plain cache can't do. Suppose one cell sets a parameter, another
does expensive work, and a third depends on the parameter:

<!-- test:skip reason="illustrative — references missing data + build_features()" -->
```python { .nb-cell }
# Cell 1
THRESHOLD = 10                       # ← change me to 20, then Run all

# Cell 2 — expensive, and unrelated to THRESHOLD
df = pd.read_csv('big.csv')
features = build_features(df)        # stays RESTORED

# Cell 3 — depends on THRESHOLD
outliers = features[features.score > THRESHOLD]
```

Change `THRESHOLD` and **Run all**: Cell 2 stays green (`RESTORED` — cash saw its
inputs didn't change), and only Cell 3 recomputes. No decorators, no dependency
graph to declare. And if you jump straight to Cell 3 after a kernel restart, cash
restores or re-runs the upstream cells it needs — restored straight from cache when
nothing changed.

> `joblib.Memory` only sees a call's arguments (and the wrapped function's own code);
> `jupyter-cache` re-runs the whole notebook when any cell changes. Cash tracks the
> lineage between statements, so it recomputes exactly what changed and nothing else.

### It follows your functions and loops

Cash keys on the **source** of what you run, not the text of a cell:

- Edit a function's body — or a helper it calls — and the statements that use it
  recompute (the badge marks it **changed**). It's not blind text matching.
- In a loop, each iteration is cached on its own — change one case and the rest
  still restore:

<!-- test:skip reason="illustrative — references undefined fetch_and_model/prices" -->
```python { .nb-cell }
for ticker in ["AAPL", "MSFT", "GOOG"]:
    prices[ticker] = fetch_and_model(ticker)   # each iteration cached separately
```

See [The notebook path](../how-it-works/notebook-path.md) for how partial hits work.

### Statement-level, not cell-level

Cash caches each **statement**, not the whole cell:

<!-- test:skip reason="illustrative — references missing data.csv" -->
```python { .nb-cell }
df     = pd.read_csv('data.csv')       # statement 1
daily  = df.resample('D').mean()       # statement 2
result = daily.rolling(7).mean()       # statement 3
```

Change statement 3 and only it re-runs — statements 1 and 2 stay cached.

*One exception: a top-level statement that mutates an object **created in an
earlier cell** (`df['x'] = ...` on an upstream `df`, `lst.append(...)` on an
upstream list) runs fresh each time — cash tracks the mutation so everything
downstream stays correct, it just doesn't replay a snapshot. The same mutation on
an object built in the same cell caches normally, and inside a loop iterations are
cached whole, mutations included.* See
[Knowing when to recompute](../how-it-works/invalidation.md).

### File changes are tracked automatically

Cash intercepts file reads (`pd.read_csv`, `np.load`, `open`, `joblib.load`, …) and
records each file's fingerprint — change the file on disk and the statements that
read it recompute, no annotation needed. See the [notebook reference](../notebook_caching_api.md).

### It survives a kernel restart

The cache lives on disk, not just in memory. Restart the kernel, re-run a cell, and
cash restores the value instead of replaying the whole chain — a fresh kernel picks
up where you left off. (`%store` and `diskcache` persist too, but don't know what's
still *valid*; cash proves freshness through lineage.)

---

## In a script

Outside notebooks, wrap any function with `@cash.cache`. It caches by the function's
**arguments** and its own **source code**, so an identical call returns instantly and
editing the body invalidates the entry:

<!-- test:skip reason="illustrative — uses a 5s sleep to stand in for real work" -->
```python
import cash

c = cash.Cash(cache_dir='./my_cache')

@c.cache
def expensive_function(x, y):
    import time
    time.sleep(5)          # stand-in for real work
    return x + y

expensive_function(1, 2)   # first call: ~5 seconds
expensive_function(1, 2)   # second call: instant
```

Results that cost more than ~0.1s persist to disk across processes and restarts;
cheaper ones are kept in memory for the session (see the [cost model](../cost-model.md)).

### Pass DataFrames and arrays — they just work

`functools.lru_cache` refuses a DataFrame (`TypeError: unhashable type: 'DataFrame'`).
Cash hashes by **content**, so unhashable inputs are fine — and two *content-equal*
frames hit the same entry, even if they're different objects:

<!-- test:skip reason="illustrative — references undefined df/weights" -->
```python
@c.cache
def summarize(df, weights):        # a pandas DataFrame and a numpy array
    return (df * weights).sum()

summarize(df, weights)             # runs
summarize(df.copy(), weights.copy())   # instant — same content, different objects
```

Built-in hashers cover pandas, numpy, polars, PyArrow, modin, and dask; register your
own for custom types with [`register_hasher`](../decorator.md).

### Edit a helper — the cache notices

Cash folds the source of the functions you *call* into the key, **transitively**:

<!-- test:skip reason="illustrative — schematic call graph" -->
```python
def clean(x):     ...                      # edit this...
def features(x):  return clean(x) + ...

@c.cache
def pipeline(x):  return features(x)       # ...and pipeline's cache invalidates
```

Editing `clean`, two calls down, invalidates `pipeline`'s cached results.
`lru_cache`, `diskcache.memoize`, and even `joblib.Memory` (which hashes only the
decorated function, not its callees) all miss this. Helpers are resolved within the
module; for cross-module dependencies, name them with `depends_on=`.

### File reads are tracked here too

The automatic file tracking from notebook mode applies in scripts as well — no
annotation required:

<!-- test:skip reason="illustrative — references missing data.csv" -->
```python
@c.cache
def load():
    return pd.read_csv("data.csv")   # change data.csv on disk → recomputes
```

For explicit control there's `@c.cache(file_depends_on="data.csv")` and `depends_on=`.

And the two paths meet: **call a `@cash.cache` function inside a notebook and its
hits show up on that cell's badge** — the same engine, either way.

The full decorator surface — `ttl`, `cache_if`, async support, iterator caching, the
purity analyzer, and `explain()` — is in the [decorator guide](../decorator.md).

---

## Configuration (both paths)

`%cash_on` takes only an optional `ttl=N`; to pick a different backend or cache
directory, construct a `Cash(backend=...)` instance before enabling the magic (in a
notebook), or decorate with its `@c.cache` (in a script). Optional backends — SQLite,
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
