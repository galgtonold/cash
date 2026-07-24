# Quick Start

!!! tip "Prefer to just try it?"
    Run cash in your browser with no install —
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb) — then come back here to wire it into your own notebooks.

## Notebook Caching (Recommended)

The easiest way to use Cash is in a Jupyter notebook:

### Step 1: Enable Cash

```python { .nb-cell }
import cash

# Enable auto-caching for all subsequent cells
%cash_on
```

`import cash` auto-registers the IPython magics, so `%load_ext cash` is not required.

!!! tip "Skip the boilerplate"
    Run `cash autoload on` once and every new IPython/Jupyter kernel starts with
    `import cash` plus `%cash_on` already executed. Use `%cash_off` to opt a single
    notebook out, or `cash autoload off` to remove the hook. See the
    [CLI reference](../cli.md) for the `--mode=available` variant if you'd
    prefer to keep `%cash_on` explicit.

### Step 2: Write Code Normally

<!-- test:skip reason="illustrative — references missing large_dataset.csv" -->
```python { .nb-cell }
import pandas as pd
import numpy as np

# This read_csv call will be cached after first run
df = pd.read_csv('large_dataset.csv')
print(f"Loaded {len(df)} rows")
```

### Step 3: Re-run and See the Magic

Run the cell again — the `read_csv` result loads instantly from cache. The badge that appeared above the cell on the first run will switch from `EXECUTED` (ochre) to `RESTORED` (green):

**First run** — Cash ran the statement and cached the result:

<iframe class="cash-badge" src="/_badges/quickstart_first_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Second run** — Cash restored the value from cache instead of recomputing:

<iframe class="cash-badge" src="/_badges/quickstart_second_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

See [Reading the Cash badge](../badges.md) for the full anatomy and every status.

## What Gets Cached?

Cash caches at the **statement level**, not the cell level. This means:

<!-- test:skip reason="illustrative — references undefined column 'old_col'/'category' on stub DataFrame" -->
```python { .nb-cell }
# Statement 1: cached independently
df = pd.read_csv('data.csv')

# Statement 2: cached independently  
df['new_col'] = df['old_col'] * 2

# Statement 3: cached independently
result = df.groupby('category').sum()
```

If you change Statement 2, only Statement 2 and 3 re-execute. Statement 1 stays cached!

## Decorator-based Caching

For caching function results outside notebooks:

<!-- test:skip reason="ends with cache_clear() which resets stats and breaks inferred hit/miss claims" -->
```python
import cash

c = cash.Cash(cache_dir='./my_cache')

@c.cache
def expensive_function(x, y):
    import time
    time.sleep(5)  # Simulate expensive computation
    return x + y

# First call: takes 5 seconds
result = expensive_function(1, 2)

# Second call: instant!
result = expensive_function(1, 2)

# Check stats — now includes recent warnings for this function.
print(expensive_function.cache_info())
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5,
#  'total_time_saved': 5.001, 'warnings': []}

# Diagnose why a call would hit or miss — pure introspection.
print(expensive_function.explain(1, 2))
# [HIT] __main__.expensive_function — hit
#   cache_key: ...
#   cached_at: ..., execution_time_saved: 5.001

# Clear cache for this function
expensive_function.cache_clear()
```

Full walkthrough including `ttl`, `cache_if`, async support,
`strict`/`assume_safe` for the purity analyzer, and iterator caching:
see the [decorator guide](../decorator.md).

### File Dependency Shorthand

```python
# test:inject: import cash; c = cash.Cash()
@c.cache(file_depends_on="data.csv")
def load_data():
    return pd.read_csv("data.csv")

# Cache auto-invalidates when data.csv changes on disk
```

### Custom Type Hashers

Cash has built-in hashing for pandas, numpy, polars, PyArrow, modin, and dask objects. For custom types:

```python
c.register_hasher(
    MyModel,
    lambda model: model.fingerprint()
)
```

## Configuration

### Change Backend

`%cash_on` only accepts an optional `ttl=N` argument — it does not take backend or cache-dir flags. To pick a different backend or cache directory, construct a `Cash(backend=...)` instance programmatically before enabling the magic. See [Configuration](configuration.md) for a full example.

### Selective Caching

<!-- test:skip reason="illustrative — references missing data.csv columns and requires matplotlib" -->
```python { .nb-cell }
# Cache a single cell
%%cash
df = pd.read_csv('data.csv')

# Skip caching for a specific statement
# @cash:no-cache
import matplotlib.pyplot as plt
plt.plot(df['x'], df['y'])
```

## File Dependency Tracking

Cash automatically tracks file reads:

```python
# If 'data.csv' changes, this re-executes automatically
df = pd.read_csv('data.csv')
```

Supported file operations:

- `pd.read_csv()`, `pd.read_excel()`, `pd.read_parquet()`, and all `pd.read_*()` functions
- `np.load()`, `np.loadtxt()`, `np.genfromtxt()`, `np.fromfile()`
- `polars.read_csv()`, `polars.read_parquet()`, `polars.scan_csv()`, and more
- `open()` and `io.open()` (the built-in file open)
- `joblib.load()`, `pickle.load()`, `json.load()`

## Statement-Level Annotations

Control caching for individual statements with comment annotations:

<!-- test:skip reason="illustrative — references undefined api/get_stock_prices/train_model/data" -->
```python { .nb-cell }
# @cash:no-cache
result = api.fetch_data()       # Always re-executed

# @cash:ttl=300
prices = get_stock_prices()     # Cached for 5 minutes

# @cash:persist
model = train_model(data)       # Always saved to disk

# @cash:allow-random
samples = np.random.randn(100)  # Suppress randomness warning
```

## Explore Cash commands

Run `%cash_help` in any notebook for the full list of magics. A few you'll use often:

- `%cash_stats` — hit/miss counts and net time saved this session (gross saving minus cash's own overhead)
- `%cash_debug on` — log detailed cache decisions to the kernel output
- `%cash_badge print` — switch to text-only badges (no HTML)

See [Debugging and monitoring](../tutorials/feature-guides/debugging-and-monitoring.md) for the full diagnostic workflow.

## What's next?

- **[Why Cash?](../why-cash.md)** — Understand when Cash helps and when it doesn't.
- **[Decorator guide](../decorator.md)** — Use `@cash.cache` in scripts and modules.
- **[Reading the Cash Badge](../badges.md)** — Decode the chips that appear above cells.
- **Tutorials by feature:** [Caching class methods](../tutorials/feature-guides/caching-class-methods.md), [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md), [Debugging and monitoring](../tutorials/feature-guides/debugging-and-monitoring.md).
- **Tutorials by use case:** [Data science](../tutorials/use-cases/data-science.md), [LLM API calls](../tutorials/use-cases/llm-api-calls.md), [Data engineering](../tutorials/use-cases/data-engineering.md), [Scientific computing](../tutorials/use-cases/scientific-computing.md).
