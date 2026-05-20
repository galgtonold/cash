# Quick Start

## Notebook Caching (Recommended)

The easiest way to use Cash is in a Jupyter notebook:

### Step 1: Enable Cash

```python
import cash

# Enable auto-caching for all subsequent cells
%cash_on
```

### Step 2: Write Code Normally

```python
import pandas as pd
import numpy as np

# This read_csv call will be cached after first run
df = pd.read_csv('large_dataset.csv')
print(f"Loaded {len(df)} rows")
```

### Step 3: Re-run and See the Magic

Run the cell again — the `read_csv` result loads instantly from cache!

## What Gets Cached?

Cash caches at the **statement level**, not the cell level. This means:

```python
# Statement 1: cached independently
df = pd.read_csv('data.csv')

# Statement 2: cached independently  
df['new_col'] = df['old_col'] * 2

# Statement 3: cached independently
result = df.groupby('category').sum()
```

If you change Statement 2, only Statement 2 and 3 re-execute. Statement 1 stays cached!

## Debug Mode

Want to see what Cash is doing?

```python
%cash_debug on
```

You'll see status badges for each statement:

- ⚙️ **EXECUTED** — Ochre chip. Cash ran the statement and cached the result (first run).
- ⚡ **CACHED** — Green chip. The result was restored from cache (cache hit!).
- ⏩ **SKIPPED** — Green chip. Already in memory from an earlier statement, no work needed.

## Decorator-based Caching

For caching function results outside notebooks:

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

# Check stats
print(expensive_function.cache_info())
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, 'total_time_saved': 0.001}

# Clear cache for this function
expensive_function.cache_clear()
```

### File Dependency Shorthand

```python
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

```python
# File-based (default)
%cash_on

# SQLite backend
%cash_on --backend sqlite

# Custom cache directory
%cash_on --cache-dir /tmp/my_cache
```

### Selective Caching

```python
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
- `open()` (built-in), `io.open()` (pathlib)
- `joblib.load()`, `pickle.load()`, `json.load()`

## Statement-Level Annotations

Control caching for individual statements with comment annotations:

```python
# @cash:no-cache
result = api.fetch_data()       # Always re-executed

# @cash:ttl=300
prices = get_stock_prices()     # Cached for 5 minutes

# @cash:persist
model = train_model(data)       # Always saved to disk

# @cash:allow-random
samples = np.random.randn(100)  # Suppress randomness warning
```

## Next Steps

- [API Reference](../api_reference.md) — All magic commands
- [Technical Architecture](../notebook_caching_technical.md) — How it works under the hood
- [Migration Guide](../migration_guide.md) — Moving from other caching solutions

## Next: read the badge

Every cell now has a Cash badge. The badge tells you what was restored, what re-ran, and *why* — including the one or two things you can change to make a stubborn cell cache. See [Reading the Cash badge](../badges.md).
