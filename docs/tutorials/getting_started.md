# Tutorial: Getting Started with Cash

This tutorial walks you through using Cash for the first time. By the end, you'll understand how Cash transparently caches your notebook computations and saves you time on every re-run.

## Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab
- Some data to work with (we'll create sample data below)

## Step 1: Install Cash

```bash
pip install cash-lib
```

## Step 2: Enable Cash in Your Notebook

Create a new notebook and add this to the first cell:

```python
import cash
%cash_on
```

`import cash` auto-registers the IPython magics, so a second `%load_ext cash` line is not needed. (`%load_ext cash` also works if you prefer the IPython convention — and if you also want `@cash.cache` for decorator-style caching, you'll need `import cash` anyway, so it's the recommended entry point.)

You'll see:

```
✅ Cash enabled. Your computations will be cached automatically.
Run %cash_help for available commands.
```

That's it — every statement you execute from now on is automatically cached.

## Step 3: Run Some Code

```python
# Cell 2
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10000, freq='h'),
    'value': np.random.randn(10000).cumsum(),
    'category': np.random.choice(['A', 'B', 'C'], 10000)
})
print(f"Created DataFrame with {len(df)} rows")
```

```python
# Cell 3
summary = df.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max']
}).round(2)
print(summary)
```

On first run, you'll see ochre **⚙️ EXECUTED** badges — Cash ran each statement and cached the results. (Internally the cache status is `COMPUTED`; the chip label reads `EXECUTED`.)

## Step 4: Experience the Cache

Now **re-run the notebook** (Kernel → Restart & Run All, or click Run All).

This time you'll see green **⚡ CACHED** badges with timing — the result was restored from cache (internal status: `RESTORED`):

```
⚡ CACHED (Saved 0.15s)   df = pd.DataFrame({...})
⚡ CACHED (Saved 0.08s)   summary = df.groupby(...)
```

The results loaded instantly from cache instead of being recomputed.

## Step 5: Change Something

Modify Cell 3 to add a filter:

```python
# Cell 3 (modified)
summary = df[df['value'] > 0].groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max']
}).round(2)
print(summary)
```

Re-run the notebook:

- **Cell 2**: ⚡ **CACHED** (green chip) — unchanged, loaded from cache
- **Cell 3**: ⚙️ **EXECUTED** (ochre chip) — code changed, recomputed

Cash knew that only Cell 3 changed, so only Cell 3 recomputed.

## Step 6: Explore Cash Commands

Run `%cash_help` to see all available commands:

```python
%cash_help
```

Useful commands to try:

```python
%cash_stats          # See cache hit/miss stats and time saved
%cash_debug on       # Show detailed caching decisions
%cash_badge print    # Switch to text-only badges (no HTML)
```

## What's Next?

- **[Data Science Workflows](data_science_workflows.md)** — Real-world patterns with pandas, sklearn, and large datasets
- **[Configuration Guide](../getting-started/configuration.md)** — Customize backends, TTLs, and storage
- **[Production Transition](production_transition.md)** — Move from notebooks to production scripts
