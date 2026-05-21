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

```python { .nb-cell }
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

### Tip: Skip the boilerplate in every notebook

If you'd rather not type those two lines at the top of every notebook, install the autoload hook once:

```bash
cash autoload on
```

Now every new IPython/Jupyter kernel starts with cash imported and `%cash_on` already executed. Use `%cash_off` to opt a single notebook out, or `cash autoload off` to remove the hook entirely. See the [CLI reference](../cli.md) for the `--mode=available` variant if you'd prefer to keep `%cash_on` explicit.

## Step 3: Run Some Code

```python { .nb-cell }
# Cell 2
import pandas as pd

# Cash automatically tracks file dependencies
# If sales.csv changes, this cell recomputes
df = pd.read_csv('sales.csv')
print(f"Loaded {len(df)} rows")
```

```python { .nb-cell }
# Cell 3
summary = df.describe()
print(summary)
```

On first run, you'll see ochre **EXECUTED** badges — Cash ran each statement and cached the results. (Internally the cache status is `COMPUTED`; the chip label reads `EXECUTED`.) The badge looks something like this:

<iframe class="cash-badge" src="/_badges/status_computed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

## Step 4: Experience the Cache

Now **re-run the notebook** (Kernel → Restart & Run All, or click Run All).

This time you'll see green **CACHED** badges with timing — the result was restored from cache (internal status: `RESTORED`):

<iframe class="cash-badge" src="/_badges/status_restored.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

The results loaded instantly from cache instead of being recomputed. See [Reading the Cash badge](../badges.md) for the full anatomy.

## Step 5: Change Something

Modify Cell 3 slightly — for example, round the output:

```python { .nb-cell }
# Cell 3 (modified)
summary = df.describe().round(2)
print(summary)
```

Re-run the notebook:

- **Cell 2**: **CACHED** (green chip) — unchanged, loaded from cache
- **Cell 3**: **EXECUTED** (ochre chip) — code changed, recomputed

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
