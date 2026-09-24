# Moving to a module

!!! info "Applies to: notebook"
    Notebook users taking their work into a `.py` module, a script or a scheduled
    job. This page is the bridge to the [decorator guide](../../decorator.md).

In a notebook, `%cash_on` caches every statement. Outside a notebook there are no
cells, so you tell cash which **functions** to cache with `@cash.cache`. The
same engine does the work: it tracks the files a function reads and the
functions it calls, and stores results in the same kind of `.cash` folder.

## The steps

1. **Find the expensive statements.** The badge shows the time of each; the
   slow loads, joins and model fits are what you want to cache.
2. **Put each in a function** in a module. Pass in everything it needs as
   arguments and return the result, instead of reading and changing notebook
   variables.
3. **Decorate the function** with `@cash.cache`.
4. **Call it** from a script, a job, or back in the notebook.

```python
# pipeline.py
import cash
import pandas as pd

@cash.cache
def load(path):
    return pd.read_csv(path)

@cash.cache
def add_features(df):
    return df.assign(amount_x2=df["amount"] * 2)
```

```python
df = load("data.csv")              # First call: cache miss
features = add_features(df)        # First call: cache miss
df = load("data.csv")              # Second call: cache hit
```

Edit `data.csv` and `load` runs again, because cash tracks the file's contents.
Edit `add_features` and it runs again, because its code is part of the key.

Called from a notebook with `%cash_on`, a decorated function's hits and misses
show on the cell's badge, so you can move code into the module step by step.

## What does not carry over

- **`# @cash:` comments do not steer a decorated function.** Use decorator
  arguments instead: `ttl=` for expiry, `file_depends_on=` for a file cash cannot
  see being read. The one comment the decorator reads is
  `# @cash:assume-safe`, which waives a purity finding for one line.
- **Every result is written to disk**, however quick it was to compute. The
  0.1 s threshold and the [cost model](../../cost-model.md) apply to notebook
  statements only.
- **The function is the unit.** Its key is its code, the functions it calls,
  its arguments and the globals it reads. Statements inside it are not cached
  one by one.
- **Side effects are not refused.** A decorated function that writes a file or
  sends a request is still cached, with a warning at its first call; a hit skips
  the effect. Keep such effects outside cached functions.
- **Prints are not replayed.** A cached call returns its value without printing
  again, where a restored notebook statement replays its output.
- **No badge.** Run a script with `CASH_SUMMARY=1` for a summary at exit, or call
  `f.explain(...)` on a function.
- **A different cache folder, maybe.** A script caches at its project root (the
  nearest folder above it with `pyproject.toml` or `.git`); a notebook caches in its working
  directory. To share one folder, set `[tool.cash] cache_dir`.

## Next

- [Decorator guide](../../decorator.md): every argument, what invalidates a
  result, and how to see what cash did.
- <a id="running-as-a-service-or-a-worker-pool"></a>[Deploying](deploying.md): services, worker pools, CI and shared backends.
