# From notebook to production

Your notebook proves the workflow. Moving to a script for production means swapping `%cash_on` for `@cash.cache`, but most of what you learned in the notebook still applies. This guide walks through the differences.

## The two APIs side by side

| | Notebook (`%cash_on`) | Script (`@cash.cache`) |
|---|---|---|
| **Caching unit** | Statement-level | Function-level |
| **Activation** | Magic command in a cell | Decorator on a definition |
| **Visibility** | Cash badges above each cell | `func.cache_info()` / `func.explain(...)` |
| **Everything else** | File tracking, purity analysis, lineage, TTL, backend choice — identical | Same |

The transition is mechanical: lift each cached statement into a function, decorate it, call it. The cache layer underneath doesn't change. Three things differ — caching unit (statement → function), activation (`%cash_on` → `@cash.cache`), visibility (badges → `cache_info()`). Everything else carries over: auto file tracking ([escape hatch](custom-file-sources.md) for non-standard access), purity analysis ([details](purity-decorators.md)), TTL and other annotations ([details](controlling-cache-behavior.md)), backend choice ([details](choosing-a-backend.md)).

## Step-by-step migration

### Step 1: Identify expensive operations

Open the notebook and look for the statements with long execution times — heavy I/O, model fits, joins on big frames. Those are your function boundaries.

```python { .nb-cell }
%cash_on
# test:inject: import pandas as pd

df = pd.read_csv('data.csv')              # I/O
features = engineer_features(df)           # compute
model = train_model(features)              # very expensive
```

### Step 2: Wrap in functions

Lift each expensive statement into a function and decorate it. For the full decorator reference (parameters, wrapper methods, gotchas), see [`@cash.cache`](../../decorator.md).

```python
# pipeline.py
from cash import cache
import pandas as pd

@cache
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@cache
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(log_amount=np.log1p(df['amount']))
```

### Step 3: Verify auto file tracking

Cash detects `open()`, `pd.read_csv`, `np.load`, and friends inside your function body and folds the file's `(mtime, size)` into the cache key automatically. Check that it picked them up:

<!-- test:expect-warning reason="load_data reads a file, so cash's impurity advisory fires on first call (it still caches) — realistic here" -->
```python
load_data("data.csv")
print(load_data.cache_info())
# {'hits': 0, 'misses': 1, 'hit_rate': 0.0, 'total_time_saved': 0.0,
#  'tracked_files': ['data.csv'], 'warnings': []}
```

If `tracked_files` is empty but you *do* read external state — for instance, you read from a URL, a database, or your own loader function — declare it explicitly with `file_depends_on=` or a custom `DataSource`. See [Custom file sources](custom-file-sources.md).

### Step 4: Choose a backend

The default (tiered RAM + file under `./.cash/`) is fine for most scripts. Production deployments usually want explicit control. See [Choosing a backend](choosing-a-backend.md) for selection logic; quick sketches:

```python
from cash import Cash, FileBackend, SQLiteBackend
from cash.backends import RedisBackend

app = Cash(backend=FileBackend("/var/cache/myapp", max_size_bytes=10_000_000_000))   # size-capped disk
app = Cash(backend=SQLiteBackend("cache.db"))                                        # single-file, transactional
app = Cash(backend=RedisBackend(host="redis.internal", port=6379, db=0))             # shared across hosts
```

### Step 5: Build a pipeline script

<!-- test:expect-warning reason="the pipeline's extract/transform steps read files, so cash's impurity advisory is expected (it still caches)" -->
```python
# run_pipeline.py
import pandas as pd
from cash import Cash, FileBackend

app = Cash(backend=FileBackend("./pipeline_cache"))

@app.cache
def extract():
    return pd.read_csv("data/raw.csv")          # auto-tracked

@app.cache
def transform(df):
    df = df.dropna()
    df['amount_norm'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    return df

@app.cache
def train(df):
    from sklearn.linear_model import LogisticRegression
    X, y = df.drop('target', axis=1), df['target']
    return LogisticRegression().fit(X, y)

if __name__ == "__main__":
    model = train(transform(extract()))
    print(extract.cache_info())
    print(transform.cache_info())
    print(train.cache_info())
```

First run: everything computes and caches. Subsequent runs: if `data/raw.csv` hasn't changed, every stage restores and `cache_info()` shows hits with non-zero `total_time_saved`.

### Step 6: Inspect what the cache is doing

`cache_info()` covers routine monitoring; when something looks off (unexpected misses, stale results), reach for `func.explain(*args)`. See [Debugging and monitoring](debugging-and-monitoring.md).

## Notebook ↔ decorator bridge

You don't have to pick one. A `@cash.cache`-decorated function called from a `%cash_on` notebook shows up in the cell badge with condensed metrics, so you can keep iterating in the notebook while the production wrappers stay in the module:

```python { .nb-cell }
%cash_on
from pipeline import train          # @cash.cache lives in the module

# test:inject: features = {"stub": True}
model = train(features)              # badge reports the decorator hit
```

## Migration checklist

- [ ] Identify expensive notebook statements
- [ ] Wrap each in a function and decorate with `@cache` / `@app.cache`
- [ ] Verify `cache_info()['tracked_files']` lists the inputs you expect
- [ ] Choose a backend for the production environment
- [ ] Add `cache_info()` to logging or end-of-run output
- [ ] Test invalidation: change an input, rerun, confirm recomputation

If applicable:

- [ ] Add `file_depends_on=` for non-standard file access (URLs, custom loaders) — see [Custom file sources](custom-file-sources.md)
- [ ] Set `ttl=` for time-sensitive data

## Related

- [`@cash.cache`](../../decorator.md) — full decorator reference
- [Choosing a backend](choosing-a-backend.md)
- [Custom file sources](custom-file-sources.md)
- [Debugging and monitoring](debugging-and-monitoring.md)
