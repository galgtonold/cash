# From notebook to production

Your notebook proves the workflow. Moving to a script for production means swapping `%cash_on` for `@cash.cache`, but most of what you learned in the notebook still applies. This guide walks through the differences.

## The two APIs side by side

| | Notebook (`%cash_on`) | Script (`@cash.cache`) |
|---|---|---|
| **Caching unit** | Statement-level | Function-level |
| **Activation** | Magic command in a cell | Decorator on a definition |
| **Visibility** | Cash badges above each cell's output | `func.cache_info()` / `func.explain(...)` |
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
import numpy as np
import pandas as pd

@cache
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@cache
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(log_amount=np.log1p(df['amount']))
```

### Step 3: Verify auto file tracking

Cash detects `open()`, `pd.read_csv`, `np.load`, and friends inside your function body and records each file's **content fingerprint** automatically. Check that it picked them up with `explain()`, which reports why the next call would hit or miss:

<!-- test:expect-warning reason="load_data reads a file, so cash's impurity advisory fires on first call (it still caches) — realistic here" -->
```python
df = load_data("data.csv")
print(load_data.cache_info())     # {'hits': 0, 'misses': 1, 'hit_rate': 0.0, ...}
print(load_data.explain("data.csv"))   # [HIT] ... — tracked files are fresh

engineer_features(df)             # the second stage, same treatment
print(engineer_features.cache_info())  # {'hits': 0, 'misses': 1, ...}
```

Now edit `data.csv` and call `explain()` again: the reason flips to `file_changed`, which is the proof the read was tracked. If it *doesn't* — because you read from a URL, a database, or your own loader — declare the dependency explicitly with `file_depends_on=` or a custom `DataSource`. See [Custom file sources](custom-file-sources.md).

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

## Running under CI, papermill, or nbconvert

Automation is usually the reason you're moving code out of the notebook in the
first place, and a headless run needs a few different settings.

<!-- claim: cash/notebook/badge_renderer/_text.py:print_text_badge @6739dc99 -->
**Use the text badge.** The HTML badge is an interactive widget; in a headless
run nothing renders it. `%cash_badge print` emits a plain-text summary after each
cell instead — readable in CI logs, in an `nbconvert` artifact, and by a coding
agent parsing the executed notebook. It's ASCII-only on purpose, so a UTF-8
kernel writing to a `cp1252` console won't crash the run.

```python { .nb-cell }
import cash
%cash_badge print
%cash_on
```

**Expect the discovery warning, and ignore it.** Without a live Jupyter Server
there is no notebook file to inspect, so cash emits
`CashNotebookDiscoveryWarning` once and disables upstream dependency tracking.
Statement-level caching still works; what you lose is cross-cell invalidation,
which a top-to-bottom automated run doesn't need. The warning is expected under
papermill, nbconvert, and CI — it's only worth investigating if you see it *in*
JupyterLab or VS Code, where it means a stale runtime.

<!-- claim: cash/core.py:Cash._surface_purity @8dc9b22e -->
**Fail the build on accidental impurity.** `@cash.cache(strict=True)` turns the
purity analyzer's warnings into `CashImpureFunctionError` at first call, so a
teammate caching a side-effecting function breaks CI instead of shipping a
silently wrong cache. Reserve it for functions you've deliberately audited —
see [`strict=`](../../decorator.md#strict-and-assume_safe-purity-gates).

**Decide whether the cache persists between runs.** A fresh container starts
cold, which is correct but slow. Options, in increasing order of setup:

- Point `CASH_CACHE_DIR` at a path your CI caches between runs.
- Share one store across runners with a Redis or S3 backend — but read
  [Sharing a cache](sharing-caches.md) first, because a file-reading *notebook
  statement* still won't hit across machines, and the paths recorded by
  `@cash.cache` have to resolve the same way on each runner.
<!-- claim: cash/__main__.py:cmd_clear @f0272005 -->
- Start clean deliberately with `python -m cash clear --all`, when a run must
  not be influenced by earlier state. The `--all` matters: bare
  `python -m cash clear` takes no default target, prints a usage line and
  **exits 1**, which in a CI step is a failed build rather than a clean start.

## Migration checklist

- [ ] Identify expensive notebook statements
- [ ] Wrap each in a function and decorate with `@cache` / `@app.cache`
- [ ] Verify tracked inputs with `explain()` — edit a file, confirm the reason becomes `file_changed`
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
