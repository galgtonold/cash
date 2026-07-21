# Data Science Workflows

Data science iteration is the killer use case for Cash. You load data once, transform it twenty different ways, and re-run the notebook constantly. Without caching, every iteration pays the full I/O + compute cost. With Cash, only the parts that changed re-run.

## The iteration-loop value proposition

The notebook workflow is: load a CSV, clean it, engineer features, train, evaluate, *change one line, repeat*. The change is usually small — a different aggregation, an extra feature, a tweaked hyperparameter — but in a vanilla notebook every cell from the change downward (and often everything above it after a kernel restart) reruns.

Cash watches each statement. When you tweak the merge on line 4 of cell 4, the groupby on line 1 of cell 4 stays cached, the CSV loads above it stay cached, and only the merge and its downstream consumers recompute. The expensive parts — the 30s parquet read, the 10s aggregation — run once across an entire afternoon of iteration.

Cash also picks the right serializer per type. For pandas DataFrames it writes Parquet automatically (`src/cash/backends/serialization.py:71-90`), so loading a 500MB frame from cache is a fast columnar read, not a pickle balloon.

## Concrete walkthrough: customer churn analysis

You're building a churn model. Load, clean, featurize, train, evaluate.

### Cell 1: Setup

```python { .nb-cell }
import cash
%cash_on
```

### Cell 2: Load Data

```python { .nb-cell }
import pandas as pd

customers = pd.read_csv('customers.csv')
transactions = pd.read_csv('transactions.csv')

print(f"Customers: {len(customers)}, Transactions: {len(transactions)}")
```

Cash detects `pd.read_csv` automatically — change `customers.csv` and the cell re-runs. The badge spells out which file invalidated the cache:

<iframe class="cash-badge" src="/_badges/miss_file_changed_customers.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

For non-pandas file readers (HDF5, custom binary formats, anything Cash doesn't intercept by default), see [Custom File Sources](../feature-guides/custom-file-sources.md).

### Cell 3: Data Cleaning

```python { .nb-cell }
customers = customers.dropna(subset=['email', 'signup_date'])
customers = customers.assign(signup_date=pd.to_datetime(customers['signup_date']))
transactions = transactions.assign(date=pd.to_datetime(transactions['date']))
```

Each statement is cached independently. Change the `dropna` logic and only that statement (plus its dependents) recomputes.

!!! tip "Use `.assign()`, not `df['col'] = ...`, for columns worth caching"
    `df['col'] = ...` mutates the frame **in place** and returns nothing, so Cash can't cache it — the statement recomputes on every run, forever. `df = df.assign(col=...)` produces a new frame and binds it to a name, which is an ordinary cacheable statement.

    This is the same in-place rule as `sort_values(inplace=True)` below, but it's easy to miss because subscript-assign doesn't *look* like a mutating method call. On a real ETL it's worth having: a tester measured roughly a fifth of their runtime going to columns that could never cache, purely because of this idiom.

    It only matters for expensive columns. `df['flag'] = 0` is not worth restructuring.

### Cell 4: Feature Engineering

```python { .nb-cell }
import numpy as np

tx_features = transactions.groupby('customer_id').agg(
    total_spend=('amount', 'sum'),
    avg_spend=('amount', 'mean'),
    tx_count=('amount', 'count'),
    last_purchase=('date', 'max'),
).reset_index()

df = customers.merge(tx_features, on='customer_id', how='left')
```

### Cell 5: Model Training

**Wrap training in a function and decorate it with `@cash.cache`.** This is the recommended way to cache model training, and the only one Cash recommends for expensive fits:

```python { .nb-cell }
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = ['total_spend', 'avg_spend', 'tx_count']
X = df[features].fillna(0)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@cash.cache(assume_safe=True)   # see the note below on assume_safe
def train_model(X_train, y_train, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train, n_estimators=100)
```

!!! note "Why `assume_safe=True` on a training function"
    `model.fit(X, y)` mutates `model` in place and its return value is discarded, so cash's purity check flags the function with a `CashImpurityWarning` — even though caching is perfectly correct here (the fitted `model` is the return value). `assume_safe=True` acknowledges you've audited it and silences the warning. Add it to any *train-and-return-the-model* function. (Plain `@cash.cache` still caches correctly; it just prints the warning.)

The decorator caches on the arguments you **pass** and on the function's **body**, so the fitted model comes back from cache when they're unchanged — including after a kernel restart — and re-trains when either differs. The value is a plain return value, so there's no ambiguity about what got cached or which name it lands on.

!!! tip "Hyperparameters in default values are keyed too"
    The cache key covers the arguments at the call site, the function body, **and** each parameter's *default value*. Editing `def train_model(..., n_estimators=100)` to `=200` and re-running re-trains and returns a 200-tree model — the two defaults are distinct cache entries, so reverting to `=100` restores the original from cache rather than re-fitting.

    Passing hyperparameters explicitly at the call site (as above) is still the clearer habit — a sweep reads better when the value is visible where it varies — but relying on a default is not a correctness trap.

!!! warning "A bare `model.fit(X, y)` is **not** cached"
    A statement like `model.fit(X_train, y_train)` mutates `model` in place and returns nothing. Cash does **not** cache it — the badge reads `NOT CACHED` with an *In-place mutation* reason, and the fit re-executes on every run.

    That's deliberate: skipping it costs nothing, because the model is never serialised, so a fit that keeps missing can't cost more than it saves. The real `.fit()` runs every time, so `model` itself is genuinely fitted.

    **It does not make aliases safe.** `backup = model` is an ordinary assignment, so Cash caches *that* statement — and a restore can rebind `backup` to a deserialised copy taken before the fit. On a warm re-run `backup is model` can be `False` with `backup` unfitted, which is not what a plain kernel does. Don't keep a second name for a model you fit in place; see the identity caveat below.

    Assignment forms are ordinary statements and **do** cache with no directive:

    ```python
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    ```

    Prefer `@cash.cache` over both — it's the path with the strongest guarantees.

<details markdown="1">
<summary>Advanced: <code># @cash:cache-fit</code> — opting a bare fit in</summary>

If you specifically want a bare in-place fit cached, ask for it per statement:

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
# @cash:cache-fit
# @cash:persist
model.fit(X_train, y_train)
```

Cash then treats the fitted estimator as the statement's cached value: the constructor's lineage plus `X_train`/`y_train` pin the key, and a hit transfers the fitted state back onto `model`. `# @cash:persist` forces it to disk so a restart restores it even if training took ten minutes. (Annotations attach to the statement directly below them, so both go on the `fit` line, not the constructor.)

**Know the identity caveat before you use it.** Restores happen per statement, so Cash cannot promise your object graph survives one:

- On a hit, `model` may be **rebound** to a restored object rather than updated in place. An alias taken earlier (`backup = model`) can be left pointing at the pre-fit, unfitted object — a wrong result that looks like a right one.
- This isn't fixable by trying harder at the `fit` statement: on a warm run-all the *constructor* statement's own cache hit rebinds `model` before the fit's transfer runs, so the aliasing is already broken upstream.
- The estimator gate is a duck-type (`fit` + `get_params`), so it admits non-sklearn estimators (xgboost, lightgbm, your own class). Each has its own pickle contract, and some never restore cleanly — which means paying serialisation on every run for nothing.

Use it when the receiver has no aliases and you've confirmed the badge actually reads `RESTORED` on a warm re-run. Otherwise use `@cash.cache`.

</details>

### Cell 6: Evaluation

```python { .nb-cell }
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### The payoff

Switch the merge in Cell 4 from `how='left'` to `how='inner'` and re-run the notebook:

- Cells 2-3 **RESTORED** from cache.
- In Cell 4, the groupby is **RESTORED** (code didn't change), only the merge **RECOMPUTES**.
- Cells 5-6 **RECOMPUTE** (new `df` → new model → new evaluation).

<iframe class="cash-badge" src="/_badges/workflows_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

Without Cash: 30s of CSV reloads + aggregations on every iteration. With Cash: everything before your change loads in milliseconds.

## Where Cash specifically helps in DS

- **Loading large CSV/parquet.** Pandas readers are intercepted automatically and DataFrames serialize as Parquet, so cached restores skip both the disk read and the pandas parsing path.
- **Feature engineering iteration.** Each statement is cached independently, so tweaking one feature doesn't invalidate the dozen others above it.
- **Model training via `@cash.cache`.** Wrap the fit in a function that *returns* the model. The trained estimator goes to disk, survives kernel restarts, and re-trains only when the data or hyperparameters change.
- **Notebook ↔ kernel-restart workflow.** When the kernel dies (OOM, package upgrade, accidental restart), the disk cache is still there. Re-running the notebook rehydrates everything Cash decided was worth persisting.

## Where to be careful in DS workflows

- **Randomness without a seed.** `df.sample(100)` or `np.random.randn(...)` without a seed produces different values per call. Cash still caches these — it warns rather than refusing, and only for calls it can see are RNG calls (`np.random.*`, `random.*`, `torch.*`); a draw hidden behind a method like `df.sample(100)` passes silently. Seed the RNG when you want reproducibility. See [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md) for `@cash:allow-random`.
- **In-place mutations to DataFrames.** `df.sort_values(..., inplace=True)` and friends mutate without returning. Cash can't detect the mutation reliably. Either return new frames (`df = df.sort_values(...)`) or mark the helper with `@stateful` — see [Purity Decorators](../feature-guides/purity-decorators.md).
- **`df['col'] = ...` never caches.** Same rule, easier to miss: subscript-assign mutates in place, so the statement recomputes every run. Use `df = df.assign(col=...)` for any column expensive enough to be worth caching. See the tip in Cell 3.
- **`datetime.now()` in transforms.** Wall-clock reads inside a cached statement bake the current time into the cache. Pull the timestamp outside the cached path, or skip caching for that statement.

## Tips for data science workflows

1. **Separate data loading from transforms.** Loading a 500MB CSV takes 10s. Put it in its own cell so it stays cached when you tweak transforms downstream.
2. **Use `@cash:persist` for trained models.** Anything that takes >10s and you want to survive a kernel restart belongs on disk, not just in the in-memory cache.
3. **Use seeded RNG for reproducible sampling.** `np.random.seed(42)` before `df.sample(...)` lets Cash treat the result as deterministic and cache it.

## Related

- [Purity Decorators](../feature-guides/purity-decorators.md) — when Cash can't auto-detect that your transform is safe.
- [Smart Persistence](../feature-guides/smart-persistence.md) — when Cash decides to write to disk.
- [Choosing a Backend](../feature-guides/choosing-a-backend.md) — for larger-than-memory datasets.
- [Debugging and Monitoring](../feature-guides/debugging-and-monitoring.md) — inspecting why a cell hit or missed.
- [Production Transition](../feature-guides/production-transition.md) — when your notebook becomes a script.
