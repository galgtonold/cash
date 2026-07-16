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
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
transactions['date'] = pd.to_datetime(transactions['date'])
```

Each statement is cached independently. Change the `dropna` logic and only that statement (plus its dependents) recomputes.

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

```python { .nb-cell }
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = ['total_spend', 'avg_spend', 'tx_count']
X = df[features].fillna(0)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
# @cash:persist
model.fit(X_train, y_train)
```

Cash caches `model.fit(...)` — the fitted estimator is the cached value, restored *in place* onto `model` on a hit, so any alias of `model` (`backup = model`) sees the fit too. An expensive fit is written to disk and restored after a kernel restart automatically; `# @cash:persist` on the `fit` line forces it to disk even when the cost model would otherwise keep it in memory — so a restart restores it even if training took ten minutes. (Annotations attach to the statement directly below them, so `# @cash:persist` goes on the `fit` line, not the constructor.)

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
- **Model training with `@cash:persist`.** The trained estimator goes to disk, survives kernel restarts, and is restored on import — see [Smart Persistence](../feature-guides/smart-persistence.md).
- **Notebook ↔ kernel-restart workflow.** When the kernel dies (OOM, package upgrade, accidental restart), the disk cache is still there. Re-running the notebook rehydrates everything Cash decided was worth persisting.

## Where to be careful in DS workflows

- **Randomness without a seed.** `df.sample(100)` or `np.random.randn(...)` without a seed produces different values per call. Cash still caches these — it warns rather than refusing, and only for calls it can see are RNG calls (`np.random.*`, `random.*`, `torch.*`); a draw hidden behind a method like `df.sample(100)` passes silently. Seed the RNG when you want reproducibility. See [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md) for `@cash:allow-random`.
- **In-place mutations to DataFrames.** `df.sort_values(..., inplace=True)` and friends mutate without returning. Cash can't detect the mutation reliably. Either return new frames (`df = df.sort_values(...)`) or mark the helper with `@stateful` — see [Purity Decorators](../feature-guides/purity-decorators.md).
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
