# Tutorial: Data Science Workflows with Cash

This tutorial covers real-world data science patterns where Cash shines: loading large datasets, exploratory analysis, feature engineering, and model training.

## Scenario: Customer Churn Analysis

You're building a churn prediction model. The workflow involves loading data, cleaning, feature engineering, training, and evaluation — and you'll iterate on each step many times.

### Cell 1: Setup

```python { .nb-cell }
import cash
%cash_on
```

### Cell 2: Load Data

```python { .nb-cell }
import pandas as pd

# Cash automatically tracks file dependencies
# If customers.csv changes, this cell recomputes
customers = pd.read_csv('customers.csv')
transactions = pd.read_csv('transactions.csv')

print(f"Customers: {len(customers)}, Transactions: {len(transactions)}")
```

!!! tip "File Tracking"
    Cash intercepts `pd.read_csv()`, `pd.read_parquet()`, `pd.read_excel()`,
    and many other file-reading functions. If the file's content changes,
    all dependent statements automatically recompute.

When the CSV changes between runs, the badge above this cell shows a `COMPUTED` row whose miss reason names the file:

<iframe class="cash-badge" src="/_badges/miss_file_changed_customers.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

### Cell 3: Data Cleaning

```python { .nb-cell }
# Multiple statements — each cached independently
customers = customers.dropna(subset=['email', 'signup_date'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
transactions['date'] = pd.to_datetime(transactions['date'])
```

If you change the `dropna` logic, only that statement and its dependents recompute. The `pd.to_datetime` calls stay cached.

### Cell 4: Feature Engineering

```python { .nb-cell }
import numpy as np

# Aggregate transaction features per customer
tx_features = transactions.groupby('customer_id').agg(
    total_spend=('amount', 'sum'),
    avg_spend=('amount', 'mean'),
    tx_count=('amount', 'count'),
    last_purchase=('date', 'max'),
    days_since_first=('date', lambda x: (x.max() - x.min()).days)
).reset_index()

# Merge with customer data
df = customers.merge(tx_features, on='customer_id', how='left')
df['days_since_last'] = (pd.Timestamp.now() - df['last_purchase']).dt.days
```

### Cell 5: Model Training

```python { .nb-cell }
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

features = ['total_spend', 'avg_spend', 'tx_count', 'days_since_last']
X = df[features].fillna(0)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# @cash:persist
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

!!! tip "@cash:persist"
    The `@cash:persist` annotation forces the model to be saved to disk.
    If the kernel restarts, the trained model is restored from disk
    instead of being retrained — even if training took 10 minutes.

### Cell 6: Evaluation

```python { .nb-cell }
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## The Iteration Loop

Here's where Cash pays off. Suppose you tweak only the merge step in Cell 4 — for example, switching from a left join to an inner join to drop customers without transactions:

```python { .nb-cell }
# Cell 4 (modified — only the merge line changed)
tx_features = transactions.groupby('customer_id').agg(
    total_spend=('amount', 'sum'),
    avg_spend=('amount', 'mean'),
    tx_count=('amount', 'count'),
    last_purchase=('date', 'max'),
    days_since_first=('date', lambda x: (x.max() - x.min()).days)
).reset_index()

# Changed: how='left' → how='inner'
df = customers.merge(tx_features, on='customer_id', how='inner')
df['days_since_last'] = (pd.Timestamp.now() - df['last_purchase']).dt.days
```

Re-run the notebook:

- Cells 2-3 are **RESTORED** from cache (CSV loading stays fast)
- In Cell 4, the groupby statement is **RESTORED** (its code didn't change), and only the merge statement **RECOMPUTES** (code changed)
- Cells 5-6 **RECOMPUTE** (new `df` → new model → new evaluation)

The Cell 4 badge shows that mix — the expensive groupby came from cache, only the merge ran:

<iframe class="cash-badge" src="/_badges/workflows_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

Without Cash: 30s to reload CSVs + recompute aggregations on every iteration.
With Cash: everything before your change loads in milliseconds.

## Working with Large Files

### Parquet Files

```python
# Cash tracks parquet files the same way as CSVs
df = pd.read_parquet('large_dataset.parquet')
```

### Multiple Data Sources

```python
# Each file is tracked independently
sales = pd.read_csv('sales_2024.csv')
products = pd.read_csv('products.csv')
regions = pd.read_parquet('regions.parquet')

# If only products.csv changes, sales and regions stay cached
combined = sales.merge(products, on='product_id').merge(regions, on='region_id')
```

## Controlling Cache Behavior

### Skip Caching for API Calls

```python
# @cash:no-cache
live_prices = api.get_current_prices()  # Always fetches fresh data
```

### Set Expiry for Volatile Data

```python
# @cash:ttl=300
market_data = fetch_market_snapshot()  # Cached for 5 minutes
```

### Prevent Randomness Issues

```python
# Unseeded random calls produce warnings — Cash can't guarantee reproducibility
# Option 1: Set a seed
np.random.seed(42)
sample = df.sample(100)

# Option 2: Suppress the warning
# @cash:allow-random
sample = df.sample(100)
```

## Debugging Cache Behavior

When something doesn't seem right:

```python
%cash_debug on
```

This raises the cash logger to DEBUG and prints labelled lines from each
subsystem. Look for these prefixes to see what happened for each statement:

- `[CACHE_KEY]` — how the cache key was constructed
- `[CACHE_HIT_DEBUG]` — why a lookup hit or missed
- `[UPSTREAM_DEBUG]` — what made an upstream cell re-run
- `[LINEAGE_DEBUG]` — the inputs detected for a statement and their resolved lineage hashes
- `[STATE]` — what the tracking state looks like at each step

Check overall statistics:

```python
%cash_stats
```

```
Cache Statistics:
  Entries: 42
  Hits: 156  |  Misses: 42  |  Hit Rate: 78.8%
  Time Saved: 4m 32s
  Storage: 128.5 MB (disk)
```

## Tips for Data Science Workflows

1. **Put `%cash_on` in the first cell** — Everything after it is cached automatically.

2. **Separate data loading from transforms** — Loading a 500MB CSV takes 10s. Put it in its own cell so it stays cached when you tweak transforms.

3. **Use `@cash:persist` for expensive computations** — Model training, large aggregations, and anything that takes >10s should be persisted to survive kernel restarts.

4. **Use `@cash:no-cache` for live data** — API calls, database queries with changing results, and real-time feeds should not be cached.

5. **Check `%cash_stats` periodically** — If your hit rate is low, you might be inadvertently invalidating caches (e.g., using `datetime.now()` in transformations).
