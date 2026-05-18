# Tutorial: Data Science Workflows with Cash

This tutorial covers real-world data science patterns where Cash shines: loading large datasets, exploratory analysis, feature engineering, and model training.

## Scenario: Customer Churn Analysis

You're building a churn prediction model. The workflow involves loading data, cleaning, feature engineering, training, and evaluation — and you'll iterate on each step many times.

### Cell 1: Setup

```python
import cash
%cash_on
```

### Cell 2: Load Data

```python
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

### Cell 3: Data Cleaning

```python
# Multiple statements — each cached independently
customers = customers.dropna(subset=['email', 'signup_date'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
transactions['date'] = pd.to_datetime(transactions['date'])
```

If you change the `dropna` logic, only that statement and its dependents recompute. The `pd.to_datetime` calls stay cached.

### Cell 4: Feature Engineering

```python
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

```python
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

```python
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## The Iteration Loop

Here's where Cash pays off. You want to try different features:

1. **Change Cell 4** to add a new feature (e.g., `std_spend`)
2. **Re-run the notebook**
3. Cells 2-3 are **RESTORED** from cache (data loading stays fast)
4. Cell 4-6 **RECOMPUTE** (new features → new model → new evaluation)

Without Cash: 30s to reload CSVs + clean data every iteration.
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

You'll see detailed output for each statement:

```
[LINEAGE_DEBUG] Statement: df = pd.read_csv('data.csv')
  Inputs: {} (no dependencies)
  File deps: {'data.csv': 'sha256:abc123...'}
  Output lineage: sha256:def456...
  Status: RESTORED (cache hit)
```

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
