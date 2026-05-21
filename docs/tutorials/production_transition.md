# Tutorial: From Notebook to Production

This tutorial shows how to transition from Cash-cached notebooks to production-ready Python scripts and services, using Cash's decorator API for function-level caching.

## The Two Faces of Cash

Cash has two APIs designed for different stages of the development lifecycle:

| Stage | API | Caching Unit | Use Case |
|-------|-----|-------------|----------|
| **Exploration** | `%cash_on` (magics) | Statement-level | Jupyter notebooks, EDA, experiments |
| **Production** | `@cash.cache` (decorator) | Function-level | Scripts, APIs, pipelines |

The transition is natural: wrap your notebook statements into functions, then decorate them.

## Step 1: Identify Cacheable Functions

In your notebook, look for expensive operations:

```python
# Notebook code (exploration)
%cash_on

df = pd.read_csv('data.csv')              # ← expensive I/O
features = engineer_features(df)           # ← expensive computation
model = train_model(features)              # ← very expensive
predictions = model.predict(test_data)     # ← quick, but model-dependent
```

## Step 2: Wrap in Functions with `@cash.cache`

Create a Python module:

```python
# pipeline.py
from cash import cache
import pandas as pd

@cache
def load_data(path: str) -> pd.DataFrame:
    """Load and validate input data."""
    df = pd.read_csv(path)
    assert len(df) > 0, "Empty dataset"
    return df

@cache
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from raw data."""
    df = df.copy()
    df['log_amount'] = np.log1p(df['amount'])
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

@cache
def train_model(features: pd.DataFrame, target_col: str = 'churned'):
    """Train a classification model."""
    from sklearn.ensemble import RandomForestClassifier
    X = features.drop(columns=[target_col])
    y = features[target_col]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
```

## Step 3: Use File Dependencies

For functions that depend on external files:

```python
from cash import Cash

app = Cash()

@app.cache(file_depends_on="data.csv")
def load_data():
    """Cache invalidates when data.csv changes."""
    return pd.read_csv("data.csv")

@app.cache(file_depends_on=["config.yaml", "schema.json"])
def load_config():
    """Multiple file dependencies."""
    import yaml, json
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    with open("schema.json") as f:
        schema = json.load(f)
    return config, schema
```

## Step 4: Choose the Right Backend

### Development (default)

```python
from cash import Cash

# TieredBackend: fast in-memory L1 + persistent file L2
app = Cash()  # Uses ~/.cash/ by default
```

### Production — File Backend

```python
from cash import Cash, FileBackend

app = Cash(backend=FileBackend(
    cache_dir="/var/cache/myapp",
    max_size_bytes=10_000_000_000  # 10 GB limit
))
```

### Production — SQLite Backend

```python
from cash.backends import SQLiteBackend

app = Cash(backend=SQLiteBackend("cache.db"))
```

### Production — Redis Backend

```python
from cash.backends import RedisBackend

app = Cash(backend=RedisBackend(
    host="redis.internal",
    port=6379,
    db=0
))
```

## Step 5: Build a Pipeline Script

```python
# run_pipeline.py
from cash import Cash, FileBackend

app = Cash(backend=FileBackend("./pipeline_cache"))

@app.cache(file_depends_on="data/raw.csv")
def extract():
    import pandas as pd
    return pd.read_csv("data/raw.csv")

@app.cache
def transform(df):
    df = df.dropna()
    df['amount_norm'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    return df

@app.cache
def train(df):
    from sklearn.linear_model import LogisticRegression
    X, y = df.drop('target', axis=1), df['target']
    model = LogisticRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    raw = extract()
    clean = transform(raw)
    model = train(clean)
    
    # Check cache stats (cache_info is attached to the decorated function, not its return value)
    print(train.cache_info())  # hits, misses, hit_rate, total_time_saved
```

On first run: everything computes and caches.
On subsequent runs: if `data/raw.csv` hasn't changed, everything restores from cache instantly.

## Step 6: Cache Introspection

The decorator API provides introspection methods:

```python
@app.cache
def my_function(x):
    return x ** 2

# Run the function
result = my_function(42)

# Introspection
print(my_function.cache_info())     # Hit/miss stats
my_function.cache_clear()           # Clear this function's cache
```

## Notebook ↔ Decorator Bridge

Cash connects both APIs. When you use `@cash.cache` inside a notebook with `%cash_on`, the decorator results appear in the notebook badges with condensed metrics:

```python
# In a notebook with %cash_on

from cash import cache

@cache
def slow_computation(x):
    import time; time.sleep(2)
    return x * 42

# The badge shows both the statement cache AND the decorator cache
result = slow_computation(10)
```

## Migration Checklist

- [ ] Identify expensive operations in notebooks
- [ ] Wrap each operation in a function with `@cache` or `@app.cache`
- [ ] Add `file_depends_on` for file-dependent functions
- [ ] Choose appropriate backend for production environment
- [ ] Add `cache_info()` calls to monitoring/logging
- [ ] Test that cache invalidation works correctly (change inputs, verify recomputation)
- [ ] Set up TTL for time-sensitive data: `@app.cache(ttl=3600)`
