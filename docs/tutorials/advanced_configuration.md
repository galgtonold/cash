# Tutorial: Advanced Configuration

This tutorial covers Cash's configuration options in depth: backends, caching policies, annotations, and performance tuning.

## Backend Selection

Cash supports multiple storage backends. Choose based on your needs:

| Backend | Persistence | Speed | Sharing | Best For |
|---------|------------|-------|---------|----------|
| `InMemoryBackend` | ❌ Kernel restart clears | ⚡ Fastest | ❌ Single process | Quick experiments |
| `FileBackend` | ✅ Survives restarts | 🔵 Fast | ⚠️ Same machine | Default for most users |
| `SQLiteBackend` | ✅ Survives restarts | 🔵 Fast | ⚠️ Same machine | Structured queries |
| `TieredBackend` | ✅ Two layers | ⚡ L1 fast, L2 persistent | ⚠️ Same machine | **Default** (recommended) |
| `RedisBackend` | ✅ Configurable | 🟢 Network-fast | ✅ Multi-process | Teams, microservices |
| `S3Backend` | ✅ Cloud-durable | 🟠 Network | ✅ Multi-region | Cloud pipelines |

### Configuring via `%cash_on`

```python
# Memory only (fastest, no persistence)
%cash_on --backend memory

# SQLite (single-file persistence)
%cash_on --backend sqlite

# Custom cache directory
%cash_on --cache-dir ./my_project_cache
```

### Configuring Programmatically

```python
from cash import Cash
from cash.backends import (
    InMemoryBackend,
    FileBackend,
    TieredBackend,
    SQLiteBackend,
)

# Tiered: in-memory L1 + file L2 (default behavior)
app = Cash(backend=TieredBackend(
    l1=InMemoryBackend(max_entries=500),
    l2=FileBackend("./cache", max_size_bytes=5_000_000_000)
))

# SQLite with custom path
app = Cash(backend=SQLiteBackend("./cache/data.db"))
```

## Statement-Level Annotations

Annotations are comment directives that control caching for individual statements.

### `@cash:no-cache` — Skip Caching

Use for non-deterministic or side-effectful operations:

```python
# @cash:no-cache
current_time = datetime.now()  # Changes every call

# @cash:no-cache
api_response = requests.get("https://api.example.com/data")

# @cash:no-cache
print(f"Debug: {some_value}")  # Side effect (output to stdout)
```

### `@cash:ttl=<seconds>` — Time-to-Live

Cache results for a limited duration:

```python
# @cash:ttl=60
fast_changing = get_stock_price("AAPL")  # Re-fetch every minute

# @cash:ttl=3600
hourly_report = generate_summary(df)  # Regenerate every hour

# @cash:ttl=86400
daily_data = fetch_daily_metrics()  # Once per day
```

### `@cash:persist` — Force Disk Storage

By default, Cash uses smart policies to decide what goes to disk. Override this for expensive results:

```python
# @cash:persist
model = train_neural_network(X, y)  # 15 min to train — save to disk!

# @cash:persist
embeddings = compute_embeddings(corpus)  # 2 GB of vectors — persist them
```

### `@cash:allow-random` — Suppress Randomness Warnings

Cash warns when random functions are used without seeds (results may not reproduce from cache):

```python
# Without annotation: Cash warns about non-reproducible result
sample = df.sample(1000)

# With annotation: silences the warning
# @cash:allow-random
sample = df.sample(1000)

# Better approach: use a seed for reproducibility
sample = df.sample(1000, random_state=42)  # No warning needed
```

## Global TTL

Set a default TTL for all statements:

```python
# Magic command
%cash_on ttl=3600  # All statements expire after 1 hour

# Or per-cell
%%cash ttl=300
# Everything in this cell expires after 5 minutes
result = compute_something()
```

## Badge Display Modes

Control how Cash shows caching status:

```python
%cash_badge html     # Rich HTML badges with colors (default in JupyterLab)
%cash_badge print    # Plain text output (good for terminals/CI)
%cash_badge off      # Silent mode — no output
```

## Debug Mode

Enable detailed logging to understand caching decisions:

```python
%cash_debug on
```

Debug output shows:

- **Lineage tracking**: What inputs each statement depends on
- **Cache key computation**: How the cache key was built
- **File dependencies**: Which files are tracked and their hashes
- **Upstream simulation**: How upstream changes are detected

```
[LINEAGE_DEBUG] Statement: summary = df.describe()
  Inputs: {'df': 'sha256:a1b2c3...'}
  File deps: {}
  Cache key: stmt:sha256:d4e5f6...
  Status: RESTORED (hit)
```

Turn off when done:

```python
%cash_debug off
```

## Cache Management

### View Statistics

```python
%cash_stats
```

### Export and Import Cache

Share cached results across notebooks or team members:

```python
# Export all cached data
%cash_export project_cache.cache

# Export specific variables only
%cash_export model_cache.cache --vars model,features,predictions

# Export lineage graph as JSON (for inspection)
%cash_export lineage.json --json
```

```python
# Import cache from file
%cash_import project_cache.cache

# Merge with existing cache (don't overwrite)
%cash_import teammate_cache.cache --merge
```

### Clear Cache

```python
%cash_clear          # Clear all cached data
%cash_clear --vars x,y  # Clear specific variables
```

## Inspect Variables

View the dependency graph and lineage of cached variables:

```python
%cash_inspect df
```

Shows:

- The code that produced the variable
- Input dependencies (other variables it depends on)
- File dependencies
- Lineage hash
- Whether it's cached in memory, on disk, or both

## Performance Tuning

### 1. Separate I/O from Computation

```python
# Good: I/O in its own statement
df = pd.read_csv('big_file.csv')   # Cached independently
result = df.groupby('col').sum()    # Cached independently

# Less optimal: combined (if you change the groupby, CSV reloads too)
result = pd.read_csv('big_file.csv').groupby('col').sum()
```

### 2. Use `@cash:persist` for Expensive Results

Kernel crashes happen. Persist anything that takes >10 seconds:

```python
# @cash:persist
trained_model = train_complex_model(X, y)  # 20 min → saved to disk
```

### 3. Monitor Hit Rates

Low hit rates (<50%) suggest code patterns that prevent caching:

```python
%cash_stats  # Check hit rate
```

Common causes of low hit rates:

- Using `datetime.now()` or `time.time()` in computations
- Unseeded random operations
- Modifying variables in-place across cells
- Importing modules that change on each load

### 4. Use `@cash:no-cache` for Cheap Operations

Don't waste cache space on trivial statements:

```python
# @cash:no-cache
x = 42  # Not worth caching

# @cash:no-cache
print(df.head())  # Side effect, don't cache
```
