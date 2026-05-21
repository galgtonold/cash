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

### Configuring Programmatically

`%cash_on` itself only accepts `ttl=N` (a global TTL in seconds). Backend
selection happens by constructing a `Cash(...)` instance and calling
`register_magic()` on it — there is no `--backend` or `--cache-dir` flag on the
magic.

```python
from cash import Cash, InMemoryBackend, FileBackend
from cash.backends.sqlite_backend import SQLiteBackend

# In-memory only (fastest, no persistence across kernel restarts)
c = Cash(backend=InMemoryBackend())
c.register_magic()

# File backend (disk-persistent, default if no backend is passed)
c = Cash(backend=FileBackend(cache_dir="./my_project_cache"))
c.register_magic()

# SQLite backend (single-file, good for sharing one cache between scripts)
c = Cash(backend=SQLiteBackend(db_path="./.cash/cache.db"))
c.register_magic()
```

For a tiered cache (in-memory L1 + on-disk L2 — the default if you call
`Cash()` with no backend argument), pass a list of backends positionally:

```python
from cash import Cash, InMemoryBackend, FileBackend
from cash.backends import TieredBackend

backend = TieredBackend([
    InMemoryBackend(max_entries=100),
    FileBackend(cache_dir="./.cash"),
])
c = Cash(backend=backend)
c.register_magic()
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

`%cash_debug on` raises the cash logger to DEBUG and prints labelled lines
from each subsystem. Look for these prefixes to see how caching decisions
were made:

- `[CACHE_KEY]` — how the cache key was constructed for a statement
- `[CACHE_HIT_DEBUG]` — why a lookup hit or missed
- `[UPSTREAM_DEBUG]` — what made an upstream cell re-run
- `[LINEAGE_DEBUG]` — the inputs detected for a statement and their resolved lineage hashes
- `[STATE]` — what the tracking state looks like at each step

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

From inside a notebook, use `%cash_repair --full` to wipe the entire cache
*and* reset in-memory tracking state. To clear only the in-memory tracking
(keep the on-disk cache), use `%cash_repair --state`:

```python
%cash_repair --full   # Clear all cache + reset in-memory state
%cash_repair --state  # Reset only in-memory state, keep cached data
```

From outside a notebook, use the `cash` CLI:

```bash
cash clear --all                  # Clear the .cash directory in the current dir
cash clear ./path/to/notebook.ipynb  # Clear the cache associated with a specific notebook
cash clear ./some/cache_dir       # Remove a specific cache directory
```

Older versions of this tutorial referenced `%cash_clear` and
`%cash_inspect` magics — those do not exist in current cash. Use
`%cash_repair` (above) or the CLI subcommands instead.

## Inspecting the Cache

To inspect cache contents from outside a notebook, use the CLI:

```bash
cash inspect                          # Inspect cache in the current directory
cash inspect path/to/notebook.ipynb   # Inspect cache for a specific notebook
```

This shows entry counts, sizes, and metadata for whatever cache directory it
discovers. (There is no in-notebook `%cash_inspect` magic — use `%cash_stats`
for session statistics or the `cash inspect` CLI for a detailed view.)

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
