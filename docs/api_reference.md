# Cash API Reference

Complete reference for all public APIs in the Cash caching framework.

## Table of Contents

- [Core API](#core-api)
- [Backends](#backends)
- [Magic Commands](#magic-commands)
- [Purity Declarations](#purity-declarations)
- [Configuration](#configuration)
- [CLI Tool](#cli-tool)
- [nbconvert Integration](#nbconvert-integration)

---

## Core API

### `Cash`

```python
from cash import Cash

cash = Cash(
    backend=None,        # CacheBackend instance (default: auto from config)
    cache_dir=None,      # str - directory for FileBackend  
    backends=None,       # List[CacheBackend] - for CascadingBackend
    compress=None,       # bool - enable gzip compression (default from config)
    register_magic=True, # bool - auto-register IPython magics
    debug=None,          # bool - enable debug logging (default from config)
    background_io=False, # bool - async backend wrapper
    use_locking=False,   # bool - enable double-checked locking
    config_path=None,    # str - path to config file
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `cash.cache(func, *, depends_on=None, dynamic_depends_on=None, file_depends_on=None, ttl=None)` | Decorator for function caching. Pass `depends_on=` / `file_depends_on=` to track `DataSource` / file dependencies. |
| `cash.register_hasher(type_, hasher_fn)` | Register a custom hasher for a specific type |
| `cash.register_file_handler(module, func, handler)` | Register a custom file tracking handler |
| `cash.show_stats()` | Print cache statistics |
| `func.cache_clear()` | Invalidate all cache entries for a wrapped function (called on the decorated function, not `cash`) |
| `cash.cleanup(max_age=None)` | Remove expired cache entries; returns count removed |
| `cash.explorer()` | Return a `CacheExplorer` for interactive browsing |

### `@cash.cache` Decorator

```python
@cash.cache
def expensive_computation(x, y):
    return x ** y

# With options
@cash.cache(ttl=3600, depends_on=[other_func])
def fetch_data(url):
    return requests.get(url).json()

# With file dependency shorthand
@cash.cache(file_depends_on="data.csv")
def load_data():
    return pd.read_csv("data.csv")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `depends_on` | `List[Callable\|DataSource]` | Static dependencies to include in cache key |
| `dynamic_depends_on` | `Callable\|List[Callable]` | Functions that return DataSources based on args |
| `file_depends_on` | `str\|List[str]` | File path(s) to track. Shorthand for `depends_on=[FileDataSource(...)]` |
| `ttl` | `int` | Time-to-live in seconds (None = never expires) |

**Attributes on decorated functions:**

| Attribute | Description |
|-----------|-------------|
| `func.cache_info()` | Returns dict with `hits`, `misses`, `hit_rate`, `total_time_saved` |
| `func.cache_clear()` | Clears all cached results for this function and resets stats |
| `func.__wrapped__` | Reference to the original unwrapped function |

### `register_hasher`

Register a custom hasher for argument types that can't be pickled or need deterministic hashing:

```python
from cash import Cash
import hashlib

c = Cash()
c.register_hasher(
    MyCustomType,
    lambda obj: hashlib.sha256(obj.serialize().encode()).hexdigest()
)
```

Cash includes built-in hashers for: pandas DataFrame/Series, numpy ndarray, polars DataFrame/Series/LazyFrame, PyArrow Table/RecordBatch, modin DataFrame, and dask DataFrame.

---

## Backends

### `InMemoryBackend`

```python
from cash import InMemoryBackend

backend = InMemoryBackend(
    max_memory_percent=0.9,
    check_interval=10,
    max_entries=None,
)
```

Fast, session-scoped cache. Lost on kernel restart. Supports smart eviction based on
memory pressure (requires `psutil`) and optional LRU eviction by entry count.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_memory_percent` | `float` | `0.9` | Memory usage fraction (0.0–1.0) at which to trigger eviction |
| `check_interval` | `int` | `10` | Number of `set` operations between memory checks |
| `max_entries` | `int \| None` | `None` | Maximum number of entries before LRU eviction (`None` = unlimited) |

### `FileBackend`

```python
from cash import FileBackend

backend = FileBackend(
    cache_dir=".cash",
    compress=False,
    max_size_bytes=None,
    flush_interval=5,
    default_ttl=None,
)
```

Persistent cache on disk. Survives kernel restarts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str` | required | Cache directory path |
| `compress` | `bool` | `False` | Enable gzip compression |
| `max_size_bytes` | `int \| None` | `None` | Max total cache size in bytes (triggers LRU eviction) |
| `flush_interval` | `int` | `5` | Seconds between metadata flush cycles |
| `default_ttl` | `int \| None` | `None` | Default TTL in seconds (`None` = no expiration) |

### `CascadingBackend`

```python
from cash import CascadingBackend, InMemoryBackend, FileBackend

backend = CascadingBackend([
    InMemoryBackend(max_entries=100),  # L1: fast
    FileBackend(cache_dir=".cash"),     # L2: persistent
])
```

Multi-tier caching. Reads check L1 first, writes go to all levels.

### `SQLiteBackend`

```python
from cash.backends.sqlite_backend import SQLiteBackend

backend = SQLiteBackend(
    db_path=".cash/cache.db",
    default_ttl=None,
    max_size_bytes=None,
    wal_mode=True,
)
```

SQLite-based persistent cache. Uses a single database file for all entries — better than
`FileBackend` for many small entries.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `".cash/cache.db"` | Path to the SQLite database file |
| `default_ttl` | `int \| None` | `None` | Default TTL in seconds (`None` = no expiration) |
| `max_size_bytes` | `int \| None` | `None` | Maximum total data size (`None` = unlimited) |
| `wal_mode` | `bool` | `True` | Use WAL journal mode for better concurrency |

### `RedisBackend` *(requires `pip install cash-lib[redis]`)*

```python
from cash.experimental import RedisBackend

backend = RedisBackend(
    host="localhost",
    port=6379,
    db=0,
    password=None,
    prefix="cash:",
    socket_keepalive=True,
    health_check_interval=30,
    retry_on_timeout=True,
    max_retries=3,
)
```

Redis-based shared cache. Connection parameters are passed directly to `redis.Redis`;
extra keyword arguments are forwarded via `**kwargs`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | Redis server hostname |
| `port` | `int` | `6379` | Redis server port |
| `db` | `int` | `0` | Redis logical database number |
| `password` | `str \| None` | `None` | Redis password, if required |
| `prefix` | `str` | `"cash:"` | Key prefix used for all cache entries |
| `socket_keepalive` | `bool` | `True` | Enable TCP keepalive on the connection |
| `health_check_interval` | `int` | `30` | Seconds between client-side health checks |
| `retry_on_timeout` | `bool` | `True` | Retry commands on socket timeouts |
| `max_retries` | `int` | `3` | Maximum retries (exponential backoff) for transient errors |
| `**kwargs` | — | — | Additional keyword args forwarded to `redis.Redis` |

### `S3Backend` *(requires `pip install cash-lib[s3]`)*

```python
from cash.experimental import S3Backend

backend = S3Backend(
    bucket="my-cache-bucket",
    prefix="cash/",
    max_pool_connections=10,
    retries=3,
)
```

S3-backed shared cache. Extra keyword arguments are forwarded to `boto3.client('s3', ...)`,
so you can pass `region_name=`, `endpoint_url=`, credentials, etc.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bucket` | `str` | required | S3 bucket name |
| `prefix` | `str` | `"cash/"` | Key prefix used for all cache objects |
| `max_pool_connections` | `int` | `10` | Connection pool size for the boto3 client |
| `retries` | `int` | `3` | Maximum retry attempts (standard mode) |
| `**kwargs` | — | — | Additional keyword args forwarded to `boto3.client('s3', ...)` |

---

## Magic Commands

### Enabling/Disabling

| Command | Description |
|---------|-------------|
| `%cash_on` | Enable auto-caching for all cells |
| `%cash_on ttl=3600` | Enable with default TTL (seconds) |
| `%cash_off` | Disable auto-caching |
| `%%cash` | Cache a single cell |
| `%%cash --ttl 300` | Cache a single cell with TTL |

### Debugging & Status

| Command | Description |
|---------|-------------|
| `%cash_debug on` | Enable debug output |
| `%cash_debug off` | Disable debug output |
| `%cash_debug json` | Enable JSON-formatted debug output |
| `%cash_debug file path.log` | Log to file in JSON format |
| `%cash_badge html` | Interactive HTML badges with live updates (default) |
| `%cash_badge print` | Text summary printed once after cell completes |
| `%cash_badge off` | No badge output |
| `%cash_status` | Show last cell execution metrics |
| `%cash_status json` | Return metrics as JSON string |
| `%cash_stats` | Show session-wide statistics |
| `%cash_stats json` | Return stats as JSON |
| `%cash_stats reset` | Reset session counters |
| `%cash_log` | View recent structured log events |
| `%cash_log 50` | View last 50 events |
| `%cash_log json` | Output log events as JSON |
| `%cash_log clear` | Clear log buffer |

### Cache Management

| Command | Description |
|---------|-------------|
| `%cash_verify` | Check cache integrity |
| `%cash_verify --fix` | Auto-fix corrupted entries |
| `%cash_repair` | Fix corrupted cache entries |
| `%cash_repair --state` | Reset tracking state |
| `%cash_repair --full` | Clear everything |
| `%cash_export results.cache` | Export cache to file |
| `%cash_export results.cache --vars x,y` | Export specific variables |
| `%cash_import results.cache` | Import cache from file |
| `%cash_import results.cache --merge` | Merge with existing cache |

### Provenance & Audit

| Command | Description |
|---------|-------------|
| `%cash_provenance x` | Show how variable `x` was computed |
| `%cash_provenance x --graph` | Include dependency graph |
| `%cash_provenance x --time` | Include computation timeline |
| `%cash_provenance x --json` | Output as JSON |
| `%cash_provenance --all` | List all tracked variables |
| `%cash_provenance --clear` | Clear provenance data |
| `%cash_audit on` | Enable audit logging (in-memory) |
| `%cash_audit on --file audit.log` | Enable with file output |
| `%cash_audit off` | Disable audit logging |
| `%cash_audit show` | Show recent audit entries |
| `%cash_audit show --json` | Show as JSON |
| `%cash_audit show cache_hit` | Filter by operation type |
| `%cash_audit summary` | Show summary statistics |
| `%cash_audit clear` | Clear audit entries |

### Benchmarking

| Command | Description |
|---------|-------------|
| `%cash_benchmark` | Benchmark next cell (3 iterations) |
| `%cash_benchmark 5` | Run 5 iterations |
| `%cash_benchmark --cold` | Clear cache before each run |
| `%cash_benchmark --compare` | Compare cached vs uncached execution |

### Module Tracking

| Command | Description |
|---------|-------------|
| `%cash_track my_module` | Track module for file changes |
| `%cash_track my_module --reload` | Force reload module |
| `%cash_track --list` | List tracked modules |
| `%cash_track --check` | Check for changes and auto-reload |

### Cache Diff & Export

| Command | Description |
|---------|-------------|
| `%cash_diff file.cache` | Compare current session with exported cache |
| `%cash_diff file.cache --vars` | Show variable-level differences |
| `%cash_export file.cache` | Export cache entries (pickle format) |
| `%cash_export file.json --json` | Export lineage as JSON (for `%cash_diff`) |
| `%cash_export file --vars x,y` | Export specific variables only |
| `%cash_import file.cache` | Import cache entries |
| `%cash_import file.cache --merge` | Merge with existing cache |

---

## Purity Declarations

Control caching behavior for specific functions using decorators.

### `@pure`

Mark a function as pure (no side effects). Pure functions are always safe to cache,
and cash will skip mutation detection when all called functions are pure.

```python
from cash import pure

@pure
def compute(x, y):
    return x + y  # No side effects — always cacheable
```

### `@stateful`

Mark a function as stateful (has side effects). Stateful functions are never cached —
statements that call them will always re-execute.

```python
from cash import stateful

@stateful
def train_model(data):
    model.fit(data)        # Side effect: modifies model state
    return model.score(data)
```

### Helper Functions

| Function | Description |
|----------|-------------|
| `is_pure(func)` | Check if a function is marked as `@pure` |
| `is_stateful(func)` | Check if a function is marked as `@stateful` |
| `analyze_function_purity(func)` | Analyze a function's purity automatically |

```python
from cash import is_pure, is_stateful, analyze_function_purity

@pure
def add(a, b):
    return a + b

is_pure(add)       # True
is_stateful(add)   # False
```

---

## Configuration

### Config File (`~/.cash/config.toml`)

```toml
[cash]
backend_type = "file"     # "memory", "file", "sqlite", "redis", "s3"
cache_dir = ".cash"
debug = false
compress = false
max_memory_entries = 1000
default_ttl = 3600

[redis]
url = "redis://localhost:6379"

[s3]
bucket = "my-cache-bucket"
prefix = "cash/"
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CASH_BACKEND` | Backend type | `memory`, `file`, `sqlite` |
| `CASH_CACHE_DIR` | Cache directory | `/tmp/cash-cache` |
| `CASH_DEBUG` | Enable debug mode | `1`, `true` |
| `CASH_COMPRESS` | Enable compression | `1`, `true` |
| `CASH_MAX_CACHE_SIZE` | Max cache size bytes | `1073741824` |
| `CASH_MAX_MEMORY_ENTRIES` | Max in-memory entries | `1000` |
| `CASH_REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `CASH_S3_BUCKET` | S3 bucket name | `my-bucket` |

### Per-Notebook Config

```python
cash = Cash(config_path="./my_notebook_config.toml")
```

### `CashConfig` Dataclass

```python
from cash import CashConfig, create_default_config

# Create default config file
create_default_config()  # Creates ~/.cash/config.toml

# Load config
config = get_config()  # Merges env > file > defaults
print(config.backend)   # "file"
print(config.cache_dir)  # ".cash"
```

---

## CLI Tool

```bash
# Show version
python -m cash version

# Show configuration info
python -m cash info

# Inspect notebook cache
python -m cash inspect notebook.ipynb

# Clear cache directories
python -m cash clear .cash
python -m cash clear  # Clears default .cash directory
```

---

## nbconvert Integration

### `CashStripPreprocessor`

Strip Cash-related outputs from notebooks before sharing or committing:

```python
# In nbconvert config (e.g., jupyter_nbconvert_config.py)
c.Exporter.preprocessors = ['cash.nbconvert.CashStripPreprocessor']
```

```python
# Programmatic usage
from cash.nbconvert import CashStripPreprocessor
import nbformat

nb = nbformat.read("notebook.ipynb", as_version=4)
preprocessor = CashStripPreprocessor(
    strip_badges=True,   # Remove badge HTML outputs
    strip_debug=True,    # Remove debug print lines
    strip_magics=False,  # Optionally remove %cash_* commands
)
nb, _ = preprocessor.preprocess(nb, {})
nbformat.write(nb, "clean_notebook.ipynb")
```

---

## Data Source Tracking

### `FileDataSource`

```python
from cash import FileDataSource

source = FileDataSource("data/input.csv")

@cash.cache(depends_on=[source])
def load_data():
    return pd.read_csv("data/input.csv")
```

When `input.csv` changes on disk, cached results are automatically invalidated.

---

## Experimental Features

Access via `cash.experimental`:

```python
from cash.experimental import (
    CacheExplorer,      # Interactive cache browser
    CacheDebugger,      # Step-through cache debugging
    AnalyticsManager,   # Cache usage analytics
    visualize_notebook,  # Notebook dependency visualization
    DependencyGraph,     # Build and query dependency graphs
    RedisBackend,       # Redis cache backend
    S3Backend,          # S3 cache backend
)
```

> ⚠️ Experimental APIs may change without notice between minor versions.

> **Note:** `TieredBackend` is used as the default backend and is fully stable.
> It is no longer experimental.
