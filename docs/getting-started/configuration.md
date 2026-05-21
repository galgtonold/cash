# Configuration

Cash can be configured through magic commands, environment variables, or the `CashConfig` class.

## Magic Command Options

### `%cash_on` — Enable Auto-Caching

```python
%cash_on              # Enable auto-caching (default backend: TieredBackend = InMemory L1 + File L2)
%cash_on ttl=3600     # Enable with a 1-hour TTL on all cached entries
```

`%cash_on` takes only an optional `ttl=N` argument. To pick a different backend or cache directory, construct a `Cash(backend=...)` instance programmatically — see [Backend Configuration](#backend-configuration) below.

### `%cash_debug` — Debug Output

```python
%cash_debug on    # Show detailed debug info
%cash_debug off   # Disable debug output
```

### `%%cash` — Per-Cell Caching

```python
%%cash
# Only this cell is cached
result = expensive_computation()
```

### `%cash_badge` — Badge Display Mode

```python
%cash_badge html    # Interactive HTML badges (default)
%cash_badge print   # Text summary after cell completes
%cash_badge off     # No badge output
```

## CashConfig Class

For programmatic configuration:

```python
from cash.config import CashConfig

config = CashConfig(
    cache_dir='./my_cache',              # Cache directory (default: '.cash')
    debug=False,                         # Debug mode
    max_cache_size=1024 ** 3,            # Max cache size in bytes (default: 1 GiB)
    compress=False,                      # Compress cached entries
    flush_interval=5,                    # Seconds between background flushes
    smart_persistence=True,              # Only persist results worth more than the I/O cost
    smart_persistence_threshold=1.0,     # Execution-time threshold (seconds) for persistence
    min_cache_savings_pct=0.20,          # Skip caching when expected savings < 20%
)
```

`CashConfig` is consumed by `get_config()` and by environment-variable / TOML loading. It is **not** auto-applied when you construct `Cash()` — to use a non-default backend, pass a backend object directly (`Cash(backend=...)`); see [Backend Configuration](#backend-configuration) below.

## Backend Configuration

### InMemoryBackend

Fastest, but data is lost when kernel restarts.

```python
from cash.backends import InMemoryBackend

backend = InMemoryBackend(max_entries=500)
```

### FileBackend (Default)

Persists cache to disk. Survives kernel restarts.

```python
from cash.backends import FileBackend

backend = FileBackend(
    cache_dir='./cache',
    max_size_bytes=1_000_000_000  # 1 GB limit
)
```

### SQLiteBackend

Single-file database backend. Good for portability.

```python
from cash.backends import SQLiteBackend

backend = SQLiteBackend(db_path='./cache.db')
```

### RedisBackend

Shared cache across notebooks/users. Requires Redis server.

```python
from cash.backends import RedisBackend

backend = RedisBackend(
    host='localhost',
    port=6379,
    db=0,
    password=None,
    prefix='cash:',
)
```

### S3Backend

Cloud storage backend. For persistent, shared caching.

```python
from cash.backends import S3Backend

backend = S3Backend(
    bucket='my-cache-bucket',
    prefix='cash/',
    max_pool_connections=10,
    retries=3,
    region_name='us-east-1',  # forwarded via **kwargs to boto3.client('s3')
)
```

### TieredBackend

Combine multiple backends with fallthrough. This is the **default backend** when no backend is specified.

```python
from cash.backends import TieredBackend, InMemoryBackend, FileBackend

backend = TieredBackend([
    InMemoryBackend(max_entries=100),   # Fast L1 cache
    FileBackend(cache_dir='./cache'),   # Persistent L2 cache
])
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CASH_CACHE_DIR` | Default cache directory | `.cash` |
| `CASH_BACKEND` | Default backend type | `file` |
| `CASH_DEBUG` | Enable debug mode | `false` |
| `CASH_MAX_CACHE_SIZE` | Max cache size in bytes | `1073741824` (1 GiB) |
| `CASH_COMPRESS` | Compress cache entries | `false` |
| `CASH_REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `CASH_S3_BUCKET` | Default S3 bucket name | (unset) |
| `CASH_MAX_MEMORY_ENTRIES` | Cap on in-memory entries (`null` = unlimited) | unset |
| `CASH_MIN_CACHE_SAVINGS_PCT` | Skip caching when expected savings fall below this fraction | `0.20` |

## File Tracking Configuration

Cash automatically tracks file reads through common libraries:

```python
# These are tracked by default:
df = pd.read_csv('data.csv')        # pandas (all read_* functions)
arr = np.load('array.npy')           # numpy
pl_df = pl.read_csv('data.csv')      # polars (read_* and scan_*)
data = joblib.load('model.pkl')      # joblib
with open('file.txt') as f:          # built-in open
    content = f.read()
import json
with open('config.json') as f:       # json
    config = json.load(f)
```

File tracking is always on for these recognized read patterns — there is no global toggle. To opt a single statement out of caching (and therefore out of file-dependency tracking for that statement), use the `# @cash:no-cache` annotation described in the [Quick Start](quickstart.md#statement-level-annotations).
