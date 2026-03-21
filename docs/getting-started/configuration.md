# Configuration

Cash can be configured through magic commands, environment variables, or the `CashConfig` class.

## Magic Command Options

### `%cash_on` — Enable Auto-Caching

```python
%cash_on                           # Default (TieredBackend: InMemory L1 + File L2)
%cash_on --backend memory          # In-memory backend only
%cash_on --backend sqlite          # SQLite backend
%cash_on --cache-dir ./my_cache    # Custom directory
```

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
    cache_dir='./my_cache',      # Cache directory
    backend_type='file',          # Backend type: 'memory', 'file', 'sqlite'
    max_size=1000,                # Max cache entries
    default_ttl=3600,             # Default TTL in seconds
    debug=False,                  # Debug mode
    enable_file_tracking=True,    # Track file dependencies
)
```

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
from cash.experimental import RedisBackend

backend = RedisBackend(url='redis://localhost:6379')
```

### S3Backend

Cloud storage backend. For persistent, shared caching.

```python
from cash.experimental import S3Backend

backend = S3Backend(
    bucket='my-cache-bucket',
    prefix='cash/',
    region='us-east-1'
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
| `CASH_CACHE_DIR` | Default cache directory | `./cash_cache` |
| `CASH_BACKEND` | Default backend type | `file` |
| `CASH_DEBUG` | Enable debug mode | `false` |
| `CASH_MAX_SIZE` | Max cache entries | `1000` |

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

To disable file tracking:

```python
from cash.config import CashConfig
config = CashConfig(enable_file_tracking=False)
```
