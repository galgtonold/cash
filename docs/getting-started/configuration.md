# Configuration

!!! info "Applies to: both paths"
    Every setting, environment variable and default, for scripts and notebooks. The *Path* column says which path each setting affects.

## Where settings come from

Cash merges its settings from six layers. A higher layer wins:

1. **Code**: `Cash(debug=True)` or `cash.configure(debug=True)`.
2. **Environment variables**: `CASH_<FIELD>`, such as `CASH_CACHE_DIR`.
3. **A file named in code**: `Cash(config_path="cash.toml")`
   ([below](#a-config-file-named-in-code)).
4. **Project file**: `[tool.cash]` in the project's `pyproject.toml`.
5. **User file**: `~/.config/cash/config.toml` (`$XDG_CONFIG_HOME` is
   honoured), or `%APPDATA%\cash\config.toml` on Windows.
6. **Defaults**, listed in the tables below.

```toml
# pyproject.toml
[tool.cash]
cache_dir = ".cash"
max_cache_size = "5GB"
```

`cash info` prints every setting in effect and the layer it came from
([CLI](../cli.md#cash-info)).

<!-- claim: cash/_location.py:project_anchor @46e903a7, cash/config.py:_anchor_cache_dir @ae7a94f7 -->
### What paths are relative to

A relative `cache_dir` is resolved against whoever wrote it:

| Written in | Resolved against |
|---|---|
| `Cash(cache_dir=...)`, `cash.configure(...)` or `CASH_CACHE_DIR` | the current working directory |
| a config file | that file's directory |
| nowhere (the `.cash` default) | the project root: the first directory above the running script that holds a `pyproject.toml`, `setup.py`, `setup.cfg` or `.git`; without one, the script's own directory. In a notebook or REPL, the current directory. |

Installed programs such as `pytest` or a console script use the project
root above the current directory, or a per-user cache directory per tool when
run outside any project. The project `pyproject.toml` is looked for from the
same root, so a script finds the same settings wherever it is started. See
[Where your cache lives](../how-it-works/storage.md#where-the-cache-folder-is) for how the folder is
chosen and capped.

<!-- claim: cash/_location.py:_marks_project @2991b315 -->
A `pyproject.toml` marks a project only when it has a `[project]`,
`[build-system]`, `[tool.poetry]` or `[tool.cash]` table.

<!-- claim: cash/config.py:CashConfig @9e5fc345 broad="the field table is a claim about every field of the dataclass" -->
## All settings

Every setting below has a `CASH_<FIELD>` environment variable, and its TOML key
is the field name. **Path** says whether it affects `@cash.cache` (decorator),
notebook caching (notebook), or both.

<!-- claim: cash/config.py:validate_value @78aa44f0, cash/config.py:parse_size @11b4b371, cash/config.py:_validated_layer @84048bbe -->
Values are checked whichever layer they come from. Strings are read as an
environment variable would be (`"true"`, `"8"`). Byte sizes also take units:
`"2GB"`, `"500MB"`, `"512MiB"` (KB/MB/GB are powers of 1000, KiB/MiB/GiB of
1024). A bad value in code raises `ValueError`. A bad value in a file or
variable is skipped with [`CONFIG-INVALID`](../warnings.md#config-invalid), and
an unknown key in a file warns
[`CONFIG-UNKNOWN-KEY`](../warnings.md#config-unknown-key).

### Storage

| Field | Env var | Default | Path | Description |
|---|---|---|---|---|
| `cache_dir` | `CASH_CACHE_DIR` | `".cash"` | both | Where the disk tier writes. Add it to `.gitignore`. |
| `compress` | `CASH_COMPRESS` | `false` | both | gzip each entry on disk. Worth it mainly for text-like values. |
| `max_cache_size` | `CASH_MAX_CACHE_SIZE` | `null` (auto) | both | Disk cap in bytes or a size such as `"5GB"`. At the cap, the entries worth least per byte (compute time per byte, raised by hits) are evicted first. A single value bigger than the cap is not written ([`CACHE-VALUE-TOO-BIG`](../warnings.md#cache-value-too-big)). Each process enforces the cap on its own writes. |
| `max_memory_entries` | `CASH_MAX_MEMORY_ENTRIES` | `null` | both | Entry-count cap for the RAM tier, evicting least recently used. `null` means no count limit; the RAM tier is still capped in bytes. |
| `flush_interval` | `CASH_FLUSH_INTERVAL` | `5` | both | Seconds between the disk tier's metadata flushes. `0` flushes after every write. |
| `file_hash_full_max_bytes` | `CASH_FILE_HASH_FULL_MAX_BYTES` | `268435456` (256 MiB) | both | Tracked files up to this size are hashed in full to check freshness. Larger files hash three sampled regions plus the timestamps, which misses a same-size edit outside those regions that keeps the modification time. |
| `shutdown_write_timeout` | `CASH_SHUTDOWN_WRITE_TIMEOUT` | `60.0` | both | Seconds a finishing process waits for its background writes before exiting without them ([`CACHE-WRITE-ABANDONED`](../warnings.md#cache-write-abandoned)). |

<!-- claim: cash/backends/adaptive_caps.py:resolve_ram_cap @02a19f23, cash/backends/adaptive_caps.py:_cgroup_memory_limit @b30940d8 -->
**What "auto" resolves to.** The disk cap is a quarter of the room on the cache
volume (free space plus what the cache already holds), between 8 GiB and
100 GiB and never above 80% of that room. The RAM cap is a fifth of the memory
the process may use (the host's, or a container's cgroup limit when smaller),
between 512 MiB and 4 GiB. `cash info` prints both.

### Notebook caching policy

These decide which notebook statements are cached and written to disk. A
`@cash.cache` result is always stored, whatever these say. See
[Cost model](../cost-model.md).

| Field | Env var | Default | Path | Description |
|---|---|---|---|---|
| `persist_all` | `CASH_PERSIST_ALL` | `false` | notebook | Cache every statement, skipping the floors below, as if each had `# @cash:persist`. |
| `min_execution_time_to_cache_seconds` | `CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS` | `0.01` | notebook | A statement faster than this is not cached at all. |
| `min_cache_savings_pct` | `CASH_MIN_CACHE_SAVINGS_PCT` | `0.20` | notebook | Fraction of the compute time a predicted restore must save for a value to be written to disk. |
| `min_cache_fixed_budget_seconds` | `CASH_MIN_CACHE_FIXED_BUDGET_SECONDS` | `0.05` | notebook | A predicted restore time below this is always acceptable, however short the compute. |
| `call_cost_floor_seconds` | `CASH_CALL_COST_FLOOR_SECONDS` | `0.003` | notebook | A call inside a statement is cached only when it runs at least this long. |
| `loop_split_max_iter_seconds` | `CASH_LOOP_SPLIT_MAX_ITER_SECONDS` | `0.006` | notebook | A loop whose iterations each take less than this may be cached as one unit instead of per call. |
| `loop_split_min_remaining_seconds` | `CASH_LOOP_SPLIT_MIN_REMAINING_SECONDS` | `0.1` | notebook | ...but only when at least this much work remains in the loop. |

### Remote files

| Field | Env var | Default | Path | Description |
|---|---|---|---|---|
| `remote_revalidate_max_age_seconds` | `CASH_REMOTE_REVALIDATE_MAX_AGE_SECONDS` | `0.0` | both | Seconds a remote file's state may be reused before the store is asked again. `0` checks on every hit. A higher value means a change can go unseen for that long; prefer `immutable=True` on a [`RemoteFileDataSource`](../api/data_sources.md) where you have one. |

### Output and switches

| Field | Env var | Default | Path | Description |
|---|---|---|---|---|
| `debug` | `CASH_DEBUG` | `false` | both | Log every cache decision, including one line per decorated call. The lines go to stderr unless your program configures `logging`. `%cash_debug on` in a notebook sets this field and also shows the notebook's debug lines. |
| `verbose` | `CASH_VERBOSE` | `false` | decorator | Only the one line per decorated call, without the other debug records. |
| `disable` | `CASH_DISABLE` | `false` | both | Run every `@cash.cache` call uncached, and make `%cash_on` decline. `CASH_DISABLE=1 pytest` checks your tests pass without the cache ([Testing your code](../tutorials/feature-guides/testing-your-code.md)). |
| `summary` | `CASH_SUMMARY` | `false` | decorator | At exit, print a per-function hit/miss table to stderr, with why each function missed: `CASH_SUMMARY=1 python model.py`. |
| `analytics` | `CASH_ANALYTICS` | `true` | notebook | Record each statement's hit, miss and timing in `analytics.db` under the per-user cache root, for the `cash.show_stats()` dashboard. `false` creates no file. |

### Backend

| Field | Env var | Default | Path | Description |
|---|---|---|---|---|
| `backend` | `CASH_BACKEND` | `"tiered"` | both | `"tiered"` is a RAM tier in front of a disk tier. `"memory"`, `"file"`, `"sqlite"`, `"redis"` or `"s3"` is that one backend. Ignored when `tiers` is set. |
| `redis_host` | `CASH_REDIS_HOST` | `"localhost"` | both | Redis server, for `backend = "redis"` or a redis tier. |
| `redis_port` | `CASH_REDIS_PORT` | `6379` | both | Redis port. |
| `redis_db` | `CASH_REDIS_DB` | `0` | both | Redis database number. |
| `redis_password` | `CASH_REDIS_PASSWORD` | `null` | both | Redis password. Prefer the environment variable to a committed file. |
| `redis_prefix` | `CASH_REDIS_PREFIX` | `"cash:"` | both | Key prefix, so several apps can share one Redis. |
| `s3_bucket` | `CASH_S3_BUCKET` | `""` | both | S3 bucket; required for `backend = "s3"`. |
| `s3_region` | `CASH_S3_REGION` | `""` | both | S3 region, such as `"us-east-1"`. Empty uses your AWS configuration. |
| `s3_prefix` | `CASH_S3_PREFIX` | `"cash/"` | both | Object key prefix. |

See [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md) for
which to pick.

### Tiers

`tiers` (path: both) replaces the stack `backend` names with your own list,
fastest first. It is set in a config file, or field by field with
`CASH_TIER_<N>_<FIELD>` variables (`<N>` counts from 0); there is no
`CASH_TIERS` variable.

```toml
[[tool.cash.tiers]]
type = "memory"
max_entries = 10000

[[tool.cash.tiers]]
type = "redis"
host = "redis.internal"

[[tool.cash.tiers]]
type = "s3"
bucket = "my-cache"
region = "us-east-1"
```

```bash
export CASH_TIER_1_HOST=prod-redis.example.com   # override one tier's host
```

A tier's `type` decides which other keys it uses (details under
[`TierConfig`](../api/config.md#cash.config.TierConfig)). A key its type does
not use does nothing, and cash warns
[`CONFIG-INVALID`](../warnings.md#config-invalid).

<!-- claim: cash/config.py:_TIER_FIELDS @3f70dfeb, cash/config.py:TierConfig.__post_init__ @afa4a855 -->
| `type` | Keys |
|---|---|
| `memory` | `max_entries`, `max_size_bytes` |
| `file` | `cache_dir`, `max_size_bytes`, `default_ttl`, `compress`, `flush_interval` |
| `sqlite` | `cache_dir`, `db_path`, `max_size_bytes`, `default_ttl`, `wal_mode` |
| `redis` | `host`, `port`, `db`, `password`, `prefix` |
| `s3` | `bucket`, `region`, `prefix` |

A key a tier leaves out comes from the top-level setting of the same meaning:
a `file` tier without `cache_dir` uses `cache_dir`, one without
`max_size_bytes` uses `max_cache_size`.

<!-- claim: cash/backends/store_notices.py:StoreNotices.too_big @0b79929c, cash/backends/redis_backend.py:RedisBackend.max_size_bytes == 10485760, cash/backends/sqlite_backend.py:SQLiteBackend.max_size_bytes == 104857600, cash/backends/sqlite_backend.py:SQLiteBackend.promotion_size_cap @36b1d2f8 -->
In a stack, a value bigger than a tier's size cap skips that tier and goes to
the others. The disk tier's cap is its whole `max_cache_size`; Redis tiers
skip values over 10 MiB, and SQLite tiers values over 100 MiB or their own
`max_size_bytes` if that is smaller. Memory and S3 tiers have no such cap.

## Config files

<!-- claim: cash/_location.py:default_user_config_path @5fbd345e, cash/_location.py:default_project_config_path @52360884 -->
The project file keeps settings under `[tool.cash]` in `pyproject.toml`. The
user file (paths above) keeps them under `[cash]`:

```toml
[cash]
debug = true
redis_host = "redis.example.com"
```

`cash.create_default_config()` writes a user file with every setting
commented out at its default. It will not overwrite an existing file unless
you pass `force=True`.

On Python 3.10, reading TOML needs `tomli`: install `cash-lib[toml]`
([`CONFIG-TOML-UNREADABLE`](../warnings.md#config-toml-unreadable)).

### A config file named in code

<!-- test:expect-warning reason="the example names a file the docs harness does not create, so CONFIG-FILE-MISSING (described below) is exactly what cash should say" -->
```python
from cash import Cash

app = Cash(config_path="./my_special_config.toml")
```

<!-- claim: cash/config.py:_resolve_config @18f97088 -->
The named file ranks above the project and user files and below environment
variables and code. That lets an installed package ship its own settings: put
a TOML file inside the package and pass
`Path(__file__).with_name("cash.toml")`. The file keeps its settings under
`[cash]` or `[tool.cash]`. A relative `cache_dir` in it is relative to the
file, and `~` is your home directory.

<!-- claim: cash/config.py:_resolve_config @18f97088 -->
A missing file warns [`CONFIG-FILE-MISSING`](../warnings.md#config-file-missing).
`cash info --config path/to/cash.toml` shows what a file resolves to.

<!-- claim: cash/__init__.py:configure @945b5c80 -->
## Changing settings at runtime

`cash.configure(...)` changes the settings of the default instance used by
`@cash.cache`; `Cash.reconfigure(...)` does the same for an instance of your
own. Neither writes to a file.

```python
import cash

cash.configure(debug=True)
cash.configure(max_cache_size="10GB")
```

Settings that shape the backend (`cache_dir`, `compress`, `max_cache_size`,
`max_memory_entries`, `flush_interval`, `backend`, `tiers`, or a connection
detail of a tier in use) make cash finish the old backend's pending writes and
build a new one. All other settings take effect on the next operation. A
connection detail for a backend not in use is kept for later.
`configure(debug=False)` also stops the log output `debug=True` started.

<!-- claim: cash/reconfigure.py:apply_overrides @3b83cdfd, cash/config.py:validated_overrides @db2d884f -->
Values are checked exactly as `Cash(...)` checks them, before anything
changes: a bad value raises `ValueError` and leaves the old settings in place.
`~` is expanded, and a relative `cache_dir` is relative to the current
directory. An instance built with a backend object (`Cash(backend=...)`) keeps
it: changing its tier settings raises `ValueError`.

## Magics that set these fields

In a notebook, two magics change these settings for the session:

- `%cash_debug on` / `off` sets `debug`.
- `%cash_persist on` / `off` sets `persist_all`.

`%cash_on ttl=N` gives every cached statement a TTL for the session; it is not
a config field. See [Magic commands](../magics.md).
