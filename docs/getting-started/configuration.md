# Configuration

Cash settles its configuration from five sources in priority order:

1. **Constructor kwargs** — `Cash(redis_host="...", debug=True)` (or any
   `CashConfig` field name).
2. **Environment variables** — `CASH_*` (every `CashConfig` field has a
   binding, plus `CASH_TIER_<N>_<FIELD>` for tier overrides).
3. **Project config** — `[tool.cash]` in the nearest `pyproject.toml`
   (walks up from the current working directory).
4. **User config** — `~/.config/cash/config.toml` on Linux/macOS, or
   `%APPDATA%\cash\config.toml` on Windows. Honours `$XDG_CONFIG_HOME`.
5. **Built-in defaults** from the `CashConfig` dataclass.

Each layer overrides the next. A single field can be set wherever is
most convenient — explicit code for one-off scripts, `pyproject.toml`
for team defaults, env vars for deployment overrides, the XDG file for
personal cross-project defaults.

## Quick reference

```python
from cash import Cash, configure

# Layer 1: constructor kwargs
cash = Cash(cache_dir="/tmp/scratch", debug=True)

# Or set in pyproject.toml [tool.cash]:
#   [tool.cash]
#   cache_dir = ".cash"
#   debug = true

# Or set via environment:
#   export CASH_CACHE_DIR=/var/cache
#   export CASH_DEBUG=true

# Or change at runtime on the default singleton:
configure(debug=True, smart_persistence_threshold=0.5)
```

## All `CashConfig` fields

Every field below is settable via every layer. The env-var column shows
the `CASH_*` binding; the TOML key matches the field name.

### Cache location & file-backend tuning

| Field | Env var | Default | Description |
|---|---|---|---|
| `cache_dir` | `CASH_CACHE_DIR` | `".cash"` | Where the default `FileBackend` writes. **Add to `.gitignore`** — this is the disk cache, not the config. |
| `compress` | `CASH_COMPRESS` | `false` | gzip data files on disk. |
| `max_cache_size` | `CASH_MAX_CACHE_SIZE` | `1073741824` (1 GiB) | LRU eviction threshold on `FileBackend`, in bytes. |
| `max_memory_entries` | `CASH_MAX_MEMORY_ENTRIES` | `null` (unlimited) | Cap on `InMemoryBackend` entries — LRU eviction when exceeded. |
| `flush_interval` | `CASH_FLUSH_INTERVAL` | `5` | Seconds between `FileBackend`'s background metadata-flush cycles. |

### Cost-aware caching policy

| Field | Env var | Default | Description |
|---|---|---|---|
| `smart_persistence` | `CASH_SMART_PERSISTENCE` | `true` | Use the cost-model promotion policy. If `false`, falls back to `_default_promotion_policy` (same rule, 1.0 s floor). |
| `smart_persistence_threshold` | `CASH_SMART_PERSISTENCE_THRESHOLD` | `1.0` (seconds) | **Legacy / no longer consulted (CAS-141).** Superseded by the cost-model comparison; kept for compatibility. |
| `min_execution_time_to_cache_seconds` | `CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS` | `0.01` | "Too cheap to cache at all" floor — statements faster than this never get a cache entry. |
| `min_cache_savings_pct` | `CASH_MIN_CACHE_SAVINGS_PCT` | `0.20` | Required savings fraction for promotion — used by both the notebook Gate A and the tier promotion policy. |
| `min_cache_fixed_budget_seconds` | `CASH_MIN_CACHE_FIXED_BUDGET_SECONDS` | `0.05` | Notebook path: always allow caching when predicted restore is below this. |

### Observability

| Field | Env var | Default | Description |
|---|---|---|---|
| `debug` | `CASH_DEBUG` | `false` | Verbose `[CACHE]` / `[UPSTREAM_DEBUG]` / `[LINEAGE_DEBUG]` log lines. |

### Backend selection — simple mode

The simplest way to pick a backend: set `backend` and the connection
fields for that backend. Cash builds it for you.

| Field | Env var | Default | Description |
|---|---|---|---|
| `backend` | `CASH_BACKEND` | `"tiered"` | One of `"tiered"`, `"memory"`, `"file"`, `"sqlite"`, `"redis"`, `"s3"`. Default builds `TieredBackend([RAM, FileBackend])` from `cache_dir`/`compress`/`max_cache_size`. |

**Redis connection (used when `backend = "redis"`):**

| Field | Env var | Default |
|---|---|---|
| `redis_host` | `CASH_REDIS_HOST` | `"localhost"` |
| `redis_port` | `CASH_REDIS_PORT` | `6379` |
| `redis_db` | `CASH_REDIS_DB` | `0` |
| `redis_password` | `CASH_REDIS_PASSWORD` | `null` |
| `redis_prefix` | `CASH_REDIS_PREFIX` | `"cash:"` |

**S3 connection (used when `backend = "s3"`):**

| Field | Env var | Default |
|---|---|---|
| `s3_bucket` | `CASH_S3_BUCKET` | `""` (required when `backend = "s3"`) |
| `s3_region` | `CASH_S3_REGION` | `""` |
| `s3_prefix` | `CASH_S3_PREFIX` | `"cash/"` |

### Backend selection — advanced (declarative tier stack)

For multi-tier setups (RAM + Redis + DISK + S3, say), set the `tiers`
list. When `tiers` is non-empty it takes precedence over `backend`.

```toml
[[tool.cash.tiers]]
type = "memory"
max_entries = 10000

[[tool.cash.tiers]]
type = "redis"
host = "redis.internal"
port = 6379
prefix = "myapp:"

[[tool.cash.tiers]]
type = "s3"
bucket = "my-cache"
region = "us-east-1"
```

Each tier's per-backend fields are documented at
[Backends](../api/backends.md).

Per-field env-var overrides for tier entries: `CASH_TIER_<N>_<FIELD>`
where `<N>` is the zero-based index. For example, to override only the
Redis tier's host without rewriting the TOML:

```bash
export CASH_TIER_1_HOST=prod-redis.example.com
```

### Per-tier size caps

Each backend declares a class-level `max_size_bytes` cap that
`TieredBackend` uses as a *promotion hint*. A value larger than the cap
quietly skips that tier but still writes to the unconstrained ones.

| Backend | `max_size_bytes` cap | Rationale |
|---|---|---|
| `InMemoryBackend` | unbounded | RAM eviction handles pressure separately. |
| `FileBackend` | unbounded | Disk has its own `max_cache_size` LRU. |
| `RedisBackend` | **10 MiB** | Redis is in-memory server-side; protocol disfavours multi-MB values. |
| `SQLiteBackend` | **100 MiB** | SQLite blobs degrade past this. |
| `S3Backend` | unbounded | S3 is fine arbitrarily large. |

Caps apply only to the **tiered pipeline**. A bare `RedisBackend()`
constructed directly accepts whatever you give it — the cap is an
opt-in promotion hint, not a hard wall.

## File locations

### `pyproject.toml` (project, version-controlled)

Most projects should put their config here so the team shares one
default. Cash walks upward from the current working directory until it
finds a `pyproject.toml`.

```toml
[tool.cash]
cache_dir = ".cash"
compress = true
max_cache_size = 5368709120   # 5 GiB

[[tool.cash.tiers]]
type = "memory"

[[tool.cash.tiers]]
type = "redis"
host = "redis.internal"
```

### `~/.config/cash/config.toml` (user, machine-private)

For personal defaults spanning all projects on a machine — e.g. your
personal Redis URL or a global `debug = true` while you're investigating
an issue. Follows the [XDG Base Directory
Spec](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html):

- Linux/macOS: `~/.config/cash/config.toml` (or `$XDG_CONFIG_HOME/cash/config.toml`)
- Windows: `%APPDATA%\cash\config.toml`

The body is a TOML file with a `[cash]` section:

```toml
[cash]
debug = true
redis_host = "redis.example.com"
```

Generate a documented template with `cash.create_default_config()`.

> ⚠️ The legacy `~/.cash/config.toml` location is **no longer read** —
> move your config to the XDG path above. (The `.cash/` directory
> remains used for the disk cache itself, which is why having config
> there too was confusing.)

### Environment variables

Every field on `CashConfig` has a `CASH_<UPPERCASE_FIELD_NAME>` env-var
binding. The full list is in the tables above. The tier-list field is
additionally overridable element-by-element with `CASH_TIER_<N>_<FIELD>`.

### Per-script overrides via `config_path`

```python
cash = Cash(config_path="./my_special_config.toml")
```

Loads the named TOML as the user-level layer (so env vars and
constructor kwargs still override it).

## Runtime mutation: `cash.configure()`

Change the active configuration of the default singleton at runtime
without restarting:

```python
import cash

# Hot fields — just update the dataclass, no rebuild.
cash.configure(debug=True)
cash.configure(smart_persistence_threshold=0.5)

# Backend-affecting fields — drain pending writes on the current
# backend, build a fresh one from the updated config, swap it in.
cash.configure(backend="redis", redis_host="prod-redis.internal")

# Pre-set stale fields — stored silently, picked up later.
cash.configure(redis_host="staging.example.com")
# ... later ...
cash.configure(backend="redis")   # uses the staging host you pre-set
```

**What it does not do**: write to `pyproject.toml` or the XDG config
file. The function is in-memory only — persistence requires editing
those files directly.

**What gets rebuilt**: the function distinguishes "hot" fields (debug,
smart-persistence policy knobs) from "backend-affecting" fields
(cache_dir, max_cache_size, max_memory_entries, flush_interval,
backend, tiers, all connection details). Hot fields just update the
dataclass. Backend-affecting fields drain the old backend's pending
writes via `shutdown()`, build a fresh backend from the new config,
and swap it in.

## Notebook-only knobs

Two settings apply only to the IPython integration and are toggled via
magic commands, not the config layer:

### `%cash_debug` — debug output on/off

```python
%cash_debug on    # detailed `[CACHE]` / `[LINEAGE_DEBUG]` traces
%cash_debug off
```

Equivalent to `cash.configure(debug=True/False)`.

### `%cash_badge` — badge display mode

```python
%cash_badge html    # interactive HTML badges (default)
%cash_badge print   # text summary after cell completes
%cash_badge off     # no badge output
```

### `%cash_on ttl=N` — global TTL

```python
%cash_on              # enable auto-caching
%cash_on ttl=3600     # 1-hour TTL on every cached entry
```

## Background writes

Every backend except RAM serialises the value on the calling thread,
then performs the actual storage write in a per-backend background
thread. `set()` returns once the bytes are captured — the disk/network
write happens asynchronously.

This means:

- `Cash` cells finish quickly even when promoting to slow tiers (S3).
- A `get()` for the same key transparently waits for any in-flight
  write — you never see stale-or-missing data.
- `delete(key)` drains pending writes for that key before deleting.
- Process exit triggers `atexit` → drains every backend's pending writes
  before the interpreter terminates. (One exception: `os._exit()` or
  `SIGKILL` skip `atexit`. Same caveat as any Python program.)
- A bare backend instance — `RedisBackend(...)` not wrapped in a `Cash`
  object — is your own to shut down. Call `backend.shutdown()` (or use
  a `try`/`finally`) for durable writes outside `Cash`'s atexit chain.

There's nothing to configure — it's the default behaviour. The legacy
`background_io` parameter and `AsyncBackendWrapper` class have been
removed.

## File tracking

Cash automatically tracks file reads through common libraries:

<!-- test:skip reason="reads files (data.csv, array.npy, model.pkl, file.txt) that don't exist in test env" -->
```python
df = pd.read_csv('data.csv')        # pandas (all read_* functions)
arr = np.load('array.npy')           # numpy
pl_df = pl.read_csv('data.csv')      # polars (read_* and scan_*)
data = joblib.load('model.pkl')      # joblib
with open('file.txt') as f:          # built-in open
    content = f.read()
```

File tracking is always on for these recognised patterns — there's no
global toggle. To opt a single statement out of caching (and therefore
out of file-dependency tracking for that statement), use the
`# @cash:no-cache` annotation described in the
[Quick Start](quickstart.md#statement-level-annotations).

## See also

- [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md) —
  decision guide for picking between RAM-only, tiered, Redis, S3, and
  custom backend stacks for your workload.
- [Smart persistence](../tutorials/feature-guides/smart-persistence.md) —
  how `smart_persistence_threshold` and the cost-model knobs interact, and
  when to override them.
