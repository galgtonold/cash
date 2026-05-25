# Choosing a backend — where Cash actually stores your results

Cash supports six backends. The default — `TieredBackend` with in-memory L1 + filesystem L2 — works for most users. Choose differently when you need cross-process sharing, network durability, or memory-only workflows.

This guide walks through each backend, when to use it, and exactly how to wire it up.

## Why this exists

Cash is one Python API on top of pluggable storage. The decorator (`@cash.cache`) and the notebook magic (`%cash_on`) don't care where bytes land — that's the backend's job. Backends differ on three axes that matter in practice:

- **Persistence** — do cached values survive a kernel restart, a process restart, a machine reboot, or none of those?
- **Speed** — RAM is ~100× faster than disk, which is ~100× faster than the network. A wrong choice here turns a "fast cache hit" into "still slow".
- **Sharing** — does a second process see what the first process cached? Cross-machine? Cross-region?

You can change backends at any time by constructing a different `Cash(...)` instance — your code doesn't change.

## Quick start — the decision tree

Walk through these questions top to bottom and stop at the first match:

- **Single user, single machine, single process** → the default `TieredBackend` (in-memory L1 + on-disk L2). Just call `Cash()` with no backend argument.
- **Single machine, multi-process or multi-script** (e.g. several Jupyter kernels, a notebook + a CLI script) → `FileBackend` or `SQLiteBackend`. SQLite is better when you have many small entries and need concurrent readers.
- **Multiple machines, low latency, shared state** (team, microservices, Airflow workers on the same VPC) → `RedisBackend`.
- **Cloud pipelines, durable, multi-region** (CI artefacts, S3-backed Lambda jobs) → `S3Backend`.
- **Throwaway experiments, persistence explicitly unwanted** (one-off notebooks, fuzzing, integration tests) → `InMemoryBackend` alone.

## The backend table

| Backend | Persistence | Speed | Sharing | Best for |
|---------|-------------|-------|---------|----------|
| `InMemoryBackend` | Kernel restart clears | Fastest | Single process | Quick experiments |
| `FileBackend` | Survives restarts | Fast | Same machine | Default for most users |
| `SQLiteBackend` | Survives restarts | Fast | Same machine | Many small entries, concurrent readers |
| `TieredBackend` | Two layers | L1 fast, L2 persistent | Same machine | **Default** (recommended) |
| `RedisBackend` | Configurable | Network-fast | Multi-process, multi-host | Teams, microservices |
| `S3Backend` | Cloud-durable | Network | Multi-region | Cloud pipelines |

## `InMemoryBackend`

```python
from cash import Cash, InMemoryBackend

c = Cash(backend=InMemoryBackend(max_entries=500))
c.register_magic()
```

A plain dict guarded by light bookkeeping. Reads and writes deep-copy by default so a downstream mutation can't poison the cache (`src/cash/backends/memory_backend.py:49-69`). Eviction has two triggers: `max_entries` enforces a hard LRU cap (`src/cash/backends/memory_backend.py:85-86`, `:210-221`), and a `psutil` memory-pressure check kicks in every `check_interval` writes when the system crosses `max_memory_percent` (`src/cash/backends/memory_backend.py:159-208`).

**Key parameters** — `max_entries` (None = unlimited), `max_memory_percent` (default 0.9 = 90% of system RAM), `check_interval` (default 10 writes between pressure checks).

**Gotcha** — kernel restart wipes everything. There is no on-disk fallback. Pair it with `TieredBackend` if you also want persistence.

## `FileBackend`

```python
from cash import Cash, FileBackend

c = Cash(backend=FileBackend(
    cache_dir="./.cash",
    compress=False,
    max_size_bytes=2 * 1024**3,   # 2 GiB cap
))
c.register_magic()
```

One file per entry under `cache_dir`, sharded by SHA-256 of the cache key (`src/cash/backends/file_backend.py:134-140`). Writes are split: serialization happens on the calling thread, the actual disk write runs on a background executor so a slow write doesn't block the cell (`src/cash/backends/file_backend.py:282-317`). A second thread flushes metadata every `flush_interval` seconds (`src/cash/backends/file_backend.py:109-113`).

Eviction is LRU on `last_access`. When `_current_size_bytes` exceeds `max_size_bytes`, the oldest entries are dropped until the cache fits under 90% of the cap (`src/cash/backends/file_backend.py:404-428`).

**Key parameters** — `cache_dir`, `compress` (gzip; usually only worth it for CSV/JSON), `max_size_bytes` (None = unlimited), `flush_interval` (seconds; 0 = flush on every write), `default_ttl` (seconds).

**Gotcha** — uses `pickle` under the hood. Never load a cache directory from an untrusted source. See `SECURITY.md`.

## `SQLiteBackend`

```python
from cash import Cash, SQLiteBackend

c = Cash(backend=SQLiteBackend(
    db_path="./.cash/cache.db",
    wal_mode=True,
    max_size_bytes=1024**3,
))
c.register_magic()
```

One SQLite database file holds every entry. Better than `FileBackend` when you have thousands of small entries — directory enumeration starts to drag, but a single indexed table doesn't. WAL journal mode is on by default for concurrent readers (`src/cash/backends/sqlite_backend.py:62-65`).

Like `FileBackend`, writes are split: serialize on the calling thread, INSERT on the background worker (`src/cash/backends/sqlite_backend.py:144-178`). Eviction is LRU based on `last_access`, triggered when total size crosses `max_size_bytes` (`src/cash/backends/sqlite_backend.py:258-285`).

**Key parameters** — `db_path`, `default_ttl`, `max_size_bytes`, `wal_mode` (default True).

**When SQLite beats File** — many small entries (thousands), concurrent reads from multiple processes, or you want one file to back up rather than a directory tree. Note the **100 MiB per-entry promotion cap** when used inside a tiered stack (`src/cash/backends/sqlite_backend.py:42-43`) — values larger than that skip SQLite and go straight to the next tier.

## `TieredBackend` (the default)

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

A list of backends ordered fastest-first. A `get` walks the list in order; on a hit, the value is promoted (written back) to every faster tier so the next read comes from RAM (`src/cash/backends/tiered_backend.py:49-76`). A `set` always writes to tier 0, then asks the promotion policy whether each subsequent tier should also get a copy.

The default policy (`_default_promotion_policy` at `src/cash/backends/tiered_backend.py:33-47`) is a two-gate filter:

1. Execution time must be at least 1.0 second — anything faster than that isn't worth persisting.
2. Re-execution must take longer than re-reading from the next tier (size / 100 MB/s).

So a `pd.read_csv` that takes 50 ms and produces a 10 MB frame stays in RAM; a model fit that takes 30 s and produces a 200 MB pickle goes to RAM **and** disk. See [Smart Persistence](smart-persistence.md) for the full policy and how `@cash:persist` overrides it.

**Key parameters** — `backends` (list, fastest first), `promotion_policy` (callable `(execution_time, size_bytes) -> bool`).

**When to depart from the default** — you don't, usually. Build a custom tier stack when you want three layers (RAM → SQLite → S3) or want to swap the L2 from file to Redis.

## `RedisBackend`

```python
from cash import Cash
from cash.backends import RedisBackend

c = Cash(backend=RedisBackend(
    host="redis.team.internal",
    port=6379,
    db=0,
    password=None,
    prefix="cash:proj-a:",
))
c.register_magic()
```

Two Redis keys per entry — `{prefix}{key}:meta` and `{prefix}{key}:data` — written together inside a pipeline so they land atomically (`src/cash/backends/redis_backend.py:128-143`). TTL, if set on the metadata, is applied via `EXPIRE` on both keys.

**Key parameters** — `host`, `port`, `db`, `password`, `prefix` (default `cash:`), plus connection-resilience knobs (`socket_keepalive`, `health_check_interval`, `retry_on_timeout`, `max_retries`).

**Caveats** —

- Requires the `redis` package (`pip install redis`). The import is lazy; if it's missing, the constructor raises `DependencyNotFoundError`.
- Everything is pickled. Don't load entries written by a different Python version into a fragile reader, and don't share a Redis with an untrusted writer.
- **10 MiB per-entry promotion cap** when inside a tiered stack (`src/cash/backends/redis_backend.py:37`). Bigger values skip Redis and land on the next tier.
- The `prefix` is the only thing keeping multiple apps from clobbering each other in the same Redis. Pick a unique one.

## `S3Backend`

```python
from cash import Cash
from cash.backends import S3Backend

c = Cash(backend=S3Backend(
    bucket="my-team-cache",
    prefix="cash/proj-a/",
    region_name="us-east-1",         # passed through to boto3.client
))
c.register_magic()
```

Two S3 objects per entry — `{prefix}{key}.meta` and `{prefix}{key}.data` — uploaded in sequence with the data going first so a partial failure never leaves a metadata pointer to a missing payload (`src/cash/backends/s3_backend.py:113-119`).

**Key parameters** — `bucket` (required), `prefix` (default `cash/`), `max_pool_connections`, `retries`, plus any kwargs accepted by `boto3.client('s3', ...)` (region, profile, credentials).

**Cost considerations** —

- Every cache miss is a GET; every cache write is a PUT. Both cost money and have round-trip latency in the tens of milliseconds. Put S3 last in your tier stack, not first.
- S3 has no native TTL on object content; metadata-level TTL still works because Cash checks it on read, but the bytes stay in the bucket until you set a lifecycle policy or call `clear()`.
- Requires `boto3` (`pip install boto3`).

## Configuration via env vars

Every backend choice is also a config option, so you can override the backend without touching code:

```bash
export CASH_BACKEND=redis
export CASH_REDIS_HOST=cache.prod
export CASH_REDIS_PORT=6379
export CASH_REDIS_PASSWORD=...
```

For an explicit multi-tier stack, use the `CASH_TIER_<N>_<FIELD>` form (`src/cash/config.py:368-428`):

```bash
export CASH_TIER_0_TYPE=memory
export CASH_TIER_0_MAX_ENTRIES=200
export CASH_TIER_1_TYPE=redis
export CASH_TIER_1_HOST=cache.prod
export CASH_TIER_2_TYPE=s3
export CASH_TIER_2_BUCKET=my-team-cache
```

The tier list, when non-empty, takes precedence over the single-backend `CASH_BACKEND` field (`src/cash/backends/factory.py:41-47`). The same fields are available under `[tool.cash]` in `pyproject.toml` and `[cash]` in `~/.config/cash/config.toml`. See the [Configuration reference](../../getting-started/configuration.md) for the full resolution order.

## Notebook vs decorator — same backend

Both the notebook magic and the decorator use the same `Cash` instance under the hood, so picking a backend works the same in either:

```python
# Decorator path
from cash import Cash, FileBackend
c = Cash(backend=FileBackend(cache_dir="./.cash"))

@c.cache
def expensive(x): ...

# Notebook path — same `c`, just register the magic too
c.register_magic()
# %cash_on now uses the FileBackend you constructed above
```

Env vars resolve to the same `CashConfig` regardless of entry point. There is no separate "notebook backend" and "decorator backend".

## API reference (compact)

| Backend | Import path | Required parameter | Key knobs |
|---|---|---|---|
| `InMemoryBackend` | `from cash import InMemoryBackend` | — | `max_entries`, `max_memory_percent` |
| `FileBackend` | `from cash import FileBackend` | `cache_dir` | `compress`, `max_size_bytes`, `flush_interval`, `default_ttl` |
| `SQLiteBackend` | `from cash import SQLiteBackend` | `db_path` | `default_ttl`, `max_size_bytes`, `wal_mode` |
| `TieredBackend` | `from cash.backends import TieredBackend` | `backends` (list) | `promotion_policy` |
| `RedisBackend` | `from cash.backends import RedisBackend` | `host` | `port`, `db`, `password`, `prefix`, retry/keepalive kwargs |
| `S3Backend` | `from cash.backends import S3Backend` | `bucket` | `prefix`, `max_pool_connections`, `retries`, boto3 kwargs |
| `CashConfig.backend` | `CASH_BACKEND` env / TOML | one of `memory`/`file`/`sqlite`/`redis`/`s3`/`tiered` | resolved by `build_backend_from_config` |
| `CashConfig.tiers` | `CASH_TIER_<N>_*` env / TOML | list of `TierConfig` | takes precedence over `backend` |

## Related

- [Configuration](../../getting-started/configuration.md) — full config resolution hierarchy (kwargs > env > project TOML > user TOML > defaults).
- [Smart Persistence](smart-persistence.md) — how the tiered promotion policy decides what to write past RAM and how `@cash:persist` overrides it.
- [Production Transition](production-transition.md) — backend selection when you move from notebook to script.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — TTL, `@cash:no-cache`, and other per-entry overrides.
- [Debugging and Monitoring](debugging-and-monitoring.md) — how to confirm your chosen backend is actually being used.
