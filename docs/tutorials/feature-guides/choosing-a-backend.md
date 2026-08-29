# Choosing a backend — where Cash actually stores your results

Cash ships seven backend classes; six of them are selectable from configuration. The default — `TieredBackend` with in-memory L1 + filesystem L2 — works for most users. Choose differently when you need cross-process sharing, network durability, or memory-only workflows.

This guide walks through each backend, when to use it, and exactly how to wire it up.

## Why this exists

Every backend exposes the same API — `@cash.cache` and `%cash_on` don't care where bytes land. They differ on three axes:

- **Persistence** — do cached values survive a kernel restart, a process restart, a machine reboot, or none of those?
- **Speed** — RAM is ~100× faster than disk, which is ~100× faster than the network. A wrong choice here turns a "fast cache hit" into "still slow".
- **Sharing** — does a second process see what the first process cached? Cross-machine? Cross-region?

You can change backends at any time by constructing a different `Cash(...)` instance — your code doesn't change.

## Quick start — the decision tree

Walk through these questions top to bottom and stop at the first match:

- **Single user, single machine, single process** → the default `TieredBackend` (in-memory L1 + on-disk L2). Just call `Cash()` with no backend argument.
- **Single machine, multi-process or multi-script** (e.g. several Jupyter kernels, a notebook + a CLI script) → `FileBackend` or `SQLiteBackend`. SQLite is better when you have many small entries and need concurrent readers.
- **Multiple machines, low latency, shared state** (team, microservices, Airflow workers on the same VPC) → `RedisBackend`. See [Sharing a cache](sharing-caches.md) for what actually produces cross-machine hits once the store is shared.
- **Cloud pipelines, durable, multi-region** (CI artefacts, S3-backed Lambda jobs) → `S3Backend`.
- **Throwaway experiments, persistence explicitly unwanted** (one-off notebooks, fuzzing, integration tests) → `InMemoryBackend` alone.

## The backend table

<!-- claim: cash/backends/__init__.py:__all__ @47f3d1b2 broad="the count and the table are a claim about the exported backend set" -->
| Backend | Persistence | Speed | Sharing | Best for |
|---------|-------------|-------|---------|----------|
| `InMemoryBackend` | Kernel restart clears | Fastest | Single process | Quick experiments |
| `FileBackend` | Survives restarts | Fast | Same machine | Default for most users |
| `SQLiteBackend` | Survives restarts | Fast | Same machine | Many small entries, concurrent readers |
| `TieredBackend` | Two layers | L1 fast, L2 persistent | Same machine | **Default** (recommended) |
| `RedisBackend` | Configurable | Network-fast | Multi-process, multi-host | Teams, microservices |
| `S3Backend` | Cloud-durable | Network | Multi-region | Cloud pipelines |
| `CascadingBackend` | Per-member | Slowest member on write | Per-member | Write-to-all mirroring (code-only; no `CASH_BACKEND` value) |

## `InMemoryBackend`

```python
from cash import Cash, InMemoryBackend

c = Cash(backend=InMemoryBackend(max_entries=500))
c.register_magic()
```

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend.__init__ @42a1a958, cash/backends/memory_backend.py:InMemoryBackend._evict @03ba0434 -->
A plain dict guarded by light bookkeeping. Reads and writes deep-copy by default so a downstream mutation can't poison the cache. Eviction has **three** triggers, and they do not use the same policy:

1. `max_entries` — a hard LRU cap, evicting oldest-accessed first (`_evict_lru`).
2. `max_size_bytes` — a soft byte cap (`_evict_to_byte_cap`).
3. A `psutil` memory-pressure check, run every `check_interval` writes, that fires when the system crosses `max_memory_percent`. This one is **not** LRU: `_evict` scores each entry as `(execution_time × access_count) / size` and drops the lowest-scoring first, so a big cheap entry goes before a small expensive one.

**Key parameters** — `max_entries` (None = unlimited), `max_size_bytes` (None = unlimited), `max_memory_percent` (default 0.9 = 90% of system RAM), `check_interval` (default 10 writes between pressure checks).

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

One file per entry under `cache_dir`, sharded by SHA-256 of the cache key. Writes are split: serialization happens on the calling thread, the actual disk write runs on a background executor so a slow write doesn't block the cell. A second thread flushes metadata every `flush_interval` seconds.

Eviction is LRU on `last_access`. When `_current_size_bytes` exceeds `max_size_bytes`, the oldest entries are dropped until the cache fits under 90% of the cap.

**Key parameters** — `cache_dir`, `compress` (gzip; usually only worth it for CSV/JSON), `max_size_bytes` (None = unlimited), `flush_interval` (seconds; 0 = flush on every write), `default_ttl` (seconds).

**Gotcha** — uses `pickle` under the hood. Never load a cache directory from an untrusted source. See `SECURITY.md`.

!!! tip "Two files per entry: what it costs, and what it doesn't"
    Measured at 20,000 entries with a 512-byte payload
    (`benchmarks/bench_backend_scale.py`, Windows/NTFS):

    | | `FileBackend` | `SQLiteBackend` |
    |---|---|---|
    | Write one more entry | 1.45 ms | **0.14 ms** |
    | Read one entry | 0.26 ms | **0.06 ms** |
    | Open the cache in a new process | 61 ms | **0.1 ms** |
    | Files on disk | 40,057 | **3** |
    | Disk used | **13.3 MB** | 21.6 MB |

    **Reads and writes do not degrade as the directory fills.** From 0 to
    20,000 entries a `FileBackend` write stays at ~1.4 ms and a read at
    ~0.25 ms — the filename comes from the key, so neither operation walks the
    directory.

    What *does* scale is **opening** the cache: 1.2 ms empty, 61 ms at 20k,
    about 3.5 µs per entry, so roughly 0.35 s at 100k and 3.5 s at 1M. That is
    the size total for the eviction cap, and it is paid once per process.

    `SQLiteBackend` is faster on every operation and is one file rather than
    tens of thousands, at the price of about 1.6× the disk at this size (its
    fixed overhead amortises: 5.0 KB per entry at 1k, 1.1 KB at 20k, against a
    flat ~670 B for `FileBackend`). `FileBackend` stays the default because a
    cache directory you can open in a file browser — and see the sizes of, and
    delete by hand — is worth real money in trust, and the per-operation
    numbers do not force the switch.

    Switch when opening the cache starts to hurt: a short script run often
    against a very large cache, where a fixed startup cost is a large share of
    the run.

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

One SQLite database file holds every entry. Better than `FileBackend` when you have thousands of small entries — directory enumeration starts to drag, but a single indexed table doesn't. WAL journal mode is on by default for concurrent readers.

Like `FileBackend`, writes are split: serialize on the calling thread, INSERT on the background worker. Eviction is LRU based on `last_access`, triggered when total size crosses `max_size_bytes`.

**Key parameters** — `db_path`, `default_ttl`, `max_size_bytes`, `wal_mode` (default True).

<!-- claim: cash/backends/sqlite_backend.py:SQLiteBackend.max_size_bytes == 104857600 -->
**When SQLite beats File** — many small entries (thousands), concurrent reads from multiple processes, or you want one file to back up rather than a directory tree. Note the **100 MiB per-entry promotion cap** when used inside a tiered stack — values larger than that skip SQLite and go straight to the next tier.

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

A list of backends ordered fastest-first. A `get` walks the list in order; on a hit, the value is promoted (written back) to every faster tier so the next read comes from RAM. A `set` always writes to tier 0, then asks the promotion policy whether each subsequent tier should also get a copy.

The default policy (`TieredBackend._default_promotion_policy`) is a two-gate filter:

1. Execution time must clear a compute floor (1.0 s for this fallback) — anything faster isn't worth persisting.
2. Re-executing must cost more than restoring, using the fitted cost model's predicted read+deserialize time (`cost_model.estimated_restore_time`) — not a raw bandwidth guess.

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

Two Redis keys per entry — `{prefix}{key}:meta` and `{prefix}{key}:data` — written together inside a pipeline so they land atomically. TTL, if set on the metadata, is applied via `EXPIRE` on both keys.

**Key parameters** — `host`, `port`, `db`, `password`, `prefix` (default `cash:`), plus connection-resilience knobs (`socket_keepalive`, `health_check_interval`, `retry_on_timeout`, `max_retries`).

**Caveats** —

- Requires the `redis` package (`pip install redis`). The import is lazy; if it's missing, the constructor raises `DependencyNotFoundError`.
- Everything is pickled. Don't load entries written by a different Python version into a fragile reader, and don't share a Redis with an untrusted writer.
<!-- claim: cash/backends/redis_backend.py:RedisBackend.max_size_bytes == 10485760 -->
- **10 MiB per-entry promotion cap** when inside a tiered stack. Bigger values skip Redis and land on the next tier.
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

Two S3 objects per entry — `{prefix}{key}.meta` and `{prefix}{key}.data` — uploaded in sequence with the data going first so a partial failure never leaves a metadata pointer to a missing payload.

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

For an explicit multi-tier stack, use the `CASH_TIER_<N>_<FIELD>` form:

```bash
export CASH_TIER_0_TYPE=memory
export CASH_TIER_0_MAX_ENTRIES=200
export CASH_TIER_1_TYPE=redis
export CASH_TIER_1_HOST=cache.prod
export CASH_TIER_2_TYPE=s3
export CASH_TIER_2_BUCKET=my-team-cache
```

<!-- claim: cash/backends/factory.py:build_backend_from_config @838026dc, cash/backends/factory.py:_build_single_backend @a583bb29 -->
The tier list, when non-empty, takes precedence over the single-backend `CASH_BACKEND` field. The same fields are available under `[tool.cash]` in `pyproject.toml` and `[cash]` in `~/.config/cash/config.toml`. See the [Configuration reference](../../getting-started/configuration.md) for the full resolution order.

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
| `RedisBackend` | `from cash.backends import RedisBackend` | — (`host` defaults to `localhost`) | `port`, `db`, `password`, `prefix`, retry/keepalive kwargs |
| `S3Backend` | `from cash.backends import S3Backend` | `bucket` (the only genuinely required one) | `prefix`, `max_pool_connections`, `retries`, boto3 kwargs |
| `CascadingBackend` | `from cash.backends import CascadingBackend` | `backends` (list) | — writes to every member, read-repairs on hit |
| `CashConfig.backend` | `CASH_BACKEND` env / TOML | one of `memory`/`file`/`sqlite`/`redis`/`s3`/`tiered` | resolved by `build_backend_from_config` |
| `CashConfig.tiers` | `CASH_TIER_<N>_*` env / TOML | list of `TierConfig` | takes precedence over `backend` |

## Related

- [Configuration](../../getting-started/configuration.md) — full config resolution hierarchy (kwargs > env > project TOML > user TOML > defaults).
- [Smart Persistence](smart-persistence.md) — how the tiered promotion policy decides what to write past RAM and how `@cash:persist` overrides it.
- [Production Transition](production-transition.md) — backend selection when you move from notebook to script.
- [Controlling Cache Behavior](controlling-cache-behavior.md) — TTL, `@cash:no-cache`, and other per-entry overrides.
- [Debugging and Monitoring](debugging-and-monitoring.md) — how to confirm your chosen backend is actually being used.
