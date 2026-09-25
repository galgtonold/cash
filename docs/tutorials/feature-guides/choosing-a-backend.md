# Choosing a backend

!!! info "Applies to: both paths"
    Anyone deciding where cached results are stored: in RAM, on local disk, or
    in a shared store.

A backend is where cash keeps entries. The default, a RAM tier in front of a
disk tier, suits one person on one machine. Choose differently when several
machines should share results, when nothing may be written to disk, or when
your values are many and small.

## Which one

Stop at the first line that fits:

- **One machine**, any number of processes: the default. Nothing to configure.
- **Many small values** (thousands under 100 KB), or one file to back up rather
  than a folder: `SQLiteBackend`.
- **Several machines sharing results**, low latency: `RedisBackend`. Read
  [Sharing a cache](sharing-caches.md) for what makes two machines' keys match.
- **Cloud jobs, durable storage across regions**: `S3Backend`.
- **Nothing written to disk** (sensitive data, throwaway runs): `InMemoryBackend`.

<!-- claim: cash/backends/__init__.py:__all__ @286e3eee broad="the count and the table are a claim about the exported backend set" -->
| Backend | Survives a restart | Shared with | Import |
|---|---|---|---|
| `TieredBackend` (**default**: RAM + file) | yes | processes on this machine | `from cash import TieredBackend` |
| `FileBackend` | yes | processes on this machine | `from cash import FileBackend` |
| `SQLiteBackend` | yes | processes on this machine | `from cash import SQLiteBackend` |
| `InMemoryBackend` | no | this process only | `from cash import InMemoryBackend` |
| `RedisBackend` | as Redis is configured | every host that reaches Redis | `from cash.backends import RedisBackend` (extra `redis`) |
| `S3Backend` | yes | every host that reaches the bucket | `from cash.backends import S3Backend` (extra `s3`) |

Every backend stores values with `pickle`, so only read a cache that people you
trust write to. See [Sharing a cache](sharing-caches.md#the-trust-boundary).

## The default: RAM in front of disk

`TieredBackend` holds a list of backends, fastest first. A read checks each tier
in order and copies a hit into the faster tiers. A write goes to the first tier;
which values also go to the slower tiers depends on the path:

=== "Decorator"

    <!-- claim: cash/backends/persistence_policy.py:PersistencePolicy.decide @dfaf7643 -->
    A decorated result goes to **every** tier whose size cap allows it, however
    cheap it was to compute. The compute floor and cost model do not apply, and
    a `promotion_policy` you pass has no effect on decorated functions. See
    [Where results are stored](../../decorator.md#where-results-are-stored).

=== "Notebook"

    A statement's result goes past RAM when it took longer than 0.1 s and
    recomputing it costs more than loading it back. `promotion_policy` replaces
    that judgement for entries the cost model has no estimate for, and
    `# @cash:persist` forces a write. See the [cost model](../../cost-model.md).

Each tier has a size cap. By default the RAM tier may use a fifth of memory
(between 512 MiB and 4 GiB) and the disk tier a quarter of the room on its
volume; `cash info` prints both. `max_cache_size` pins the disk cap.

To build a stack by hand:

```python
from cash import Cash, FileBackend, InMemoryBackend, TieredBackend

app = Cash(backend=TieredBackend([
    InMemoryBackend(max_entries=100),
    FileBackend(cache_dir="./.cash"),
]))
```

`Cash(backends=[...])` builds the same thing from a list.

## `FileBackend`

```python
from cash import Cash, FileBackend

app = Cash(backend=FileBackend(
    cache_dir="./.cash",
    max_size_bytes=2 * 1024**3,   # 2 GiB cap
))
```

<!-- claim: cash/backends/file_eviction.py:FileEvictor.evict @d859ee8a, cash/backends/file_eviction.py:FileEvictor.rebuild_queue @a5ec5c6f -->
One file per entry in `cache_dir`, which you can inspect and delete by hand.
The disk write runs on a background thread. Reads and writes stay fast as the
folder fills, because the file name comes from the key. When the folder passes
`max_size_bytes`, entries are evicted down to 90% of it, least valuable per
byte first: cheap to recompute, large, and rarely read go first.

**Key parameters:** `cache_dir`, `max_size_bytes` (None: unlimited),
`compress` (gzip; worth it mostly for text), `flush_interval`, `default_ttl`.

## `SQLiteBackend`

```python
from cash import Cash, SQLiteBackend

app = Cash(backend=SQLiteBackend(
    db_path="./.cash/cache.db",
    max_size_bytes=1024**3,
))
```

One database file holds every entry, in WAL mode so several processes can read
at once. Eviction is least recently used, once the total passes
`max_size_bytes`. Which of the two disk backends is faster depends on value
size (measured at 100,000 entries):

| Value size | Faster write | Faster read |
|---|---|---|
| 512 B to 128 KB | SQLite, 2 to 4 times | SQLite, 3 to 6 times |
| around 512 KB | about equal | about equal |
| 1 MB and up | File, and the gap grows with size | File |

Cached DataFrames, arrays and models are usually megabytes, which is why the
default uses files.

<!-- claim: cash/backends/sqlite_backend.py:SQLiteBackend.max_size_bytes == 104857600 -->
**Key parameters:** `db_path`, `max_size_bytes`, `default_ttl`, `wal_mode`
(default True). Inside a tier stack, a single value over 100 MiB skips SQLite
and goes to the next tier.

## `InMemoryBackend`

```python
from cash import Cash, InMemoryBackend

app = Cash(backend=InMemoryBackend(max_size_bytes=512 * 1024**2))
```

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend._evict @2bbbead9, cash/backends/memory_backend.py:InMemoryBackend._evict_to_byte_cap @2e17ed9e -->
A dictionary in this process: everything is gone when the process ends. It
evicts on three triggers: `max_entries` (least recently used first),
`max_size_bytes`, and system memory above `max_memory_percent` (default 90%).
The last two evict the least valuable entries per byte first.

Values are copied in and out, so a cached value is held **twice** in RAM: once
as your variable and once in the cache. That copy is what keeps a later change
to your object from changing the cached value. If your values are large
relative to RAM, set `max_size_bytes` explicitly.

## `RedisBackend`

```python
from cash import Cash
from cash.backends import RedisBackend

app = Cash(backend=RedisBackend(
    host="redis.internal",
    port=6379,
    db=0,
    prefix="cash:proj-a:",
))
```

Each entry is two Redis keys, written together. A ttl becomes a Redis
`EXPIRE`. Give each project its own `prefix`: it is the only thing that keeps
two projects in one Redis apart.

<!-- claim: cash/backends/redis_backend.py:RedisBackend.max_size_bytes == 10485760, cash/backends/redis_backend.py:RedisBackend.lock @cfdf2e01 -->
- Install the extra: `pip install "cash-lib[redis]"`. Without it the backend
  raises `DependencyNotFoundError`.
- Inside a tier stack, a single value over 10 MiB skips Redis.
- Used on its own, it is the one backend whose `use_locking=True` lock spans
  processes and hosts. Behind a RAM tier, it locks per process only; see
  [Deploying](deploying.md#sharing-a-cache-between-processes-and-hosts).

## `S3Backend`

```python
from cash import Cash
from cash.backends import S3Backend

app = Cash(backend=S3Backend(
    bucket="my-team-cache",
    prefix="cash/proj-a/",
    region_name="us-east-1",   # passed through to boto3
))
```

One object per entry. Reading an entry's metadata fetches only its first 8 KB.
Every miss is a GET and every write a PUT, both billed and both tens of
milliseconds, so put S3 last in a tier stack. S3 does not delete expired
entries by itself: cash skips them on read, and a bucket lifecycle rule or
`cash.cleanup()` removes them. Install the extra:
`pip install "cash-lib[s3]"`.

## Configuring without code

Every backend is also a configuration setting, in `[tool.cash]`, in
`~/.config/cash/config.toml`, or as an environment variable:

```bash
export CASH_BACKEND=redis
export CASH_REDIS_HOST=cache.prod
export CASH_REDIS_PREFIX=team-analytics:
```

A tier stack uses `CASH_TIER_<N>_<FIELD>`:

```bash
export CASH_TIER_0_TYPE=memory
export CASH_TIER_1_TYPE=redis
export CASH_TIER_1_HOST=cache.prod
export CASH_TIER_2_TYPE=s3
export CASH_TIER_2_BUCKET=my-team-cache
```

<!-- claim: cash/backends/factory.py:build_backend_from_config @2784fa8a, cash/backends/factory.py:tier_specs @428b03c6 -->
A tier list replaces the stack `CASH_BACKEND` names, and a setting a tier leaves
out comes from the top-level field of the same meaning. See
[Configuration](../../getting-started/configuration.md).

## Using your backend

The decorator and the notebook read the same configuration, so a backend set in
`[tool.cash]` or with `CASH_*` variables applies to both.

=== "Decorator"

    Pass the backend to your own instance and decorate with it:

    ```python
    from cash import Cash, FileBackend

    app = Cash(backend=FileBackend(cache_dir="./.cash"))

    @app.cache
    def expensive(x):
        return x ** 2

    expensive(3)   # first call: computes
    expensive(3)   # cache hit
    ```

=== "Notebook"

    Set the backend in `[tool.cash]` in `pyproject.toml`, or with `CASH_*`
    variables, before you start the kernel. `%cash_on` then uses it:

    ```toml
    [tool.cash]
    backend = "sqlite"
    ```

## Related

- [Sharing a cache](sharing-caches.md): Redis and S3 across machines.
- [Deploying](deploying.md): locking, workers and production settings for decorated code.
- [Configuration](../../getting-started/configuration.md): every field and variable.
- [Where your cache lives](../../how-it-works/storage.md): tiers, eviction and serialization in depth.
