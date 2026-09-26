---
search:
  boost: 2
---

# Configuration

!!! info "Applies to: both paths"
    Every setting, environment variable and default, for scripts and notebooks. A few settings are marked for one path only.

This page lists every cash setting, the places you can set it, and which
value wins when several of them do.

## Where settings come from (precedence) { #where-settings-come-from }

cash merges its settings from six layers. A higher layer wins:

1. **Code**: `Cash(debug=True)` or `cash.configure(debug=True)`.
2. **Environment variables**: `CASH_<FIELD>`, such as `CASH_CACHE_DIR`.
3. **A file named in code**: `Cash(config_path="cash.toml")`
   ([below](#a-config-file-named-in-code)).
4. **Project file**: `[tool.cash]` in the project's `pyproject.toml`.
5. **User file**: `~/.config/cash/config.toml` (`$XDG_CONFIG_HOME` is
   honoured), or `%APPDATA%\cash\config.toml` on Windows.
6. **Defaults**, listed in the tables below.

```toml title="pyproject.toml"
[tool.cash]
cache_dir = ".cash"
max_cache_size = "5GB"
```

`cash info` prints every setting in effect and the layer it came from
([CLI](../cli.md#cash-info)).

<!-- claim: cash/_location.py:project_anchor @46e903a7, cash/config.py:_anchor_cache_dir @edf1f957 -->
### What paths are relative to

A relative `cache_dir` is resolved against whoever wrote it:

| Written in | Resolved against |
|---|---|
| `Cash(cache_dir=...)`, `cash.configure(...)` or `CASH_CACHE_DIR` | the current working directory when it is read |
| a config file | that file's directory |
| nowhere (the `.cash` default) | the project root: the first directory above the running script that holds a `pyproject.toml`, `setup.py`, `setup.cfg` or `.git`; without one, the script's own directory. In a notebook or REPL, the current directory. |

A tier's own `cache_dir` or `db_path` follows the same rule.

For installed programs such as `pytest` or a console script:

- The project root is the one above the current directory. Outside any
  project, each tool gets its own directory under the per-user cache root.
- The project `pyproject.toml` is looked for from the same root, so a
  script finds the same settings wherever it is started.

[Where your cache lives](../how-it-works/storage.md#where-the-cache-folder-is)
says how the folder is chosen and capped.

<!-- claim: cash/_location.py:_marks_project @2991b315 -->
A `pyproject.toml` marks a project only when it has a `[project]`,
`[build-system]`, `[tool.poetry]` or `[tool.cash]` table.

<!-- claim: cash/config.py:CashConfig @9e5fc345 broad="the field table is a claim about every field of the dataclass" -->
## All settings

Every setting is named by its field. In a TOML file the key is the field
name. Its environment variable is `CASH_` plus the field name in capitals:
`cache_dir` is `CASH_CACHE_DIR`, `max_cache_size` is `CASH_MAX_CACHE_SIZE`.
A setting applies to both paths unless it is marked
<span class="md-tag">decorator only</span> or
<span class="md-tag">notebook only</span>.

<!-- claim: cash/config.py:validate_value @b391f4e9, cash/config.py:parse_size @11b4b371, cash/config.py:_validated_layer @0460d759 -->
Values are checked whichever layer they come from:

| Value | Accepted |
|---|---|
| Any setting, as a string | Read as an environment variable would be: `"true"`, `"8"`. |
| Byte sizes | A number of bytes, or a string with a unit: `"2GB"`, `"500MB"`, `"512MiB"`. KB/MB/GB are powers of 1000, KiB/MiB/GiB of 1024. At least 1; leave unset for no cap. |
| Entry counts | At least 1; leave unset for no cap. |
| Seconds and other numbers | At least 0. |
| `min_cache_savings_pct` | Between 0 and 1. |
| `cache_dir` | Any path except the empty string. |
| A `CASH_*` variable set to `""` | Counts as unset (`CASH_CACHE_DIR=${X:-}`). |

What happens on an error depends on where the value was set:

| Set in | A bad value | A name that is not a setting |
|---|---|---|
| Code: `Cash(...)`, `cash.configure(...)` | Raises `ValueError`. | Raises `ValueError` naming the closest setting (`Cash(ttl=60)`, a misspelt name, an unknown key in a tier table). |
| A config file | Skipped, with [`CONFIG-INVALID`](../warnings.md#config-invalid). | Warns [`CONFIG-UNKNOWN-KEY`](../warnings.md#config-unknown-key). |
| An environment variable | Skipped, with [`CONFIG-INVALID`](../warnings.md#config-invalid). | `CASH_TIER_<N>_*` warns `CONFIG-UNKNOWN-KEY`; another `CASH_*` name is ignored, since other tools share the prefix. |

### Storage

| Field | Default | Description |
|---|---|---|
| `cache_dir` | `".cash"` | Where the disk tier writes. Add it to `.gitignore`. |
| `compress` | `false` | gzip each entry on disk. Worth it mainly for text-like values. |
| `max_cache_size` | `null` (auto) | Disk cap in bytes or a size such as `"5GB"`. At the cap, the entries worth least per byte (compute time per byte, raised by hits) are evicted first. A single value bigger than the cap is not written ([`CACHE-VALUE-TOO-BIG`](../warnings.md#cache-value-too-big)). Each process enforces the cap on its own writes. cash shows the cap, and where it comes from, when caching starts ([Where your cache lives](../how-it-works/storage.md#where-the-cache-folder-is)). |
| `max_memory_entries` | `null` | Entry-count cap for the RAM tier, evicting least recently used. `null` means no count limit; the RAM tier is still capped in bytes. |
| `flush_interval` | `5` | Seconds between the disk tier's metadata flushes. `0` flushes after every write. |
| `file_hash_full_max_bytes` | `268435456` (256 MiB) | Tracked files up to this size are hashed in full to check freshness. Larger files hash three sampled regions plus the timestamps, which misses a same-size edit outside those regions that keeps the modification time. |
| `shutdown_write_timeout` | `60.0` | Seconds a finishing process waits for its background writes before exiting without them ([`CACHE-WRITE-ABANDONED`](../warnings.md#cache-write-abandoned)). `0` does not wait, and does not warn. |

<!-- claim: cash/backends/adaptive_caps.py:resolve_ram_cap @02a19f23, cash/backends/adaptive_caps.py:_cgroup_memory_limit @b30940d8 -->
**What "auto" resolves to.** The disk cap is a quarter of the room on the cache
volume (free space plus what the cache already holds), between 8 GiB and
100 GiB and never above 80% of that room. The RAM cap is a fifth of the memory
the process may use (the host's, or a container's cgroup limit when smaller),
between 512 MiB and 4 GiB. `cash info` prints both.

### Notebook caching policy

These apply to the notebook only: they decide which statements are cached
and written to disk. A `@cash.cache` result is always stored, whatever these
say. See
[Cost model](../cost-model.md).

| Field | Default | Description |
|---|---|---|
| `persist_all` | `false` | Cache every statement, skipping the floors below, as if each had `# @cash:persist`. |
| `min_execution_time_to_cache_seconds` | `0.01` | A statement faster than this is not cached at all. |
| `min_cache_savings_pct` | `0.20` | Fraction of the compute time a predicted restore must save for a value to be written to disk. |
| `min_cache_fixed_budget_seconds` | `0.05` | A predicted restore time below this is always acceptable, however short the compute. |
| `call_cost_floor_seconds` | `0.003` | A call inside a statement is cached only when it runs at least this long. |
| `loop_split_max_iter_seconds` | `0.006` | A loop whose iterations each take less than this may be cached as one unit instead of per call. |
| `loop_split_min_remaining_seconds` | `0.1` | ...but only when at least this much work remains in the loop. |

### Remote files

| Field | Default | Description |
|---|---|---|
| `remote_revalidate_max_age_seconds` | `0.0` | Seconds a remote file's state may be reused before the store is asked again. `0` checks on every hit. A higher value means a change can go unseen for that long; prefer `immutable=True` on a [`RemoteFileDataSource`](../api/data_sources.md) where you have one. |

### Output and switches

| Field | Default | Description |
|---|---|---|
| `debug` | `false` | Log every cache decision, including one line per decorated call. The lines go to stderr unless your program configures `logging`. `%cash_debug on` in a notebook sets this field and also shows the notebook's debug lines. |
| `verbose` | `false` | <span class="md-tag">decorator only</span> Only the one line per decorated call, and the disk cache's cap and first eviction, without the other debug records. |
| `disable` | `false` | Run every `@cash.cache` call uncached, and make `%cash_on` decline. `CASH_DISABLE=1 pytest` checks your tests pass without the cache ([Testing your code](../tutorials/feature-guides/testing-your-code.md)). |
| `summary` | `false` | <span class="md-tag">decorator only</span> At exit, print a per-function hit/miss table to stderr, with why each function missed: `CASH_SUMMARY=1 python model.py`. |
| `analytics` | `true` | <span class="md-tag">notebook only</span> Record each statement's hit, miss and timing in `analytics.db` under the per-user cache root, for the `cash.show_stats()` dashboard. `false` creates no file. |

### Backend

| Field | Default | Description |
|---|---|---|
| `backend` | `"tiered"` | `"tiered"` is a RAM tier in front of a disk tier. `"memory"`, `"file"`, `"sqlite"`, `"redis"` or `"s3"` is that one backend. Ignored when `tiers` is set. |
| `redis_host` | `"localhost"` | Redis server, for `backend = "redis"` or a redis tier. |
| `redis_port` | `6379` | Redis port. |
| `redis_db` | `0` | Redis database number. |
| `redis_password` | `null` | Redis password. Prefer the environment variable to a committed file. |
| `redis_prefix` | `"cash:"` | Key prefix, so several apps can share one Redis. |
| `s3_bucket` | `""` | S3 bucket; required for `backend = "s3"`. |
| `s3_region` | `""` | S3 region, such as `"us-east-1"`. Empty uses your AWS configuration. |
| `s3_prefix` | `"cash/"` | Object key prefix. |

See [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md) for
which to pick.

### Tiers

`tiers` replaces the stack `backend` names with your own list,
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
# override one tier's host
export CASH_TIER_1_HOST=prod-redis.example.com
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

<!-- claim: cash/config.py:_resolve_config @21f63b6a -->
The named file ranks above the project and user files and below environment
variables and code. That lets an installed package ship its own settings: put
a TOML file inside the package and pass
`Path(__file__).with_name("cash.toml")`. The file keeps its settings under
`[cash]` or `[tool.cash]`. A relative `cache_dir` in it is relative to the
file, and `~` is your home directory.

<!-- claim: cash/config.py:_resolve_config @21f63b6a -->
A missing file warns [`CONFIG-FILE-MISSING`](../warnings.md#config-file-missing).
`cash info --config path/to/cash.toml` shows what a file resolves to.

<!-- claim: cash/__init__.py:configure @9f9c6e1a -->
## Changing settings at runtime

`cash.configure(...)` changes the settings of the default instance used by
`@cash.cache`; `Cash.reconfigure(...)` does the same for an instance of your
own. Neither writes to a file.

```python
import cash

cash.configure(debug=True)
cash.configure(max_cache_size="10GB")
```

When a change takes effect:

- **Settings that shape the backend** (`cache_dir`, `compress`,
  `max_cache_size`, `max_memory_entries`, `flush_interval`, `backend`,
  `tiers`, or a connection detail of a tier in use): cash finishes the old
  backend's pending writes and builds a new one.
- **A connection detail for a backend not in use**: kept for later.
- **All other settings**: from the next operation.
  `configure(debug=False)` also stops the log output `debug=True` started.

<!-- claim: cash/_active.py:publish_settings @035400df -->
In worker processes (`multiprocessing`, `ProcessPoolExecutor`, joblib):

- A worker started afterwards runs with what `cash.configure(...)` and
  `cash.disabled()` set, whatever the start method.
- A worker already running keeps the settings it started with.
- An instance of your own is built again in each worker by your code, so its
  `reconfigure(...)` stays in the process that called it.

See [Threads and processes](../tutorials/feature-guides/thread-safety.md#across-processes-pool-processpoolexecutor-joblib).

<!-- claim: cash/reconfigure.py:apply_overrides @b8ecc0a2, cash/config.py:validated_overrides @fecbc461 -->
Values are checked exactly as `Cash(...)` checks them, before anything
changes:

- A bad value raises `ValueError` and leaves the old settings in place.
- So does a backend that cannot be built (`backend="s3"` without
  `s3_bucket`, a Redis tier without the `redis` package); the old backend
  keeps working.
- `~` is expanded. A relative `cache_dir` is relative to the current
  directory, made absolute at once so a later `os.chdir()` does not move the
  cache.
- An instance built with a backend object (`Cash(backend=...)`) keeps it:
  changing its tier settings raises `ValueError`.

## Magics that set these fields

In a notebook, two magics change these settings for the session:

- `%cash_debug on` / `off` sets `debug`.
- `%cash_persist on` / `off` sets `persist_all`.

`%cash_on ttl=N` gives every cached statement a TTL for the session; it is not
a config field. See [Magic commands](../magics.md).

## Related

- [Where your cache lives](../how-it-works/storage.md): how the cache folder
  is chosen, and what happens at the size cap.
- [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md):
  which `backend` or `tiers` stack to pick.
- [Command-line interface](../cli.md#cash-info): `cash info` shows every
  setting in effect and where it came from.
- [Configuration API](../api/config.md): `CashConfig` and `TierConfig` field
  by field.
