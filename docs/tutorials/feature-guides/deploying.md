# Deploying `@cash.cache`

!!! info "Applies to: decorator"
    Services, scheduled jobs, worker pools, CI and libraries that use
    `@cash.cache`.

The defaults suit one developer running scripts on one machine. This page
covers what changes when the code runs unattended, in several processes, on
several hosts, or inside someone else's program.

## Production configuration

Configure a deployed job from the environment, so the code stays the same
everywhere. Every setting has a `CASH_*` variable
([Configuration](../../getting-started/configuration.md) lists them all):

```bash
export CASH_CACHE_DIR=/var/cache/myapp   # absolute: no doubt where entries go
export CASH_MAX_CACHE_SIZE=20GB          # pin the disk cap on a shared volume
export CASH_SUMMARY=1                    # one summary per run, in the job log
```

`cash info`, run with the same environment, prints the folder, backend and cap
the job will use. Check it once in the container or on the host.

<!-- claim: cash/config.py:CashConfig.max_cache_size == None -->
**The disk cap.** Without `max_cache_size`, the disk tier may use a quarter of
the room on its volume. When it is full, cash evicts the entries worth least per
byte first: cheap to recompute, large, rarely read. On a volume other programs
also use, pin the cap. Each process enforces the cap on its own writes, so
several workers writing at once can briefly overshoot it.

**A deploy keeps the cache.** Entries are keyed by code, so after a deploy only
the functions whose code (or helpers, or data) changed recompute. You don't
need to clear the cache on deploy.

**The kill switch.** `CASH_DISABLE=1` makes every cached function run its body
and touch no cache, for every `Cash` instance in the process. Use it to rule
cash out when you debug a production problem.

## A default lifetime for every entry

There is no top-level ttl setting: a decorator's `ttl=` covers one function. To
give every entry a lifetime, declare the tier stack with a `default_ttl` on the
disk tier. It applies to every entry written without a `ttl=`:

```toml
[[tool.cash.tiers]]
type = "memory"

[[tool.cash.tiers]]
type = "file"
default_ttl = 3600          # seconds
```

<!-- claim: cash/backends/factory.py:tier_specs @428b03c6 -->
From the environment: `CASH_TIER_0_TYPE=memory`, `CASH_TIER_1_TYPE=file`,
`CASH_TIER_1_DEFAULT_TTL=3600`. A decorator's own `ttl=` takes precedence, and
as with `ttl=`, the shorter of the value an entry was written with and the
current one wins. `cash clear --expired` (or `cash.cleanup()`) frees the disk
that expired entries still hold.

## Services and worker processes

<!-- claim: cash/backends/clear_watch.py:ClearWatcher.cleared @04551b75 -->
- **Clearing a live service.** `cash clear --all`, `--function` and `--entry`
  reach processes that are still running: each checks at most once a second
  whether its cache folder was cleared, and drops what it holds in RAM. No
  restart is needed.
- **Workers that hold nothing in RAM.** `CASH_BACKEND=file` keeps every entry
  on disk only. Each hit reads the file, and a worker's memory stays flat.
- **Worker processes** (`multiprocessing`, `ProcessPoolExecutor`, joblib,
  gunicorn workers) share the disk cache and keep their own RAM tier, counters
  and warnings. See [Threads and processes](thread-safety.md#across-processes-pool-processpoolexecutor-joblib).
- **Concurrent callers.** Without locking, two callers that miss the same key at
  once both compute it. `Cash(use_locking=True)` makes one compute while the
  others wait; the next section says which locks span processes.
- **Network reads.** A cached function that calls an API or a database warns
  [`KEY-NETWORK-READ`](../../warnings.md#key-network-read) until it has a
  `ttl=`. Give every such function a freshness plan before it ships.

## Sharing a cache between processes and hosts

On one machine, processes that use the same cache folder share entries with no
extra setup. Across hosts, point every host at a Redis or S3 backend; see
[Sharing a cache](sharing-caches.md) for what makes keys match on two machines,
and [Choosing a backend](choosing-a-backend.md) for the options.

<!-- claim: cash/backends/_base.py:CacheBackend.lock @2c1d7483, cash/backends/redis_backend.py:RedisBackend.lock @cfdf2e01 -->
**Locking across processes needs Redis as the backend itself.**
`use_locking=True` gives one computation per key within a process on every
backend. Only a bare `RedisBackend` also locks across processes and hosts:

| Backend | `use_locking=True` locks |
|---|---|
| `CASH_BACKEND=redis`, or `Cash(backend=RedisBackend(...))` | across processes and hosts |
| A tier stack with RAM in front of Redis (`[[tool.cash.tiers]]`, `CASH_TIER_*`, `Cash(backends=[...])`) | **within each process only** |
| Any other backend | within each process only |

So a RAM tier in front of Redis buys fast repeat hits and costs cross-process
locking. Pick what your workload needs: for a paid API that several workers may
call with the same arguments at once, use a bare `RedisBackend`.

## CI

- **Prove the tests, not the cache.** Run the suite once with `CASH_DISABLE=1`,
  and give each run its own cache folder. See
  [Testing your code](testing-your-code.md).
- <!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @21132aa4 -->
  **Fail the build on side effects.** `@cash.cache(strict=True)` raises
  `CashImpureFunctionError` on the first call of a function with a side effect
  cash would skip on a hit. It honours `# @cash:assume-safe` on lines you have
  checked, so it works as a standing gate that fails only on new findings. See
  [Side effects](../../decorator.md#side-effects).
- <!-- claim: cash/__main__.py:cmd_clear @a08b9044 -->
  **Decide whether the cache survives between runs.** A fresh runner starts
  cold, which is correct but slow. To keep it warm, point `CASH_CACHE_DIR` at a
  folder your CI caches between runs, or share a Redis or S3 backend. To start
  clean on purpose, run `cash clear --all`. Plain `cash clear` exits with an
  error, which fails the step.
- **File paths must match.** An entry that read files records their paths.
  Runners that check out to different paths recompute those entries.

## Using cash inside a library

A library that caches with the module-level `cash.cache` shares the
application's cache and configuration: the app's `cash.configure(...)` calls,
its `[tool.cash]` table and its `CASH_*` variables all apply to your functions.
Own a private instance instead:

<!-- claim: cash/core.py:Cash.__init__ @777322f9 -->
<!-- test:skip reason="illustrative: a two-file library layout" -->
```python
# mylib/_cache.py
from cash import Cash

cache = Cash(register_magic=False)

# mylib/features.py
from mylib._cache import cache

@cache.cache
def build_features(path):
    ...
```

- **`register_magic=False`.** A `Cash()` created inside IPython registers the
  `%cash_*` notebook commands to itself. If your library is imported in a
  notebook, that takes the commands over from the user's own session.
  `register_magic=False` prevents it.
- **Where entries go.** Leave `cache_dir` out and the library caches in the
  project it runs in, or, when run from outside any project, in a per-user
  cache folder. A `cache_dir` passed in code beats `CASH_CACHE_DIR`, so leave it
  out if your users should be able to move the cache.
- **What to tell your users.** `CASH_DISABLE=1` turns your cache off,
  `CASH_CACHE_DIR` moves it, and `cash clear --function NAME` clears one of your
  functions. Mention them in your own docs.

## Related

- [The `@cash.cache` guide](../../decorator.md)
- [Threads and processes](thread-safety.md)
- [Sharing a cache](sharing-caches.md)
- [Choosing a backend](choosing-a-backend.md)
- [Configuration](../../getting-started/configuration.md)
