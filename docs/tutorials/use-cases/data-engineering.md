# ETL and Data Engineering Pipelines

ETL pipelines re-process the same data over and over during development, debugging, and incremental backfills. Cash turns each pipeline step into a cached function, so re-running after fixing a bug downstream skips everything upstream that didn't change.

## Why this matters in data engineering

- **Iteration speed.** Debugging a tail-end transform shouldn't require re-reading 100GB of source data. With each step cached, the failing step is the only one that re-runs while you fix it.
- **Idempotency.** A pipeline that's already run against a given input shouldn't re-run when invoked a second time on the same input. Cash gives you that for free — same args, same source files, cache hit.
- **Partial backfill.** Fix the bug, re-run the pipeline, only affected steps recompute. No bespoke "skip steps 1-3" logic in your orchestrator.
- **Reproducibility.** Cached intermediate states preserve the exact inputs and outputs of past runs, which is useful when an analyst asks "what did this report look like last Tuesday?"

## The pattern: pipeline-as-functions

Each step is a `@cash.cache`'d function. The result is a dependency graph the cache figures out for you:

```python
import cash
import pandas as pd

@cash.cache
def extract(source_path):
    return pd.read_parquet(source_path)

@cash.cache
def normalize(df):
    df = df.copy()
    df['amount'] = df['amount'].fillna(0)
    return df

@cash.cache
def aggregate(df):
    return df.groupby('region').agg(total=('amount', 'sum'))

def run(source):
    raw = extract(source)
    clean = normalize(raw)
    agg = aggregate(clean)
    agg.to_parquet('output.parquet')

if __name__ == "__main__":
    run("s3://bucket/raw.parquet")
```

Change the `aggregate` function and re-run: `extract` and `normalize` are cache hits; only `aggregate` re-executes. Change the source file and re-run: `extract` misses (the file mtime moved), and everything downstream cascades.

The first run on a fresh cache reads the parquet, normalises, aggregates, and writes. Every subsequent run with the same source path returns the aggregated frame in milliseconds — the whole pipeline collapses to three cache lookups and one parquet write.

## File-based source data

Pandas readers (`read_parquet`, `read_csv`, `read_json`) are intercepted automatically and the source file's mtime folds into the cache key. Touch the file and `extract` misses on the next call. For non-pandas readers — Arrow, HDF5, custom binary formats — declare the dependency explicitly with `file_depends_on=` or `dynamic_depends_on=`. See [Custom File Sources](../feature-guides/custom-file-sources.md).

## Database and API sources

Cash doesn't auto-track SQL connections or HTTP endpoints — there's no mtime to watch. Two practical patterns:

**Snapshot to file first.** Land the query result on disk, then key your transforms off the snapshot:

```python
import cash
import pandas as pd

@cash.cache(file_depends_on="snapshot.parquet")
def load_snapshot():
    return pd.read_parquet("snapshot.parquet")
```

A cron or Airflow task refreshes `snapshot.parquet`; everything downstream invalidates automatically when it changes.

**Use a `DataSource` to track a version.** When you can name the freshness signal (a table's `last_modified` row, an API version, a config hash), wrap it in a `DataSource` subclass and pass a resolver to `dynamic_depends_on=`. The resolver must return a `DataSource` instance — passing a raw string or version number is silently ignored. See [Dynamic Dependencies](../feature-guides/dynamic-dependencies.md) for the correct subclass shape and a worked example.

## Choosing a backend for pipelines

- **Single machine, dev or batch.** The default tiered backend (in-memory L1, disk L2) handles everything up to ~100GB of cached state.
- **Shared dev/staging server.** `SQLiteBackend` for L2 — concurrent reads, single-file deployment, no daemon.
- **Distributed workers (Spark, Dask, Ray).** `RedisBackend` so all workers see the same cache and the same hits.
- **Cloud and cross-region.** `S3Backend` as L2 — workers in any region hit the same bucket and the same cached parquet files.

See [Choosing a Backend](../feature-guides/choosing-a-backend.md) for the decision tree and configuration snippets.

## Idempotency and backfills

The same pipeline invoked twice with the same arguments is a cache hit on every step. That's the idempotency property — useful for orchestrators that may retry tasks, useful for humans who want to re-run the script without thinking about it.

For backfills, encode the period as an argument:

```python
@cash.cache
def extract(source_path, date):
    return pd.read_parquet(f"{source_path}/dt={date}")

@cash.cache
def normalize(df):
    ...
```

Now `extract(src, "2026-01-15")` and `extract(src, "2026-01-16")` are independent cache entries. Backfilling January re-runs only the dates that haven't been cached; resuming after a partial run picks up where it left off.

Partial re-execution after a code change works the same way: edit `normalize`, and `extract` is still a hit for every date — only `normalize` and downstream steps recompute.

This is the workflow Cash optimises for: you've spent two hours running a 30-day backfill, the aggregation step has a bug, you fix it, you re-run the whole script. Extraction is a hit on all 30 days. Normalisation is a hit on all 30 days. Only the aggregation re-runs. The 2-hour pipeline finishes in seconds.

## Schema changes

When the *code* changes, Cash sees the new function source and invalidates downstream automatically. The trickier case is when the schema changes but the code that consumes it doesn't — a new column appears in the source table, but `normalize` still does `df['amount'].fillna(0)` and doesn't notice. Two options:

- **Re-snapshot and rely on file mtime.** If your snapshot file rewrites whenever the upstream schema moves, the file mtime change cascades through every cached step.
- **Bump a version dependency.** Wrap "schema version" in a `DataSource` subclass and pass it via `dynamic_depends_on=` (see the feature guide). Increment the version on schema changes; every cached step re-runs.

## Monitoring

In a pipeline run, you want to know which steps hit and which missed:

```python
print(extract.cache_info())
print(normalize.cache_info())
print(aggregate.cache_info())
# CacheInfo(hits=1, misses=0, ...)
```

In a notebook, `%cash_stats` prints a summary across every tracked function. On the CLI, `cash inspect` reads the cache directory and lists entries with sizes and timestamps. See [Debugging and Monitoring](../feature-guides/debugging-and-monitoring.md) for the full surface — `f.explain()` is especially useful in CI when you want to know *why* a step missed.

In production, log `cache_info()` at the end of each pipeline run. A sudden drop in hit rate is usually the first signal that something changed upstream — a schema migration, a new file landing, a clock skew — well before any downstream metric notices.

## Production deployment

Cash slots into orchestrators without ceremony — each Airflow task, Prefect flow, or Dagster op just calls Cash-decorated functions. Two things to set up:

- **Shared cache directory.** If workers are distributed, point Cash at shared storage (NFS for on-prem, S3 for cloud). Otherwise each worker has its own cache and you lose cross-worker hits.
- **TTL on freshness-sensitive steps.** For data that goes stale on a known cadence (daily exchange rates, hourly inventory snapshots), set `ttl=` so the cache expires automatically and the next call re-fetches.

```python
@cash.cache(ttl=3600)  # refresh every hour
def fetch_exchange_rates():
    return requests.get("https://api.exchangerate.host/latest").json()
```

See [Production Transition](../feature-guides/production-transition.md) for the notebook-to-script handover and the production-readiness checklist.

## Caveats

- **Don't cache the write step.** `to_parquet`, `to_csv`, `write_table` — these are side effects with no useful return value. Cache the *computation* that produces the frame; leave the write outside the cached function.
- **Large intermediate states bloat disk.** A 50GB intermediate cached after every step adds up fast. For cheap or transient transforms (a `df.rename(columns=...)` that runs in milliseconds), skip caching with `# @cash:no-cache` and let it recompute. See [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md).
- **Mutable state.** Cash assumes pure transforms. If a function mutates its input (`df.fillna(0, inplace=True)`), the cached result may not match what callers see on a miss. Defensive `df.copy()` at the top of each step is cheap insurance.
- **Don't cache the client object.** Database connections, S3 clients, Spark sessions — initialize them at module scope, not inside a cached function. The client isn't a function of its arguments and serializing it usually doesn't even work.
- **Watch out for non-deterministic transforms.** Anything that calls `datetime.now()`, `uuid.uuid4()`, or unseeded random sampling inside a cached step will bake the first observed value into the cache. Either lift the non-determinism out (pass `now` as an argument) or skip caching that step. See [Controlling Cache Behavior](../feature-guides/controlling-cache-behavior.md).

## Related

- [Custom File Sources](../feature-guides/custom-file-sources.md) — declare non-pandas file readers as cache dependencies.
- [Dynamic Dependencies](../feature-guides/dynamic-dependencies.md) — track per-call dependencies via a `DataSource` resolver.
- [Choosing a Backend](../feature-guides/choosing-a-backend.md) — picking storage for single-machine vs distributed pipelines.
- [Production Transition](../feature-guides/production-transition.md) — moving from notebook to scheduled job.
- [Debugging and Monitoring](../feature-guides/debugging-and-monitoring.md) — `cache_info()`, `f.explain()`, and CLI inspection.
