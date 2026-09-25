# Data engineering

!!! info "Applies to: decorator"
    Anyone building ETL pipelines and backfills from Python functions.

A pipeline re-processes the same data again and again while you develop, debug
and backfill it. Make each step a cached function, and a re-run after a fix
skips every step whose code and inputs didn't change.

## The pattern: one cached function per step

```python
import cash
import pandas as pd

@cash.cache
def extract(source_path):
    return pd.read_parquet(source_path)

@cash.cache
def normalize(df):
    df = df.copy()
    df["amount"] = df["amount"].fillna(0)
    return df

@cash.cache
def aggregate(df):
    return df.groupby("region").agg(total=("amount", "sum"))

def run(source):
    agg = aggregate(normalize(extract(source)))
    agg.to_parquet("output.parquet")      # the write stays outside the cache

if __name__ == "__main__":
    run("s3://bucket/raw.parquet")
```

Edit `aggregate` and re-run: `extract` and `normalize` are hits, and only
`aggregate` runs. Change the source data and `extract` misses; its new result
is a new argument for `normalize`, so everything downstream recomputes.

## Source data

<!-- claim: cash/tracking/reader_patches.py:FileDependencyRegistry._initialize_defaults @56d3c684 -->
Files read inside a step are tracked by content, with nothing to declare:
pandas and polars readers, pyarrow's `csv`, `parquet`, `feather` and `json`
readers, `open()`. An `s3://` or `gs://` read is tracked by the object's ETag or
version. A `touch` that leaves the bytes alone does not invalidate. Readers
cash can't see (`h5py`, a `pyarrow.fs` file system) need
`file_depends_on=`; see [File dependencies](../feature-guides/custom-file-sources.md).

Databases and APIs have no file to check. Three ways to handle them:

- **Snapshot to a file.** A scheduled task writes the query result to disk, and
  a cached step reads it. The read is tracked, so a new snapshot invalidates
  everything downstream:

    ```python
    @cash.cache
    def load_snapshot():
        return pd.read_parquet("snapshot.parquet")
    ```

- **Track a version.** If you can read a freshness signal (a table's
  `last_modified`, a schema version), wrap it in a `DataSource` and return it
  from a `dynamic_depends_on=` resolver. The resolver must return a
  `DataSource`: a raw string or number makes the call run uncached, with a
  [`KEY-DYNAMIC-DEP-FAILED`](../../warnings.md#key-dynamic-dep-failed) warning.
  See [Dynamic dependencies](../feature-guides/dynamic-dependencies.md).
- **Give it a lifetime.** A step that queries or fetches directly warns
  [`KEY-NETWORK-READ`](../../warnings.md#key-network-read), because cash can't
  see when the data changes. A `ttl=` answers that and silences the warning:

    ```python
    @cash.cache(ttl=3600)   # refresh hourly
    def fetch_exchange_rates():
        return requests.get("https://api.exchangerate.host/latest").json()
    ```

## Backfills

Make the period an argument:

```python
@cash.cache
def extract_day(source_path, date):
    return pd.read_parquet(f"{source_path}/dt={date}")
```

Each date is its own entry. A backfill computes only the dates not yet cached,
and a run that stopped halfway resumes where it left off. Fix a bug in a later
step and re-run the whole month: every `extract_day` is a hit, and only the
fixed step and those after it run again.

## Seeing which steps ran

Each cached function counts its own hits and misses:

```python
import cash

@cash.cache
def load_day(date):
    return {"date": date, "rows": 1000}

load_day("2026-01-15")          # first call: computes
load_day("2026-01-16")          # a different date: computes
load_day("2026-01-15")          # cache hit

print(load_day.cache_info())
# {'hits': 1, 'misses': 2, 'hit_rate': 0.333..., ...}
```

On a re-run of processed dates, `hit_rate` should be close to 1.0; the step
where it drops is where the re-run stopped being free. Log `cache_info()` at the
end of each run: a sudden drop is often the first sign that something upstream
changed. `f.explain(...)` says why a call would miss, and `CASH_SUMMARY=1`
prints a table for the whole run; see
[Seeing what cash did](../../decorator.md#seeing-what-cash-did).

## Running it in production

Each Airflow task, Prefect flow or Dagster op just calls the cached functions.
For workers on several machines, point them at a shared backend (Redis or S3),
or each keeps its own cache. See [Deploying](../feature-guides/deploying.md) and
[Choosing a backend](../feature-guides/choosing-a-backend.md).

## Caveats

<!-- claim: cash/analysis/file_effects.py:SideEffectVisitor @07c1a65b broad="the write-detection claim is about the visitor as a whole" -->
- **Keep writes out of cached steps.** `to_parquet` and `to_csv` are effects: a
  hit would skip them. Cash warns if a cached step writes. Cache the step that
  builds the frame and write it outside, as `run` does above.
- **Leave cheap steps undecorated.** Every cached result is written to disk. A
  rename that takes milliseconds isn't worth a 5 GB entry.
- **Don't change arguments in place.** A step that runs
  `df.fillna(0, inplace=True)` on its input is not stored, so it runs every
  time. Copy first, as `normalize` does; see
  [Results cash refuses to store](../../decorator.md#results-cash-refuses-to-store).
- **Create clients at module level.** Database connections, S3 clients and Spark
  sessions are not results. Build them once, outside cached functions.
- **Pass the clock in.** A step that reads `datetime.now()` stores the first
  value it saw, with a warning. Pass `now` as an argument instead.

## Related

- [File dependencies](../feature-guides/custom-file-sources.md)
- [Dynamic dependencies](../feature-guides/dynamic-dependencies.md)
- [Deploying](../feature-guides/deploying.md)
