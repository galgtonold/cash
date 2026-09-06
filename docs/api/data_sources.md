# Data sources

<!-- claim: cash/data_source.py:FileDataSource @4099fc64 broad="the mtime contract is a property of the whole class", cash/remote_source.py:RemoteFileDataSource @16c56807 broad="the scheme list and validator contract are properties of the whole class" -->
Objects that contribute to a cache key by reporting a **token representing
their current state** (an mtime, a version, a content digest) — the cached
entry invalidates when that token changes. Two are bundled:
`FileDataSource` tracks a local file's mtime, and `RemoteFileDataSource` tracks
a remote object by the validator its store maintains. Custom subclasses extend
the same pattern to databases, API endpoints, etc.

## Imports

```python
from cash import FileDataSource         # the bundled file-mtime source
from cash import RemoteFileDataSource   # s3://, gs://, az://, http(s)://
from cash.data_source import DataSource  # ABC for writing your own
```

::: cash.FileDataSource
    options:
      members:
        - __init__
        - get_id
        - has_changed
        - update_state

### Example

```python
from cash import Cash, FileDataSource

c = Cash()
source = FileDataSource("data/input.csv")

@c.cache(depends_on=[source])
def load_data():
    return pd.read_csv("data/input.csv")

load_data()             # computes, recording the file's current state
load_data()             # hits — the file hasn't changed
```

When `input.csv` changes on disk, cached results are automatically
invalidated. For the simpler one-off case, prefer
`@c.cache(file_depends_on="data/input.csv")` — same behavior, less
typing.

---

::: cash.RemoteFileDataSource
    options:
      members:
        - __init__
        - get_id
        - has_changed
        - state_token
        - update_state

### Example

<!-- test:skip reason="illustrative — requires a reachable bucket" -->
```python
from cash import Cash, RemoteFileDataSource

c = Cash()

@c.cache(depends_on=[RemoteFileDataSource("s3://bucket/events.parquet")])
def load_events():
    return read_via_boto3("bucket", "events.parquet")
```

Reads cash can already see — `pd.read_parquet("s3://bucket/key")` and friends —
are tracked this way **automatically**; declare a source explicitly only for the
ones it can't see. See
[Remote objects](../tutorials/feature-guides/custom-file-sources.md#remote-objects-tracked-by-the-stores-own-validator)
for the full story, including `immutable=` and the failure behaviour.

!!! note "`http(s)://` needs no extra install"
    It resolves through the standard library. Other schemes go through fsspec
    and its filesystem for that scheme (`pip install "cash-lib[s3]"` plus
    `s3fs` for `s3://`, `gcsfs` for `gs://`); a missing one raises
    `DependencyNotFoundError` rather than silently recomputing forever.

---

## Custom data sources

To track something other than a file as a cache dependency, subclass
`DataSource`. The contract is three abstract methods, plus an optional
`state_token()` hook:

::: cash.data_source.DataSource
    options:
      members:
        - get_id
        - has_changed
        - update_state
        - state_token

!!! warning "`has_changed()` must return a state *token*, not a `bool`"
    Despite the name, the value `has_changed()` returns is folded into the cache
    key — so it must **change when the data changes** (a version, a digest, a
    max-id). A plain `bool` only has two states and can't track changes: cash
    warns with `CashCacheIneffectiveWarning` and the cache never invalidates.
    (`FileDataSource` returns a bool from `has_changed()`, but its state token is
    the file mtime — the base `state_token()` picks up its `_get_mtime` — which
    is why it works.) This is the same contract as
    [`dynamic_depends_on=`](../tutorials/feature-guides/dynamic-dependencies.md).

### Example: tracking a database table

```python
from cash.data_source import DataSource

class DBTableSource(DataSource):
    def __init__(self, connection, table_name):
        self.conn = connection
        self.table = table_name

    def get_id(self):
        return f"db_table:{self.table}"

    def has_changed(self):
        # The state TOKEN folded into the cache key — a value that moves when
        # the table changes, not a bool. (max_id, row_count) shifts whenever
        # rows are added or removed.
        row = self.conn.execute(
            f"SELECT MAX(id), COUNT(*) FROM {self.table}"
        ).fetchone()
        return (row[0], row[1])

    def update_state(self):
        pass   # token-based tracking keeps no internal state to update
```

Pass via `depends_on=`, then **call it** — a `DataSource` only proves itself when
the token is actually read:

<!-- test:expect-warning reason="reading the module-global `conn` is unhashable, so cash advises it can't invalidate on it — expected here, the DataSource is what tracks change" -->
```python
@c.cache(depends_on=[DBTableSource(conn, "users")])
def user_summary():
    return conn.execute("SELECT COUNT(*) FROM users").fetchone()

user_summary()          # computes, and records the current token
user_summary()          # hits — the table hasn't moved
```

Cash may warn here that `user_summary` reads a module global (`conn`) it can't
hash, so *changes to that global* won't invalidate the entry. That's expected
for a database example and not a problem: the connection isn't the data, and the
`DataSource` is what notices when the table moves.

Insert a row and the next call recomputes, because `(max_id, count)` changed.
(If `has_changed()` returned a `bool` instead, this is the moment cash would warn
that the entry can never invalidate — which is why the example exercises it
rather than only defining it.)
