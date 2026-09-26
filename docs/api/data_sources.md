---
search:
  boost: 0.5
---

# Data sources

For the decorator: objects passed in `depends_on=` whose state becomes part
of the cache key. The entry is recomputed when that state changes.

```python
from cash import DataSource, FileDataSource, RemoteFileDataSource
```

cash already tracks the files and remote objects a cached function reads
through common readers (`open`, `pd.read_csv("s3://...")`). Declare a source
only for what it cannot see; for a local file, `file_depends_on="path"` is
shorter.

<!-- claim: cash/file_source.py:FileDataSource @69335436 broad="the content-token contract is a property of the whole class", cash/remote_source.py:RemoteFileDataSource @754fe5e0 broad="the scheme list and validator contract are properties of the whole class" -->
::: cash.FileDataSource
    options:
      members: false

```python
from cash import Cash, FileDataSource

c = Cash()

@c.cache(depends_on=[FileDataSource("data/input.csv")])
def load_data():
    return pd.read_csv("data/input.csv")

load_data()  # computes
# a hit, until the file's content changes; a touch alone does not
load_data()
```

::: cash.RemoteFileDataSource
    options:
      members: false

`http(s)://` needs no extra package. Other schemes go through fsspec and the
filesystem package for the scheme (`s3fs` for `s3://`, `gcsfs` for `gs://`);
a missing one raises `DependencyNotFoundError`. The
[remote objects guide](../tutorials/feature-guides/custom-file-sources.md#remote-objects-tracked-by-the-stores-own-validator)
covers `immutable=` and `max_age=`.

<!-- test:skip reason="needs a reachable bucket" -->
```python
from cash import Cash, RemoteFileDataSource

c = Cash()

EVENTS = RemoteFileDataSource("s3://bucket/events.parquet")

@c.cache(depends_on=[EVENTS])
def load_events():
    return read_via_boto3("bucket", "events.parquet")
```

## Custom data sources

Subclass `DataSource` and implement its two methods.

::: cash.DataSource
    options:
      members:
        - get_id
        - state_token

`state_token()` must return a value that changes when the data does: a
version, a digest, a maximum id. A `bool` cannot, so the entry would never be
recomputed; cash warns
[`KEY-BOOL-STATE-TOKEN`](../warnings.md#key-bool-state-token) if it sees one.

```python
from cash import DataSource

class DBTableSource(DataSource):
    def __init__(self, connection, table_name):
        self.conn = connection
        self.table = table_name

    def get_id(self):
        return f"db_table:{self.table}"

    def state_token(self):
        # Changes whenever rows are added or removed.
        row = self.conn.execute(
            f"SELECT MAX(id), COUNT(*) FROM {self.table}"
        ).fetchone()
        return (row[0], row[1])
```

<!-- test:expect-warning reason="the body reads the module-global conn, which cash cannot hash; the DataSource is what tracks the table" -->
```python
@c.cache(depends_on=[DBTableSource(conn, "users")])
def user_summary():
    return conn.execute("SELECT COUNT(*) FROM users").fetchone()

user_summary()  # computes and records the token
user_summary()  # hit, until the table's token changes
```

cash may warn that `user_summary` reads a global (`conn`) it cannot hash.
That is expected here: the connection is not the data, and the source
tracks the table.
