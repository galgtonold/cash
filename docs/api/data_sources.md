# Data sources

Objects that contribute to a cache key by reporting "have I changed
since you last saw me?". The bundled `FileDataSource` tracks file
mtime; custom subclasses extend the same pattern to databases, URLs,
API endpoints, etc.

## Imports

```python
from cash import FileDataSource       # the bundled file-mtime source
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
```

When `input.csv` changes on disk, cached results are automatically
invalidated. For the simpler one-off case, prefer
`@c.cache(file_depends_on="data/input.csv")` — same behavior, less
typing.

---

## Custom data sources

To track something other than a file as a cache dependency, subclass
`DataSource`. The contract is three methods:

::: cash.data_source.DataSource
    options:
      members:
        - get_id
        - has_changed
        - update_state

### Example: tracking a database table

```python
import hashlib
from cash.data_source import DataSource

class DBTableSource(DataSource):
    def __init__(self, connection, table_name):
        self.conn = connection
        self.table = table_name
        self._last_max_id = self._current_max_id()

    def _current_max_id(self):
        row = self.conn.execute(
            f"SELECT MAX(id), COUNT(*) FROM {self.table}"
        ).fetchone()
        return (row[0], row[1])

    def get_id(self):
        return f"db_table:{self.table}"

    def has_changed(self):
        return self._current_max_id() != self._last_max_id

    def update_state(self):
        self._last_max_id = self._current_max_id()
```

Pass via `depends_on=`:

```python
@c.cache(depends_on=[DBTableSource(conn, "users")])
def user_summary():
    return conn.execute("SELECT COUNT(*) FROM users").fetchone()
```
