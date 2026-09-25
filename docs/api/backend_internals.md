# Internals

For both paths, and only if you write your own backend: the base class, the
metadata it receives, and the serializers. They are importable from
`cash.backends` but are not part of the public API that
[Versioning](../versioning.md) promises to keep, and may change in any
release. To pick a bundled backend, see [Backends](backends.md).

```python
from cash.backends import (
    CacheBackend, EntryMetadata, MetadataDict,
    Serializer, PickleSerializer, ParquetSerializer, get_serializer,
)
```

## Writing a backend

<!-- claim: cash/backends/_base.py:CacheBackend @663d2688 broad="the page documents the ABC as a whole contract" -->
Subclass `CacheBackend` and implement `get`, `set`, `delete`, `clear` and
`list_entries`. Every other method has a default. The contract:

- `get(key)` returns `(metadata, value)`, or `(None, None)` for a missing
  or unreadable entry.
- `set()` stores the value with its metadata, adding `key`, `created_at`,
  `last_access` and `access_count` when they are missing.
- Storage failures raise `CacheBackendError`. A failed `set()` cleans up
  what it partly wrote before raising.
- Treat metadata as an opaque dict: store every key you are given and read
  keys with `metadata.get(...)`. A missing `ttl` means "use my default".

```python
import time

from cash import Cash
from cash.backends import CacheBackend

class DictBackend(CacheBackend):
    source_label = "DICT"
    cost_kind = "ram"

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key, (None, None))

    def set(self, key, value, metadata=None, serializer=None):
        meta = dict(metadata or {}, key=key)
        now = time.time()
        meta.setdefault("created_at", now)
        meta.setdefault("last_access", now)
        meta.setdefault("access_count", 0)
        self.store[key] = (meta, value)

    def delete(self, key):
        self.store.pop(key, None)

    def clear(self):
        self.store.clear()

    def list_entries(self):
        return [meta for meta, _ in self.store.values()]

c = Cash(backend=DictBackend())

@c.cache
def square(x):
    return x * x

square(3)
square(3)
assert square.cache_info()["hits"] == 1
```

Override a default when your backend can do better:

| Override | When |
|---|---|
| `peek_metadata(key)` | `get()` records an access (a use count, a last-used time). `explain()` reads through this and must change nothing. |
| `entry_count()` | You can count entries without reading them. The default reads every entry's metadata, and `%cash_on` calls it each time it runs. |
| `local_dir` (property) | Entries live in a local directory; cash keeps its per-function records beside them. |
| `default_ttl` (property) | Your backend takes a default TTL. Return it here; cash reads it only through this property. |
| `shutdown()` | You write in the background; wait for the writes here. |

Class attributes: `source_label` names the tier where an entry says where it
came from (`RAM`, `DISK`); `cost_kind` is how the cost model predicts restore
times from it (`"ram"`, `"disk"`, `"redis"` or `"s3"`; default `"disk"`);
`max_size_bytes` is the largest value a `TieredBackend` copies into it
(`None`: any size).

::: cash.backends.CacheBackend
    options:
      members:
        - get
        - set
        - delete
        - clear
        - list_entries
        - entry_count
        - peek_metadata
        - get_metadata
        - cleanup_expired
        - tier_labels
        - lock
        - shutdown

## Metadata

A backend receives and returns metadata as a plain `MetadataDict`
(`dict[str, Any]`). `CacheMetadata` is the typed view cash builds before a
write and reads after a read. Its `to_dict()` leaves out unset fields, so an
entry written without a `ttl` has no `ttl` key. `EntryMetadata` lists every
key the bundled backends read or write.

::: cash.decorator.cache_metadata.CacheMetadata
    options:
      members: false

## Serializers

A backend either uses the `serializer` that `set()` is given, as the bundled
ones do, or always uses one of its own. `get_serializer(value)` picks Parquet
for a pandas DataFrame when pyarrow or fastparquet is installed, and pickle
otherwise. Loading a pickle runs code; see [Security](backends.md#security).

::: cash.backends.Serializer
    options:
      members:
        - serialize
        - deserialize

::: cash.backends.PickleSerializer
    options:
      members: false

::: cash.backends.ParquetSerializer
    options:
      members: false

::: cash.backends.get_serializer
