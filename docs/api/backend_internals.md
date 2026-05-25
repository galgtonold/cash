# Backend internals

This page is for users **writing their own backend** or contributing
fixes to the bundled ones. End-users picking a backend should go to
[Backends](backends.md) instead.

## Imports

```python
from cash.backends import CacheBackend, CacheMetadata
from cash.backends._base import PendingWrites
from cash.backends.serialization import (
    Serializer,
    PickleSerializer,
    CloudPickleSerializer,
    ParquetSerializer,
    get_serializer,
)
```

## Writing a backend in three steps

1. Subclass `CacheBackend` (below) and implement the five abstract
   methods (`get`, `set`, `delete`, `clear`, `list_entries`).
2. Stamp standard metadata via `_init_metadata()` at the start of
   `set()` — keeps your backend's entries compatible with the
   `list_entries` consumers (UI, cleanup, badge).
3. Honour the error contract: return `(None, None)` from `get()` on
   miss, raise `CacheBackendError` from anywhere else for
   infrastructure failures.

If your backend touches the network or disk, also pull in
`PendingWrites` (below) so `set()` can return fast and the real I/O
happens off the calling thread.

If your stored format isn't bytes-friendly (e.g. you want Parquet on
disk for DataFrames), see the `Serializer` hierarchy and the
`get_serializer()` dispatch below.

---

::: cash.backends.CacheBackend
    options:
      members:
        - get
        - set
        - delete
        - clear
        - list_entries
        - cleanup_expired
        - get_metadata
        - tier_labels
        - lock
        - shutdown

---

The typed-dict shape every backend's `metadata` parameter and
`list_entries` row conforms to. All fields are optional (`total=False`)
because not every backend populates every field — always use
`metadata.get(key)`, not direct subscript.

::: cash.backends.CacheMetadata

### Standard fields

| Field | Type | When populated |
|---|---|---|
| `key` | `str` | Always (stamped by `_init_metadata`) |
| `created_at` | `float` | Always (unix ts) |
| `last_access` | `float` | Updated by backends that track LRU |
| `access_count` | `int` | Updated by backends that track frequency |
| `size` | `int` | Backends that track size (`FileBackend`, `SQLite`, RAM with `max_size_bytes`) |
| `storage` | `list[str]` | Tier labels for the entry (e.g. `["RAM", "DISK"]`) |
| `ttl` | `int` | Set by `Cash.cache(ttl=...)` |
| `execution_time` | `float` | Set by the decorator; used by `TieredBackend`'s smart-persistence policy |
| `outputs` | `list[str]` | Set by the notebook statement processor |
| `lineage_hash` | `str` | Set by the notebook statement processor |
| `source` | `str` | Per-backend identifier (e.g. `'RAM'`, `'disk'`) |

Cash itself adds a few decorator-specific fields too — `serializer_cls`,
`args_hash`, `state_hash`, `func_name`, `auto_file_deps`,
`iterator_storage`, `n_chunks`, `materialized_iterator`. These are
internal: a custom backend should round-trip whatever metadata it's
given without inspecting it.

---

Per-backend background-write scheduler. Used by the bundled
`FileBackend`, `SQLiteBackend`, `RedisBackend`, and `S3Backend` so a
slow `set()` doesn't block the user's calling thread.

::: cash.backends._base.PendingWrites
    options:
      members:
        - submit
        - wait
        - drain
        - wait_all
        - pending_count
        - shutdown

### Usage pattern

The recommended pattern in a custom backend:

```python
from cash.backends import CacheBackend
from cash.backends._base import PendingWrites

class MyBackend(CacheBackend):
    def __init__(self):
        self._writes = PendingWrites()

    def set(self, key, value, metadata=None, serializer=None):
        # Serialise on the calling thread — must NOT happen in the
        # worker, or post-set() mutation can corrupt the cached copy.
        payload = serializer.serialize(value) if serializer else pickle.dumps(value)

        # Dispatch the actual write to the background.
        self._writes.submit(key, self._write_payload, key, payload, metadata)

    def get(self, key):
        # Wait for any pending write for THIS key before reading,
        # so we observe the latest value.
        self._writes.wait(key)
        ...

    def delete(self, key):
        self._writes.drain(key)  # discard any pending write for key
        ...

    def shutdown(self):
        self._writes.shutdown(wait=True)
```

---

Cash ships three serialisers and a dispatch function. Custom backends
can either accept a `serializer` parameter (the bundled approach — the
caller picks) or hard-wire one (e.g. a Parquet-only backend).

::: cash.backends.serialization.Serializer
    options:
      members:
        - serialize
        - deserialize

::: cash.backends.serialization.PickleSerializer
    options:
      members: false

::: cash.backends.serialization.CloudPickleSerializer
    options:
      members: false

::: cash.backends.serialization.ParquetSerializer
    options:
      members: false

::: cash.backends.serialization.get_serializer
