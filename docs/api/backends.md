---
search:
  boost: 0.5
---

# Backends

For both paths: the storage classes. Choosing a backend through
[configuration](../getting-started/configuration.md#backend) is usually
enough; build one yourself for tests or an unusual stack, and pass it as
`Cash(backend=...)` or `Cash(backends=[...])`.

```python
from cash import (
    FileBackend,
    InMemoryBackend,
    SQLiteBackend,
    TieredBackend,
)

from cash.backends import RedisBackend, S3Backend  # pip install 'cash-lib[redis]' / 'cash-lib[s3]'
```

To write a backend of your own, see [Internals](backend_internals.md).

## Security

<!-- claim: cash/backends/serialization.py:PickleSerializer @eb457c2f broad="the pickle-executes-code warning is about the serializer as a whole", cash/backends/serialization.py:get_serializer @76cf2c1b -->
!!! danger "Loading a cache runs code"
    `FileBackend`, `SQLiteBackend`, `RedisBackend` and `S3Backend` store
    values with `pickle` (a DataFrame may use Parquet instead). Loading a
    pickle runs code embedded in it, so a cache is only as safe as whoever
    wrote to it. cash does not sandbox this, and no setting makes an
    untrusted cache safe.

    | Cache | Safe to load? |
    |---|---|
    | Your own `.cash` directory | Yes, as safe as your own code. |
    | A cache from a trusted teammate, on machines you control | Yes, like running their code. |
    | A Redis server or S3 bucket others can write to | Only if every writer is trusted. |
    | A cache downloaded from a stranger | No: that is running their code. |

    To share results across a trust boundary, export the data (Parquet,
    CSV, `np.save`) instead of the cache.

::: cash.InMemoryBackend
    options:
      members:
        - __init__

::: cash.FileBackend
    options:
      members:
        - __init__

::: cash.SQLiteBackend
    options:
      members: false

::: cash.TieredBackend
    options:
      members:
        - __init__

::: cash.backends.redis_backend.RedisBackend
    options:
      members: false

::: cash.backends.s3_backend.S3Backend
    options:
      members: false
