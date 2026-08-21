# Backends

The concrete backend classes users instantiate. The recommended way to
pick one is via [configuration](../getting-started/configuration.md);
direct instantiation is useful for tests and advanced setups.

To write your own backend or contribute fixes, see
[Backend internals](backend_internals.md).

## Security

<!-- claim: cash/backends/serialization.py:PickleSerializer @eb457c2f broad="the pickle-executes-code warning is about the serializer as a whole", cash/backends/serialization.py:get_serializer @76cf2c1b -->
!!! danger "Cached values are pickled — loading a cache runs code"
    Every persistent backend (`FileBackend`, `SQLiteBackend`,
    `RedisBackend`, `S3Backend`) serializes values with **`pickle`**
    (or `cloudpickle`). Deserialization executes arbitrary code embedded
    in the payload, so **reading from a cache is only as safe as the party
    that wrote it.**

    | Scenario | Safe? |
    |---|---|
    | Your own local `.cash/` directory | ✅ As safe as your own code |
    | A cache exported by a trusted teammate on infra you control | ✅ Treat like running their `.py` |
    | A Redis/S3 store other tenants can write to | ⚠️ Only if every writer is trusted |
    | A `.cash/` directory downloaded from the internet / a stranger | ❌ Do not load — this is remote code execution |

    Cash does **not** sandbox deserialization. If you need to move results
    across a trust boundary, re-export the underlying data (Parquet, CSV,
    `np.save`, …) rather than shipping the pickle cache. There is no flag
    that makes loading an untrusted cache safe.

## Imports

```python
# Top-level (preferred for the common backends):
from cash import (
    InMemoryBackend,
    FileBackend,
    SQLiteBackend,
    CascadingBackend,
)

# Default backend (auto-built by Cash() when you pass nothing):
from cash.backends.tiered_backend import TieredBackend

# Experimental — extras-gated:
#   pip install 'cash-lib[redis]'
#   pip install 'cash-lib[s3]'
from cash.backends.redis_backend import RedisBackend
from cash.backends.s3_backend import S3Backend
```

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
      members:
        - __init__

::: cash.CascadingBackend
    options:
      members:
        - __init__

::: cash.backends.tiered_backend.TieredBackend
    options:
      members:
        - __init__

---

The two backends below ship as **experimental** under
`cash.experimental` — their configuration surfaces tend to evolve
faster than the core. Install the relevant extra:

```bash
pip install 'cash-lib[redis]'
pip install 'cash-lib[s3]'
```

::: cash.backends.redis_backend.RedisBackend
    options:
      members:
        - __init__

::: cash.backends.s3_backend.S3Backend
    options:
      members:
        - __init__
