# Backends

The concrete backend classes users instantiate. The recommended way to
pick one is via [configuration](../getting-started/configuration.md);
direct instantiation is useful for tests and advanced setups.

To write your own backend or contribute fixes, see
[Backend internals](backend_internals.md).

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
