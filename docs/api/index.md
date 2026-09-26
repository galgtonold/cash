---
search:
  boost: 0.5
---

# API reference

For both paths: every public name, generated from the docstrings. Import each
one from the top-level package, for example `from cash import Cash, pure`, or
use it as `cash.X`. The guides explain when to use what; these pages list
parameters and return values.

| Page | Covers |
|---|---|
| [Cash class](cash.md) | `Cash` and its methods, the methods every cached function gets (`explain`, `cache_info`, `cache_clear`), `CacheExplanation`, and the module-level `configure`, `disabled`, `reset_session` and `cleanup`. |
| [Backends](backends.md) | `InMemoryBackend`, `FileBackend`, `SQLiteBackend`, `TieredBackend`, and the remote `RedisBackend` and `S3Backend` from `cash.backends`. |
| [Purity markers](purity.md) | `pure`, `stateful`, `opaque`, `is_pure`, `is_stateful`. |
| [Configuration API](config.md) | `CashConfig`, the tier entry `TierConfig`, `get_config`, `create_default_config`. |
| [Data sources](data_sources.md) | `DataSource`, `FileDataSource`, `RemoteFileDataSource`. |
| [Exceptions and warnings](exceptions.md) | Every exception and warning class, and how to filter warnings. |
| [Notebook integration](notebook.md) | `CashStripPreprocessor`, for stripping cash output with nbconvert. |
| [Inspection tools](inspection.md) | `CacheExplorer`, returned by `Cash.explorer()`. |
| [Internals](backend_internals.md) | For writing your own backend: `CacheBackend`, the metadata it receives, and the serializers. Not covered by the API guarantee. |

Each warning code is explained on [Warnings](../warnings.md), each setting on
[Configuration](../getting-started/configuration.md), and each command on
[Command-line interface](../cli.md).
