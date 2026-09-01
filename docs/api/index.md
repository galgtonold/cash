# API Reference

This section is the **auto-generated** reference for every public symbol
in the `cash` package. Content is pulled directly from docstrings in the
source — if you spot an issue, file it against the corresponding
`src/cash/...` file, not the page itself.

## How it's organised

| Page | Covers |
|---|---|
| [Cash class](cash.md) | `Cash`, the lazily-created module-level singleton, `configure`, `reset_session`, and `CacheExplanation` (returned by `f.explain()`). |
| [Backends](backends.md) | Concrete backends users instantiate: `InMemoryBackend`, `FileBackend`, `SQLiteBackend`, `CascadingBackend`, plus the experimental `RedisBackend` and `S3Backend`. |
| [Backend internals](backend_internals.md) | The `CacheBackend` ABC, the `CacheMetadata` dataclass (and the plain-dict wire format backends actually see), `Serializer` hierarchy, `PendingWrites` helper. Read this if you want to write your own backend or contribute fixes. |
| [Purity & annotations](purity.md) | `@pure`, `@stateful`, `mark_pure`, `mark_stateful`, `is_pure`, `is_stateful`, `analyze_function_purity`. |
| [Configuration](config.md) | `CashConfig` dataclass, `get_config`, `create_default_config`. |
| [Data sources](data_sources.md) | `FileDataSource` and the `DataSource` ABC for custom dependency tracking. |
| [Exceptions & warnings](exceptions.md) | The full hierarchy of `CashError` subclasses and `CashWarning` subclasses. |
| [Notebook integration](notebook.md) | `CashStripPreprocessor` (nbconvert), `CacheStatus`, `ExecutionResult`, `CashMagics`, `CodeAnalyzer`. The Python-side hooks used by tooling around the magics. |
| [Experimental](experimental.md) | `CacheExplorer`, `CacheDebugger`, `DependencyGraph`, `AnalyticsManager`, `visualize_notebook`. Useful but unstable APIs under `cash.experimental`. |

## Looking for narrative guides?

The pages here are reference material — they show every parameter and
return type. For walkthroughs that put symbols in context:

- [Decorator guide](../decorator.md) — `@cash.cache` with examples and gotchas
- [Notebook caching](../notebook_caching_api.md) — `%cash_on` and friends
- [Purity tutorial](../tutorials/feature-guides/purity-decorators.md) — `@pure`, `@stateful`, decorator-side analyzer
- [Caching class methods](../tutorials/feature-guides/caching-class-methods.md) — recipe for stateful receivers
- [Configuration page](../getting-started/configuration.md) — how the config layers interact

## Filtering Cash warnings

All Cash-emitted warnings inherit from `cash.CashWarning` (itself a
`UserWarning` subclass). To silence one category:

```python
import warnings, cash
warnings.filterwarnings("ignore", category=cash.CashImpurityWarning)
```

To promote one to an error (useful in CI):

```python
warnings.filterwarnings("error", category=cash.CashImpurityWarning)
```

See the [Exceptions & warnings](exceptions.md) page for the full
hierarchy.
