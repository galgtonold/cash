# Cash - Smart Caching for Python

Cash is a transparent caching library for Python with two primary use cases:

1. **Jupyter Notebook Caching** — Automatically cache cell results with `%cash_on`
2. **Decorator-based Caching** — Cache function results with `@cash.cache`

## Key Features

- **Statement-level caching** — Each line caches independently, not just whole cells
- **Automatic dependency tracking** — Changes propagate through the computation graph
- **File dependency tracking** — Re-run when data files change
- **Control structure caching** — Per-iteration loop caching and branch caching for conditionals
- **Mutation detection** — AST-based detection of in-place mutations prevents stale cache reads
- **Randomness detection** — Warns about unseeded random calls that break cache determinism
- **Side effect detection** — File writes, network calls, and DB operations flagged as uncacheable
- **Function tracking** — Changes to helper functions automatically invalidate dependent caches
- **Decorator ↔ notebook bridge** — `@cash.cache` calls are tracked in notebook badges with condensed metrics
- **Built-in type hashing** — Native hashing for pandas, numpy, polars, PyArrow, modin, dask
- **Custom type hashers** — `register_hasher()` for domain-specific or non-picklable types
- **Auto import tracking** — Local module imports tracked; changing a helper file invalidates caches
- **Purity declarations** — `@pure` and `@stateful` decorators for fine-grained caching control
- **Module-qualified function keys** — No collisions between same-named functions in different modules
- **Multiple backends** — TieredBackend (default), InMemory, File, SQLite, Redis, S3
- **Provenance tracking** — Full computational history for every variable
- **Zero configuration** — Just add `%cash_on` to your notebook

## Quick Example

```python
import cash

# Enable auto-caching for all cells
%cash_on

# First run: executes normally (~5 seconds)
df = pd.read_csv('large_file.csv')
result = df.groupby('category').sum()

# Second run: instant from cache ✅
```

Cash shows a badge above each cell summarizing what it did. The example below — restored upstream `df`, restored intermediate `features`, recomputed `preds` because a new `features` lineage invalidated its cache — gives you the visual vocabulary at a glance:

--8<-- "docs/_badges/anatomy_hero.html"

See [Reading the Cash badge](badges.md) for the full anatomy.

## Installation

```bash
pip install cash-lib
```

For optional backends:

```bash
pip install cash-lib[redis]    # Redis backend
pip install cash-lib[s3]       # S3 backend
pip install cash-lib[all]      # Everything
```

## How It Works

Cash intercepts notebook cell execution and:

1. **Analyzes** each statement's inputs and outputs using AST
2. **Computes** a cache key from the code + input lineage hashes
3. **Checks** the cache backend for existing results
4. **Executes** on cache miss, **restores** on cache hit
5. **Tracks** variable lineage for dependency invalidation

```
lineage(result) = hash(code + sorted(input_lineages) + file_deps)
```

## Documentation

- [Getting Started](getting-started/quickstart.md) — Installation and first steps
- [API Reference](api_reference.md) — All magic commands and backends
- [Architecture](notebook_caching_technical.md) — Technical deep dive
- [Migration Guide](migration_guide.md) — Moving from lru_cache, joblib, etc.
- [Contributing](contributing.md) — Development setup, testing, and code style
