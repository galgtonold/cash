# Cash: Smart Caching for Python & Jupyter Notebooks

[![PyPI version](https://img.shields.io/pypi/v/cash-lib.svg)](https://pypi.org/project/cash-lib/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/cash-lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`cash` is a smart caching library for Python with two primary use cases:

1. **Decorator-based caching** (`@cash.cache`) for functions with automatic dependency tracking
2. **Jupyter notebook caching** via IPython magics (`%cash_on`) with statement-level granularity

## Why Cash?

Data scientists spend hours re-running notebooks — loading the same CSVs, recomputing the same transformations, re-training models that haven't changed. Cash eliminates this waste:

| Scenario | Without Cash | With Cash | Savings |
|----------|-------------|-----------|--------|
| Re-running 50-cell notebook 10×/day | Each run: 3 min | First run: 3 min, subsequent: ~15s | **~29 min/day** |
| Cloud notebook (SageMaker/Vertex) @ $0.50/hr | 10 runs × 3 min = 30 min | 3 min + 9 × 15s = ~5 min | **~$50/year per user** |
| Team of 10 sharing preprocessed data | Each person runs full pipeline | Export/import cached results | **90% compute reduction** |
| ML experiment iteration | Full pipeline re-run per tweak | Only changed steps re-execute | **5-10× faster iteration** |

**For a team of 10, Cash can save ~$75K/year in productivity** — and cloud compute savings stack on top.

### How is this different?

- **Not cell-level — statement-level.** If you change one line in a cell, only that line and its dependents recompute.
- **Not manual — automatic.** No `pickle.dump()`, no `if os.path.exists(...)` guards. Just add `%cash_on`.
- **Not blind — dependency-aware.** Cash tracks which variables depend on which. Change `config` → only cells using `config` recompute.
- **Not brittle — file-aware.** CSVs, Parquet files, and data dependencies are tracked automatically. File changes trigger recomputation.

## Features

- 🚀 **Statement-level notebook caching** - Each statement is cached independently, not the whole cell
- 🔗 **Automatic dependency tracking** - Changes propagate through the dependency chain automatically
- 📁 **File dependency tracking** - Cache invalidates when data files (CSV, Parquet, etc.) change
- 🔄 **Upstream re-execution** - Changed upstream cells are automatically re-executed
- 🎯 **Decorator ↔ notebook bridge** - `@cash.cache` calls are tracked in notebook badges with condensed metrics
- 🧬 **Built-in type hashing** - Native support for pandas, numpy, polars, PyArrow, modin, and dask objects
- 🔧 **Custom type hashers** - `register_hasher()` for non-picklable or domain-specific types
- 📦 **Auto import tracking** - Local module imports are tracked; changing a helper file invalidates caches
- 💾 **Multiple backends** - TieredBackend (default), InMemory, File, SQLite, Redis, S3
- 🎯 **Zero-config** - Just add `%cash_on` to your notebook

## Installation

```bash
pip install cash-lib
```

## Quick Start

### Notebook Caching (Primary Use Case)

Add `%cash_on` to the first cell of your Jupyter notebook:

```python
# Cell 1
%load_ext cash
%cash_on
```

```python
# Cell 2 - This gets cached automatically
import pandas as pd
df = pd.read_csv("large_dataset.csv")  # File dependency tracked!
```

```python
# Cell 3 - Cached based on Cell 2's state
summary = df.describe()
print(summary)
```

When you re-run the notebook:
- ✅ If nothing changed → restored from cache instantly
- 🔄 If `large_dataset.csv` changed → Cell 2 & 3 re-execute
- ⚡ If only Cell 3 code changed → Cell 2 stays cached, Cell 3 re-executes

### Decorator-based Caching

```python
from cash import cache

@cache
def expensive_computation(x):
    # This result is cached based on code + arguments
    return x ** 2 + sum(range(x))

result = expensive_computation(1000000)

# Introspection
print(expensive_computation.cache_info())
# {'hits': 0, 'misses': 1, 'hit_rate': 0.0, 'total_time_saved': 0.0}

expensive_computation.cache_clear()  # Clear cached results for this function
```

### Custom Instance with File Backend

```python
from cash import Cash, FileBackend

app = Cash(backend=FileBackend("./my_cache"))

@app.cache
def load_and_process(path):
    import pandas as pd
    return pd.read_csv(path).describe()
```

### File Dependency Shorthand

```python
from cash import Cash

app = Cash()

@app.cache(file_depends_on="data.csv")
def load_data():
    return open("data.csv").read()
# Cache automatically invalidates when data.csv changes
```

### Custom Type Hashers

```python
from cash import Cash

app = Cash()

# Register a hasher for custom types that can't be pickled
app.register_hasher(
    MyModel,
    lambda model: model.get_fingerprint()
)

@app.cache
def predict(model, data):
    return model.predict(data)
```

## Notebook Magic Commands

| Command | Description |
|---------|-------------|
| `%cash_on` | Enable automatic caching for all cells |
| `%cash_on ttl=3600` | Enable with 1-hour TTL |
| `%cash_off` | Disable automatic caching |
| `%cash_debug on/off` | Toggle debug output |
| `%cash_badge html/print/off` | Set badge display mode |
| `%cash_status` | Show last cell execution metrics |
| `%cash_stats` | Show session-wide statistics (hits, misses, time saved) |
| `%cash_log` | View recent structured log events |
| `%cash_verify` | Check cache integrity |
| `%cash_verify --fix` | Check and fix corrupted entries |
| `%cash_repair` | Remove corrupted cache entries |
| `%cash_repair --state` | Reset in-memory tracking (keep cache) |
| `%cash_repair --full` | Clear all cache and state |
| `%cash_provenance var` | Show history of a variable |
| `%cash_provenance --time` | Timeline of all computations |
| `%cash_audit on/off` | Enable/disable audit logging |
| `%cash_track module` | Track external module file changes |
| `%cash_export file` | Export cache to a file |
| `%cash_export file --json` | Export lineage as JSON (for `%cash_diff`) |
| `%cash_import file` | Import cache from a file |
| `%cash_diff file` | Compare current session with exported cache |
| `%cash_diff file --vars` | Show variable-level differences |
| `%cash_benchmark` | Benchmark a statement (iterations, timing) |
| `%%cash` | Cell magic for explicit caching |

## Backend Comparison

| Backend | Speed | Persistence | Sharing | Use Case |
|---------|-------|-------------|---------|----------|
| `TieredBackend` | ⚡ Fastest | Yes (L2) | No | **Default** — InMemory L1 + File L2 with smart persistence |
| `InMemoryBackend` | ⚡ Fastest | No | No | Session-only, development |
| `FileBackend` | 🏃 Fast | Yes | No | Persistent local caching |
| `SQLiteBackend` | 🏃 Fast | Yes | No | Many small entries, atomic ops |
| `CascadingBackend` | 🏃 Fast | Mixed | No | Custom multi-tier setups |
| `RedisBackend`* | 🌐 Network | Yes | Yes | Distributed teams |
| `S3Backend`* | 🌐 Network | Yes | Yes | Cloud workflows |

*Available via `from cash.experimental import RedisBackend, S3Backend`

## How It Works

Cash uses **lineage-based caching** at the statement level:

1. **AST Analysis**: Each statement is parsed to identify inputs and outputs
2. **Lineage Hashing**: A hash is computed from the code + all input dependencies
3. **Mutation Check**: In-place mutations (`.append()`, `+=`, etc.) are detected to avoid stale reads
4. **Cache Lookup**: If the lineage hash matches a cached entry, restore from cache
5. **Decorator Bridge**: `@cash.cache` calls inside statements are logged and shown in badges
6. **Upstream Detection**: Before running a cell, check if upstream cells have changed
7. **File Tracking**: File reads (pandas, numpy, open) are intercepted for dependency tracking
8. **Side Effect Detection**: File writes, network calls, and DB operations are flagged as uncacheable

### Cache Key Format
```
stmt:{sha256(code + sorted(input_lineage_hashes) + file_dependency_hashes + func_source_hashes)}
```

## Troubleshooting

### Cache seems stale
```python
%cash_verify        # Check integrity
%cash_repair        # Fix corrupted entries
%cash_repair --full # Nuclear option: clear everything
```

### Debug what's happening
```python
%cash_debug on      # Enable debug output
# Run your cell...
%cash_debug off     # Disable when done
```

### Variable not restored correctly
```python
%cash_repair --state  # Reset tracking state, keep cache
# Re-run your cells from top
```

## Purity Declarations

Control caching behavior for specific functions:

```python
from cash import pure, stateful

@pure
def compute(x, y):
    return x + y  # Always safe to cache, skip mutation checks

@stateful
def train_model(data):
    model.fit(data)  # Has side effects, always re-execute
```

## Experimental Features

Advanced features are available via `cash.experimental`:

```python
from cash.experimental import CacheExplorer, CacheDebugger, AnalyticsManager
```

These features may change in future versions.

## License

MIT
