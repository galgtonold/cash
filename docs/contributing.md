# Contributing to Cash

Thank you for your interest in contributing to Cash! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Clone and Install

```bash
git clone https://github.com/galgtonold/cash.git
cd cash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### Verify Setup

```bash
pytest tests/ -v --tb=short
```

## Project Structure

```
src/cash/
├── core.py                 # Main Cash class, decorator-based caching
├── config.py               # Configuration management
├── analytics.py            # Cache usage analytics
├── data_source.py          # FileDataSource for dependency tracking
├── logging.py              # Structured logging (JSON, file output)
├── nbconvert.py            # nbconvert preprocessor for stripping badges
├── utils.py                # Utility functions
├── __init__.py             # Public API exports
├── __main__.py             # CLI tool (python -m cash)
├── backends/               # Pluggable storage backends
│   ├── backend.py          # Base classes + InMemory/File/Cascading backends
│   ├── sqlite_backend.py   # SQLite backend
│   ├── redis_backend.py    # Redis backend (experimental)
│   ├── s3_backend.py       # S3 backend (experimental)
│   ├── tiered_backend.py   # Multi-tier backend (experimental)
│   ├── serialization.py    # Pickle/CloudPickle serializers
│   └── lazy.py             # LazyProxy for deferred deserialization
├── notebook/               # Jupyter integration
│   ├── magics.py           # IPython magic commands
│   ├── cache_key.py        # Unified cache key computation (single source of truth)
│   ├── statement_processor.py  # Statement-level caching
│   ├── upstream.py         # Upstream cell tracking & virtual restore
│   ├── analysis.py         # AST-based code analysis
│   ├── annotations.py      # @cash: directive parser
│   ├── file_tracker.py     # File dependency tracking
│   ├── function_tracker.py # Function source tracking & module hot reload
│   ├── control_structures.py  # Loop/conditional caching
│   ├── mutation_detector.py   # In-place mutation detection
│   ├── side_effects.py     # Side effect detection (file writes, network, etc.)
│   ├── randomness.py       # Unseeded randomness detection
│   ├── purity.py           # @pure/@stateful decorators
│   ├── provenance.py       # Variable provenance tracking
│   └── audit.py            # Audit logging for compliance
├── ui/                     # Display components
│   ├── explorer.py         # Interactive cache browser
│   ├── debugger.py         # Cache state debugger
│   ├── visualizer.py       # Notebook dependency visualization
│   ├── dashboard.py        # Analytics dashboard
│   └── graph.py            # Dependency graph utilities
└── experimental/           # Experimental feature namespace
    └── __init__.py         # Lazy imports for experimental APIs
```

## Testing

### Test Structure

- `tests/test_notebook/` — Unit tests (mock IPython)
- `tests/test_notebook_integration/` — Integration tests (real notebooks)
- `tests/` — Core library tests

### Running Tests

```bash
# All tests
pytest tests/ -v --tb=short

# Unit tests only
pytest tests/test_notebook/ -v

# Integration tests only
pytest tests/test_notebook_integration/ -v

# Specific test file
pytest tests/test_notebook/test_magics.py -v

# Specific test
pytest tests/test_notebook/test_magics.py::TestCashMagics::test_basic_caching -v

# With coverage
pytest tests/ --cov=cash --cov-report=term-missing
```

### Writing Tests

#### Unit Tests

Use the `magics_fixture` for testing notebook components:

```python
def test_feature(magics_fixture):
    magics, shell, backend = magics_fixture
    shell.user_ns['x'] = 10
    magics.cash("", "y = x * 2")
    assert shell.user_ns['y'] == 20
```

#### Integration Tests

Use the `nb_runner` fixture for end-to-end notebook tests:

```python
def test_feature(nb_runner):
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "print(f'Result: {y}')"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Result: 20" in nb_runner.get_output(3)
```

### Test Isolation

!!! warning
    Unit tests mock IPython in `sys.modules`. Never run integration tests
    in the same pytest session after unit tests without the cleanup fixtures.

## Code Style

- **Type hints** on all public methods
- **Docstrings** for all public APIs
- **PEP 8** formatting
- **No trailing whitespace**

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for your changes
3. Ensure all tests pass: `pytest tests/ -v --tb=short`
4. Update documentation if needed
5. Submit PR with clear description

## Architecture Guidelines

- **Lineage hashes** encode full dependency chains
- **Cache keys** include code + input lineages + file hashes
- **Statement-level** granularity, not cell-level
- **Pluggable backends** via `CacheBackend` abstract class
