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

A directory-level map (browse `src/cash/` for the individual modules — the
layout below is intentionally coarse so it doesn't drift as files move within a
package):

```
src/cash/
├── core.py             # The Cash class + @cash.cache decorator
├── config.py           # CashConfig; TOML / env / programmatic resolution
├── data_source.py      # FileDataSource and the DataSource protocol
├── remote_source.py    # RemoteFileDataSource: s3/gs/az/http, keyed on the
│                       #   store's own validator (ETag, version id, generation)
├── dependency_state.py # Folds source/dep/helper state into the cache key
├── purity_analyzer.py  # Static purity/impurity analysis for the decorator
├── source_norm.py      # Normalizes source before hashing (drops comments,
│                       #   blank lines, indentation width) so a reformat keeps
│                       #   your cache
├── analytics.py        # Cache-usage analytics
├── effectiveness.py    # Measures whether caching actually paid off
├── graph.py            # Dependency-graph utilities
├── nbconvert.py        # nbconvert preprocessor (strips badges / magics)
├── logging.py          # Structured logging
├── exceptions.py       # Public exception types
├── utils.py            # Shared internal helpers
├── _agent_guide.py     # The text cash.help() returns — a byte-for-byte copy
│                       #   of docs/for-coding-agents.md, kept in step by a test
├── __main__.py         # CLI entry point (python -m cash)
│
├── backends/           # Pluggable storage: _base.py (abstract CacheBackend),
│                       #   memory / file / cascading / tiered / sqlite / redis /
│                       #   s3, plus serialization and lazy (LazyProxy)
├── notebook/           # Jupyter integration
│   ├── ipython/        #   magics, cell executor, argument parsing
│   ├── statement/      #   statement-level caching
│   ├── control_structures/  # per-iteration loop / branch caching
│   ├── upstream/       #   upstream simulation & virtual restore
│   ├── badge_renderer/ #   the HTML / text cell badge
│   └── *.py            #   analysis, annotations, file/function tracking,
│                       #   cost_model, consumables, randomness, purity,
│                       #   provenance, audit, …
├── ui/                 # Interactive display components (explorer, debugger, …)
├── experimental/       # Lazy-imported experimental APIs
└── labextension/       # PREBUILT JupyterLab extension (cash-live-cells), shipped
                        #   in the wheel. Source is TypeScript under
                        #   labextension/ at the repo root; this is its build
                        #   output and is committed, so a plain `pip install`
                        #   drops it in without Node
```

The default backend is `TieredBackend([InMemoryBackend, FileBackend])` — RAM in
front of disk. Redis and S3 are optional-dependency backends.

The labextension is the part most easily missed: it pushes your unsaved cell
sources to the kernel so an upstream check can see edits you haven't saved.
What it covers and what it doesn't is documented under
[editing without saving](known-limitations.md#editing-without-saving).

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

# With coverage (pytest-cov is not a project dependency; use the
# `coverage` tool the [tool.coverage.*] config in pyproject.toml targets)
coverage run -m pytest tests/
coverage report -m
```


!!! warning "A bare `pytest tests/` needs more than `[dev]`"
    `tests/docs/` stubs third-party modules by monkeypatching the real
    ones, so those packages must be importable — without
    `pip install -e ".[dev,docs-test]"` several doc pages fail on a real
    import rather than skipping. CI never runs the suite bare; it uses
    `--ignore=tests/test_notebook_integration --ignore=tests/test_wheel_gate
    --ignore=tests/docs` for the fast job and a separately-provisioned job
    for `tests/docs/`. Mirror that locally, or install both extras.

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
