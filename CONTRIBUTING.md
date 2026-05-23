# Contributing to Cash

Thank you for your interest in contributing to Cash! This guide will help you get started.

## Development Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/galgtonold/cash.git
cd cash

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Required Dependencies
```bash
pip install pytest pytest-subtests nbclient jupyter_client ipykernel pandas numpy
```

## Running Tests

### Unit Tests
```bash
# Run all unit tests
pytest tests/test_notebook/ tests/test_ui/ -v --tb=short

# Run specific test file
pytest tests/test_notebook/test_magics.py -v

# Run specific test
pytest tests/test_notebook/test_magics.py::TestCashMagics::test_basic_caching -v

# With debug output
pytest tests/test_notebook/test_magics.py -v -s
```

### Integration Tests
```bash
# Run integration tests (requires Jupyter kernel)
pytest tests/test_notebook_integration/ -v --tb=short
```

### Full Test Suite
```bash
# REQUIRED before submitting any PR
pytest tests/ -v --tb=short
```

### Test Architecture

**Unit tests** (`tests/test_notebook/`) use a mock IPython shell:
```python
def test_feature(cash_magics, mock_shell, cash_instance):
    cash_magics._auto_cache_enabled = True
    cash_magics._execute_cell("x = 42")
    assert mock_shell.user_ns['x'] == 42
```

**Integration tests** (`tests/test_notebook_integration/`) use real notebooks:
```python
def test_flow(nb_runner):
    nb_runner.create_notebook(["x = 10", "y = x * 2", "print(y)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "20" in nb_runner.get_output(3)
```

### Important: Test Isolation

Some unit tests mock IPython at module level. The test framework handles this with:
1. **Test ordering**: IPython-mocking tests run last among unit tests
2. **Cleanup fixtures**: `cleanup_sys_modules_between_tests` removes stale mocks
3. **Module-scoped cleanup**: Problematic test files restore `sys.modules` state

⚠️ **Do NOT run integration tests individually after running IPython-mocking unit tests** without restarting pytest.

## Project Structure

```
src/cash/
├── __init__.py           # Public API (stable exports)
├── core.py               # Main Cash class, decorator-based caching
├── data_source.py        # FileDataSource for file dependency tracking
├── analytics.py          # Usage analytics (experimental)
├── utils.py              # Utilities (notebook cell reading, etc.)
├── backends/             # Storage backends
│   ├── backend.py        # InMemory, File, Cascading, AsyncWrapper
│   ├── tiered_backend.py # Tiered multi-level caching
│   ├── redis_backend.py  # Redis backend
│   ├── s3_backend.py     # AWS S3 backend
│   └── serialization.py  # Data serialization
├── notebook/             # Jupyter notebook integration
│   ├── magics.py         # IPython magic commands (%cash_on, %%cash)
│   ├── statement_processor.py  # Statement-level cache processing
│   ├── upstream.py       # Upstream dependency detection & re-execution
│   ├── analysis.py       # AST-based code analysis
│   ├── file_tracker.py   # File read interception
│   └── control_structures.py  # Loop/conditional caching
├── experimental/         # Experimental features namespace
└── ui/                   # UI components (experimental)
    ├── explorer.py       # Cache browser widget
    ├── debugger.py       # Cache state debugger
    ├── visualizer.py     # Dependency graph visualization
    └── graph.py          # Graph utilities
```

## Adding a New Backend

1. Subclass `CacheBackend` from `cash.backends.backend`:
```python
from cash.backends import CacheBackend

class MyBackend(CacheBackend):
    def get(self, key): ...
    def set(self, key, value, metadata=None): ...
    def delete(self, key): ...
    def clear(self): ...
    def list_entries(self): ...
```

2. Add tests in `tests/test_backends/`
3. Register in `cash/backends/__init__.py`

## Adding File Tracking for a New Library

Use `Cash.register_file_handler()`:
```python
def my_handler(original_func, track_callback):
    def wrapper(path, *args, **kwargs):
        track_callback(str(path))
        return original_func(path, *args, **kwargs)
    return wrapper

cash.register_file_handler("mylib", "read_data", my_handler)
```

## Code Style

- Follow PEP 8
- Use type hints for public APIs
- Prefix internal functions/methods with `_`
- Write docstrings for all public methods
- Keep imports organized (stdlib → third-party → local)

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v --tb=short`)
- [ ] New tests added for new features
- [ ] No regressions in integration tests
- [ ] Documentation updated (README, docstrings)
- [ ] Roadmap updated if applicable (`.github/ROADMAP.md`)
