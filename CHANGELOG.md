# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0b1] - Beta Release

### Changed
- Version bumped to 0.5.0b1 for public beta release
- Development status updated from Alpha to Beta
- Added `%cash_help` magic for quick-reference command card
- Expanded documentation with tutorials and use-case guides

## [0.3.0] - Decorator–Notebook Bridge

### Added

- **Decorator–Notebook Bridge**: `@cash.cache` decorator calls inside notebook cells are now tracked and displayed in badges
  - Call logging via `Cash._log_decorator_call()` with thread-safe append
  - `Cash.drain_decorator_calls()` for atomic retrieval of call events
  - Badge integration showing per-function hit/miss counts with condensed display for many calls
  - Decorator metrics (hits, misses, time saved) visible in `%cash_status` output

- **`cache_info()` and `cache_clear()`**: Per-function introspection on decorated functions
  - `func.cache_info()` returns `{'hits', 'misses', 'hit_rate', 'total_time_saved'}`
  - `func.cache_clear()` clears all cache entries for a function and resets stats
  - `func.__wrapped__` preserved via `functools.wraps`

- **`register_hasher(type_, hasher_fn)`**: Custom type hasher registration
  - Priority chain: `_cash_hash` attr → registered hashers → built-in hashers → pickle
  - Enables caching functions with non-picklable argument types

- **Built-in type hashers**: Native hashing for pandas DataFrame/Series, numpy ndarray, polars DataFrame/Series/LazyFrame, PyArrow Table/RecordBatch, modin DataFrame, dask DataFrame

- **`file_depends_on` parameter**: Shorthand for `@cash.cache(file_depends_on="data.csv")`, equivalent to `depends_on=[FileDataSource("data.csv")]`

- **Automatic import source invalidation**: Local module imports are tracked; changing a helper file invalidates dependent caches with transitive module dependency expansion

- **Opaque call pattern warnings**: Warnings when decorated functions are called with arguments that can't be hashed

- **`cleanup(max_age)` method**: Remove expired cache entries by age or stored TTL

- **`explorer()` method**: Returns `CacheExplorer` instance for interactive cache browsing

- **`register_file_handler()` method**: Extensible file tracking for custom libraries

### Fixed

- **Transitive notebook-level invalidation**: Changing a `@cash.cache` decorated function now correctly invalidates all notebook cells that depend on it, not just direct callers
- **Module-qualified function keys**: `Cash._get_func_key(func)` now uses `f"{func.__module__}.{func.__qualname__}"` to prevent collisions when different modules define functions with the same qualname

### Changed

- Default backend is now `TieredBackend` (InMemory L1 + FileBackend L2) with smart persistence policy
- `_analyzed.discard()` called when function source hash changes, forcing dependency graph rebuild

## [0.2.0] - 2025-02-06

### Added
- **Configuration System** (`cash.config`):
  - Global config file support (`~/.cash/config.toml`)
  - Environment variable support (`CASH_BACKEND`, `CASH_DEBUG`, `CASH_CACHE_DIR`, etc.)
  - `get_config()`, `CashConfig`, `create_default_config()` API
  - Config precedence: env vars > config file > defaults

- **SQLite Backend** (`cash.backends.sqlite_backend`):
  - Single-file cache storage using SQLite
  - WAL mode for concurrent access
  - TTL expiration, LRU eviction, max size limits
  - Thread-safe with entry counting and size tracking

- **FileBackend TTL**:
  - `default_ttl` parameter for automatic cache expiration
  - Per-entry TTL override via metadata
  - Expired entries auto-deleted on access

- **Collaboration Magics**:
  - `%cash_export <file>` - export cache entries to portable file
  - `%cash_import <file>` - import cache entries (with `--merge` mode)
  - `%cash_stats` - session-wide statistics (JSON output, reset)

- **Mutation Detection** (`cash.notebook.mutation_detector`):
  - AST-based detection of in-place mutations (append, extend, etc.)
  - Detection-only mode (does not affect lineage)
  - **Mutation-aware caching**: early detection before cache lookup
  - `get_top_level_mutated_variables()` excludes class/function body internals
  - Pure side-effects on non-output variables correctly marked as uncacheable

- **Purity Declaration System** (`cash.notebook.purity`):
  - `@cash.pure` decorator marks functions as pure (no side effects)
  - `@cash.stateful` decorator marks functions as stateful (always re-execute)
  - `is_pure()` / `is_stateful()` helper functions for checking markers
  - Pure functions skip mutation detection for better performance
  - Stateful functions skip caching entirely to ensure correctness
  - Integrated before skip-optimization to prevent stale @stateful results

- **Function Tracking** (`cash.notebook.function_tracker`):
  - Track function source code changes for cache invalidation
  - Function source hashes included in cache keys and lineage
  - `%cash_track` magic for monitoring imported module files
  - **Hot reload notification**: badge shows "🔄 Function changed" with orange highlight

- **Module Hot Reload**:
  - `%cash_track my_module` to watch for file changes
  - `%cash_track --check` auto-detects and reloads changed modules
  - `.pyc` cache invalidation for reliable reload

- **Structured Logging** (`cash.logging`):
  - JSON formatter for machine-readable log output
  - In-memory log handler with event type filtering
  - `%cash_debug json` for JSON console output
  - `%cash_debug file <path>` for file-based logging
  - `%cash_log` magic to view/filter/clear recent events

- **CLI Tool** (`python -m cash`):
  - `cash version` - show version info
  - `cash info` - show configuration details
  - `cash inspect <notebook>` - show cache statistics
  - `cash clear [dir]` - clear cache directories

- **nbconvert Integration** (`cash.nbconvert`):
  - `CashStripPreprocessor` strips badges, debug output from notebooks
  - Optional magic command stripping for clean exports

- **Documentation**:
  - API reference (`docs/api_reference.md`)
  - Migration guide from lru_cache, joblib, pickle (`docs/migration_guide.md`)
  - Architecture Decision Records (`docs/architecture_decisions.md`)

- **CI/CD**:
  - GitHub Actions CI (Python 3.10-3.13 × Linux/macOS/Windows)
  - PyPI publish workflow (release + Test PyPI)
  - Pre-commit hooks (ruff, file checks)
  - Docker support (Dockerfile + docker-compose.yml)

- **Community**:
  - CODE_OF_CONDUCT.md, SECURITY.md
  - Issue templates (bug report, feature request)
  - Pull request template

- **Badge UX**: Loop iteration grouping with collapsed display, loop variable values shown per iteration

- **Simulation Optimization**: Incremental upstream simulation caching

- **Polars Support**: File tracking for polars read/scan functions

- **CloudPickle Serializer**: Support for lambda functions and closures

- **Error Recovery Magics**:
  - `%cash_verify` - check cache integrity
  - `%cash_repair` - repair corrupted entries

- **Benchmarking**: `%cash_benchmark` magic for performance testing

- **Provenance Tracking** (`cash.notebook.provenance`):
  - `ProvenanceTracker` records variable history with full dependency chain
  - `%cash_provenance` magic: `--all`, `--graph`, `--time`, `--json`, `--clear`
  - Transitive dependency/dependent graph traversal
  - JSON export for external analysis

- **Audit Logging** (`cash.notebook.audit`):
  - `AuditLogger` with `AuditEntry` dataclass for compliance tracking
  - `%cash_audit on/off/show/summary/clear` magic
  - In-memory buffer (max 5000 entries) + optional file output
  - Filter by operation type or variable name

- **Lazy Deserialization** (`cash.backends.lazy`):
  - `LazyProxy` class defers deserialization until value access
  - `FileBackend.get_metadata()` for metadata-only lookups

- **AST Parse Caching**: LRU cache for parsed ASTs in upstream checker

- **Script Caching Demo**: Example showing `@cash.cache` decorator usage in Python scripts

- **Comprehensive Test Suite**: 1474 tests (unit + integration), 81% coverage
  - Library compatibility tests: sklearn, matplotlib, numpy, pandas
  - Data science workflow integration tests (sklearn pipelines, pandas → sklearn, CSV cascades)
  - Type hints for all public API and key internal modules
  - Purity declaration tests (45 tests)
  - Coverage boost tests: serialization, control structures, backends, config, purity, graph, annotations, nbconvert, tiered, SQLite, analysis

- **Documentation Site** (MkDocs Material):
  - Landing page with feature overview and key concepts
  - Getting Started guides: installation, quick start, configuration
  - Contributing guide with development setup and testing
  - API reference, architecture decisions, migration guide

- **Cache Diff** (`%cash_diff`):
  - Compare current session lineage with exported cache file
  - Shows only-in-current, only-in-other, changed, identical variable counts
  - `--vars` flag for variable-level detail
  - Supports both JSON and pickle cache file formats

- **JSON Cache Export** (`%cash_export --json`):
  - Export lineage metadata as JSON for %cash_diff interoperability
  - `%cash_export file.json --json` exports lineage without cache values
  - `--vars` filter works with JSON export

### Changed
- Configuration: `compress` and `debug` parameters now default to `None` (use config)
- Backend creation extracted to `_create_default_backend()` method
- Python version requirement updated to >=3.10
- Package metadata updated with proper classifiers and keywords
- Optional dependencies: `pip install cash-lib[redis]`, `[s3]`, `[all]`, etc.

### Fixed
- 10 test failures from Phase 1.1 cleanup
- `__all__` bug in `cash/__init__.py`
- IPython mock test isolation issues

## [0.1.0] - Initial Release

### Added
- Core `Cash` class with decorator-based caching
- Jupyter notebook integration via IPython magics (`%cash_on`, `%%cash`)
- Statement-level caching with automatic dependency tracking
- Pluggable backends: InMemory, File, Redis, S3, Tiered/Cascading
- File dependency tracking (pandas, numpy, builtins, etc.)
- Smart persistence policy for tiered caching
- Interactive badge display for cache status
- Upstream dependency detection and re-execution
