# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0b2] - Unreleased

### Added
- `@cash.cache` now supports `async def` functions. Awaited results are
  cached; auto-file-dep tracking works correctly under concurrent
  `asyncio.gather`. (Async generators emit a `CashCacheIneffectiveWarning`
  and are returned unwrapped — full async-gen caching is planned for
  a later release.)
- `cash.CashWarning`, `cash.CashCacheIneffectiveWarning`,
  `cash.CashCacheStoreFailedWarning` exposed at the top level. Filter
  via standard `warnings.filterwarnings(...)` — e.g. set
  `CashCacheIneffectiveWarning` to `error` in CI to fail the build
  when a deploy introduces an unpicklable arg.
- New tutorial: `docs/caching-class-methods.md` — recipe for caching
  methods on stateful objects (`Loader`, services, database wrappers)
  via `cash.register_hasher`.
- `@cash.cache(cache_if=callable)` — optional predicate that receives
  the function's return value and returns a bool. When false, the
  result is returned to the caller as normal but not stored in the
  cache. Useful for skipping the caching of negative results
  (`cache_if=lambda r: r is not None`). Predicate exceptions are
  caught (debug-logged) and treated as false. Works on both sync and
  async functions.
- `@cash.cache` now caches functions that return one-shot iterators
  (Python generators, `map`/`filter` results, custom iterators). The
  iterator is eagerly materialized into a list, the list is cached,
  and each call returns a fresh iterator over the cached values.
  Generator-specific methods (`.send`, `.throw`) are not supported on
  the cached wrapper. Not suitable for infinite or streaming
  generators — see `docs/caching-class-methods.md` for the trade-off.
- `cash.register_hasher(T, fn)` now hashes `fn`'s source (or
  bytecode) at registration and embeds the hash in the cache key.
  Changing the body of a registered hasher invalidates dependent
  cache entries, even when the new hasher's output coincidentally
  matches.
- `@cash.cache(chunk_max_items=..., chunk_max_bytes=...)` — iterator
  results are now stored in chunks. Defaults are 1M items and 1GB
  bytes; iterators below these thresholds land in a single chunk and
  behave indistinguishably from a list. Larger iterators are split
  across multiple backend keys and the retrieval iterator reads them
  lazily. RAM bounded by chunk size on both write and read. Chunked
  storage is on by default with no opt-in required.
- `f.explain(*args, **kwargs)` — every `@cash.cache`-decorated function
  now exposes an `explain()` method that returns a `CacheExplanation`
  describing whether the next call with those args would hit or miss
  the cache, and *why*. Reasons include `hit`, `key_uncomputable`
  (unhashable arg), `no_entry` (with `source_changed` detection when a
  sibling entry exists), `ttl_expired`, and `file_changed` (with the
  list of changed paths). Pure introspection — never calls the
  function, mutates stats, or writes to the backend. Available on
  async-wrapped functions too. `CacheExplanation` is exported from the
  top-level `cash` package.
- `f.cache_info()` now includes a `warnings` key — a rolling log of
  recent `CashWarning` emissions for that function (capped at the
  last 20). Lets users discover silent misbehavior after the fact
  even when `warnings.simplefilter` swallowed the stderr emission.
  `f.cache_clear()` now also resets this log and forgets dedup marks
  so future misbehavior re-warns.
- **Purity analyzer on the decorator** — `@cash.cache` now AST-walks
  the decorated function body and its module-bounded helpers on
  first call, flagging known-impure calls (`requests.post`,
  `os.system`, file-write methods, `logging.info`, …), scope
  mutations (`global`, `nonlocal`, attribute/subscript assignment),
  explicit dynamism (`eval`/`exec`/`compile`, `getattr(obj, name)()`
  with non-constant `name`, calling a parameter as a function), and
  discarded calls to non-known-pure callees. Surfaced as a one-shot
  `CashImpurityWarning` per `(function, reason)`. Two opt-in modes:
  - `@cash.cache(strict=True)` — raises `CashImpureFunctionError`
    on first call if any issue is found. Also promotes opaque
    callees (no source) to issues. Use in CI to fail builds that
    introduce caching of side-effecting code.
  - `@cash.cache(assume_safe=True)` — silences the warning when
    you've audited the function and know caching is correct (e.g.
    a memoized API call where the side effect is idempotent). The
    analyzer still runs because helper source hashes feed the
    cache key.
  Mutually exclusive — passing both raises `ValueError` at
  decoration time.
- `cash.mark_pure(func)` / `cash.mark_stateful(func)` — module-level
  helpers to annotate third-party callables you've audited. Sets
  the existing `_cash_pure` / `_cash_stateful` attributes the
  analyzer respects. Returns *func* unmodified (no wrapping), so
  it's safe to call on C extensions and callable instances.
- `CashImpurityWarning` (subclass of `CashCacheIneffectiveWarning`)
  and `CashImpureFunctionError` (subclass of `CashError`) exported
  from the top-level `cash` package.
- **Latent-bug fix: helper-source-hash cache invalidation.**
  Previously, editing a plain helper called from a `@cash.cache`d
  function did NOT invalidate the cache — only edits to `@cash.cache`d
  callees were tracked. Now the same analyzer walk captures source
  hashes (and module-resolution paths) of every analyzed user-code
  helper. On every call, helpers are re-resolved from `sys.modules`
  and re-hashed, with current hashes folded into the cache key.
  This catches both cross-process edits (new run = new hash = new
  key) and in-process redefinitions (notebook cell rerun, REPL
  rebind, hot-reload). Per-call overhead is ~5-30μs for typical
  helper counts. The fallback to the recorded snapshot kicks in
  when re-resolution fails (helper deleted/renamed since analysis).
- Purity analyzer now recurses through **closure variables** in
  addition to `__globals__`. A `@cash.cache`d function defined
  inside another function (e.g. a factory pattern) gets its sibling
  helpers analyzed for impurity, scope mutations, and dynamic
  patterns. Closure helpers contribute to the cache-key state hash
  via the analysis-time snapshot (they have no stable
  `sys.modules` path for re-resolution, so per-call invalidation
  defers to the snapshot — which is the right behavior since
  closures are re-created fresh each time the enclosing function
  runs).

### Changed
- Ineffective-cache and store-failure events now emit
  `warnings.warn(...)` instead of `logger.warning(...)`, deduplicated
  per `(category, function, argument type)`. Users who relied on
  silent failure should add `warnings.filterwarnings("ignore",
  category=cash.CashWarning)` to their startup code.
- Three previously-silent failure paths now emit
  `CashCacheIneffectiveWarning` instead of a `logger.debug` /
  `logger.warning` line that nobody read: a `cache_if=` predicate that
  raises (was: silent skip), backend lock acquisition failure (was:
  proceeded unlocked with only a debug log), and a stored entry whose
  metadata fails validation (was: silently treated as miss). Same
  per-`(category, function, reason)` dedup as the existing warnings.
- `FileAccessTracker` now uses `contextvars.ContextVar` for active-
  tracker dispatch. Concurrent `asyncio.gather` and threaded callers
  are correctly isolated. No user-facing API change.
- `cache_if` interaction with iterator-returning functions: predicate
  is honored when the result fits in a single chunk. For multi-chunk
  results, `cache_if` is bypassed and a one-shot
  `CashCacheIneffectiveWarning` fires at the chunk_0 → chunk_1
  transition. To keep gating active on large iterators, lower
  `chunk_max_items` / `chunk_max_bytes` or materialize manually.

### Backward compatibility
- v1 iterator cache entries (written by 0.5.0b2 prior to chunked
  storage, with `metadata['materialized_iterator']=True`) continue
  to read correctly via a legacy code path. Old entries are
  eventually replaced by chunked entries on the next compute miss;
  no migration is required.
- The `CachedIterator` class has been renamed to `_ListCachedIterator`
  internally. The old name is kept as a deprecation-friendly alias
  in `cash.core` for one release and will be removed in 0.6.0.

### Not yet supported
- `@cash.cache` on `async def gen(): yield ...` (async generators)
  emits `CashCacheIneffectiveWarning` and returns the function
  unwrapped.
- `use_locking=True` combined with an async function emits
  `CashCacheIneffectiveWarning` and proceeds unlocked.

## [0.5.0b1] - Beta Release

### Added
- **Bug-report button** in the badge header with a budget-aware URL builder that auto-fills a GitHub issue with the failing cell, environment info, and the most recent metrics (without exceeding GitHub's URL length cap).
- **Per-iteration caching for upstream loop re-execution.** When upstream simulation has to re-run a loop, each iteration is now cached individually instead of treating the whole loop as one cache unit. Editing a loop body or extending the iterable only re-runs the affected iterations.
- **Forward-probe skip optimization in upstream simulation.** Before scheduling upstream cells to repair broken variables, Cash now probes the current cell to see whether its disk cache hits would restore the same variables. If so, the upstream re-execution is skipped entirely.
- **File-dependency path fallback.** When a project is moved (e.g. Google Drive path change, repo cloned to a new machine), absolute paths in cache metadata no longer cause full recomputation. `cash.utils.resolve_file_dep_path()` resolves stale paths via CWD-relative basename and suffix matches, and the resolver is wired into all cache-validation paths (`statement_processor`, `magics` restore, `upstream` checks).
- **`uncacheable_reasons` on metrics.** The badge and text-mode output now explain *why* a statement was not cached (`@cash:no-cache annotation`, `Input variable missing lineage`, ...).
- **Storage tier display in COMPUTED badges.** Each computed row now shows where the value landed (`RAM`, `RAM+DISK`, ...) with a hover explanation. Falls back to friendly labels (`- no outputs`, `- trivial`) when storage info is genuinely unavailable.
- **`%cash_help` magic** for a quick-reference command card.
- **`%cash_feedback` magic** that points at the issue tracker and discussions.
- **Welcome message on `%cash_on`** with actionable next steps.
- Expanded documentation with tutorials and use-case guides.

### Changed
- Version bumped to 0.5.0b1 for public beta release.
- Development status updated from Alpha to Beta.
- `[pandas]` / `[all]` / `[dev]` extras now require `pyarrow>=13.0` so DataFrame hashing produces stable, cross-platform results.
- `TieredBackend.set()` now propagates the resolved storage destinations back to the caller's metadata dict (previously the badge couldn't tell where a value landed).
- Status-based labels for upstream auto-execution loop groups in the badge.
- Upstream auto-exec loop groups render with full per-iteration detail in the badge.
- Cached simulation state is restored before the first changed cell regardless of whether the change was a code-hash mismatch or a stale file dependency (previously a stale file dep dropped *all* cached state).

### Fixed
- **Downstream overwrites of loop-produced variables** are now detected by upstream simulation; previously they could mask staleness.
- **Single-unit fallback for small loops with expensive iterations** is no longer triggered, so per-iteration caching stays effective.
- **Progress step lag** in the badge during long-running cells.
- **Bug report `RichOutput` repr** crash fixed.
- Internal `__iteration_context__` / `control_context` comments are stripped from the bug-report URL so reported code matches what the user wrote.
- Drop the inline `_cashBadgeExp` expand/collapse persistence script that caused state drift when cells were re-rendered mid-execution.
- **Windows console emoji crash.** `import cash; %load_ext cash; %cash_on` no longer raises `UnicodeEncodeError` from a vanilla `python.exe` shell on Windows (cp1252). A new `cash.utils.safe_text()` helper passes UTF-8 streams through unchanged and downgrades each emoji to a short ASCII fallback when the active stream cannot encode it (`✅` → `[OK]`, `⚙️` → `[run]`, ...). Jupyter kernels are UTF-8 so notebook users were unaffected.
- **`@cash:no-cache` annotation crash.** The fast path on a cached-skip annotation referenced `metrics` before it was defined, raising `UnboundLocalError` on first hit; the dict is now initialised before the append.
- **Decorator `execution_time` always 0 on Windows.** `@cash.cache` recorded per-call timings with `time.time()`, which has ~16 ms resolution on Windows and produced zero-duration entries that broke the call log; switched to `time.perf_counter()` (nanosecond resolution everywhere).
- **`%cash_benchmark --compare` dropped the `Speedup` line on coarse timers.** Same Windows resolution issue caused `mean_uncached` to round to 0; switched to `perf_counter`, and now print `Speedup: n/a (timings below timer resolution)` instead of silently omitting the line.
- **File-dep cache invalidation missed same-mtime rewrites.** On filesystems with coarse mtime granularity (HFS+/APFS, some ext4 configs) two back-to-back rewrites of the same file produce identical mtimes and the cache stayed valid. `file_dependencies` metadata now records both mtime and size, and all five validation paths check both. Existing on-disk caches load fine — they just lose the size check until they're re-written.

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
