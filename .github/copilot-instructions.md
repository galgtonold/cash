# Agent Instructions for Cash

> **Scope:** This file is the single source of truth for AI coding assistants working in this repo —
> GitHub Copilot, Claude Code (loaded via `CLAUDE.md` → this file), Cursor, Codex, etc.
> If you change conventions, change them here, not in the per-tool config.

## Project Overview
Cash is a smart caching library for Python with two primary use cases:
1. **Decorator-based caching** (`@cash.cache`) for functions with automatic dependency tracking
2. **Jupyter notebook caching** via IPython magics (`%cash_on`) with statement-level granularity

**Current Status:** Public Beta (v0.5.0b1). See [`.github/planning/INITIAL_RELEASE_ROADMAP.md`](planning/INITIAL_RELEASE_ROADMAP.md) for the active release plan and [`.github/planning/BETA_ROADMAP.md`](planning/BETA_ROADMAP.md) for the broader beta roadmap.

**Development Priority:** Focus on the active phase in `INITIAL_RELEASE_ROADMAP.md`. New features go into `BETA_ROADMAP.md` Phase 2+ unless they unblock the release.

## Commit messages
- **Never include `Co-Authored-By: Claude ...`** or any other AI-attribution trailer in commit messages. Author the commit normally.
- Use Conventional-Commits-ish prefixes (`feat:`, `fix:`, `test:`, `chore:`, `build:`, `docs:`, `refactor:`) plus an optional scope (`feat(badge): ...`).
- Subject ≤ 72 chars. Body wraps at 72. Lead with *why*, not *what*.
- One logical change per commit. If a diff touches three concerns, split it.

## Architecture

### Core Components (`src/cash/`)
- **`core.py`** - Main `Cash` class, entry point for decorator-based caching
- **`backends/`** - Pluggable storage backends (InMemoryBackend, FileBackend, Redis, S3, Tiered)
- **`notebook/`** - Jupyter integration (the most complex subsystem)

### Notebook Subsystem (`src/cash/notebook/`)
The notebook caching is statement-level, not cell-level. Key modules:
- **`magics.py`** - IPython magic commands (`%cash_on`, `%%cash`), orchestrates execution
- **`cache_key.py`** - **Unified cache key computation** (single source of truth for all cache key generation)
- **`statement_processor.py`** - Processes individual statements, handles cache lookup/store
- **`upstream.py`** - Detects and re-executes changed upstream cells, manages lineage simulation
- **`analysis.py`** - AST-based code analysis for inputs/outputs detection
- **`annotations.py`** - Parses `@cash:` comment directives (no-cache, ttl, persist, allow-random)
- **`file_tracker.py`** - Intercepts file reads (pandas, numpy, polars, open, joblib, etc.) for dependency tracking
- **`function_tracker.py`** - Tracks function source code changes, module hot reload
- **`control_structures.py`** - Per-iteration caching for loops and conditionals
- **`mutation_detector.py`** - AST-based detection of in-place mutations
- **`side_effects.py`** - Detection of file writes, network calls, and other side effects
- **`randomness.py`** - Unseeded random call detection and seed tracking
- **`purity.py`** - `@pure` and `@stateful` decorator system
- **`provenance.py`** - Variable computation history and dependency graphs
- **`audit.py`** - Compliance audit logging

### Key Data Flows
1. **Lineage Tracking**: Each variable gets a lineage hash = `hash(code + sorted(input_lineages) + file_deps)`
2. **Cache Keys**: `stmt:{hash(code + input_lineage_hashes)}` for statement-level caching
3. **Upstream Simulation**: Before running a cell, simulate all upstream cells to detect stale variables

## Development Patterns

### Testing Requirements
**Every feature must have both unit tests AND integration tests before completion. Same goes for every bug fix.**

#### Unit Tests (`tests/test_notebook/`)
- Use `magics_fixture` from `conftest.py` for mock IPython shell testing
- Test individual components in isolation (statement processor, upstream checker, etc.)
- Always use `tmp_path` for file-based test data to ensure isolation

```python
# Example unit test pattern
def test_feature_name(magics_fixture):
    magics, shell, backend = magics_fixture
    # Setup
    shell.user_ns['x'] = 10
    # Execute
    magics.cash("", "y = x * 2")
    # Assert
    assert shell.user_ns['y'] == 20
```

#### Integration Tests (`tests/test_notebook_integration/`)

**Two approaches available:**

##### 1. New API: `nb_runner` fixture (Recommended for new tests)
- Uses real notebook files that cash reads naturally (no mocking)
- Supports cell modification with proper change detection
- Supports selective cell execution with kernel persistence
- Reference notebooks in `tests/test_notebook_integration/reference_notebooks/`

```python
# Example using nb_runner with reference notebook
from tests.test_notebook_integration.conftest import REFERENCE_NOTEBOOKS_DIR

def test_example(nb_runner):
    nb_runner.load(REFERENCE_NOTEBOOKS_DIR / "financial_demo.ipynb")
    nb_runner.start_kernel()  # with_cash=True by default
    nb_runner.run_all()
    assert "AAPL" in nb_runner.get_output(3)
    
    # Modify a cell and re-run (cash detects the change!)
    nb_runner.set_cell_source(2, "x = 100  # modified")
    nb_runner.run_cells([2, 3])
    output = nb_runner.get_output(3)

# Example with programmatic notebook creation
def test_programmatic(nb_runner):
    nb_runner.create_notebook([
        "x = 10",
        "y = x * 2",
        "print(f'Result: {y}')"
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Result: 20" in nb_runner.get_output(3)
```

### Bug Reproduction Workflow
When debugging notebook-related bugs:
1. Create a reproduction test notebook
2. Use the built-in VS Code notebook tools (`edit_notebook_file`, `run_notebook_cell`, `read_notebook_cell_output`, `configure_python_notebook`) to execute cells and observe behavior. **DO NOT use the external Jupyter MCP server** — use the VS Code built-in notebook editing/execution tools only.
3. Once the issue is reproduced, write an integration test using `nb_runner` that programmatically verifies the fix.

### Interactive Notebook Testing Workflow
When performing interactive user-testing (cell-by-cell execution with edits):
- Use `edit_notebook_file` (with `editType: "insert"`) to add cells
- Use `edit_notebook_file` (with `editType: "edit"`) to modify cell source
- Use `edit_notebook_file` (with `editType: "delete"`) to remove cells
- Use `configure_python_notebook` to set up the Python kernel
- Use `run_notebook_cell` to execute individual cells
- Use `read_notebook_cell_output` to inspect cell results
- Use `copilot_getNotebookSummary` to get cell IDs and status
- **NEVER use open_browser_page or any MCP jupyter tools for notebook work**


### Pre-Completion Checklist
**Before reporting a feature or fix as complete, ALWAYS:**
1. Run ALL unit tests: `pytest tests/test_notebook/ -v --tb=short`
2. Run relevant integration tests by looking at file names and run them (max 10 most relevant ones). Never run all integration tests during development - it's too slow and makes it harder to iterate quickly.
3. Run the specific feature tests with `-s` for debug output
4. Verify no regressions in related test files


Use parallel test execution (`-n auto`) for integration tests, but be cautious of random failures.

### Test Isolation
**CRITICAL: Unit tests mock IPython in sys.modules, which breaks integration tests.**

Some unit tests (`test_statement_lineage.py`, `test_output_order.py`) mock IPython modules at import time. To prevent sys.modules pollution:

1. **Test ordering**: `conftest.py` has `pytest_collection_modifyitems` to run integration tests AFTER unit tests
2. **Module cleanup**: Auto-fixture `cleanup_sys_modules_between_tests` removes IPython mocks before integration tests
3. **Per-module cleanup**: Problematic test files have module-scoped fixtures to restore sys.modules state

**DO NOT run integration tests individually after running unit tests without restarting pytest** - the cleanup only happens during full test runs.

### Test Commands

#### Available Markers
| Marker | Description | Example Files |
|--------|-------------|---------------|
| `core` | Basic caching flow | `test_basic_flow.py`, `test_financial_demo_scenario.py` |
| `loops` | For/while loop caching | `test_control_structure_caching.py`, `test_loop_side_effects_integration.py` |
| `control` | If/else branch caching | `test_control_structure_caching.py`, `test_stress_batch6_ifelse_regression.py` |
| `upstream` | Upstream simulation/restore | `test_upstream_*.py`, `test_out_of_order_execution.py` |
| `files` | File dependency tracking | `test_file_invalidation_real.py`, `test_read_csv_caching.py`, `test_chdir_*.py` |
| `modules` | Module import/reload | `test_module_caching.py`, `test_module_reload_integration.py` |
| `mutations` | Mutation detection | `test_mutation_accumulator_init.py`, `test_accumulator_add_item.py` |
| `badges` | Badge display | `test_badge_integration.py` |
| `restore` | Disk restore after restart | `test_disk_restore_after_restart.py` |
| `libraries` | Real-world library integration | `test_real_world_libraries.py` |
| `stress` | Comprehensive regression | `test_stress_batch1-6.py` (1000+ tests, do not run) |

#### Timeout Configuration
All integration tests have a **30-second timeout** configured in `pyproject.toml`. If a test exceeds this, it fails with a timeout error. Individual tests can override:
```python
@pytest.mark.timeout(60)  # Allow 60 seconds for this specific test
def test_slow_operation(nb_runner):
    ...
```

### Test Fixtures

#### Unit Tests
- Use `magics_fixture` from `conftest.py` for unit tests with mock IPython shell
- Always use `tmp_path` for file-based test data to ensure isolation

#### Integration Tests
- **`nb_runner`** (NEW - recommended): Uses real notebook files, supports cell modification
  - `nb_runner.create_notebook([...])` - create notebook programmatically
  - `nb_runner.load(path)` - load existing .ipynb file
  - `nb_runner.start_kernel()` - start kernel (with_cash=True by default)
  - `nb_runner.run_all()` / `run_cell(n)` / `run_cells([n, m])` - execute cells (1-based indexing)
  - `nb_runner.get_output(n)` - get text output from cell n
  - `nb_runner.set_cell_source(n, code)` - modify cell (cash detects changes)
  - `nb_runner.reset_cash_state()` - clear cash's internal tracking state
  
- **`notebook_runner`** (legacy): Uses mocking, for backward compatibility only

#### Reference Notebooks
Store test notebooks in `tests/test_notebook_integration/reference_notebooks/` for reuse.

### Debugging
Enable debug output in notebooks:
```python
%cash_on
%cash_debug on
```
Debug output prefixes: `[UPSTREAM_DEBUG]`, `[LINEAGE_DEBUG]`, `[ALREADY_EXECUTED]`, `[TIMING_PROXY]`

## Critical Conventions

### Lineage System
- **`variable_lineage`**: Maps var_name â†’ lineage hash (full dependency chain)
- **`executed_cell_codes`**: Maps var_name â†’ code that last produced it
- **`executed_input_lineages`**: Maps var_name â†’ {input_var: lineage} used when computing
- **`_cash_hash`**: Attribute attached to objects with their lineage hash

### Cache Key Format
Statement cache keys: `stmt:{sha256(code + ':'.join(sorted(input_lineages)) + file_hash_component)}`

### âš ï¸ Unified Cache Key Computation (CRITICAL ARCHITECTURAL RULE)
**All cache key computation MUST go through `compute_cache_key()` in `cash.notebook.cache_key`.** This is the single source of truth for building statement cache keys. Never duplicate cache key logic in other modules.

**Why this matters:** Cache keys are computed in multiple contexts â€” runtime execution (`statement_processor.py`), upstream simulation (`upstream.py` `_update_virtual_lineage`), virtual restore (`upstream.py` `_try_virtual_restore`), and skip checks. Any divergence between these computations causes cache misses or stale data after kernel restarts. This has caused critical bugs multiple times in the past.

**Call sites that use `compute_cache_key()`:**
1. `_analyze_and_hash()` in `statement_processor.py` â€” runtime cache key
2. `_update_virtual_lineage()` in `upstream.py` â€” simulation forward propagation
3. `_try_virtual_restore()` in `upstream.py` â€” backward restore from disk
4. Skipped statement checking in `upstream.py` â€” verifying skipped stmts

**Input lineage priority order** (in `compute_cache_key()`):
1. `virtual_lineage` (simulation context â€” checked FIRST, reflects current simulated code)
2. `variable_lineage` (runtime state â€” may hold stale lineages from previous execution)
3. `_cash_hash` attribute on the object in `user_ns`
4. `compute_hash_fn` fallback (content-based hashing)

**Module lineage propagation:** When `_update_virtual_lineage()` processes import statements (`import X` / `from X import Y`), it copies module output lineages to `self.variable_lineage`. This ensures modules are available for downstream runtime cache key computation even after kernel restart (when imports are "skipped stmts" that never go through `process()`).

**When modifying cache key logic:** Change ONLY `compute_cache_key()` in `cache_key.py`. All call sites will automatically pick up the change. Add tests in `tests/test_notebook/test_virtual_restore_modules.py`.

### Skip Optimization Logic
Before executing a statement, check if it was already computed:
1. Code matches `executed_cell_codes[var]`
2. Output's `_cash_hash` matches stored lineage (not externally modified)
3. No file dependencies OR file deps haven't changed
4. Input lineages match `executed_input_lineages[var]`
5. Special case: self-assignment (`df = df.sort_values()`) - check output lineage, not input

### Metrics for Badge Display
When returning from `process()` in `statement_processor.py`, include:
- `status`: 'COMPUTED', 'RESTORED', or 'SKIPPED'
- `code`: The statement code
- `outputs`: List of output variable names
- `total_time`, `saved_time`, `storage`, etc.

## Performance Guidelines
- **Test execution time**: Always monitor test execution times. If tests take too long (>30s per test or >5min for a test suite), investigate and optimize
- **Integration test optimization**: Use programmatic notebook creation over file-based notebooks when possible
- **Kernel pooling**: DISABLED - caused hanging and zombie processes. Each test gets a fresh kernel
- **Quick feedback loop**: Run subset of tests during development, full suite before completion

## Common Pitfalls
- **File paths**: Always normalize paths (`path.replace('\\', '/')`) for cross-platform cache key consistency
- **Windows file locking**: Use retry loops when deleting temp directories in tests
- **nbclient execution order**: Cells run sequentially; can't skip cells to test specific orders
- **Duplicate cells**: If two cells have identical code, use cell IDs for disambiguation
- **Zombie processes**: After killing Python processes, may need to reconfigure environment

## Project Management
- **Roadmaps**: All planning docs live in `.github/planning/` (gitignored).
  - `INITIAL_RELEASE_ROADMAP.md` — active release plan
  - `BETA_ROADMAP.md` — broader beta roadmap, Phase 2+ features
  - `COMMUNITY_OUTREACH.md`, `VIDEO_SCRIPT.md` — launch materials
- **Before adding features**: Check the active roadmap. If it's not there and not a release blocker, propose it before building.
- **Breaking changes**: Document in `CHANGELOG.md` under the upcoming version and call them out in the PR description.
- **Version control**: Commit in logical chunks with clear messages (see *Commit messages* section above).

## Release Process

When the user asks to **cut a release** / **bump the version** / **prepare release X.Y.Z**:

### 1. Pick the version
- Beta: `0.5.0b1` → `0.5.0b2` → `0.5.0rc1` → `0.5.0`
- Bug-fix: `0.5.0` → `0.5.1`
- New feature, no breaks: `0.5.0` → `0.6.0`
- Breaking change: `0.5.0` → `1.0.0` (only after we've earned `1.0`)

### 2. Auto-generate the CHANGELOG entry from `git log`
**Do not write the changelog by hand.** Derive it from commits since the previous tag.

```bash
PREV=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
RANGE=${PREV:+$PREV..HEAD}
git log --no-merges --pretty=format:"%h %s" $RANGE
```

Then group the commits by their Conventional-Commits prefix into Keep-a-Changelog sections:

| Commit prefix | CHANGELOG section |
|---|---|
| `feat:` / `feat(...)`: | **Added** |
| `fix:` / `fix(...)`: | **Fixed** |
| `refactor:`, `perf:`, `build:`, `chore:` (with user-visible impact) | **Changed** |
| `BREAKING CHANGE:` in body, or `!` in prefix (`feat!:`) | **Breaking** (top of section) |
| `test:`, `ci:`, `chore:` (no user-visible impact) | **Omit** unless they materially change behaviour |
| `docs:` | Mention only if user-facing docs changed |

Rewrite each line so it reads as a user-facing change, not a commit subject. Strip the prefix, expand abbreviations, write in past tense or imperative as is consistent with the rest of the file. Add a *why* clause when the commit subject doesn't explain it.

Insert the new section at the top of `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD`. Keep the prior `[0.5.0b1]`, `[0.3.0]`, ... sections intact.

### 3. Bump version in `pyproject.toml`
Edit the single `version = "..."` line. No other files need to change (we don't store the version in source).

### 4. Verify
- `pytest tests/test_notebook -x --timeout=30` (unit suite green)
- `python -m build` (sdist + wheel build cleanly; inspect contents)
- `pip install dist/cash_lib-X.Y.Z*.whl` in a fresh venv → `python -c "import cash; print(cash.__version__)"`

### 5. Commit & tag
```bash
git add pyproject.toml CHANGELOG.md
git commit -m "release: X.Y.Z"
git tag vX.Y.Z
```

Push only after the user explicitly confirms — pushing the tag triggers the PyPI publish workflow.

### 6. Verify post-publish
- `pypi.org/project/cash-lib/X.Y.Z/` is live with correct metadata
- `pip install cash-lib==X.Y.Z` resolves on a clean machine
