# Agent Instructions for Cash

> **Scope:** This file is the single source of truth for AI coding assistants working in this repo —
> GitHub Copilot, Claude Code (loaded via `CLAUDE.md` → this file), Cursor, Codex, etc.
> If you change conventions, change them here, not in the per-tool config.

## Project Overview
Cash is a smart caching library for Python with two primary use cases:
1. **Decorator-based caching** (`@cash.cache`) for functions with automatic dependency tracking
2. **Jupyter notebook caching** via IPython magics (`%cash_on`) with statement-level granularity

**Current Status:** Preparing the first public release, `v0.1.0` (nothing has been published to PyPI yet). The single source of truth for all planning is the **Linear** workspace (team `Cash`, issue prefix `CAS`) — see the *Project Management* section below.

**Development Priority:** Work the `v0.5.0 Release` project in Linear (milestones: *Beta hardening*, *Launch readiness*). New features default to the `Post-Beta (0.6+)` project unless they unblock the release.

## Commit messages
- **Never include `Co-Authored-By: Claude ...`** or any other AI-attribution trailer in commit messages. Author the commit normally.
- Use Conventional-Commits-ish prefixes (`feat:`, `fix:`, `test:`, `chore:`, `build:`, `docs:`, `refactor:`) plus an optional scope (`feat(badge): ...`).
- Subject ≤ 72 chars. Body wraps at 72. Lead with *why*, not *what*.
- One logical change per commit. If a diff touches three concerns, split it.

## Architecture

### Core Components (`src/cash/`)
- **`core.py`** - Main `Cash` class, entry point for decorator-based caching
- **`backends/`** - Pluggable storage backends: `InMemoryBackend`, `FileBackend`, `SQLiteBackend`, `RedisBackend`, `S3Backend`, plus two multi-backend composites, `TieredBackend` (promotion + read-repair) and `CascadingBackend`. `factory.py` maps a `TierConfig.type` string (`memory` / `file` / `sqlite` / `redis` / `s3` / `tiered`) to the class.
- **`notebook/`** - Jupyter integration (the most complex subsystem)

### Notebook Subsystem (`src/cash/notebook/`)
The notebook caching is statement-level, not cell-level. The four biggest clusters are **packages**, not modules — see the ADR referenced on each.

**Packages:**
- **`ipython/`** - The IPython adapter (ADR-013). `magics.py` holds `CashMagics` (`%cash_on`, `%%cash`); `admin.py` the admin magics (`%cash_status`, `%cash_clear`, ...); `cell_executor.py` the `CellExecutor` that both entry points delegate to. Public surface: `CashMagics`.
- **`statement/`** - `StatementProcessor` and its four siblings (ADR-011): `freshness.py` (`CacheFreshnessChecker`), `file_deps.py` (`StatementFileDeps`), `lineage.py` (`StatementLineageBuilder`), `restore.py` (`StatementRestorer`), plus `derivation_edges.py` (numpy-view / live-reference lineage bumps, CAS-115/89). Public surface: `StatementProcessor`, `ProcessResult`.
- **`upstream/`** - Upstream detection + lineage simulation (ADR-010): `checker.py` (`UpstreamChecker`), `simulator.py` (`NotebookSimulator`), `virtual_lineage.py`, `mismatch_classifier.py`, `reexecution_planner.py`. Public surface: `UpstreamChecker`, `UpstreamResult`, `NotebookSimulator`.
- **`control_structures/`** - Per-iteration caching for loops and conditionals (ADR-012): `processor.py` orchestrates, `for_handler.py` / `if_handler.py` / `try_handler.py` are the strategies, `helpers.py` the shared lineage/badge/error helpers.
- **`badge_renderer/`** - `BadgeView` IR (`view.py`, `view_builder.py`) + the renderers under `renderers/` (HTML v3, text) and `theme.py`.

**Modules:**
- **`cache_key.py`** - **Unified cache key computation** (single source of truth for all cache key generation)
- **`analysis.py`** - AST-based code analysis for inputs/outputs detection
- **`cacheability.py`** - Pure-AST cacheability analysis. **Folds the former `mutation_detector.py` (in-place mutations) and `side_effects.py` (file writes, network calls) into one module** — both names are gone.
- **`cacheability_decision.py`** - The runtime merge (`decide_cacheability`): AST analysis + annotations + `@stateful` + forbidden-function scan → `(cacheable, reasons)`
- **`annotations.py`** - Parses `@cash:` comment directives. Six of them: `persist`, `no-cache`, `allow-random`, `cache-fit`, `cache-calls`, and `ttl=N`. Each hyphenated name also accepts a run-together alias (`nocache`, `allowrandom`, `cachefit`, `cachecalls`). An unknown directive is silently dropped — no warning, no log line.
- **`lineage_store.py`** - `LineageStore`: the single seam for reading/writing variable lineage; owns the resolution priority ladder
- **`restore.py`** - `Restorer`: **variable**-granular cache restoration (distinct from `statement/restore.py`, which is statement-granular)
- **`consumables.py`** - Classification + divergence probing for consumable, unrestorable inputs (generators, file handles)
- **`file_tracker.py`** - Intercepts file reads (pandas, numpy, polars, open, joblib, etc.) for dependency tracking
- **`file_dep_snapshot.py`** - Pure helpers for file-dep snapshots (`{path: {mtime, size, hash}}`) and the **content-authoritative** freshness check shared by the decorator and notebook paths (CAS-98/CAS-10/CAS-119)
- **`function_tracker.py`** - Tracks function source code changes, module hot reload
- **`module_invalidator.py`** - Invalidates caches downstream of a changed local module
- **`object_hashing.py`** - Pure `compute_hash` / sizing helpers for arbitrary objects
- **`randomness.py`** - Unseeded random call detection and seed tracking
- **`purity.py`** - `@pure` and `@stateful` decorator system
- **`provenance.py`** - Variable computation history and dependency graphs
- **`cache_status.py`** - Cache status enum + execution result types
- **`cost_model.py`** - Tuned serialise/deserialise cost model behind the cache-or-not decision
- **`server_discovery.py`** - Jupyter Server integration: notebook path discovery and cell reading
- **`audit.py`** - Compliance audit logging
- **`_protocols.py`** - `TrackingState` + the subsystem's Protocol types
- **`_trace.py`** - Opt-in decision tracing for the upstream checker/simulator

### JupyterLab extension (`labextension/`)

The one part of this repo that is not Python. It pushes the notebook's live
(unsaved) cell sources over the `cash_live_cells` comm that
`notebook/live_cells.py` receives, which is the only way cash sees an edit that
has not been written to the `.ipynb` yet.

**Node is never on your critical path.** The built bundle is committed under
`src/cash/labextension/`, so `pip install -e .`, `pytest`, and building the wheel
all work with no JavaScript toolchain installed. You need Node only if you change
`labextension/src/index.ts` — then `cd labextension && npm install && npm run
build`, and commit the regenerated `src/cash/labextension/`.

`comm.commsOverSubshells = 'disabled'` in that file is **load-bearing** and is
guarded twice (a build-time script and
`tests/test_notebook/test_labextension_packaging.py`). Read
`labextension/README.md` before touching it.

### Key Data Flows
1. **Lineage Tracking**: Each variable gets a lineage hash = `hash(code + sorted(input_lineages) + file_deps)`
2. **Cache Keys**: `stmt:{hash(code + input_lineage_hashes)}` for statement-level caching
3. **Upstream Simulation**: Before running a cell, simulate all upstream cells to detect stale variables

## Development Patterns

### Testing Requirements
**Every feature must have both unit tests AND integration tests before completion. Same goes for every bug fix.**

#### Unit Tests (`tests/test_notebook/`)
- Use a `magics_fixture` for mock IPython shell testing. It is **not** a shared conftest fixture — each test module defines its own (27 files under `tests/test_notebook/` do). Copy the pattern from a neighbouring test file rather than importing it.
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

Use the `nb_runner` fixture:
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

#### `get_output` vs `peek` — pick by what you are claiming

Assert on **`get_output(cell)`** when the claim is about **what the user sees**.
Assert on **`nb_runner.peek(expr)`** when the claim is about **kernel state**.
Both are legitimate; conflating them is the bug.

`get_output` returns text captured when that cell last ran, and a cached
statement's stdout is *replayed on a hit* — so a printed value can report what
was on screen when the entry was written rather than what the variable holds
now. Measured during CAS-260: a printed reading made a broken arm look correct
and sent a round of that investigation down a false trail; the out-of-band read
reversed the conclusion.

```python
assert "Result: 20" in nb_runner.get_output(3)   # the user sees this
assert nb_runner.peek("y") == "20"               # the kernel holds this
assert nb_runner.peek("len(rows)") == "3"        # any expression works
```

`peek` runs outside the notebook's cells with `store_history=False`, so nothing
about it is cached, replayed, or added to the notebook. A bare name is wrapped
as `globals().get(name)`, so an undefined one reads as `"None"` rather than
raising into silence.

**Do not instrument a cached callee with a counter it writes itself** (a global
list it appends to). Such a write is captured and restored on a hit, so the
counter reads the same whether the call ran or was served, *and* it costs the
call its reuse. Count executions with `os.open`/`os.write` — not
`builtins.open`, which `FileAccessTracker` patches into a file dependency,
changing the entry every run and silently disabling what you are measuring.

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


**Parallel by default.** `pyproject.toml` sets `addopts = "-n 16 --dist loadscope"`, so every `pytest` run is parallel (pytest-xdist ships in the `dev` extra). Integration tests spend ~90% of wall-clock blocked on per-test Jupyter kernel boot/IPC, so a serial run pins one core and leaves the rest idle. The worker count is pinned (not `-n auto`) because the suite is boot-throttle-bound, not CPU-bound: a sweep found ~2× the boot throttle (16 ≈ 2×`CASH_TEST_BOOT_THROTTLE`) both fastest and most stable, while oversubscribing only queues kernel boots and destabilizes the run. `loadscope` keeps each module/class on a single worker, preserving the per-file ordering unit tests rely on. For interactive debugging (pdb, `-s`, single-stepping) disable parallelism with `-n0`.

### Test Isolation
Unit tests use **real IPython** with a `MockShell(Configurable)` (see `conftest.py` and `tests/test_notebook/`), not a `sys.modules['IPython'] = MagicMock()` mock. Do not reintroduce module-level IPython mocking — it pollutes `sys.modules` for whatever module xdist schedules next on the same worker and resurfaces the cross-test contamination this suite was cleaned up to avoid.

**Process-global IPython singleton.** `InteractiveShell.instance()` registers a process-wide singleton, so `get_ipython()` returns a live shell for the rest of the worker even after the test that created it finishes. Any test that calls `.instance()` (the overhead-benchmark drivers, the docs harness) MUST clear it in teardown via `InteractiveShell.clear_instance()` — otherwise downstream tests that assume no active shell break (e.g. `reset_session()` re-creates the global Cash; `Cash()` auto-registers magics). `tests/test_benchmarks_overhead/conftest.py` and `tests/docs/_harness.py` already do this.

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
- Define a `magics_fixture` in the test module itself (see the note under *Unit Tests* above — it is per-file, not shared)
- Always use `tmp_path` for file-based test data to ensure isolation

#### Integration Tests
- **`nb_runner`**: Uses real notebook files, supports cell modification
  - `nb_runner.create_notebook([...])` - create notebook programmatically
  - `nb_runner.load(path)` - load existing .ipynb file
  - `nb_runner.start_kernel()` - start kernel (with_cash=True by default)
  - `nb_runner.run_all()` / `run_cell(n)` / `run_cells([n, m])` - execute cells (1-based indexing)
  - `nb_runner.get_output(n)` - get text output from cell n
  - `nb_runner.set_cell_source(n, code)` - modify cell (cash detects changes)
  - `nb_runner.reset_cash_state()` - clear cash's internal tracking state

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
- **`variable_lineage`**: Maps var_name → lineage hash (full dependency chain)
- **`executed_cell_codes`**: Maps var_name → code that last produced it
- **`executed_input_lineages`**: Maps var_name → {input_var: lineage} used when computing
- **`_cash_lineage_hash`**: Attribute attached to objects with their lineage hash. (There is no `_cash_hash` — the shorter name appears only inside a test name.)

### Cache Key Format
Statement cache keys: `stmt:{sha256(code + ':'.join(sorted(input_lineages)) + file_hash_component)}`

### ⚠️ Unified Cache Key Computation (CRITICAL ARCHITECTURAL RULE)
**All cache key computation MUST go through `compute_cache_key()` in `cash.notebook.cache_key`.** This is the single source of truth for building statement cache keys. Never duplicate cache key logic in other modules.

**Why this matters:** Cache keys are computed in multiple contexts — runtime execution (`statement/processor.py`), upstream simulation (`upstream/virtual_lineage.py` `_update_virtual_lineage`), virtual restore (`upstream/virtual_lineage.py` `_try_virtual_restore`), and skip checks. Any divergence between these computations causes cache misses or stale data after kernel restarts. This has caused critical bugs multiple times in the past.

**Call sites that use `compute_cache_key()`:**
1. `_analyze_and_hash()` in `statement/processor.py` — runtime cache key
2. `_update_virtual_lineage()` in `upstream/virtual_lineage.py` — simulation forward propagation
3. `_try_virtual_restore()` in `upstream/virtual_lineage.py` — backward restore from disk
4. Skipped statement checking in `upstream/virtual_lineage.py` — verifying skipped stmts

**Input lineage priority order** (in `compute_cache_key()`):
1. `virtual_lineage` (simulation context — checked FIRST, reflects current simulated code)
2. `variable_lineage` (runtime state — may hold stale lineages from previous execution)
3. `_cash_lineage_hash` attribute on the object in `user_ns`
4. `compute_hash_fn` fallback (content-based hashing)

**Module lineage propagation:** When `_update_virtual_lineage()` processes import statements (`import X` / `from X import Y`), it copies module output lineages to `self.variable_lineage`. This ensures modules are available for downstream runtime cache key computation even after kernel restart (when imports are "skipped stmts" that never go through `process()`).

**When modifying cache key logic:** Change ONLY `compute_cache_key()` in `cache_key.py`. All call sites will automatically pick up the change. Add tests in `tests/test_notebook/test_virtual_restore_modules.py`.

### Skip Optimization Logic
Before executing a statement, check if it was already computed:
1. Code matches `executed_cell_codes[var]`
2. Output's `_cash_lineage_hash` matches stored lineage (not externally modified)
3. No file dependencies OR file deps haven't changed
4. Input lineages match `executed_input_lineages[var]`
5. Special case: self-assignment (`df = df.sort_values()`) - check output lineage, not input

### Metrics for Badge Display
When returning from `process_statement()` in `statement/processor.py`, include:
- `status`: 'COMPUTED', 'RESTORED', or 'SKIPPED'
- `code`: The statement code
- `outputs`: List of output variable names
- `total_time`, `saved_time`, `storage`, etc.

## Performance Guidelines
- **Test execution time**: Always monitor test execution times. If tests take too long (>30s per test or >5min for a test suite), investigate and optimize
- **Integration test optimization**: Use programmatic notebook creation over file-based notebooks when possible
- **Kernel pooling**: DISABLED - caused hanging and zombie processes. Each test gets a fresh kernel
- **Quick feedback loop**: Run subset of tests during development, full suite before completion

## Prove a new test can fail

A test written alongside a fix must be shown to fail **without** it:

```bash
python scripts/fails_first.py tests/test_core/test_my_new_guard.py
```

It stashes `src/`, runs the tests, restores, and exits non-zero if they all
passed — i.e. if they never exercised the fix. This is not ceremony; vacuously
green tests have shipped here repeatedly, in four recurring shapes:

1. **The mechanism never engages.** Cross-process persistence has a ~0.1 s
   compute floor, so a cached function cheaper than that is never written to
   disk — a staleness test over two processes then passes whether or not the
   bug exists. Use `tests.conftest.ABOVE_PERSISTENCE_FLOOR_S` for the sleep, and
   assert the body ran exactly once across runs so you *know* it cached.
2. **Empty input trivially satisfies the assertion.** "Is this output
   encodable / valid / clean?" is true of an empty string, so a harness that
   silently executed nothing looks green. Assert the input is non-empty first.
3. **A different gate is substituted for the real one.** `mkdocs build
   --strict` checks links and nav and never executes a python fence; only
   `pytest tests/docs/` does. Passing one says nothing about the other.
4. **State is checked instead of behaviour.** Asserting a policy object exists
   passes even when nothing calls it. Drive the behaviour across the boundary
   you care about.

For an exclusion or filter, add a **positive control** in the same test (assert
the thing that must survive is still there), or the assertion passes when
everything is excluded.

## Writing documentation
- **Never cite a line number.** `` `core.py:1234` `` is banned in any published
  page and fails `tests/docs/test_doc_claims.py`. Name the **symbol**
  (`Cash._compute_with_lock`, `MUTATING_METHODS`) — it moves with the code. The
  ban started as a ratchet over 22 grandfathered pins; 20 of them had already
  rotted onto unrelated code. The one exempt form appends the commit the line
  was read at (`` `src/cash/core.py:1234@8e5f4ce` ``), which names a fixed
  snapshot and so cannot rot — use it only for claims genuinely *about* history.
- **Anchor every mechanism claim** to the source that decides it:
  `<!-- claim: cash/core.py:Cash.cache @? -->`, then `python scripts/claims.py
  --pin` fills the digest. Prefer a **value** anchor (`== 0.01`) whenever the
  prose quotes a constant — it verifies itself forever with no human in the loop.
- **Before changing code, run `python scripts/claims.py --report <src file>`** to
  see which doc claims rest on it.
- Full authoring guide: `tests/docs/README.md`.

## Common Pitfalls
- **File paths**: Always normalize paths (`path.replace('\\', '/')`) for cross-platform cache key consistency
- **Windows file locking**: Use retry loops when deleting temp directories in tests
- **nbclient execution order**: Cells run sequentially; can't skip cells to test specific orders
- **Duplicate cells**: If two cells have identical code, use cell IDs for disambiguation
- **Zombie processes**: After killing Python processes, may need to reconfigure environment

## Project Management

**Linear is the single source of truth.** All tasks, bugs, follow-ups, and roadmap live in the Linear workspace (team `Cash`, issue prefix `CAS`), accessed via the Linear MCP. Do **not** create or resurrect roadmap markdown. The old `planning/ROADMAP.md` no longer exists in the working tree at all (neither `planning/` nor `planning/archive/` is present) — if you need that history, read it out of git, and do not recreate the directory.

**Behavioral rules for AI assistants (do these without being asked):**
1. **Look in Linear by default.** At the start of planning work, query Linear for the active `v0.5.0 Release` project plus In Progress / Todo issues before proposing what to do. Don't ask the user to paste issue state — fetch it.
2. **Auto-create on flag.** The moment you notice deferred work, an out-of-scope fix, a real bug, or a "do this later" — create a Linear issue immediately (don't let it die in chat). Default: status Backlog, the right workstream label, a best-guess priority. Better an imperfectly-triaged issue than a lost one.
3. **Cross-check on reference.** When the user mentions an issue — by id (`CAS-123`) or by description — look it up in Linear and work from its current state/comments, not from memory.
4. **Reflect progress.** Move an issue to In Progress when you start it and Done when it's verified. Reference the `CAS-id` in the commit subject/body where applicable so commits ↔ issues cross-link.

**Structure:**
- **Projects** = finite, goal-bearing efforts: `v0.5.0 Release`, `Post-Beta (0.6+)`.
- **Labels** = the five perennial workstreams + helpers: `correctness`, `cache-perf`, `release`, `Bug`, `Improvement` (the five), plus `tech-debt`, `xfail`, `docs`, `Feature`.
- **`xfail` label** is load-bearing: an issue with it maps to a `pytest.mark.xfail` in the suite; closing the issue means flipping that marker to a passing test. Some `xfail` issues are *documented limitations* (e.g. CAS-9) — read the body before "fixing".

**Other:**
- **Breaking changes**: Document in `CHANGELOG.md` under the upcoming version and call them out in the PR description.
- **Version control**: Commit in logical chunks with clear messages (see *Commit messages* section above).

## Release Process

When the user asks to **cut a release** / **bump the version** / **prepare release X.Y.Z**:

### 1. Pick the version
The first public release is `0.1.0`. Development ran through internally-numbered
versions up to `0.5.0b2`, none of which were ever published; versioning restarted
at `0.1.0` so the public series begins where users actually join it. Do **not**
resurrect the old numbers — `0.2.0`, `0.3.0` and `0.5.0*` are historical only and
are recorded under *Pre-release development history* in `CHANGELOG.md`.

- Bug-fix: `0.1.0` → `0.1.1`
- New feature, no breaks: `0.1.0` → `0.2.0`
- Breaking change: while `0.x`, a break goes in a minor bump (`0.1.0` → `0.2.0`)
  and MUST be called out under **Breaking**; the API is not yet stable
- `1.0.0` only once the API is one we're willing to freeze

Prefer plain final versions over pre-release suffixes. `pip install cash-lib`
ignores pre-releases unless the user passes `--pre`, so a `bN`/`rcN` release is
invisible to most people — that is a deliberate choice, not a default.

### 1b. Clear the doc-claim queue

Every claim in the docs is anchored to the source that decides it. Between
releases, fingerprint drift is reported but not enforced — this is where it is
enforced.

```bash
python scripts/claims.py --queue
```

It must print `No drifted claims.` For each entry it does print, re-read the
claim against the code shown by `--accept <page>`, then either fix the prose or
re-pin with `--accept <page> --yes`.

Do this **before** the CHANGELOG: a claim found wrong is often a **Fixed**
entry in its own right.

The `build` job in `.github/workflows/publish.yml` — the workflow every
release actually runs — sets `CASH_CLAIMS_STRICT=1` and re-runs
`tests/docs/test_claim_anchors.py::test_no_fingerprint_drift`, promoting
drift from advisory to blocking before the package is built. That is the real
gate; running `--queue` above by hand is what keeps you from finding out
about a non-empty queue only when the build fails.

### 2. Write the CHANGELOG entry FROM the `git log` (curate, don't transcribe)

The `git log` is the **source of truth, not the output**. Read the whole range,
then write release notes a user would actually want — do **not** mechanically
emit one line per commit.

```bash
PREV=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
RANGE=${PREV:+$PREV..HEAD}
git log --no-merges --pretty=format:"%h %s" $RANGE
```

- **Ground every entry in the log.** If a commit's subject doesn't make the
  user-facing effect clear, read its diff. Never describe a change that isn't in
  the range and never invent one — anchoring to the log is the whole point, and
  it is what keeps the notes honest. (This is what "don't write it by hand" always
  meant: don't write from memory — write from the log.)
- **Synthesize, don't transcribe.** A single change that landed as five commits
  (implementation + three follow-up fixes + a test) is **one** entry. Group
  related work by theme, not by commit boundary.
- **Omit internal churn.** Test-only, CI, refactor and build-plumbing commits
  with no user-visible effect don't belong in the notes.
- **Lead with impact.** Say what changed for the user and *why* it matters, in
  plain language — not the commit subject.

Sort the surviving entries into Keep-a-Changelog sections. Use the commit
prefixes as a **hint**, not a rule:

| Commit prefix | CHANGELOG section |
|---|---|
| `feat:` / `feat(...)`: | **Added** |
| `fix:` / `fix(...)`: | **Fixed** |
| `refactor:`, `perf:`, `build:`, `chore:` (with user-visible impact) | **Changed** |
| `BREAKING CHANGE:` in body, or `!` in prefix (`feat!:`) | **Breaking** (top of section) |
| `test:`, `ci:`, `chore:` (no user-visible impact) | **Omit** unless they materially change behaviour |
| `docs:` | Mention only if user-facing docs changed |

Insert the new section at the top of `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD`, above `## [0.1.0]`. Leave the *Pre-release development history* block and everything under it untouched — those entries are a frozen record, not a running log. **The user reviews the entry before it's committed.**

### 3. Bump the version
Edit the single `__version__ = "..."` line in `src/cash/__init__.py`. That is the **single source of truth** — `pyproject.toml` declares `dynamic = ["version"]` and hatchling reads it from there at build time (`[tool.hatch.version]`), so the wheel metadata, `cash.__version__`, and `cash version` can never disagree. Do **not** add a `version =` line back to `pyproject.toml`.

### 4. Build & verify — **always into an empty `dist/`**

`python -m build` **adds** to `dist/`; it never clears it. Left alone, `dist/`
accumulates every build you have ever made, and old versions sit there looking
exactly like fresh ones. Clear it *first*, every time:

```bash
rm -rf dist/            # MANDATORY — never build on top of an existing dist/
python -m build
ls dist/                # MUST list exactly two files, both X.Y.Z: the wheel + the sdist
```

If `ls dist/` shows any version other than the one you are releasing, stop and
clean it out before going near an upload — do not "just skip" the extra files.

- `pytest tests/test_notebook -x --timeout=30` (unit suite green)
- `twine check dist/*`
- `pip install dist/cash_lib-X.Y.Z-py3-none-any.whl` in a fresh venv →
  `python -c "import cash; print(cash.__version__)"` prints `X.Y.Z`
- In that same fresh venv, `pip install "jupyterlab>=4,<5"` then
  `jupyter labextension list` → must show `cash-live-cells vN.N.N enabled ok
  (python, cash-lib)`. A wheel that installs fine but registers no extension is
  the failure mode here, and nothing short of this command catches it: the
  bundle is committed build output, so it can be stale or absent while every
  Python check stays green.

### 5. Commit & tag

The version's single source of truth is `src/cash/__init__.py` (`__version__`);
`pyproject.toml` reads it via `dynamic = ["version"]` and must NOT be edited.

```bash
git add src/cash/__init__.py CHANGELOG.md
git commit -m "release: X.Y.Z"
git tag vX.Y.Z
```

Push only after the user explicitly confirms.

**Pushing the tag does NOT publish.** `.github/workflows/publish.yml` triggers on
`release: types: [published]` (or a manual `workflow_dispatch`), not on a tag
push — so a tag alone leaves you with no release and nothing on PyPI. Publishing
requires creating a GitHub Release from the tag, which is the deliberate
irreversible-action gate: see step 6.

### 6. Publish

**The normal path is the `Publish to PyPI` workflow** (`.github/workflows/publish.yml`),
which builds from a fresh checkout and therefore *cannot* pick up local strays.
Prefer it. Publish by hand only if that workflow is broken.

If you must publish by hand: **never `twine upload dist/*`.** The glob uploads
whatever happens to be in the directory. Name the two files explicitly, with the
version in the filename, so the command can only ever publish what you intend:

```bash
twine upload dist/cash_lib-X.Y.Z-py3-none-any.whl dist/cash_lib-X.Y.Z.tar.gz
```

**Why this is not negotiable:** a PyPI upload is irreversible. A version can never
be re-uploaded, even after you delete it — the name is burned forever. A `dist/*`
glob over a stale directory publishes a real, wrong release. This is not
hypothetical: this repo's `dist/` sat for months holding `cash_lib-0.2.0` **and** a
`0.5.0b1` built from an older tree (258 KB vs the real 490 KB), so `twine upload
dist/*` would have shipped 0.2.0 — and a `0.5.0b1` that was not the code we
believed it was. `rm -rf dist/` before the build plus an explicit versioned
filename at upload defeats both, independently.

### 7. Verify post-publish
- `pypi.org/project/cash-lib/X.Y.Z/` is live with correct metadata
- `pip install cash-lib==X.Y.Z` resolves on a clean machine
