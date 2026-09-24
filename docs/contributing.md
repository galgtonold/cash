# Contributing to Cash

!!! info "Applies to: both paths"
    Contributors: setting up, finding your way around the code, and getting a change through CI.

## Setup

You need Python 3.10 or later and Git.

```bash
git clone https://github.com/galgtonold/cash.git
cd cash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,docs-test]"
pip install ruff==0.15.8 pre-commit   # the ruff version CI uses
pre-commit install                     # runs ruff on each commit
```

`docs-test` holds the packages the documentation's examples import. Without it,
`pytest tests/docs` fails on real imports.

## The code

Cash has two engines: the decorator (`@cash.cache`) and the notebook engine
(`%cash_on`). They share the hashing, effect rules, file tracking and storage,
but each builds its own keys. [How Cash works](how-it-works/overview.md)
describes both.

```
src/cash/
├── __init__.py         # the public API (cash.__all__) and the default Cash instance
├── __main__.py         # the `cash` command line
├── core.py             # the Cash class: registries, configuration, the cache decorator front
├── decorator/          # what a cached call does: its key (code, globals,
│                       #   closures, arguments, seed, files), the call itself,
│                       #   explain() and the warnings it raises
├── notebook/           # the notebook engine (only the magics loaders import it)
│   ├── ipython/        #   the magics and the cell executor
│   ├── statement/      #   statement-level caching
│   ├── control_structures/  # per-iteration loop and branch caching
│   ├── upstream/       #   upstream simulation and restore
│   ├── badge_renderer/ #   the HTML and text badge
│   └── *.py            #   the statement cache key, lineage store, call units,
│                       #   provenance, live cell sources, ...
├── analysis/           # static analysis: statement inputs and outputs,
│                       #   # @cash: annotations, mutations, cacheability,
│                       #   the purity analysis of a decorated function
├── tracking/           # what a computation reads at run time: files,
│                       #   function source, modules, randomness
├── backends/           # storage: memory, file, tiered, SQLite, Redis, S3;
│                       #   serialization, the entry format, eviction, size caps
├── ui/                 # the dashboard and the cache explorer
├── labextension/       # the prebuilt JupyterLab extension (source in labextension/ at the repo root)
├── config.py           # CashConfig and how settings are resolved
├── effects.py          # which calls write files, send requests, read the clock or environment
├── effect_observer.py  # side effects a cached function performs on its first call
├── purity.py           # @pure, @stateful and the known-pure registry
├── dependency_state.py # the state hash: own source, dependencies, helpers
├── object_hashing.py   # content hashes and sizes of values
├── source_norm.py      # normalises source before hashing (comments, blank lines)
├── cost_model.py       # predicted serialise and restore time per type and backend
├── effectiveness.py    # notices when caching costs more than it saves
├── data_source.py      # the DataSource protocol
├── file_source.py      # FileDataSource for a local file
├── remote_source.py    # RemoteFileDataSource for s3, gs, az and http
├── diagnostics.py      # the stable code of every warning
├── exceptions.py       # public exceptions and warnings
├── analytics.py        # cache-usage analytics
├── graph.py            # the function dependency graph (Cash.graph)
├── nbconvert.py        # nbconvert preprocessor that strips badges and magics
├── reconfigure.py      # cash.configure() on a running instance
├── _agent_guide.py     # the text cash.help() returns; identical to docs/for-coding-agents.md
└── _*.py, *.py         # small helpers: logging, console output, paths,
                        #   where the project and cache are, clocks, type sets
```

The JupyterLab extension sends unsaved cell sources to the kernel, so the
upstream check can see edits not yet saved. Node is needed only to rebuild it
after changing `labextension/src/index.ts`; the build output is committed.

## Before you push

CI runs these on every push and pull request; run the ones your change touches.

```bash
ruff check .
ruff format --check .

# Unit tests (the default is 16 xdist workers; add -n 0 -s to debug)
pytest tests/ --ignore=tests/test_notebook_integration --ignore=tests/test_wheel_gate --ignore=tests/docs -m "not perf"

# Integration tests: the files named after what you changed, then the core set
pytest tests/test_notebook_integration/loops/ -v
pytest @tools/test_selection/core_set.txt

# Docs
pytest tests/docs
```

The integration core set is the few hundred integration tests that together
cover every line, feature and step sequence of the whole suite; the whole
suite runs nightly. Don't run the whole suite while iterating. To re-pick the
core set after large changes, see `tools/test_selection/`.

**Show that a new test fails without your fix:**

```bash
python scripts/fails_first.py tests/test_core/test_your_change.py
```

It stashes your changes under `src/`, runs the tests, and fails if they pass
anyway. A test can pass vacuously when the mechanism never engages (a cached
function faster than the persistence floor never reaches disk; sleep
`tests.conftest.ABOVE_PERSISTENCE_FLOOR_S`), when empty input satisfies the
assertion, or when it checks state instead of behaviour.

**Changing behaviour the docs describe:**

```bash
python scripts/claims.py --report cash/cost_model.py   # claims resting on a file
python scripts/claims.py --pin                         # fill new `@?` anchors
python scripts/claims.py --queue                       # claims to re-read
python scripts/doc_numbers.py --update                 # refresh test counts and similar
```

[`tests/docs/README.md`](https://github.com/galgtonold/cash/blob/main/tests/docs/README.md)
explains claim anchors, skipped examples and derived numbers.

## Writing tests

| Folder | What goes there |
|---|---|
| `tests/test_core/` | the decorator, keys, hashing, configuration |
| `tests/test_backends/` | storage backends |
| `tests/test_notebook/` | notebook unit tests, with a real IPython shell |
| `tests/test_ui/`, `tests/test_cli/` | the dashboard and explorer; the `cash` command |
| `tests/test_tooling/` | CI workflows, test selection, repository hygiene |
| `tests/test_notebook_integration/<feature>/` | real kernels over real notebooks, one folder per feature |
| `tests/docs/` | the documentation's examples and claims |
| `tests/test_wheel_gate/` | the built wheel in a fresh environment; skipped unless switched on |

Name a new file after the behaviour it checks and put it in the folder of the
feature it covers.

A notebook unit test uses the fixtures in `tests/conftest.py` (`mock_shell`,
`cash_magics`, `cash_instance`, `clean_backend`, `statement_processor`) and
runs a cell with `run_cash_cell`:

```python
from tests._cell_driver import run_cash_cell

def test_feature(cash_magics, mock_shell):
    mock_shell.user_ns["x"] = 10
    run_cash_cell(cash_magics, "y = x * 2")
    assert mock_shell.user_ns["y"] == 20
```

Pass `cells=[...]` when the upstream check needs the notebook's other cells.
Read state through `cash_magics.tracking_state` or
`cash_magics.cash_status("dict")`, not private attributes.

An integration test drives a real kernel with the `nb_runner` fixture:

```python
def test_feature(nb_runner):
    nb_runner.create_notebook(["x = 10", "y = x * 2", "print(f'Result: {y}')"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "Result: 20" in nb_runner.get_output(3)
```

Helpers such as `shows_cached` come from `tests._nbharness`.

The root conftest gives every test its own `CASH_CACHE_DIR` and fails a test
that leaves a `.cash/` folder in the checkout, or that discards a cache write.
A test that calls `InteractiveShell.instance()` must call
`InteractiveShell.clear_instance()` in teardown.

## Code style

- `ruff check` and `ruff format`, as configured in `pyproject.toml`.
- Type hints and docstrings on public functions and methods.
- Every statement cache key goes through `compute_cache_key()` in
  `cash.notebook.cache_key`, and every lineage write through `LineageStore`.
  Keys contain code and input lineages; files reach a statement key only
  through the lineage of the variable they were read into.

## Pull requests

1. Branch from `main`.
2. Add tests, and show they fail without the fix.
3. Run the checks above.
4. Update the documentation the change affects.
5. Describe what changed and why.

Commit messages use Conventional Commits (`fix:`, `feat:`, `docs:`, ...).
