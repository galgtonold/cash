---
search:
  boost: 0.5
---

# How cash is tested

!!! info "Applies to: both paths"
    Contributors, and anyone deciding how far to trust cash's results.

What the test suites cover, where each one runs, and how to run them
yourself. Figures on this page are derived from the repository by
`scripts/doc_numbers.py` and checked in CI; they are current as of
<!-- docnum:version -->0.13.0<!-- /docnum -->.

## The suites

**<!-- docnum:tests_total -->~20,450<!-- /docnum --> tests** in
<!-- docnum:test_files -->~2,030<!-- /docnum --> files:

| Suite | Size | What it covers |
|---|---|---|
| Unit | <!-- docnum:tests_unit -->~15,140<!-- /docnum --> | keys, lineage, hashing, backends, the decorator, the notebook engine with a real IPython shell |
| Notebook integration | <!-- docnum:tests_integration -->~4,720<!-- /docnum --> | real kernels running real notebooks, one folder per feature |
| Docs | <!-- docnum:tests_docs -->~620<!-- /docnum --> | the documentation's examples and claims ([below](#the-docs-are-tested-too)) |

## What runs where

| Check | When | Platforms | What it runs |
|---|---|---|---|
| Unit | every push to main, every pull request | Linux, Windows and macOS, Python 3.10 to 3.14: <!-- docnum:platforms -->15<!-- /docnum --> combinations | the unit suite, without the tests marked `perf` |
| Polars 1 | every push and pull request | Linux, Python 3.12 | the unit tests named for polars, on the newest polars 1 (the others run on polars 2) |
| Integration core set | every push to main, every pull request | Linux, Python 3.10 to 3.14 | about 340 integration tests (`tools/test_selection/core_set.txt`), chosen to cover every line, feature and step sequence the whole suite covers |
| Integration, full | nightly | Linux, Python 3.12, in 6 parallel shards | the whole integration suite |
| Docs | every push and pull request | Linux, Python 3.12 | the docs suite, and a check that the derived numbers are current |
| Lint | every push and pull request | Linux | `ruff check .` and `ruff format --check .` |
| Benchmarks | weekly, and when `benchmarks/` changes | Linux | the benchmark tooling tests, and the `perf` tests, which never block |
| Release | when a release is published | Linux | the claim-drift check, blocking; then the build and `twine check` |

A push to a working branch (`claude/**`, `cext/**`) runs a smaller tier so it
never queues: the unit suite on Linux with Python 3.13 in 4 shards, and on
Windows and macOS with Python 3.10 and 3.14 (Windows in 2 shards each), and the
integration core set on Python 3.10, 3.13 and 3.14; the other checks run as above.

Two rules apply to every test run. `xfail_strict` is on, so a test marked as
an expected failure fails the build once it passes. And a test fails if any
cache write was discarded while it ran, so a cache that silently stops writing
cannot pass.

## New tests must fail first

`scripts/fails_first.py` checks out the last commit into a temporary
worktree, runs the new tests against that unfixed source, and fails if they
pass there. A
test that passes without the fix proves nothing and is rewritten.

## The docs are tested too

- **Python examples run.** The docs suite finds every `.md` file under
  `docs/`, `examples/` and the repository root and runs its Python code blocks.
  For a decorated function, it checks that the hits and misses match the
  example's comments (`# cache hit`, `# cache miss`).
- **Claims are pinned to the code.** About
  <!-- docnum:claims -->~500<!-- /docnum --> statements about behaviour carry an
  anchor naming the source that decides them, with a fingerprint of that
  source. When the code changes, the claim is listed for re-reading. Pull
  requests report drift without failing; a release fails on it.
- **Numbers are derived.** Test counts, the CI matrix and the version are
  written by `scripts/doc_numbers.py`, and CI fails when one is out of date.
- **Links, settings and names are checked.** Internal links and anchors,
  config fields, environment variables and magic names in the docs must exist
  in the code.

Prose without an anchor is not checked by anything, and an anchor covers one
function, not the functions it calls.

## Packaging

The suites run against an editable install with the `dev` extra, which adds
every library cash works with (pandas, polars, redis and the rest).
`scripts/wheel_gate.py` checks the package a user gets: it builds a wheel,
installs it into a fresh virtual environment, and drives a real Jupyter server
through kernel restarts. It is run by hand; CI skips it.

## Running the suites

```bash
# the unit suite
pytest tests/ --ignore=tests/test_notebook_integration \
    --ignore=tests/test_wheel_gate --ignore=tests/docs
# the integration core set
pytest @tools/test_selection/core_set.txt
# the whole integration suite (slow)
pytest tests/test_notebook_integration
pytest tests/docs
python scripts/claims.py --queue
python scripts/fails_first.py <your new test file>
```

## Running the tests a change touches

`tools/test_selection/impacted.py` reads the diff from `origin/main` (plus
uncommitted changes) and picks the tests whose recorded coverage runs a
changed line:

```bash
# print the node ids, one per line
python tools/test_selection/impacted.py
# run them: -n for workers, pytest arguments after --
python tools/test_selection/impacted.py --run -n 8
# without a map: the tests that import what changed
python tools/test_selection/impacted.py --file-level
```

It reads a coverage map: one row per test and file, with the lines that test
ran, and the commit it was built at. The map lives outside the repository, in
`~/.cache/cash-test-map/map.sqlite` unless `--map` names another. Build it
with the whole suite, once a night on a fast machine:

```bash
python tools/test_selection/impacted.py build -- -n 16
```

That runs every test CI runs, apart from `perf` and the wheel gate, under
per-test coverage, Jupyter kernels included, and takes about 20 to 30 minutes
on 16 cores. Without a map the tool stops and says how to build one;
`--file-level` instead picks the tests that import the changed modules.

What it picks, and says it picked, on stderr:

- **A changed line** selects the tests that ran it. A `def` line or module
  constant runs at import, once per process, so a change there counts as a
  change to its function, or to the whole file.
- **Lines no test runs** (new code, an untested branch) widen to the tests of
  the enclosing function or class, then of the file, then to the tests that
  import the module, then to tests named after it.
- **C sources** (`src/cash/_c/`) are not traced. They select the tests that run
  the Python lines calling into `cash._native`, the tests that import it, and
  `tests/test_core/watchers/`.
- **Changed test files** run whole; a changed `conftest.py` runs its folder.
- **A map from an older commit** is read through `git diff`: changed lines are
  carried to the map's line numbers, and lines that moved too far to place widen
  to their file, with a count.

## Related

- [Contributing](../contributing.md): setup, and which tests to run for a
  change.
- [Writing cache-safe cells](../known-limitations.md) and
  [Decorator limitations](../decorator-limitations.md): the cases cash is
  known to get wrong or not see, in a notebook and with `@cash.cache`.
