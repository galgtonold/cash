# Agent Instructions for Cash

> **Scope:** the single source of truth for AI coding assistants in this repo:
> GitHub Copilot, Claude Code (via `CLAUDE.md`), Cursor, Codex and others.
> Change conventions here, not in per-tool config.

## Project overview

Cash is a Python caching library with two entry points:

1. **`@cash.cache`**, a decorator that caches function results and tracks what they depend on.
2. **`%cash_on`**, IPython magics that cache Jupyter notebooks statement by statement.

It is published on PyPI as `cash-lib`. The version lives only in `src/cash/__init__.py`
(`__version__`); released versions and their notes are in `CHANGELOG.md`. Cash is
still `0.x`, so the cache format may change between minor versions.

Read these instead of relying on this file for details:

- `docs/contributing.md`: setup, the directory map, running and writing tests.
- `docs/how-it-works/`: how keys, lineage, invalidation and storage work.
- `docs/architecture_decisions.md`: the ADRs (not published on the docs site).
- `pyproject.toml` (`[tool.pytest.ini_options]`) and `tests/conftest.py`,
  `tests/test_notebook_integration/conftest.py`: test settings, markers and fixtures.
  The kernel runner behind the integration fixtures is the `tests/_nbharness/` package.

## Commit messages

- **Never include `Co-Authored-By: Claude ...`** or any other AI-attribution trailer. Author the commit normally.
- Conventional-Commits prefixes (`feat:`, `fix:`, `test:`, `chore:`, `build:`, `ci:`, `docs:`, `refactor:`), optional scope (`feat(badge): ...`).
- Subject at most 72 characters, body wrapped at 72. Lead with *why*.
- One logical change per commit.
- Never reference the private tracker (see *Project management*).

## Architecture

- **`src/cash/core.py`**: the `Cash` class and `@cash.cache`.
- **`src/cash/backends/`**: storage backends. `factory.py` maps a `TierConfig.type`
  (`memory` / `file` / `sqlite` / `redis` / `s3` / `tiered`) to a class. The default is
  `TieredBackend([InMemoryBackend, FileBackend])`.
- **`src/cash/tracking/`** and **`src/cash/analysis/`**, plus `purity.py`,
  `object_hashing.py` and `cost_model.py` at the top level: the layer that the
  decorator and the notebook share. `tracking/` records what a computation depends
  on at run time (file reads and snapshots, function source, randomness);
  `analysis/` is static analysis (statement inputs and outputs, `# @cash:`
  annotations, cacheability).
- **`src/cash/notebook/`**: the notebook subsystem. Its large parts are packages:
  `ipython/` (`CashMagics`, the cell executor), `statement/` (`StatementProcessor`
  and its siblings), `upstream/` (`UpstreamChecker`, `NotebookSimulator`,
  virtual lineage), `control_structures/` (per-iteration loop and branch caching)
  and `badge_renderer/`. `cache_key.py` and `lineage_store.py` hold the rules below.
- **Layering:** `notebook/` imports `core` and the shared layer, never the reverse.
  Outside `notebook/`, only the magics loaders (`Cash.register_magic`,
  `load_ipython_extension`, `nbconvert`) import it, so `@cash.cache` runs without
  the notebook package (`tests/test_core/test_decorator_without_notebook.py`).

### JupyterLab extension (`labextension/`)

The only non-Python part. It sends the notebook's unsaved cell sources over the
`cash_live_cells` comm that `notebook/live_cells.py` receives, the only way cash
sees an edit not yet written to the `.ipynb`.

Node is never needed to install, test or build the wheel: the bundle is
committed under `src/cash/labextension/`. Only after changing
`labextension/src/index.ts`, run `cd labextension && npm install && npm run build`
and commit the regenerated bundle. `comm.commsOverSubshells = 'disabled'` in that
file is load-bearing and guarded by a build script and
`tests/test_notebook/test_labextension_packaging.py`; read `labextension/README.md`
before touching it.

## Critical conventions

### Unified cache-key computation

**Every statement cache key goes through `compute_cache_key()` in
`cash.notebook.cache_key`.** Never build a key anywhere else.

Keys are computed at runtime (`_analyze_and_hash` in `statement/processor.py`),
during upstream simulation and virtual restore (`_update_virtual_lineage` and
`try_virtual_restore` in `upstream/virtual_lineage.py`) and for call units
(`call_unit.py`). If two of these disagree, a kernel restart turns into cache
misses or stale values; that has caused critical bugs more than once. To change
the key, change only `compute_cache_key()`, and add tests next to
`tests/test_notebook/test_virtual_restore_modules.py`.

A key is `{namespace}:{sha256(...)}`, where the namespace is `stmt` for statements
and `call` for call units, and the hash covers the statement's source hash, the
lineage of each input, the source of called functions and modules, and the
statement's occurrence index in the cell.

### Lineage

- A variable's **lineage hash** stands for its full dependency chain: the code that
  produced it plus the lineages of its inputs.
- `LineageStore` (`notebook/lineage_store.py`) is the one place lineage is read and
  written. `LineageStore.resolve` applies the priority order: the simulation's
  `virtual_lineage` first, then the store (`variable_lineage`), then the object's
  `_cash_lineage_hash` attribute, then `compute_hash_fn(value)`.
- The simulator also tracks `executed_cell_codes` (variable to the code that last
  produced it) and `executed_input_lineages` (variable to the input lineages used).

## Testing

- **Every feature and bug fix needs a unit test, and an integration test when it
  touches notebook behaviour.**
- Unit tests live in feature folders: `tests/test_core/`, `test_backends/`,
  `test_notebook/`, `test_ui/`, `test_cli/`, `test_tooling/` (CI, test selection,
  hygiene), plus `tests/docs/`. Each is a package; put a new file in the folder
  of the feature it pins, named after the behaviour, not at the top of `tests/`.
- Notebook unit tests use a real IPython with the one `MockShell` in
  `tests/conftest.py`, through its fixtures: `mock_shell`, `clean_backend`,
  `cash_instance`, `cash_magics` (as `%load_ext cash` leaves it) and
  `statement_processor`. Run a cell with `run_cash_cell(cash_magics, code)` from
  `tests/_cell_driver.py`; pass `cells=[...]` when the upstream check needs the
  notebook's cells. Do not copy the shell or build another `CashMagics` over an
  in-memory Cash: a test that needs other settings (a disk cache, `persist_all`,
  a second kernel over the same cache dir) builds that one difference on the
  shared fixtures and says why. Read state through `magics.tracking_state` and
  `%cash_status` (`cash_status("dict")`) where they have it, not private
  attributes. Use `tmp_path` for files. Never mock `IPython` in `sys.modules`.
- The root conftest points `CASH_CACHE_DIR` at a per-test directory under the
  pytest base temp, and fails a test that leaves `.cash/` in the checkout. A test
  of the default cache location must set or clear `CASH_CACHE_DIR` itself.
- Integration tests live in `tests/test_notebook_integration/<feature>/`, one
  folder per feature (`basics`, `loops`, `upstream`, `restart`, `files`,
  `calls`, `language`, ...; each folder's `__init__.py` says what it covers).
  Add a test to the file whose name says the behaviour it checks, or start a
  file named that way; never name a file after the sweep or review that found
  the bug.
- Integration tests use `nb_runner`, which
  drives a real kernel over a real `.ipynb`: `create_notebook`, `load`,
  `start_kernel`, `run_all` / `run_cells` (1-based), `set_cell_source`,
  `get_output`, `peek` (`tests/_nbharness/runner.py`). Import helpers such as
  `shows_cached` from `tests._nbharness`, never from a conftest. Assert on
  `get_output(n)` for what the user sees and on `nb_runner.peek(expr)` for
  kernel state: a cache hit replays stdout, so printed
  output can describe the value from when the entry was written.
- Do not count executions of a cached callee with a counter it writes itself; the
  write is restored on a hit. Use `os.open`/`os.write` (not `builtins.open`, which
  the file tracker turns into a dependency).
- `InteractiveShell.instance()` creates a process-wide shell. A test that calls it
  must call `InteractiveShell.clear_instance()` in teardown, as
  `tests/docs/_harness.py` and `benchmarks/tests/conftest.py` do.
- Settings live in `pyproject.toml`: xdist workers and scheduler, reruns for
  kernel-boot failures, the 30 s per-test timeout (override with
  `@pytest.mark.timeout(...)`), strict xfails, and the markers. Use `-n 0` to
  debug. Integration workers reuse one warm kernel per worker; mark a test
  `fresh_kernel` when it needs a real restart, or set `CASH_TEST_REUSE_KERNEL=0`
  to rule out cross-test contamination.
- `perf` marks wall-clock threshold tests; CI's unit job skips them.
  `benchmarks/tests/` tests the benchmark tooling and runs separately.

### Choosing integration tests

Never run the whole integration suite while iterating; it takes a long time.
Run the files whose names match what you changed, at most about ten; the
feature folders (`tests/test_notebook_integration/loops/` and so on) are where
to look for them. For a
broader check, run the core set, the same one CI runs on every push:

```bash
pytest @tools/test_selection/core_set.txt
```

The core set is the few hundred integration tests that together cover every
line, feature, feature pair and step sequence the whole suite covers. The whole
suite runs nightly in shards (`.github/workflows/nightly.yml`); to run one
shard locally, add `-p tools.test_selection.shard --shard=2/6` to
`python -m pytest tests/test_notebook_integration`.

To re-pick the core set after the suite has changed a lot:

```bash
python tools/test_selection/run_baseline.py   # the whole suite once, with per-test coverage (slow)
python tools/test_selection/select_core.py    # re-pick from an existing baseline
```

`select_core.py` greedily picks the passing tests that add new covered lines,
features, feature pairs or step sequences per second of runtime, and writes
`.testsel/core_set.txt` plus a report. Copy both over the committed ones in
`tools/test_selection/`.

### Before reporting work as done

1. Run the unit tests for the area you touched (`pytest tests/test_notebook/` for notebook work).
2. Run the relevant integration tests, chosen as above.
3. Show that a new test fails without the fix: `python scripts/fails_first.py <test file>`.
   It runs the tests against the last commit's `src/` in a temporary worktree and fails if any pass anyway. Common ways a
   test passes vacuously: the mechanism never engages (a cached function faster
   than the persistence floor is never written to disk; sleep
   `tests.conftest.ABOVE_PERSISTENCE_FLOOR_S`), empty input satisfies the
   assertion, a different gate stands in for the real one, or state is checked
   instead of behaviour. Give every filter or exclusion a positive control.

### Debugging

`%cash_debug on` prints the cache decisions, with prefixes such as `[UPSTREAM]`,
`[UPSTREAM_DEBUG]`, `[CACHE_KEY]`, `[CONTROL]` and `[TIMING_PROXY]`. In VS Code,
reproduce notebook bugs with the built-in notebook tools, not a Jupyter MCP
server, then pin the fix with an `nb_runner` test. Keep throwaway scripts in
`scratch/` at the repo root (gitignored).

## Writing documentation

- **Never cite a line number** (`` `core.py:1234` ``) in a published page;
  `tests/docs/test_doc_claims.py` fails on it. Name the symbol instead. The one
  exception pins a commit (`` `src/cash/core.py:1234@8e5f4ce` ``), for claims about history.
- **Anchor every mechanism claim** to the source that decides it:
  `<!-- claim: cash/core.py:Cash.cache @? -->`, then `python scripts/claims.py --pin`.
  Prefer a value anchor (`== 0.01`) when the prose quotes a constant.
- Before changing code, `python scripts/claims.py --report <src file>` lists the
  claims that rest on it.
- Full guide: `tests/docs/README.md`.

## Project management

**Two trackers, different jobs.** Use the right one:

| tracker | what goes there |
| --- | --- |
| **`galgtonold/cash-tracker`** (private) | Everything internal: bugs, follow-ups, tech debt, roadmap. This is the single source of truth for planning. |
| **`galgtonold/cash`** (public) | Only issues filed by outside users — the README's bug-report link, the badge's "Report a bug" button, the links at the end of `%cash_help`. Do not file internal work here. |

Board: <https://github.com/users/galgtonold/projects/1> (private) — views *All work*, *Board*, *High priority*, *Correctness*, *Docs*.

**Never reference a private tracker issue from the public repo.** No `Fixes cash-tracker#12` in a commit that lands on `galgtonold/cash`, no private issue numbers in public PR descriptions or CHANGELOG entries. A public reader cannot open them, so it reads as noise. Describe the change on its own terms instead.

Do **not** create or resurrect roadmap markdown. The old `planning/ROADMAP.md` no longer exists in the working tree at all (neither `planning/` nor `planning/archive/` is present) — if you need that history, read it out of git, and do not recreate the directory.

**Behavioral rules for AI assistants (do these without being asked):**
1. **Look in the tracker by default.** Before proposing what to work on, run
   `gh issue list --repo galgtonold/cash-tracker --state open --label prio:high`
   (and drop the label filter for the full picture). Don't ask the user to paste issue state — fetch it.
2. **Auto-create on flag.** The moment you notice deferred work, an out-of-scope fix, a real bug, or a "do this later" — file it immediately (don't let it die in chat):
   `gh issue create --repo galgtonold/cash-tracker --title "..." --body-file <file> --label "prio:medium" --label "type:bug"`
   Best-guess priority is fine. Better an imperfectly-triaged issue than a lost one.
3. **Cross-check on reference.** When the user mentions an issue — by number or description — read it (`gh issue view <n> --repo galgtonold/cash-tracker`) and work from its current state and comments, not from memory. **Ticket claims are unreliable**: roughly a third describe code that has since changed. Verify against the source before acting on a description.
4. **Close with evidence.** When work is verified, close the issue with a comment naming what you checked (`--reason completed --comment "..."`). Don't close silently, and don't delete.

**Historical Linear ids.** Issues lived in Linear (team `Cash`, prefix `CAS`) until 2026-08-21, when all 68 open ones were migrated. Old commit messages and `CHANGELOG.md` entries still cite ids of the form `CAS-<n>`; those are **not** GitHub issue numbers. Every migrated issue carries a canonical footer instead, so map an old id to its issue with (put the number in for `<n>`):

```bash
gh issue list --repo galgtonold/cash-tracker --state all --limit 300 \
  --json number,title,body \
  --jq '.[] | select(.body | contains("Migrated from Linear `CAS-<n>`")) | "#\(.number) \(.title)"'
```

Match on the **backticked** id, not a bare search: `--search` returns every issue that merely mentions the id, and an unanchored regex matches a longer id that starts with the same digits. The footer format is uniform across all 68 precisely so this stays a one-line exact lookup — keep it that way when filing new issues.

Nothing else in the repo cites a tracker id or a user-testing round: `tests/test_tooling/test_repo_hygiene.py` fails on one anywhere outside `CHANGELOG.md`. Say what a change protects instead.

**Structure:**
- **Priority** — `prio:high` / `prio:medium` / `prio:low`. Every issue has exactly one.
- **Type** — `type:bug`, `type:feature`, `type:improvement`, `type:docs`, `type:tech-debt`.
- **Workstreams** — `correctness` (can serve a stale or wrong result), `cache-perf`, `release`.
- **Area** — `area:notebook`, `area:decorator`, `area:backends`, `area:badge`, `area:tests`. Optional.
- **`known-limitation`** is load-bearing (it replaces Linear's `xfail`): an issue with it maps to a `pytest.mark.xfail` in the suite, and closing it means flipping that marker to a passing test. Some are *documented limitations we do not intend to fix* — read the body before "fixing" one.

**Other:**
- **Breaking changes**: Document in `CHANGELOG.md` under the upcoming version and call them out in the PR description.
- **Version control**: Commit in logical chunks with clear messages (see *Commit messages* section above).

## Release process

When asked to cut a release or bump the version:

### 1. Pick the version

- Bug fixes only: bump the patch number.
- New features: bump the minor number.
- Breaking change: while `0.x`, bump the minor number and list it under **Breaking**.
- Go to 1.0 only once the API is one we are willing to freeze.

Never reuse a version that was published. The numbers under *Pre-release
development history* in `CHANGELOG.md` were never published. Prefer final
versions over `bN`/`rcN`: pip ignores pre-releases without `--pre`.

### 2. Clear the doc-claim queue

```bash
python scripts/claims.py --queue
```

It must print `No drifted claims.` For each entry, re-read the claim against the
code (`--accept <page>` shows it), then fix the prose or re-pin with
`--accept <page> --yes`. Do this before the CHANGELOG: a wrong claim is often a
**Fixed** entry. `publish.yml` re-runs this check with `CASH_CLAIMS_STRICT=1` and
blocks the release on drift.

### 3. Write the CHANGELOG entry from `git log`

```bash
PREV=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
git log --no-merges --pretty=format:"%h %s" ${PREV:+$PREV..HEAD}
```

The log is the source, not the output. Read the whole range (and the diff when a
subject is unclear), then write notes a user wants:

- Ground every entry in the log; never describe a change that is not in the range.
- One entry per user-visible change, however many commits it took.
- Leave out test, CI, refactor and build changes with no user-visible effect.
- Lead with what changed for the user and why it matters.

Sort into Keep-a-Changelog sections, using prefixes as a hint: `feat` → **Added**,
`fix` → **Fixed**, user-visible `refactor`/`perf`/`build`/`chore` → **Changed**,
`!` or `BREAKING CHANGE:` → **Breaking** (first). Add the section as
`## [X.Y.Z] - YYYY-MM-DD` below `## [Unreleased]`, above the previous release.
Leave *Pre-release development history* untouched. **The user reviews the entry
before it is committed.**

### 4. Bump the version

Edit only `__version__ = "..."` in `src/cash/__init__.py`. `pyproject.toml` reads it
(`dynamic = ["version"]`, `[tool.hatch.version]`); never add a `version =` line there.
`tests/test_core/test_docs_version_currency.py` fails if a user-facing page names
another version.

### 5. Build and verify, always into an empty `dist/`

```bash
rm -rf dist/            # python -m build adds to dist/, it never clears it
python -m build
ls dist/                # exactly two files, both X.Y.Z: the wheel and the sdist
```

- `pytest tests/test_notebook -x`
- `twine check dist/*`
- In a fresh venv, `pip install dist/cash_lib-X.Y.Z-py3-none-any.whl`, then
  `python -c "import cash; print(cash.__version__)"` prints `X.Y.Z`.
- In that venv, `pip install "jupyterlab>=4,<5"` and `jupyter labextension list`
  must show `cash-live-cells ... enabled OK (python, cash-lib)`. Nothing else
  catches a wheel that installs but registers no extension.

### 6. Commit and tag

```bash
git add src/cash/__init__.py CHANGELOG.md
git commit -m "release: X.Y.Z"
git tag vX.Y.Z
```

Push only after the user confirms. Pushing the tag does not publish:
`publish.yml` runs on a published GitHub Release (or a manual dispatch).

### 7. Publish

Publish by creating the GitHub Release, which runs `publish.yml` from a fresh
checkout. Only if that workflow is broken, upload by hand, naming both files
explicitly and never `dist/*`: a PyPI version can never be uploaded again, and a
stale `dist/` has held old builds that a glob would have published.

```bash
twine upload dist/cash_lib-X.Y.Z-py3-none-any.whl dist/cash_lib-X.Y.Z.tar.gz
```

Afterwards, check that `pypi.org/project/cash-lib/X.Y.Z/` is live and
`pip install cash-lib==X.Y.Z` resolves on a clean machine.
