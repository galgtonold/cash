# Docs tests

This folder checks that the documentation is true: its examples run and do
what the prose says, and its claims about Cash still match the source.

## Running locally

```bash
pytest tests/docs/ tests/test_core/test_agent_guide_sync.py -n0 -o addopts=""
```

`test_agent_guide_sync.py` belongs in the run because it also reads a page
under `docs/` (see [Pages other tests own](#pages-other-tests-own)). Install
the `docs-test` extra first; the examples import the packages it holds. CI
runs this folder in the `docs-parity` job of `.github/workflows/ci.yml` on
every push and pull request.

## Examples run as programs

`test_tutorials.py::test_doc_page` collects every `.md` file under `docs/`
(except `docs/superpowers/`), under `examples/`, and at the repository root
that contains a ` ```python ` fence. For each page it:

1. extracts every ` ```python ` fence in order, including one indented in a
   content tab, admonition or list item (its body is dedented);
2. drops the fences marked to skip and those in `_harness.PENDING_FENCES`
   (below), joins the rest into one script and runs it
   in a fresh namespace (top-level `await` works; `tests/docs/conftest.py`
   supplies SDK mocks);
3. finds the `@cash.cache` functions and works out the hits and misses the
   page implies, from how often each is called with each argument tuple and
   from comments such as `# First call: cache miss` or `# cache hit`;
4. compares that with each function's `cache_info()`.

A page with a `{ .nb-cell }` fence or a `%magic` line runs instead in an
in-process IPython shell with cash's magics loaded, one fence per cell, as a
notebook would.

### Annotations

Put these HTML comments on the line directly before a fence:

| Annotation | Effect |
|---|---|
| `<!-- test:skip reason="..." -->` | the fence is not run; `reason=` is required |
| `<!-- test:expect-raises -->` | the fence may raise; the page carries on |
| `<!-- test:expect-warning -->` | the fence may emit a Cash warning; otherwise one fails the page |

`<!-- test:allow-unexercised reason="..." -->` anywhere on a page allows a
cached function that the page defines but never calls. Skip counts appear in
the run's summary.

Run an example yourself before you paste it, and write the hit and miss
comments next to the calls they describe.

### Checking a badge

`<!-- test:expect-badge first=EXECUTED rerun=CACHED -->` before a notebook
cell asserts the badge on its first run and, with `rerun=`, on an immediate
second run (CACHED, EXECUTED, MIXED or SKIPPED; a bare word means `first=`).
Put it at the top of a stack of annotations: the other readers stop at a
comment they do not know. The docs conftest caps `time.sleep` at 1 ms, so
assert `rerun=CACHED` only on real compute or a `# @cash:persist` statement.

`CASH_DOCS_RERUN_NB_CELLS=1` reruns every `{ .nb-cell }` fence that is not
only imports and definitions, and fails when the rerun reads EXECUTED. It is
opt-in while a few cells still fail it; mark a cell that really runs again
with `<!-- test:expect-badge rerun=EXECUTED -->`.

### Fences waiting on a page edit

`_harness.PENDING_FENCES` lists fences that do not run as written, by page
and the start of the fence's code, each with a reason. They are skipped until
the page is fixed; then delete the entry. `test_harness.py` fails on an entry
that matches no fence, so the list only shrinks. Do not add to it: fix the
page, or give the fence a `test:skip` with a reason.

## Other checks

- **Page checks** read the pages without running them: `test_doc_claims.py`
  (environment variables, configuration defaults, magics, internal links and
  anchors, test names), `test_api_references_resolve.py`,
  `test_mermaid_diagrams.py`, `test_warning_codes_documented.py` and the
  other `test_*.py` files.
- **Generated assets**: `test_badge_examples_fresh.py`,
  `test_badge_images_fresh.py`, `test_brand_assets_fresh.py` and
  `test_try_cash_notebooks_in_sync.py` fail when a committed file no longer
  matches the script that builds it, and name the script to re-run.
- **Example notebooks**: `test_example_notebooks.py` checks every notebook in
  `examples/` for the standard setup cell and none of `%%time`,
  `%load_ext cash`, debug toggles or mtime sleeps, and that the examples'
  links into the docs resolve. With `CASH_RUN_EXAMPLE_NOTEBOOKS=1` (the
  `example-notebooks` CI job) it also runs the allow-listed notebooks top to
  bottom in real kernels, offline.
- **Tables tied to data**: `test_benchmarks_table_matches_frozen_data.py`
  (the restore table in `docs/benchmarks.md`) and
  `test_cacheability_checker_matches.py` (the table in
  `docs/how-it-works/safety.md` against `docs/javascripts/cacheability-checker.js`).
- `test_harness.py` tests the harness itself, with fixtures under `_fixtures/`.

## No line numbers

`test_no_line_pinned_source_references` fails a published page that cites a
source line (`` `core.py:1234` ``). Name the symbol instead
(`Cash._compute_with_lock`): it moves with the code and a claim anchor can
check it. The one allowed form pins the commit the line was read at,
`` `src/cash/core.py:1234@8e5f4ce` ``, and
`test_commit_pinned_references_resolve` checks that the commit and line exist.
Use it only for a claim about that snapshot.

## Numbers the docs quote

Test counts, the claim count, the platform count and the version are wrapped
in markers:

```markdown
Roughly <!-- docnum:tests_total -->~8,750<!-- /docnum --> tests.
```

`test_doc_numbers.py` fails when a marked value no longer matches the
repository. Refresh them with:

```bash
python scripts/doc_numbers.py --update   # --list shows every value, --check only checks
```

## Claim anchors

A claim anchor ties a sentence about Cash's behaviour to the source that
decides it:

```markdown
<!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
Cash keys a call on the function source plus its arguments.
```

Anchor any sentence that states a default, a threshold, an invalidation rule,
what a flag does, or what is cached and what is not. Paths are relative to
`src/`; one comment can hold several targets, separated by commas.

| Form | Example | Checks |
|---|---|---|
| Fingerprint | `cash/core.py:Cash.cache @7a77d1c5` | the symbol exists and its source is unchanged |
| Value | `cash/config.py:CashConfig.max_cache_size == None` | the literal in the source equals the one written |
| Existence | `cash/backends/redis.py:RedisBackend` | the symbol exists |

Use a value anchor when the sentence quotes a constant. Write a fingerprint
as `@?` and fill it with the tool, never by hand. Anchor the narrowest symbol:
an anchor on a module or class needs `broad="reason"`.

```bash
python scripts/claims.py --pin                        # fill every @?
python scripts/claims.py --queue                      # drifted claims, with nearby unanchored prose
python scripts/claims.py --accept docs/page.md        # show a page's drifted code
python scripts/claims.py --accept docs/page.md --yes  # re-pin it
python scripts/claims.py --report cash/cost_model.py  # claims resting on a source file
```

When a fingerprint drifts, read the sentence against the new code, fix the
sentence if it is wrong, then re-pin. Drift is reported in the `docs-parity`
job summary without failing it; the release build in
`.github/workflows/publish.yml` sets `CASH_CLAIMS_STRICT=1` and fails on it.
Unresolved targets and wrong values fail every run.

`tools/claims/claim_manifest.json` records how many anchors each audited page
has; a page may gain anchors but not lose them. When you rewrite a page and
its count changes on purpose, update its entry. The library lives in
`tools/claims/` so `scripts/claims.py` does not import from `tests/`.

### Limits

- A fingerprint covers one symbol. If the behaviour lives in something that
  symbol calls, the callee can change while the anchor stays green. Anchor
  the symbol whose body decides the behaviour, not the entry point.
- A symbol inside `if TYPE_CHECKING:` or `try:`/`except` cannot be anchored,
  nor can a tuple-unpacked constant (`X, Y = 1, 2`).
- A value containing a comma, or an expression such as `8 * _GIB`, needs a
  fingerprint anchor.
- Put a claim anchor above any `test:` annotation, not between it and its
  fence, or the annotation no longer reaches the fence.
- An anchor inside a code fence is ignored.
- Prose with no anchor is not checked at all. `--queue` lists unanchored lines
  that name a drifted symbol, as a hint.
- `test_doc_claims.py` and `test_claim_coverage.py` use "claim" for a fence's
  expected hits and misses, which is unrelated to claim anchors.

## Pages other tests own

`docs/for-coding-agents.md` must be identical to
`cash._agent_guide.AGENT_GUIDE`, which `cash.help()` returns;
`tests/test_core/test_agent_guide_sync.py` checks this. It cannot carry an
HTML comment, so it has no claim anchors.
