# Docs feature-parity tests

These tests run every Python code fence in the Cash tutorials as a real
Python program and assert that the documented cache hit/miss behavior
actually holds.

## How it works

For each markdown file under `docs/tutorials/feature-guides/` (and later
under `use-cases/`), `tests/docs/test_tutorials.py::test_doc_page`:

1. Extracts every ` ```python ` fence in source order (`_harness.extract_fences`),
   including indented ones inside a content tab, admonition or list item
   (their body is dedented)
2. Skips fences annotated with `<!-- test:skip reason="..." -->`, and the
   ones listed in `_harness.PENDING_FENCES` (see below)
3. Concatenates the remaining fences into a single executable script
4. Compiles with `PyCF_ALLOW_TOP_LEVEL_AWAIT` so async examples work
5. Runs the script in a fresh namespace (autouse fixtures from
   `tests/docs/conftest.py` provide SDK mocks where needed)
6. Parses the script with `ast` to find `@cash.cache`-decorated functions
7. Infers the expected `(hits, misses)` per function based on:
   - The number of times each function is called with each unique arg tuple
   - Inline comment hints (`# cache hit`, `# cache miss`, `# First call:`, etc.)
   - Function-level markers like `@cash.stateful`
8. Reads each decorated function's `cache_info()` and asserts the actual
   hits/misses match the inferred expectation

If the doc says "first call computes, second call hits" but the code
actually misses both times (because someone broke `@cash.cache`), the
test fails with the markdown filename and line range.

A page with a `{ .nb-cell }` fence or a `%magic` line runs through an
in-process IPython shell with cash's magics loaded instead, one fence per
cell, so it behaves as a notebook would.

## Checking a badge

A notebook cell can assert the badge a reader sees:

```markdown
<!-- test:expect-badge first=EXECUTED rerun=CACHED -->
```

`first=` is the badge on the cell's first run; `rerun=` runs the cell again
at once and checks that badge. A bare word means `first=`. The words are the
badge's own: CACHED, EXECUTED, MIXED, SKIPPED. Put this annotation at the
top of a stack of annotations: the other annotation readers stop at a comment
they do not know.

The docs conftest caps `time.sleep` at 1 ms, so a cell that sleeps to stand in
for slow work is too cheap to cache here and reads EXECUTED on a rerun. Assert
CACHED only on a statement that clears the cost floor for real, or carries
`# @cash:persist`.

`CASH_DOCS_RERUN_NB_CELLS=1` reruns every `{ .nb-cell }` fence that is not
only imports and definitions, and fails when the rerun reads EXECUTED. It is
opt-in because a few cells on the current pages still fail it. Annotate a
cell that really does run again with `<!-- test:expect-badge rerun=EXECUTED -->`.

## Fences waiting on a page edit

`_harness.PENDING_FENCES` lists fences that do not run as written, keyed by
page and the start of the fence's code, each with the reason. They are
skipped until the page is fixed; then delete the entry.
`test_harness.py::test_every_pending_fence_entry_still_matches_a_fence` fails
on an entry that matches no fence, so the list only shrinks. Do not add to it:
fix the page, or give the fence a `test:skip` with a reason.

## Skipping a fence

When a fence genuinely can't be tested (illustrative output, bash
command, signature shape, deliberate anti-pattern), annotate the line
immediately before it:

```markdown
<!-- test:skip reason="signature illustration only" -->
```

The `reason=` attribute is required. Skip counts show up in the
end-of-run terminal summary, so accumulating skips becomes visible.

## Running locally

```bash
pytest tests/docs/ -v
```

Unit tests of the harness itself (extractor, claim inference, etc.)
live in `tests/docs/test_harness.py` and use synthetic markdown fixtures
under `tests/docs/_fixtures/`.

## CI status

The `docs-parity` job in `.github/workflows/ci.yml` runs this whole folder on
every push and pull request and blocks the merge when a test fails. Claim
drift is the one exception: see [When a claim drifts](#when-a-claim-drifts).

## What else lives here

- **Drift tests for generated doc assets**: `test_badge_examples_fresh.py`,
  `test_badge_images_fresh.py`, `test_brand_assets_fresh.py` and
  `test_try_cash_notebooks_in_sync.py` fail when a committed file under
  `docs/` or `examples/` no longer matches the script that builds it. Each
  names the script to re-run.
- **Example notebooks**: `test_example_notebooks.py` checks every notebook in
  `examples/` for the standard setup cell and no `%%time`, `%load_ext cash`,
  debug toggles or mtime sleeps. With `CASH_RUN_EXAMPLE_NOTEBOOKS=1` (the
  `example-notebooks` CI job) it also runs the notebooks on its allow-list
  top to bottom in real kernels, offline.
- **Page-level checks**: `test_doc_claims.py` (env vars, config defaults,
  magics, internal links, cited line numbers and test names),
  `test_api_references_resolve.py`, `test_mermaid_diagrams.py` and the other
  `test_*.py` files lint the published pages without executing them.
- **The claim-anchor library** is not here: it lives in `tools/claims/`
  (`anchors.py` and `claim_manifest.json`), so `scripts/claims.py` can use it
  without importing from `tests/`.

## Authoring conventions

When you add a new tutorial:

1. Comments inside the fence drive claim inference. Use comments like
   `# First call: cache miss` and `# Second call: cache hit` next to
   actual function calls — they're machine-readable.
2. If a fence is illustrative (no real call to assert), annotate with
   `<!-- test:skip reason="..." -->`.
3. Don't paste an example you haven't actually run yourself first. The
   harness will catch you.
4. **Never cite a line number.** See below.

## Pages that other tests own

`docs/for-coding-agents.md` is **byte-identical** to
`cash._agent_guide.AGENT_GUIDE` — one guide, two surfaces — and
`tests/test_core/test_agent_guide_sync.py` asserts the equality. It therefore
**cannot carry a claim anchor**: any HTML comment you add fails that test.
Ground its claims in the `AGENT_GUIDE` string instead. Its manifest entry
records this in a `note` field.

The general lesson, learned the expensive way: `tests/docs/` is not the only
suite that reads `docs/`. Before pushing a docs change, run

```bash
pytest tests/docs/ tests/test_core/test_agent_guide_sync.py -n0 -o addopts=""
```

A green `tests/docs/` alone let an anchor through that reddened all 15
test-matrix jobs in CI.

## Never cite a line number

`test_no_line_pinned_source_references` bans a bare line number in any
published page — `` `core.py:1234` ``, `` `core.py,1234` ``, and the
split-span `` (`core.py`, `:1234`) `` form all fail the build.

This was a burn-down ratchet with 22 grandfathered pins. All 22 are gone and
**20 of them had rotted** — pointing at a docstring, at a matplotlib comment,
at an entropy-reseed guard, and in one case at a different test than the one
whose code the page quoted verbatim. They don't drift one at a time; they
drift together, whenever something is inserted above them, and a stale line
number still reads as authoritative. That is what makes it worse than no
citation.

**Name the symbol instead** — `Cash._compute_with_lock`, `MUTATING_METHODS`,
`single_unit_policy.should_run_as_single_unit`. It moves with the code,
`--report` can find it when the code changes, and a [claim
anchor](#claim-anchors) can re-verify it.

The one exempt form carries the commit the line was read at:

```markdown
`src/cash/core.py:1234@8e5f4ce`
```

That names a fixed snapshot — `git show 8e5f4ce:src/cash/core.py` resolves it
forever — so it cannot rot. `test_commit_pinned_references_resolve` checks
that the commit is real and that the file actually had that many lines at that
commit (fully, on a complete clone; on CI's shallow clone it verifies the pins
whose commits are present). Reach for this only when the claim is genuinely
*about* a historical state — an ADR, a post-mortem, a CHANGELOG note. For
"here is where this behaviour lives", the symbol is strictly better.

## Claim anchors

Prose is the one thing the fence harness cannot check, and it is where every
doc failure in this repo has actually lived. A **claim anchor** links a prose
claim to the source that decides it:

```markdown
<!-- claim: cash/core.py:Cash.cache @7a77d1c5 -->
Cash keys a call on the function source plus its arguments.
```

Anchor a claim whenever it asserts **how cash behaves** — a default, a
threshold, an invalidation rule, what a flag does, what is cached versus
skipped. Motivation, comparisons and narration need no anchor.

### Three forms

| Form | Example | Checks |
|---|---|---|
| Fingerprint | `cash/core.py:Cash.cache @7a77d1c5` | resolves, and its source is unchanged |
| Value | `cash/config.py:CashConfig.max_cache_size == None` | the documented literal **equals** the one in source |
| Existence | `cash/backends/redis.py:RedisBackend` | resolves only |

Prefer the **value** form whenever the claim quotes a constant. A fingerprint
proves only that someone looked; `== 0.01` proves the number is right forever.

### Authoring

Write `@?` and let the tool fill the digest — never copy a hash by hand:

```bash
python scripts/claims.py --pin
```

Paths are relative to `src/`. One comment may carry several targets, separated
by commas (so a value containing a comma, like a tuple, needs a fingerprint
anchor instead). Anchor the **narrowest** node: a class-level anchor fires on
every unrelated edit inside it, and the checker rejects one unless it carries
`broad="reason"`.

### When a claim drifts

The `docs-parity` job (`.github/workflows/ci.yml`) reports drift in the job
summary on every PR, but does not fail the build on it. The `build` job in
`.github/workflows/publish.yml` — the workflow every release runs — sets
`CASH_CLAIMS_STRICT=1` and re-runs
`tests/docs/test_claim_anchors.py::test_no_fingerprint_drift`, which turns
that same drift into a build failure before a package is ever built. To clear
an entry, read the claim against the current source and then re-pin:

```bash
python scripts/claims.py --accept docs/page.md          # dry run: shows the code
python scripts/claims.py --accept docs/page.md --yes    # re-pin
```

Re-pinning without reading is worse than having no mechanism at all — it
manufactures assurance that nobody checked. The dry run exists to make reading
the default.

### The unpinned prose beside it

Everything above only ever reads a sentence that carries an anchor. Prose with
no anchor is invisible to it however false it goes — the JupyterLab live-cell
branch falsified quickstart's "Google Colab is the exception" warning, and the
queue saw nothing. It was found by hand with `grep`.

That branch falsified a second statement, `magics.md`'s `%%cash` behaviour
list, and **that one is a different failure that this section does not fix.**
It was not unanchored: `docs/magics.md` pins `CashMagics.cash`, and the anchor
stayed green because that method's own body never changed. The behaviour had
moved in a callee several hops down (`_read_notebook_code_cells`), so a green
anchor sat over a false sentence and read as assurance. One-hop callee depth
would not have caught it either — the intermediate is unchanged too; reaching
it needs transitive call-graph fingerprints and the noise that implies.

Be precise about which of the two you are looking at, because they want
opposite fixes: unanchored prose wants an anchor, an anchor-depth miss already
has one and wants a *deeper* one. The rules below reach the `%%cash` section
only by luck of wording — its intro happens to name `%cash_on` — and they
report a line 17 above the false bullet. Treat that as co-location, not
detection.

So `--queue` now prints, under each drifted target, the published prose that
talks about that same code and pins nothing:

```
cash/notebook/ipython/magics.py:CashMagics.cash_on
  docs/magics.md
    :271  names ... but pins nothing: 'processing as `%cash_on` (upstream simulation, ...)'
  docs/getting-started/quickstart.md
    :116  closed enumeration on a page that pins ...: '**Google Colab is the exception**: ...'
```

Two rules feed it, because the two real misses needed different ones: a line
that **names** the symbol (outside any section already anchored to it), and a
line that **closes an enumeration** ("the only", "the exception", "no other")
on a page that anchors the target but in a section that anchors nothing. The
second exists because the quickstart sentence names no symbol at all, so
nothing name-based can reach it.

This is triage, not a gate — it decides nothing, fails nothing, and changes no
exit code. Read it while you are re-reading the drifted claim, and anchor
whichever lines turn out to be claims. Expect it to be chatty for a widely
documented symbol: `%cash_on` is named on 26 pages, and it lists all of them.

### Limitations

- **A fingerprint is one symbol deep, so an anchor can be green over false
  prose.** The claim pins the symbol it names; if the documented behaviour
  actually lives in something that symbol calls, the callee can change while
  the fingerprint holds and nothing asks anyone to re-read. This is the
  `magics.md` case above, and it is the more insidious of the two failures
  because a green anchor reads as a check that passed rather than a check that
  never ran. Closing it needs transitive call-graph fingerprints; nothing here
  approximates that. When you anchor a claim, pin the symbol whose **body**
  decides the behaviour, not the entry point a reader would name.
- **Only direct children are walked.** A symbol defined inside
  `if TYPE_CHECKING:` or a `try:`/`except ImportError:` block cannot be
  anchored — `resolve()` only descends through a definition's immediate
  children, not into nested conditional bodies.
- **A tuple-unpacked constant cannot be anchored.** `X, Y = 1, 2` has no
  single `ast.Assign`/`ast.AnnAssign` target named `X` or `Y` on its own;
  write it as two separate assignments if it needs a value anchor.
- **A value containing a comma needs a fingerprint anchor instead** — one
  claim comment's targets are split on `,`, so `== (1, 2)` would parse as two
  targets.
- **Anchor placement matters when a fence follows.** Put the anchor **above**
  any `<!-- test:skip reason="..." -->` or `<!-- test:expect-* -->`
  annotation, not between it and the fence it annotates.
  `_annotations.py`'s backward walk stops at the first non-blank,
  non-`test:`-comment line, so a claim anchor sitting between the annotation
  and the fence silently breaks the annotation's link to that fence.
- **"Claim" is overloaded.** `test_doc_claims.py` and `test_claim_coverage.py`
  use "claim" for a different concept entirely — a fence's inferred cache
  hit/miss expectation. That is unrelated to the prose claim anchors this
  section describes.
- **An anchor inside a code fence is an example, not a live claim.** It is
  ignored by the parser, by `--pin`/`--accept`, and by the false-assurance
  guards — write one there only to illustrate the anchor syntax itself, never
  expecting it to be checked or filled in.

Working on source rather than docs? Check what your change touches first:

```bash
python scripts/claims.py --report cash/cost_model.py
```
