# Docs feature-parity tests

These tests run every Python code fence in the Cash tutorials as a real
Python program and assert that the documented cache hit/miss behavior
actually holds.

## How it works

For each markdown file under `docs/tutorials/feature-guides/` (and later
under `use-cases/`), `tests/docs/test_tutorials.py::test_doc_page`:

1. Extracts every ` ```python ` fence in source order (`_harness.extract_fences`)
2. Skips fences annotated with `<!-- test:skip reason="..." -->`
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

PR1 ships with `continue-on-error: true` in `.github/workflows/ci.yml`
so the docs-parity job's red builds don't block other PRs while the
harness stabilizes. Flip to `false` after ~2 weeks of green runs.

## Scope rollout

- **PR1 (this PR):** harness + 3 proof-of-harness pages (Custom
  Hashers, Dynamic Dependencies, Async Caching). All plain Python; no
  external-service mocks needed.
- **PR2 (next):** expand to all 13 feature guides. Add `numpy` /
  pandas / file fixtures as needed.
- **PR3 (later):** use cases. Add `mock_anthropic`, `mock_openai`,
  `mock_redis_backend`, `mock_s3`, `sample_customers_csv` fixtures.
  Wire nb-cell pages through `tests/test_notebook_integration/conftest.py::KernelPool`.

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
`ForLoopHandler._should_execute_loop_as_single_unit`. It moves with the code,
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

### Limitations

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
python scripts/claims.py --report cash/notebook/cost_model.py
```
