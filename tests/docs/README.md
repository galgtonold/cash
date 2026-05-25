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
