# Annotations (`@cash:` directives)

Annotations are `#`-comment directives that tweak Cash's per-statement caching decisions. Drop one on a line near a statement and Cash will pick it up the next time that cell runs — useful for forcing a cache that the cost model would otherwise skip, opting out of caching entirely, or overriding the default TTL for a specific value.

## At a glance

| Directive | Aliases | Takes value | Effect |
|---|---|---|---|
| `# @cash:persist` | — | no | Force-cache this statement, even if the cost model would skip it. |
| `# @cash:no-cache` | `nocache` | no | Never cache this statement. Wins over `persist`. |
| `# @cash:ttl=N` | — | non-negative int (seconds) | Override the default TTL for this statement. |
| `# @cash:allow-random` | `allowrandom` | no | Parsed but **currently inert** — see [below](#cash-allow-random-alias-allowrandom). |

A minimal example:

```python
# @cash:persist
model = train_model(data)        # forced to disk, even if "too cheap"

# @cash:ttl=300
quotes = fetch_quotes("AAPL")    # expires from cache after 5 minutes

# @cash:no-cache
ts = datetime.utcnow()           # never cached
```

## Grammar

Annotations are matched by a single regex at [`src/cash/notebook/annotations.py:33`][regex]:

```python
ANNOTATION_PATTERN = re.compile(r'#\s*@cash:([\w-]+)(?:=(\d+))?')
```

It's applied with `re.search` (not `re.match`), so the directive can appear **anywhere on the line** — including trailing on a normal code line.

A few details that bite people:

- **`@cash:` is case-sensitive.** `# @Cash:persist` is silently ignored. Only the directive *name* after the colon is lower-cased ([`annotations.py:45`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), so `# @cash:PERSIST` works.
- **No space is allowed between `:` and the directive.** `# @cash: persist` (note the space) **silently does nothing**. This is one of the most common mistakes — see [Common mistakes](#common-mistakes).
- **Whitespace before `@cash:` is fine.** `#@cash:persist`, `# @cash:persist`, and `#   @cash:persist` all match.
- **`=N` only accepts digits.** `# @cash:ttl=60` works. `# @cash:ttl=`, `# @cash:ttl=abc`, and `# @cash:ttl=-5` all silently no-op (the regex requires `\d+`).
- **Unknown directives silently drop.** `# @cash:typo` produces no warning and no log line. Spell-check your directives.

## Directives

### `# @cash:persist`

Forces a statement to be cached on disk even when the cost model would normally skip it. Useful when:

- The statement is fast (under `min_execution_time_to_cache_seconds`, default 0.01 s) but you want the cache for downstream determinism.
- You're benchmarking restore overhead and want to guarantee the value is on disk.
- A downstream cell depends on the value being available across kernel restarts.

```python
# @cash:persist
cheap_constant = compute_constants()    # would normally be skipped; now forced
```

Behind the scenes: the parser sets `CacheAnnotation(persist=True)` ([`annotations.py:49`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), and `_parse_annotation` in the statement processor turns that into `force_persist=True` ([`statement_processor.py:566`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement_processor.py)), which bypasses the cost-model skip logic downstream.

If both `persist` and `no-cache` apply to the same statement, **`no-cache` wins** (see [Merging](#merging-multiple-annotations)).

### `# @cash:no-cache` (alias: `nocache`)

Disables caching entirely for a statement. The statement still executes; its result simply isn't stored, and on the next run it executes again.

```python
# @cash:no-cache
now = datetime.utcnow()    # always fresh
```

This is the right directive when a statement has observable side effects or produces values that must always be recomputed (timestamps, monotonic counters, "fire and forget" prints).

Behind the scenes: the cacheability decision short-circuits at [`cacheability_decision.py:71-72`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/cacheability_decision.py):

```python
if annotation is not None and annotation.no_cache:
    return False, ['@cash:no-cache annotation']
```

Because this check runs before any other reason-source, `no-cache` wins over `persist`, the cost model, and anything else.

### `# @cash:ttl=N`

Overrides the default TTL (time-to-live, in seconds) for this statement. The cached value will be treated as expired once `N` seconds have elapsed since it was stored.

```python
# @cash:ttl=300
quotes = fetch_quotes("AAPL")    # cache for 5 minutes
```

Notes:

- `N` is a **non-negative integer**. Negative numbers and decimals are not parsed (`\d+` only). `ttl=0` is accepted as a valid value but means "immediately expired" — every run will be a miss.
- If multiple `ttl=` annotations apply to the same statement, **the last one wins** (see [Merging](#merging-multiple-annotations)).
- TTL only governs *cache freshness*. A statement with `no-cache` won't be cached at all, so its `ttl=` is irrelevant.

Behind the scenes: the annotation sets `CacheAnnotation.ttl` ([`annotations.py:54-58`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), which `_parse_annotation` reads and uses as `effective_ttl` ([`statement_processor.py:564-565`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement_processor.py)).

### `# @cash:allow-random` (alias: `allowrandom`)

!!! warning "Parsed but currently inert"
    The parser recognises `@cash:allow-random` and sets the `allow_random` field on the `CacheAnnotation` ([`annotations.py:52-53`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), but no runtime code reads that field. Specifically, `_parse_annotation` at [`statement_processor.py:554-568`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement_processor.py) only consumes `ttl`, `persist`, and `no_cache`, and the `check_and_warn_randomness` helper that would have honoured it ([`randomness.py:320`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/randomness.py)) is not called from anywhere in `src/`.

    Practically: **the annotation has no effect today.** Unseeded-randomness warnings still appear regardless of whether you add `# @cash:allow-random`. Wiring it up is tracked for a future release.

If you need to suppress randomness warnings today, the most reliable workaround is to seed the RNG explicitly (`np.random.seed(0)`, `rng = np.random.default_rng(0)`) — then Cash treats the call as deterministic and there's nothing to warn about.

## Lookback and scoping

Cash needs to associate each annotation with a specific statement. It does this in [`parse_annotations_in_range`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py) at `annotations.py:62-105`, which walks two directions from a top-level AST node:

### Backward walk

Starting from the line **immediately above** the statement and walking up, the parser collects annotations until it hits a stop condition. For each line:

1. **Blank line** → STOP. The lookback block is over. ([`annotations.py:94-95`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py))
2. **Code line** (non-blank, doesn't start with `#`) → STOP. ([`annotations.py:85-86`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py))
3. **Plain comment line** (`# hello`, no `@cash:`) → CONTINUE walking. Plain comments are *transparent*; they don't break the binding.
4. **Annotation comment** (`# @cash:directive`) → merge into the running annotation, then continue walking.

So this works (the plain comment in the middle is fine):

```python
# @cash:persist
# Train the production model — see ADR-12
# @cash:ttl=3600
model = train_model(data)
```

But this does **not** (the blank line ends the lookback before the annotation is reached):

```python
# @cash:persist

model = train_model(data)    # the annotation does NOT apply
```

### Forward walk

After the backward walk, the parser also scans the lines *inside* the statement's range (`start_line..end_line` inclusive, [`annotations.py:99-103`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)). This lets you put an annotation inside the body of a compound statement:

```python
for i in range(10):
    # @cash:no-cache
    result = compute(i)     # whole for-loop is no-cache
```

Verified by [`tests/test_notebook/test_annotations.py:125-134`](https://github.com/galgtonold/cash/blob/main/tests/test_notebook/test_annotations.py).

Because the forward walk runs *after* the backward walk, an in-body annotation overrides a header annotation when they conflict (see [Merging](#merging-multiple-annotations)).

### Trailing annotations

Because the regex uses `re.search`, an annotation can ride on the end of a code line:

```python
x = expensive_call()    # @cash:persist
```

That's a perfectly valid placement. Don't overuse it — the multi-line form above the statement is usually easier to read.

## Merging multiple annotations

When several annotations apply to a single statement (stacked above, on the line, or inside a compound body), Cash merges them with `CacheAnnotation.merge` ([`annotations.py:19-26`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)):

| Field | Merge rule |
|---|---|
| `persist` | logical OR |
| `no_cache` | logical OR |
| `allow_random` | logical OR |
| `ttl` | "other wins if set" — order-sensitive |

That means:

- Boolean flags are sticky: once `persist=True` shows up in the block, the whole block is `persist=True`.
- **`ttl` order matters**: the *last* `ttl=` the parser sees wins. The forward walk runs after the backward walk, so an in-body `ttl=` overrides a header `ttl=`.
- Mixing `persist` and `no-cache` is allowed, but `no-cache` short-circuits cacheability at [`cacheability_decision.py:71`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/cacheability_decision.py), so `persist` becomes a no-op.

Examples:

```python
# @cash:persist
# @cash:ttl=60
model = train()                  # persist=True, ttl=60

# @cash:ttl=60
# @cash:ttl=3600
model = train()                  # ttl=3600 (last one wins)

# @cash:persist
# @cash:no-cache
model = train()                  # no_cache wins; never cached
```

## Common mistakes

These are the failure modes that produce *no error and no warning* — the annotation just doesn't take effect. Worth memorising.

### Space after the colon

```python
# @cash: persist          # WRONG — silently ignored
model = train()
```

The regex is `#\s*@cash:([\w-]+)`. `\w` does not match a space, so `@cash: persist` doesn't match the directive name capture, and there's no fallback to forgive it. Write `# @cash:persist` (no space after the colon).

### Wrong case for `@cash:`

```python
# @Cash:persist           # WRONG — silently ignored
model = train()
```

The literal `@cash:` is case-sensitive (the regex isn't compiled with `IGNORECASE`). The *directive name* is lower-cased after the match, so `# @cash:PERSIST` works, but the `@cash:` prefix has to be lowercase.

### `ttl=` with no value (or non-digits)

```python
# @cash:ttl=               # WRONG — silently ignored
# @cash:ttl=5m             # WRONG — silently ignored ("5m" isn't all digits)
# @cash:ttl=-30            # WRONG — minus sign isn't a digit
model = train()
```

Use a plain integer in seconds: `# @cash:ttl=300`.

### Typo'd directive name

```python
# @cash:perist             # WRONG — typo, silently ignored
model = train()
```

Unknown directives produce no error. Watch your spelling, especially `persist` and `no-cache`.

### Annotation *below* the statement

```python
model = train()
# @cash:persist            # WRONG — lookback is backward-only from the statement
```

The parser walks backward from the statement looking for annotations *above* it (and forward into compound-statement bodies). An annotation on a sibling line *below* a top-level statement binds to the next statement, not the one above it.

### Blank line between annotation and statement

```python
# @cash:persist

model = train()            # WRONG — blank line broke the binding
```

A blank line terminates the backward walk. Keep annotations directly above the statement (plain comments between them are fine, blank lines are not).

## Edge cases

### Compound statements bind to the whole block

A single annotation on a `for`/`if`/`while`/`try` block applies to the **entire compound statement** — it's one cache entry, not one per iteration or branch.

```python
# @cash:persist
for i in range(10):
    results.append(compute(i))
```

The for-loop as a whole is the cached statement. Verified by [`tests/test_notebook/test_annotations.py:125-134`](https://github.com/galgtonold/cash/blob/main/tests/test_notebook/test_annotations.py).

### Annotation in a string literal

The parser doesn't tokenize — it pattern-matches each line as text. A `# @cash:no-cache` substring inside a Python string literal would technically match the regex and bind to the following statement:

```python
docstring = """
Example:
    # @cash:no-cache
    x = 1
"""
y = 2                       # would get no_cache=True applied
```

This is unlikely to come up in practice (the string literal would have to be on a line that doesn't itself start with a non-comment token before the `#`), but if you're seeing inexplicable `@cash:no-cache annotation` reasons in your badge, check for stray annotation strings in docstrings or test fixtures.

### Extra leading hashes

```python
## @cash:persist            # MATCHES — `re.search` finds `# @cash:` starting at index 1
model = train()
```

Because the regex is applied with `re.search`, `## @cash:persist` still binds — the engine finds `# @cash:persist` starting at position 1. This is harmless and rarely intentional, but worth knowing if you're writing emphasised section comments.

### `ttl=0`

`# @cash:ttl=0` parses (digits are non-negative), and it means "expires immediately". Every run will be a cache miss. Useful for benchmarks, otherwise probably not what you want.

## Source map

For source-diving:

- Grammar regex: [`src/cash/notebook/annotations.py:33`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- `parse_annotation_line`: [`annotations.py:35-60`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- Lookback semantics: [`annotations.py:62-105`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- Merge rules: [`annotations.py:19-26`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- Consumer (statement processor): [`statement_processor.py:554-568`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement_processor.py)
- `no-cache` short-circuit: [`cacheability_decision.py:71-72`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/cacheability_decision.py)
- Tests: [`tests/test_notebook/test_annotations.py`](https://github.com/galgtonold/cash/blob/main/tests/test_notebook/test_annotations.py)

[regex]: https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py
