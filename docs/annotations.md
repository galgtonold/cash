# Annotations (`@cash:` directives)

Annotations are `#`-comment directives that tweak Cash's per-statement caching decisions. Drop one on a line near a statement and Cash will pick it up the next time that cell runs — useful for forcing a cache that the cost model would otherwise skip, opting out of caching entirely, or overriding the default TTL for a specific value.

## At a glance

| Directive | Aliases | Takes value | Effect |
|---|---|---|---|
| `# @cash:persist` | — | no | Force-cache this statement, even if the cost model would skip it. |
| `# @cash:no-cache` | `nocache` | no | Never cache this statement. Wins over `persist`. |
| `# @cash:ttl=N` | — | non-negative int (seconds) | Override the default TTL for this statement. |
| `# @cash:allow-random` | `allowrandom` | no | Suppress the unseeded-randomness warning for this statement. Advisory only — see [below](#cashallow-random-alias-allowrandom). |
| `# @cash:cache-fit` | `cachefit` | no | Opt a bare `estimator.fit(X, y)` in to caching. Off by default — see [below](#cashcache-fit-alias-cachefit). |
| `# @cash:no-cache-calls` | `nocachecalls` | no | Turn off caching the expensive **call inside** a statement. **On by default** — see [below](#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls). |
| `# @cash:cache-calls` | `cachecalls` | no | Legacy. Parses without error but does nothing — call-level caching no longer needs opting in. |

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

Annotations are matched by a single regex at [`src/cash/notebook/annotations.py`][regex]:

<!-- test:skip reason="source-code excerpt: references re module without import" -->
```python
ANNOTATION_PATTERN = re.compile(r'#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\d+))?')
```

It's applied with `re.search` (not `re.match`), so the directive can appear **anywhere on the line** — including trailing on a normal code line.

A few details that bite people:

<!-- claim: cash/notebook/annotations.py:ANNOTATION_PATTERN @8370c735, cash/notebook/annotations.py:parse_annotation_line @7b6bd448 -->
- **`@cash:` is case-sensitive.** `# @Cash:persist` is silently ignored. Only the directive *name* after the colon is lower-cased ([`annotations.py` — `ANNOTATION_PATTERN`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), so `# @cash:PERSIST` works.
- **A space after the colon is fine.** `# @cash: persist` and `# @cash:persist` both match (the pattern allows `\s*` after the colon), as does spacing around `=` — `# @cash:ttl = 60` works.
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

<!-- claim: cash/notebook/annotations.py:parse_annotation_line @7b6bd448, cash/notebook/statement/processor.py:StatementProcessor._parse_annotation @70e15ddd -->
Behind the scenes: the parser sets `CacheAnnotation(persist=True)` ([`annotations.py` — `parse_annotation_line`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), and `_parse_annotation` in the statement processor turns that into `force_persist=True` ([`statement/processor.py` — `StatementProcessor._parse_annotation`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement/processor.py)), which bypasses the cost-model skip logic downstream.

If both `persist` and `no-cache` apply to the same statement, **`no-cache` wins** (see [Merging](#merging-multiple-annotations)).

See [Cost model and smart persistence](cost-model.md) for the full explanation of the persistence decision.

### `# @cash:no-cache` (alias: `nocache`)

Disables caching entirely for a statement. The statement still executes; its result simply isn't stored, and on the next run it executes again.

```python
# @cash:no-cache
now = datetime.utcnow()    # always fresh
```

This is the right directive when a statement has observable side effects or produces values that must always be recomputed (timestamps, monotonic counters, "fire and forget" prints).

<!-- claim: cash/notebook/cacheability_decision.py:decide_cacheability @894ac130 -->
Behind the scenes: the cacheability decision short-circuits at [`cacheability_decision.py` — `decide_cacheability`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/cacheability_decision.py):

<!-- test:skip reason="source-code excerpt: has return outside function" -->
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

<!-- claim: cash/notebook/annotations.py:parse_annotation_line @7b6bd448 -->
Behind the scenes: the annotation sets `CacheAnnotation.ttl` ([`annotations.py` — `parse_annotation_line`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)), which `_parse_annotation` reads and uses as `effective_ttl` ([`statement/processor.py` — `StatementProcessor._parse_annotation`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement/processor.py)).

### `# @cash:allow-random` (alias: `allowrandom`)

When a statement draws from an RNG that hasn't been seeded, Cash raises a
`CashRandomnessWarning` telling you the cached value may not be reproducible:

```
CashRandomnessWarning: Unseeded randomness detected: numpy.random.rand() at line 1.
Cached results may not be reproducible. Consider calling seed() first or use
@cash:allow-random to suppress.
```

`# @cash:allow-random` silences that warning for the statement it applies to:

```python
# @cash:allow-random
noise = np.random.rand(1000)   # no warning
```

!!! info "Advisory only — it does not change caching"
    `allow-random` suppresses a *warning*. It does **not** change whether the
    statement is cached, and it is **not** an opt-out from caching. An unseeded
    random statement is cacheable by default, with or without the annotation —
    that is deliberate, and the annotation only tells Cash you already know.

    If you want the statement to re-run every time, that's a different
    directive: [`# @cash:no-cache`](#cashno-cache-alias-nocache).

### `# @cash:cache-fit` (alias: `cachefit`)

Opts a bare, in-place `estimator.fit(X, y)` statement in to caching. **Without it,
such a statement is not cached** — the badge reads `NOT CACHED` with an *In-place
mutation* reason and the fit re-executes on every run:

<!-- test:skip reason="illustrative: needs a real sklearn estimator and training data" -->
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X, y)                  # NOT cached — re-executes every run

# @cash:cache-fit
clf.fit(X, y)                  # cached; a hit restores the fitted estimator
```

The default is deliberate, and it is the *safe* default in both directions:

- **It cannot cost you anything.** A skipped statement is never serialised, so a
  fit that keeps missing can't be a net loss.
- **It cannot give you a wrong object.** The real `.fit()` runs and mutates the
  receiver, so `model` is genuinely fitted and any alias of it (`backup = model`)
  sees the fit, exactly as plain Python behaves.

Only the **bare-expression** form is affected. An assignment is an ordinary
statement and caches with no directive:

<!-- test:skip reason="illustrative: needs a real sklearn estimator and training data" -->
```python
clf = clf.fit(X, y)                                    # caches
m = RandomForestClassifier(n_estimators=100).fit(X, y) # caches
```

The gate is a duck-type — the method must be `fit`/`partial_fit` and the receiver
must expose a callable `fit` **and** a callable `get_params` — so `lst.append(x)`
and an arbitrary object that merely happens to have a `fit` method are never
swept in.

!!! warning "The identity caveat — read before opting in"
    A cache hit may **rebind** the receiver rather than update it in place. An
    alias taken earlier (`backup = model`) can then be left pointing at the
    pre-fit, unfitted object — a wrong result that looks like a right one.

    Cash restores **per statement**, and that is why this can't simply be fixed:
    on a warm run-all the *constructor* statement's own cache hit rebinds `model`
    before the `fit` statement's in-place transfer runs, so the alias graph is
    already broken upstream of the fit. Per-statement restore cannot preserve
    alias identity compositionally.

    The duck-type gate also admits the whole sklearn-compatible universe
    (xgboost, lightgbm, your own estimator), each with its own `__getstate__`
    contract; some never restore cleanly, which means paying serialisation every
    run for nothing.

    Opt in when the receiver has no aliases and you've confirmed the badge really
    reads `RESTORED` on a warm re-run.

For expensive training, prefer the decorator — wrap the fit in a function that
*returns* the model. It caches on the arguments, has no identity caveat, and is
the recommended ML path:

<!-- test:skip reason="illustrative: needs a real sklearn estimator and training data" -->
```python
@cash.cache(assume_safe=True)   # fit()'s discarded return trips the purity check; safe here
def train_model(X, y, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X_train, y_train)
```

An unseeded opted-in fit warns that the cached model is a frozen replay (the
`.fit()`'s internal randomness is invisible to the AST scanner);
[`# @cash:allow-random`](#cashallow-random-alias-allowrandom) suppresses it.

### Call-level caching (default) and `# @cash:no-cache-calls` (alias: `nocachecalls`)

Cash caches **one level down** by default, from the statement to the
expensive call inside it, not just the statement itself. This is on
automatically — no directive needed — which is exactly what makes it help in
the one place statement-level caching structurally cannot: a cheap wrapper
around slow work.

<!-- test:skip reason="illustrative: `compute` and `items` are the reader's own" -->
```python { .nb-cell }
for x in items:
    results.append(compute(x))     # compute(x) is cached; the append re-runs
```

No annotation appears above that loop. Prior to this being the default, the
same effect required an explicit `# @cash:cache-calls` on the header; that
spelling still parses (so an old notebook doesn't error) but does nothing —
see [Common mistakes](#common-mistakes) if you're migrating a notebook that
still has it lying around and wondering why it's a no-op now.

If you need to turn this off — for a statement whose callee has side effects
the analyzer can't see, or while debugging a caching-related surprise —
`# @cash:no-cache-calls` is the escape hatch:

<!-- test:skip reason="illustrative: `compute` and `items` are the reader's own" -->
```python { .nb-cell }
# @cash:no-cache-calls
for x in items:
    results.append(compute(x))     # compute(x) re-executes every run, like before this feature existed
```

Two shapes benefit, and neither can be fixed at statement level:

- **A call inside an in-place mutation.** `results.append(compute(x))` mutates an
  object that already exists, so cash refuses to cache the statement — there is
  no snapshot that would reproduce an append. By default `compute(x)` is
  cached anyway and the append still executes, which is both faster *and* more
  faithful: the mutation genuinely happens on every run.
- **A call inside an accumulator fold.** `s += compute(x)` reads `s`, so each
  iteration's key encodes every iteration before it and reordering the list
  re-runs everything after the first change. A call cache keys on arguments, not
  on execution history, so it is order-independent by construction. See
  [Reordering a loop's items](known-limitations.md#reordering-a-loops-items-re-runs-the-tail).

**What is eligible.** One rule: a call qualifies when it does **not** read the
statement's assignment or mutation target. If it does, it *is* the fold and
nothing order-independent can be pulled out of it:

| Statement | Cached call |
|---|---|
| `s += compute(x)` | `compute(x)` |
| `out.append(compute(x))` | `compute(x)` |
| `prices[t] = compute(t)` | `compute(t)` |
| `s = merge(s, x)` | none — the call reads `s` |
| `df.sort_values(inplace=True)` | none — the mutation *is* the work |

Calls already wrapped in `@cash.cache` are left alone (they are on this path
already), and builtins are skipped so a hot loop doesn't pay for a cache key per
`len()`.

**Bound methods are deliberately not intercepted.** `model.predict(x)` looks like
an obvious candidate, but caching a method puts `self` in the key, and
[caching class methods](tutorials/feature-guides/caching-class-methods.md)
documents why that needs your judgement rather than cash's guess: an unpicklable
receiver silently fails to cache, a heavy `self.df` gets pickled on every call,
and two logically-identical instances miss each other. Self-mutating methods
(`counter.next()`, `cursor.fetchone()`) would be frozen outright. Decorate the
method yourself, with a `register_hasher` for its type, when you want that.

**If nothing is eligible, nothing happens.** No warning, no error, no badge note
— for most statements there is no expensive sub-call to find, and that is the
overwhelmingly common case under default-on. (An earlier, opt-in version of
this feature warned when a directive matched nothing; that warning no longer
exists, because "matched nothing" stopped being a signal the user did
anything wrong the moment interception became unconditional.)

`# @cash:no-cache-calls` attaches to the statement below it, and the backward
scan stops at the first non-comment line — same rules as every other
directive. To turn interception off for a whole loop rather than one
statement, put it on the **header**:

<!-- test:skip reason="illustrative: contrasts directive placement" -->
```python { .nb-cell }
s = 0
# @cash:no-cache-calls              <- on the header: reaches the body
for x in items:
    s += compute(x)
```

It also propagates from a cell's **leading comment block** — the very first
lines of the cell, before any code — to every top-level statement in that
cell, the same way [`# @cash:no-cache`](#cashno-cache-alias-nocache) does.
That matters more now than it used to: under the old opt-in directive,
forgetting it just meant a statement missed out on a speed-up. Under
default-on, the failure direction inverts — forgetting the *opt-out* on one
statement in a cell where you meant to disable it for all of them means that
one statement stays intercepted, which can matter if its callee is one you
don't trust interception's purity judgement on (see the warning below).

Hits show up on the badge alongside decorated calls — it is the same cache —
but tagged so you can tell an intercepted call from one you hand-decorated
with `@cash.cache`:

```text
  @cash.cache:
    compute() [intercepted]: 2/3 cached (0.402s)
```

!!! warning "On by default — know the risk"
    Cash's statement path judges a statement's callees only against the
    forbidden-function scan and explicit `@stateful` marks. Routing a call
    through the cache applies the *stricter* decorator gate — but it also means
    statements cash previously declined to cache for an unrelated reason (the
    mutation) are purity-judged for the first time, **automatically, for every
    statement**, not just ones you opted in. If `compute` has side effects the
    analyzer can't see, a cached call will skip them **without you asking it
    to**. This risk is not smaller than it was when the feature was opt-in —
    only the default moved. Check the badge shows the hits you expect, and
    reach for `# @cash:no-cache-calls` (per statement or per cell) the moment a
    callee's purity is something you're not sure of.

### The two warnings

Cash raises a *different* warning when it serves you a cached value that came
from an unseeded draw:

```
CashRandomnessWarning: Unseeded randomness restored from cache: numpy.random.rand()
at line 1. The value you are seeing is a replay of an earlier run, not a fresh
draw - re-running will not change it. Use @cash:no-cache to re-run it every time,
seed the RNG for real reproducibility, or @cash:allow-random to suppress.
```

The distinction matters, because they are different claims:

| when | warning | what it means |
|---|---|---|
| the statement is **computed** | `Unseeded randomness detected` | advice about the code you wrote — this *may* not reproduce |
| the value is **restored** | `Unseeded randomness restored from cache` | a fact about the number in front of you — it *is* a replay |

The second is the one that catches a frozen Monte Carlo. On a restore, the value
on screen is not what your code would produce if it ran — so re-running to "see
how much it varies" measures nothing.

!!! warning "A value can be frozen without being cached"

    Caching is not the only thing that freezes a draw. To keep a re-executed
    draw position-correct, cash rewinds the RNG to where the cell started — so
    the statement genuinely re-runs and still returns the same number. A *cheap*
    draw (`r = random.random()`) is under the [persistence
    floor](cost-model.md) and is never cached at all, so it is frozen by the
    rewind alone and only ever raises the first warning, never the "restored
    from cache" one.

    So don't read "no restore warning" as "this varies". If you want a draw to
    genuinely redraw on every run, say so — `# @cash:no-cache` switches off the
    rewind as well as caching, and is the only thing that does.

Both fire **once per statement per session**, not once per run: re-running an
unchanged cell a third time stays quiet, and editing the statement makes it warn
again. A random draw inside a loop body warns once, not once per iteration.
`# @cash:allow-random` suppresses both.

To silence the warning globally rather than per statement, filter the class —
it's exported from the package root and belongs to the `CashWarning` family:

```python
import warnings
import cash

warnings.filterwarnings("ignore", category=cash.CashRandomnessWarning)
```

Seeding the RNG is the other way to silence it, and the better one when you want
reproducibility rather than just quiet.

Cash recognises two ways of being random, because reproducibility is established
differently for each:

**Module globals** — `np.random.rand()`, `random.random()`, `torch.rand()`. Here
seeding is a property of the *module*: after `np.random.seed(0)` or
`random.seed(0)`, Cash treats subsequent draws from that module as deterministic
for the rest of the session and stops warning about them. Tracking is per module
(`random`, `numpy.random`, `torch`, `tensorflow.random`), so seeding
`numpy.random` does not silence a `random.random()` call.

**Generator objects** — `rng = np.random.default_rng()` and the draws you take
off it. Here reproducibility is a property of the *object*, fixed by the
constructor, so Cash reads it from there:

```python
rng = np.random.default_rng()     # unseeded
x = rng.standard_normal(1000)     # warns

rng = np.random.default_rng(42)   # seeded
y = rng.standard_normal(1000)     # no warning
```

The same applies to `np.random.Generator(...)`, `np.random.RandomState(...)` and
`random.Random(...)`. Because a generator owns its state, **`np.random.seed()`
does not silence a `default_rng()` draw** — the two channels are independent, and
seeding the global RNG says nothing about your Generator. Pass the seed to the
generator instead.

!!! warning "Detection is rooted at the RNG, not at the draw"
    Cash finds these by following the generator from where it is constructed, so
    a draw off an object it never saw constructed is **not** detected and caches
    silently. Two cases in particular: `df.sample()`, and a generator reached
    through an attribute (`self.rng.normal()`).

    A generator passed **into a function** *is* followed — both
    `price(rng=default_rng())` and `price(default_rng(), n)` warn, provided cash
    scanned that function's `def` this session and can see it draw off that
    parameter. It stays silent when it cannot be sure: a callee it never scanned,
    one that isn't a plain Python function (a class, a builtin, a `partial`), an
    argument landing in `*args`, or anything after a `*unpacking`. Each of those
    is a missing warning, never a false one.

    Seed explicitly, or use [`# @cash:no-cache`](#cashno-cache-alias-nocache),
    when a statement must genuinely re-run every time. The warning is advisory —
    it never changes what gets cached.

## Lookback and scoping

<!-- claim: cash/notebook/annotations.py:parse_annotations_in_range @f1299a54 -->
Cash needs to associate each annotation with a specific statement. It does this in [`parse_annotations_in_range`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py) which walks two directions from a top-level AST node:

### Backward walk

Starting from the line **immediately above** the statement and walking up, the parser collects annotations until it hits a stop condition. For each line:

1. **Blank line** → STOP. The lookback block is over. ([`annotations.py` — `parse_annotations_in_range`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py))
2. **Code line** (non-blank, doesn't start with `#`) → STOP. ([`annotations.py` — `parse_annotations_in_range`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py))
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

After the backward walk, the parser also scans the lines *inside* the statement's range (`start_line..end_line` inclusive, [`annotations.py` — `parse_annotations_in_range`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)). This lets you put an annotation inside the body of a compound statement:

```python
for i in range(10):
    # @cash:no-cache
    result = compute(i)     # this statement re-runs every iteration
```

Verified by [`test_annotations.py` — `test_for_loop_with_internal_annotation`](https://github.com/galgtonold/cash/blob/main/tests/test_notebook/test_annotations.py).

Because the forward walk runs *after* the backward walk, an in-body annotation overrides a header annotation when they conflict (see [Merging](#merging-multiple-annotations)).

### Scoping inside control structures

**Annotation granularity follows cache granularity.** Cash caches `for` and `if`
bodies *per statement*, so a directive inside one is scoped to the statement it
sits on — not to the whole block:

```python
for i in range(10):
    # @cash:no-cache
    fresh = compute(i)      # re-runs every iteration
    heavy = i * i           # still cached per-iteration
```

To scope a directive to the **whole loop**, put it on the header. Every statement
inside then inherits it:

```python
# @cash:no-cache
for i in range(10):
    fresh = compute(i)      # re-runs
    heavy = i * i           # also re-runs — inherited from the loop
```

`while` and `with` blocks (and a `for` containing `break`/`continue`) execute as a
**single cache unit**, so there is no finer entry for a directive to attach to —
one anywhere inside applies to the whole unit:

<!-- test:skip reason="illustrative loop over undefined convergence state" -->
```python
while not converged:
    # @cash:no-cache
    step()                  # the entire while block is no-cache
```

Directives merge from the outside in: an enclosing structure's directive is
combined with the statement's own using the rules under
[Merging](#merging-multiple-annotations).

!!! note "Directives work at any nesting depth"
    An annotation on a statement inside a loop, an `if`, a `with` or a `try` is
    honoured exactly as it would be at the top level — there is no need to hoist
    code out of a block to annotate it.

### Trailing annotations

Because the regex uses `re.search`, an annotation can ride on the end of a code line:

```python
x = expensive_call()    # @cash:persist
```

That's a perfectly valid placement. Don't overuse it — the multi-line form above the statement is usually easier to read.

## Merging multiple annotations

<!-- claim: cash/notebook/annotations.py:CacheAnnotation.merge @ff86f11c -->
When several annotations apply to a single statement (stacked above, on the line, or inside a compound body), Cash merges them with `CacheAnnotation.merge` ([`annotations.py` — `CacheAnnotation.merge`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)):

| Field | Merge rule |
|---|---|
| `persist` | logical OR |
| `no_cache` | logical OR |
| `allow_random` | logical OR |
| `cache_fit` | logical OR |
| `cache_calls` | logical OR (parsed but inert — see [above](#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls)) |
| `no_cache_calls` | logical OR |
| `ttl` | "other wins if set" — order-sensitive |

That means:

- Boolean flags are sticky: once `persist=True` shows up in the block, the whole block is `persist=True`.
- **`ttl` order matters**: the *last* `ttl=` the parser sees wins. The forward walk runs after the backward walk, so an in-body `ttl=` overrides a header `ttl=`.
- Mixing `persist` and `no-cache` is allowed, but `no-cache` short-circuits cacheability at [`cacheability_decision.py` — `decide_cacheability`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/cacheability_decision.py), so `persist` becomes a no-op.
- `no-cache` also wins over call-level interception: caching the expensive call inside a no-cache statement would honour the letter while breaking the intent, the same reasoning that makes it win over `persist`. You never need to write `no-cache-calls` alongside `no-cache` — the statement-level directive already covers it.

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

### Space after the colon — *not* a mistake

```python
# @cash: persist          # OK — this works
# @cash:persist           # OK — equivalent
# @cash:ttl = 60          # OK — spacing around '=' is allowed too
```

The pattern is `#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\d+))?` — the `\s*` after the colon absorbs the space, so both spellings take effect. (Earlier versions of this page listed the spaced form as a common mistake; that was wrong, and the regex quoted alongside it was stale.)

### Wrong case for `@cash:`

```python
# @Cash:persist           # WRONG — silently ignored
model = train()
```

The literal `@cash:` is case-sensitive (the regex isn't compiled with `IGNORECASE`). The *directive name* is lower-cased after the match, so `# @cash:PERSIST` works, but the `@cash:` prefix has to be lowercase.

### `ttl=` with no value (or non-digits)

```python
# @cash:ttl=               # WRONG — silently ignored
# @cash:ttl=-30            # WRONG — minus sign isn't a digit, silently ignored
model = train()
```

!!! danger "A unit suffix is silently *truncated*, not rejected"
    The value pattern is `(\d+)`, and it is not anchored to the end of the
    directive — so it matches the leading digits and drops the rest:

    ```python
    # @cash:ttl=5m          # parses as ttl=5  -- FIVE SECONDS, not five minutes
    # @cash:ttl=2h          # parses as ttl=2  -- two seconds
    ```

    There is no warning. If you write `ttl=5m` meaning five minutes you get a
    value 60× smaller than you intended, and the only symptom is a cache that
    keeps missing. **TTL is always in seconds and takes digits only** — write
    `# @cash:ttl=300`.

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

### A header annotation covers every statement in the block

A single annotation on a `for`/`if`/`while`/`try` **header** applies to every statement inside the block — each one inherits it:

```python
# @cash:persist
for i in range(10):
    results.append(compute(i))    # persisted on every iteration
```

This is about *which statements the directive reaches*, not cache granularity. Granularity is unchanged: a `for`/`if` body is still cached per statement (and a `for` per iteration), while `while`/`with` execute as a single unit — see [Scoping inside control structures](#scoping-inside-control-structures).

### A cell-header opt-out does not reach past an intervening statement

`# @cash:no-cache` and `# @cash:no-cache-calls` also propagate from a **cell's
leading comment block** — the very first lines of the cell, before any real
code — to every top-level statement in that cell. That is a *different*
mechanism from the control-structure header inheritance above, and the two
don't compose the way you might expect:

<!-- test:skip reason="illustrative: `compute` and `items` are the reader's own" -->
```python
# @cash:no-cache-calls
out = []                          # <- reached by the cell-header mechanism
for x in items:
    out.append(compute(x))        # <- NOT reached: this loop's own header
                                   #    annotation is resolved separately, by
                                   #    scanning locally upward from `for`,
                                   #    and that scan stops at `out = []`
```

The loop's own annotation is resolved by a **local backward scan starting at
the `for` line**, not by consulting the cell header — so it stops the moment
it hits `out = []`, the same as it would stop at any other code line. The
cell-header opt-out reached `out = []` (a plain top-level statement) but
never reached the loop at all.

**What to do:** put the opt-out directly above the statement or loop you mean
to cover, with nothing in between:

<!-- test:skip reason="illustrative: `compute` and `items` are the reader's own" -->
```python
out = []
# @cash:no-cache-calls
for x in items:
    out.append(compute(x))        # now covered
```

Or, if the intent really is "nothing in this cell should be intercepted",
make the loop the first thing in the cell so the two mechanisms coincide.

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

- Grammar regex: [`src/cash/notebook/annotations.py`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- `parse_annotation_line`: [`annotations.py` — `parse_annotation_line`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- Lookback semantics: [`annotations.py` — `parse_annotations_in_range`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- Merge rules: [`annotations.py` — `CacheAnnotation.merge`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py)
- Consumer (statement processor): [`statement/processor.py` — `StatementProcessor.process_statement`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/statement/processor.py)
- `no-cache` short-circuit: [`cacheability_decision.py` — `decide_cacheability`](https://github.com/galgtonold/cash/blob/main/src/cash/notebook/cacheability_decision.py)
- Tests: [`tests/test_notebook/test_annotations.py`](https://github.com/galgtonold/cash/blob/main/tests/test_notebook/test_annotations.py)

## See also

- [Controlling cache behavior](tutorials/feature-guides/controlling-cache-behavior.md) —
  end-to-end walkthrough of the annotation directives alongside `cache_if=`,
  `ttl=`, and `force_persist` in a realistic notebook.

[regex]: https://github.com/galgtonold/cash/blob/main/src/cash/notebook/annotations.py
