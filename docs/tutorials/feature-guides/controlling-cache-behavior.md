# Controlling cache behavior — overriding Cash's defaults per statement

Cash decides what to cache by analyzing your code, but sometimes you need to override that decision — skip a statement entirely, give it a time limit, force it to disk, or silence a warning. This guide covers every knob.

## Why this exists

The notebook layer makes a verdict on every statement: cache it, refuse to cache it, persist it to disk, or just keep it in RAM. The verdict is conservative by design — Cash would rather decline to cache a statement than replay a stale result. But the analyzer doesn't know that *you* know:

- The function with the print statement is fine to cache; the print is debug noise.
- The API call you're caching for 60 seconds shouldn't expire never; you want fresh data each minute.
- The 200 MB model you just trained should hit disk, even though the smart-persistence policy would normally leave it in RAM.
- The `np.random.randn` call is intentional and you don't need a warning every cell run.

Four general-purpose comment annotations and a magic-level TTL cover all of those cases. Two more specialised directives are covered elsewhere: the ML-specific [`# @cash:cache-fit`](../../annotations.md#cashcache-fit-alias-cachefit), and [`# @cash:no-cache-calls`](../../annotations.md#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls), the opt-out for cash's default behavior of caching the *calls inside* a statement instead of just the statement itself — the fix for an accumulator loop that can never cache as a whole. (`# @cash:cache-calls` is the old opt-in spelling of that same feature; it still parses, but does nothing now that the behavior is on by default.) They live as `# @cash:<directive>` comments on or immediately above the statement, and they're picked up by the same parser regardless of whether you came in via `%cash_on` or `%%cash`.

## Quick start

```python { .nb-cell }
import cash
%cash_on ttl=3600  # default: cache entries expire after an hour

# @cash:no-cache
print(df.describe())            # side effect — don't cache

# @cash:ttl=60
price = fetch_stock("AAPL")     # re-fetch every minute, ignores the global hour

# @cash:persist
model = train_xgb(X, y)         # 12 min to fit — force to disk

# @cash:allow-random
noise = np.random.rand(1000)    # we know it's unseeded; don't warn us
```

<!-- claim: cash/notebook/annotations.py:parse_annotation_line @519e5ecd, cash/notebook/annotations.py:ANNOTATION_PATTERN @95980cce -->
That's the everyday language — six directives in total, counting the two specialised ones above. Stack annotations on consecutive lines above a statement (Cash walks backwards through comment lines until it hits a blank or a non-comment).

## The four annotations

### `@cash:no-cache` — skip caching entirely

Use when the statement is non-deterministic, side-effectful, or just cheaper to recompute than to look up.

```python { .nb-cell }
# test:inject: import requests
# @cash:no-cache
current_time = datetime.now()

# @cash:no-cache
api_response = requests.get("https://api.example.com/data")

# @cash:no-cache
print(f"Debug: {some_value}")
```

<!-- claim: cash/notebook/cacheability_decision.py:decide_cacheability @894ac130 -->
The decision-merge layer short-circuits as soon as it sees this annotation — `decide_cacheability` returns `(False, ['@cash:no-cache annotation'])` before consulting anything else. The badge shows the statement as NOT CACHED with that exact reason string.

<iframe class="cash-badge" src="/_badges/not_cached_explicit.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

### `@cash:ttl=<seconds>` — give it an expiration date

Cache the result, but discard it if it's older than `<seconds>` next time you ask.

```python { .nb-cell }
# @cash:ttl=60
fast_changing = get_stock_price("AAPL")    # one minute

# @cash:ttl=3600
hourly_report = generate_summary(df)       # one hour

# @cash:ttl=86400
daily_data = fetch_daily_metrics()         # one day
```

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._parse_annotation @70e15ddd, cash/core.py:Cash._validate_ttl @98fd97a4 -->
The annotation TTL overrides the global TTL set by `%cash_on ttl=N` or `%%cash ttl=N`. `_parse_annotation` does the merge: if `annotation.ttl is not None`, the effective TTL becomes that value; otherwise the global TTL applies.

The check itself is in `Cash._validate_ttl`: on a lookup hit, `_validate_ttl` compares `time.time() - metadata['timestamp']` against the TTL and raises `CacheExpiredError` when it's stale. Stale entries fall through to recompute as if the cache had missed.

### `@cash:persist` — force it onto disk

Cash's default tiered backend (`InMemoryBackend` over `FileBackend`) uses a promotion policy that only writes through to disk when the execution-time-times-savings math works out. The default cut-off is a **0.1 s** compute floor *and* re-execution slower than re-reading (a conservative 1.0 s floor applies only when smart persistence is turned off). See [Smart Persistence](smart-persistence.md) for the full policy.

When you know better — anything that takes more than a few seconds to recompute and you can't afford to lose to a kernel crash — `@cash:persist` overrides the policy:

```python { .nb-cell }
# @cash:persist
model = train_neural_network(X, y)         # 15 min — save it

# @cash:persist
embeddings = compute_embeddings(corpus)    # 2 GB of vectors — persist them
```

The annotation sets `force_persist = True`, which the post-execute path threads into the tiered backend so promotion runs unconditionally.

### `@cash:allow-random` — accept non-reproducibility

<!-- claim: cash/notebook/randomness.py:check_and_warn_randomness @acebcb9b, cash/notebook/randomness.py:MODULE_ALIASES @1d79cda4, cash/notebook/randomness.py:RANDOM_FUNCTIONS @928168d0 -->
Cash scans every statement for unseeded calls to known RNG functions (`numpy.random.randn`, `torch.rand`, `random.choice`, dozens more — full list in `RANDOM_FUNCTIONS`) and raises a `CashRandomnessWarning` when it finds one. The reasoning: a cached `np.random.rand(1000)` won't match what a fresh re-execution would produce, so cache hits are silently non-reproducible.

Two fixes. Seed it:

```python { .nb-cell }
np.random.seed(42)
noise = np.random.rand(1000)  # no warning, fully reproducible
```

Seeding is tracked per module for the rest of the session, so one `np.random.seed(42)` quiets every later `np.random.*` draw — but not a `random.random()` one.

Or, if non-reproducibility is exactly what you want (you're exploring, you'll re-roll deliberately), suppress the warning:

```python { .nb-cell }
# @cash:allow-random
noise = np.random.rand(1000)
```

The annotation flips `suppress_warning=True` in `check_and_warn_randomness`; **the cell still caches either way.** `allow-random` is advisory — it changes what Cash *says*, never what it *stores*. Unseeded randomness has never blocked caching, and adding the annotation doesn't opt you out of it; if you want the statement to re-run every time, use `@cash:no-cache`.

The warning fires once per statement per session, so a re-run of an unchanged cell won't nag you and a loop won't warn per iteration.

!!! note "What the scanner can and can't see"
    Detection is name-based: Cash recognises calls rooted at a known RNG
    *module* (`np.random.*`, `random.*`, `torch.*`, `tf.random.*`), including
    through aliases and `from ... import`. It does **not** see randomness hiding
    behind a method on your own objects — `df.sample(1000)` and
    `model.fit(X, y)` draw from the global NumPy RNG but produce no warning,
    because Cash can't tell those methods from any other. Treat the warning as a
    helpful catch, not a guarantee that seeded code is the only quiet code.

## RNG state is replayed across cache hits

<!-- claim: cash/notebook/randomness.py:capture_object_rng_states @b4e01b79, cash/notebook/randomness.py:restore_object_rng_states @fa5d1e25 -->
A cache hit restores more than the value. If you hold your own RNG object — an
`np.random.Generator`, an `np.random.RandomState`, or a `random.Random` —
its internal state is captured alongside the cached statement and **replayed**
when that statement is restored. Draws taken *after* a cached statement
therefore match what a full re-run would have produced:

<!-- test:skip reason="harness stubs np.random with _FakeRandomState, which has no default_rng" -->
```python { .nb-cell }
rng = np.random.default_rng(0)
a = rng.random(3)     # cache this statement...
b = rng.random(3)     # ...and b still matches a full re-run
```

Without the replay, restoring `a` from cache would leave `rng` un-advanced and
`b` would silently draw `a`'s numbers. Cash captures
`Generator.bit_generator.state` / `RandomState.get_state()` / `Random.getstate()`
and re-injects it on the hit, so the carrier ends on the same post-state the
original execution left it in. Module-global RNG state (`random`,
`numpy.random`, `torch`) is captured and restored the same way.

Cache entries written before this behaviour existed carry no object-RNG state;
they restore unchanged rather than erroring.

## Global TTL — `%cash_on ttl=N` and `%%cash ttl=N`

Two ways to set a default TTL for every cached statement in scope:

```python { .nb-cell }
%cash_on ttl=3600
# Every statement from now on expires after 1 hour
# unless overridden by @cash:ttl=...
```

```python { .nb-cell }
%%cash ttl=300
# Just this cell — entries expire after 5 minutes.
# Saves and restores the global TTL on entry/exit.
result = compute_something()
```

`%cash_on ttl=N` sets `self._global_ttl` on the magic. `%%cash` parses the same `ttl=N` arg locally and swaps the global TTL in/out around the cell, so the cell-scoped value doesn't leak out.

A per-statement `# @cash:ttl=N` annotation always wins over both: the merge logic in `_parse_annotation` favors the annotation's TTL whenever it's set.

## Function-level controls on `@cash.cache`

The decorator path has its own knobs that mirror some of the annotations:

```python
import cash

c = cash.Cash()

@c.cache(ttl=3600)                 # same as @cash:ttl=3600 on a statement
def fetch_daily_summary():
    return load_yesterdays_metrics()

@c.cache(assume_safe=True)         # silence the impurity warning after audit
def fetch_user(uid):
    return requests.get(f"https://api/{uid}").json()

@c.cache(strict=True)              # fail loudly in CI if analyzer finds issues
def critical_function(x):
    return ...
```

`ttl` here works identically to the statement annotation — `_validate_ttl` is the same code path. `assume_safe` and `strict` are about purity, not freshness; see [Purity Decorators](purity-decorators.md) for the full breakdown. They're mutually exclusive at decoration time.

To waive one statement rather than the function, annotate it — `# @cash:assume-safe` on the audited line. It is honoured under `strict=True` too, and unlike the flag it does not cover code added afterwards.

## Randomness detection — what gets flagged

`RandomnessDetector` keeps a session-wide set of `seeded_modules`. When it sees a seed call (`np.random.seed(42)`, `torch.manual_seed(0)`, `random.seed(...)`), it marks the module as seeded and stops warning about subsequent calls to its RNG functions. When it sees an unseeded call, it emits a `CashRandomnessWarning`.

Tracked module aliases:

- `np` → `numpy` (so `np.random.randn` resolves to `numpy.random.randn`)
- `tf` → `tensorflow`
- Full alias map in `MODULE_ALIASES`; imports detected via `visit_Import` and `visit_ImportFrom` so `import numpy.random as nr` and `from random import choice` both work.

Tracked functions:

- `random.*` — `random`, `randint`, `choice`, `sample`, `shuffle`, …
- `numpy.random.*` — `rand`, `randn`, `choice`, `permutation`, plus distributions like `beta`, `binomial`, `normal`, …
- `torch.*` — `rand`, `randn`, `randint`, `randperm`, `rand_like`, …
- `tensorflow.random.*` / `tf.random.*` — `uniform`, `normal`, `truncated_normal`, …

The full set is `RANDOM_FUNCTIONS`. Anything outside this set isn't checked — third-party RNG libraries are silently allowed.

## Precedence and edge cases

The merge for cacheability has one absolute winner: **`@cash:no-cache` short-circuits everything else**. If it's set, the statement is not cached, full stop — no TTL check, no persist, no purity scan.

After that, the reason-source order is:

1. `@cash:no-cache` annotation
2. Forbidden function calls (e.g. `input()`)
3. `@stateful` function calls
4. In-place mutations / side effects detected by the AST visitor
5. Inputs missing lineage

The first source that triggers wins; later sources are not consulted.

For the annotations that *don't* skip caching:

<!-- claim: cash/notebook/annotations.py:CacheAnnotation.merge @b2421117 -->
- `@cash:persist` + `@cash:ttl=N` compose freely — a statement can be both forced-to-disk and time-limited. `CacheAnnotation.merge` ORs the persist flags and overrides the TTL, so stacking on consecutive lines works:

  ```python { .nb-cell }
  # @cash:persist
  # @cash:ttl=86400
  daily_model = train_lightgbm(X, y)
  ```

- `@cash:allow-random` is purely advisory — it suppresses warnings but does not influence the cacheability decision. You can combine it with anything.

- Per-statement `@cash:ttl=N` overrides the global `%cash_on ttl=N` / `%%cash ttl=N` whenever it's set, even when its value is *longer* than the global (`StatementProcessor._parse_annotation` assigns `effective_ttl = annotation.ttl` whenever it is not `None`).

- A negative or non-integer TTL: the regex captures the whole value (`\S*`) and the parser then requires ASCII digits, so `ttl=-30`, `ttl=abc` and `ttl=5m` set no TTL. They are **not** silent -- each warns and names the directive it could not read. The wide capture is what makes that possible: a `\d+` value group would match only the `5` of `ttl=5m` and silently mean *five seconds*, a 60x error whose only symptom was a cache that kept missing. See [Annotations - common mistakes](../../annotations.md#ttl-with-no-value-or-non-digits).

## API reference

| Annotation | Triggers (regex `#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\S*))?`) | Effect |
|---|---|---|
| `# @cash:no-cache` | directive=`no-cache` (alias `nocache`) | Sets `CacheAnnotation.no_cache=True`. Short-circuits `decide_cacheability` to return `(False, ['@cash:no-cache annotation'])`. |
| `# @cash:ttl=N` | directive=`ttl`, value=`N` (captured wide, then required to be ASCII digits) | Sets `CacheAnnotation.ttl=N`. Overrides global `_global_ttl` for this statement. Checked at lookup time by `_validate_ttl`. |
| `# @cash:persist` | directive=`persist` | Sets `CacheAnnotation.persist=True`. Forces tiered-backend promotion to the persistent tier regardless of the smart-persistence policy. |
| `# @cash:allow-random` | directive=`allow-random` (alias `allowrandom`) | Sets `CacheAnnotation.allow_random=True`. `check_and_warn_randomness` suppresses `CashRandomnessWarning` for the statement. |
| `%cash_on ttl=N` | line-magic flag | Sets `self._global_ttl` on the magic. Applies to every statement unless overridden by `@cash:ttl=...`. |
| `%%cash ttl=N` | cell-magic flag | Swaps `_global_ttl` in for the duration of the cell, then restores it. |
| `@c.cache(ttl=N)` | decorator kwarg | Same TTL semantics, applied to function-level caching. |

All annotation parsing lives in `src/cash/notebook/annotations.py`. The single regex pattern is `ANNOTATION_PATTERN = re.compile(r'#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\S*))?')` — the value group is deliberately wide so a malformed value is *rejected by name* rather than silently truncated.

## Related

- [Annotations](../../annotations.md) — short reference card for every annotation.
- [Purity Decorators](purity-decorators.md) — `@pure`, `@stateful`, and the `assume_safe`/`strict` decorator modes.
- [Smart Persistence](smart-persistence.md) — the default policy that `@cash:persist` overrides.
- [Reading the Cash Badge](../../badges.md) — how each annotation shows up in the badge (skip reasons, TTL expiration).
- [Choosing a backend](choosing-a-backend.md) — where `@cash:persist` actually writes through to.
