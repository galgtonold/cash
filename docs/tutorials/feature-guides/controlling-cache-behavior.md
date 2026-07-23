# Controlling cache behavior — overriding Cash's defaults per statement

Cash decides what to cache by analyzing your code, but sometimes you need to override that decision — skip a statement entirely, give it a time limit, force it to disk, or silence a warning. This guide covers every knob.

## Why this exists

The notebook layer makes a verdict on every statement: cache it, refuse to cache it, persist it to disk, or just keep it in RAM. The verdict is conservative by design — Cash would rather decline to cache a statement than replay a stale result. But the analyzer doesn't know that *you* know:

- The function with the print statement is fine to cache; the print is debug noise.
- The API call you're caching for 60 seconds shouldn't expire never; you want fresh data each minute.
- The 200 MB model you just trained should hit disk, even though the smart-persistence policy would normally leave it in RAM.
- The `np.random.randn` call is intentional and you don't need a warning every cell run.

Four comment annotations and a magic-level TTL cover all of those cases. They live as `# @cash:<directive>` comments on or immediately above the statement, and they're picked up by the same parser regardless of whether you came in via `%cash_on` or `%%cash`.

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

That's the whole language. Stack annotations on consecutive lines above a statement (Cash walks backwards through comment lines until it hits a blank or a non-comment, `src/cash/notebook/annotations.py`).

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

The decision-merge layer short-circuits as soon as it sees this annotation — `decide_cacheability` returns `(False, ['@cash:no-cache annotation'])` before consulting anything else (`src/cash/notebook/cacheability_decision.py`). The badge shows the statement as NOT CACHED with that exact reason string.

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

The annotation TTL overrides the global TTL set by `%cash_on ttl=N` or `%%cash ttl=N`. `_parse_annotation` does the merge: if `annotation.ttl is not None`, the effective TTL becomes that value; otherwise the global TTL applies (`src/cash/notebook/statement/processor.py`).

The check itself is at `src/cash/core.py`: on a lookup hit, `_validate_ttl` compares `time.time() - metadata['timestamp']` against the TTL and raises `CacheExpiredError` when it's stale. Stale entries fall through to recompute as if the cache had missed.

### `@cash:persist` — force it onto disk

Cash's default tiered backend (`InMemoryBackend` over `FileBackend`) uses a promotion policy that only writes through to disk when the execution-time-times-savings math works out. The default cut-off is a **0.1 s** compute floor *and* re-execution slower than re-reading (a conservative 1.0 s floor applies only when smart persistence is turned off). See [Smart Persistence](smart-persistence.md) for the full policy.

When you know better — anything that takes more than a few seconds to recompute and you can't afford to lose to a kernel crash — `@cash:persist` overrides the policy:

```python { .nb-cell }
# @cash:persist
model = train_neural_network(X, y)         # 15 min — save it

# @cash:persist
embeddings = compute_embeddings(corpus)    # 2 GB of vectors — persist them
```

The annotation sets `force_persist = True` (`src/cash/notebook/statement/processor.py`), which the post-execute path threads into the tiered backend so promotion runs unconditionally.

### `@cash:allow-random` — accept non-reproducibility

Cash scans every statement for unseeded calls to known RNG functions (`numpy.random.randn`, `torch.rand`, `random.choice`, dozens more — full list at `src/cash/notebook/randomness.py`) and raises a `CashRandomnessWarning` when it finds one. The reasoning: a cached `np.random.rand(1000)` won't match what a fresh re-execution would produce, so cache hits are silently non-reproducible.

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

The annotation flips `suppress_warning=True` in `check_and_warn_randomness` (`src/cash/notebook/randomness.py`); **the cell still caches either way.** `allow-random` is advisory — it changes what Cash *says*, never what it *stores*. Unseeded randomness has never blocked caching, and adding the annotation doesn't opt you out of it; if you want the statement to re-run every time, use `@cash:no-cache`.

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

`%cash_on ttl=N` sets `self._global_ttl` on the magic (`src/cash/notebook/ipython/magics.py`). `%%cash` parses the same `ttl=N` arg locally and swaps the global TTL in/out around the cell (`src/cash/notebook/ipython/magics.py`), so the cell-scoped value doesn't leak out.

A per-statement `# @cash:ttl=N` annotation always wins over both: the merge logic in `_parse_annotation` favors the annotation's TTL whenever it's set (`src/cash/notebook/statement/processor.py`).

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

`ttl` here works identically to the statement annotation — `_validate_ttl` is the same code path. `assume_safe` and `strict` are about purity, not freshness; see [Purity Decorators](purity-decorators.md) for the full breakdown. They're mutually exclusive at decoration time (`src/cash/core.py`).

## Randomness detection — what gets flagged

`RandomnessDetector` (`src/cash/notebook/randomness.py`) keeps a session-wide set of `seeded_modules`. When it sees a seed call (`np.random.seed(42)`, `torch.manual_seed(0)`, `random.seed(...)`), it marks the module as seeded and stops warning about subsequent calls to its RNG functions. When it sees an unseeded call, it emits a `CashRandomnessWarning`.

Tracked module aliases:

- `np` → `numpy` (so `np.random.randn` resolves to `numpy.random.randn`)
- `tf` → `tensorflow`
- Full alias map at `src/cash/notebook/randomness.py`; imports detected via `visit_Import` and `visit_ImportFrom` so `import numpy.random as nr` and `from random import choice` both work.

Tracked functions:

- `random.*` — `random`, `randint`, `choice`, `sample`, `shuffle`, …
- `numpy.random.*` — `rand`, `randn`, `choice`, `permutation`, plus distributions like `beta`, `binomial`, `normal`, …
- `torch.*` — `rand`, `randn`, `randint`, `randperm`, `rand_like`, …
- `tensorflow.random.*` / `tf.random.*` — `uniform`, `normal`, `truncated_normal`, …

The full set is `RANDOM_FUNCTIONS` at `src/cash/notebook/randomness.py`. Anything outside this set isn't checked — third-party RNG libraries are silently allowed.

## Precedence and edge cases

The merge for cacheability has one absolute winner: **`@cash:no-cache` short-circuits everything else**. If it's set, the statement is not cached, full stop — no TTL check, no persist, no purity scan (`src/cash/notebook/cacheability_decision.py`).

After that, the reason-source order is:

1. `@cash:no-cache` annotation
2. Forbidden function calls (e.g. `input()`)
3. `@stateful` function calls
4. In-place mutations / side effects detected by the AST visitor
5. Inputs missing lineage

The first source that triggers wins; later sources are not consulted (`src/cash/notebook/cacheability_decision.py`).

For the annotations that *don't* skip caching:

- `@cash:persist` + `@cash:ttl=N` compose freely — a statement can be both forced-to-disk and time-limited. The annotation merger ORs persist flags and overrides TTL (`src/cash/notebook/annotations.py`), so stacking on consecutive lines works:

  ```python { .nb-cell }
  # @cash:persist
  # @cash:ttl=86400
  daily_model = train_lightgbm(X, y)
  ```

- `@cash:allow-random` is purely advisory — it suppresses warnings but does not influence the cacheability decision. You can combine it with anything.

- Per-statement `@cash:ttl=N` overrides the global `%cash_on ttl=N` / `%%cash ttl=N` whenever it's set, even when its value is *longer* than the global (`statement/processor.py:591-592`).

- A negative or non-integer TTL: the regex `\d+` only matches non-negative integers (`src/cash/notebook/annotations.py`), and bad values return `None` from `parse_annotation_line` and are silently dropped.

## API reference

| Annotation | Triggers (regex `#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\d+))?`) | Effect |
|---|---|---|
| `# @cash:no-cache` | directive=`no-cache` (alias `nocache`) | Sets `CacheAnnotation.no_cache=True`. Short-circuits `decide_cacheability` to return `(False, ['@cash:no-cache annotation'])`. |
| `# @cash:ttl=N` | directive=`ttl`, value=`N` (digits only) | Sets `CacheAnnotation.ttl=N`. Overrides global `_global_ttl` for this statement. Checked at lookup time by `_validate_ttl`. |
| `# @cash:persist` | directive=`persist` | Sets `CacheAnnotation.persist=True`. Forces tiered-backend promotion to the persistent tier regardless of the smart-persistence policy. |
| `# @cash:allow-random` | directive=`allow-random` (alias `allowrandom`) | Sets `CacheAnnotation.allow_random=True`. `check_and_warn_randomness` suppresses `CashRandomnessWarning` for the statement. |
| `%cash_on ttl=N` | line-magic flag | Sets `self._global_ttl` on the magic. Applies to every statement unless overridden by `@cash:ttl=...`. |
| `%%cash ttl=N` | cell-magic flag | Swaps `_global_ttl` in for the duration of the cell, then restores it. |
| `@c.cache(ttl=N)` | decorator kwarg | Same TTL semantics, applied to function-level caching. |

All annotation parsing lives in `src/cash/notebook/annotations.py`. The single regex pattern is `ANNOTATION_PATTERN = re.compile(r'#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\d+))?')` (`annotations.py`).

## Related

- [Annotations](../../annotations.md) — short reference card for every annotation.
- [Purity Decorators](purity-decorators.md) — `@pure`, `@stateful`, and the `assume_safe`/`strict` decorator modes.
- [Smart Persistence](smart-persistence.md) — the default policy that `@cash:persist` overrides.
- [Reading the Cash Badge](../../badges.md) — how each annotation shows up in the badge (skip reasons, TTL expiration).
- [Choosing a backend](choosing-a-backend.md) — where `@cash:persist` actually writes through to.
