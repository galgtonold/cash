# Using cash — a guide for coding agents

cash caches expensive Python results so slow notebooks and scripts iterate fast.
Reach for it **only** on genuinely slow, deterministic work. This page is the
minimal correct path; it is also what `cash.help()` returns.

## Two front-ends, one engine

- **Notebook** — first cell: `import cash` then `%cash_on`. Every statement caches
  by *(its code + upstream inputs + file deps)*. Editing a cell or any upstream
  input auto-invalidates everything downstream.
- **Script / module** — `@cash.cache` on an expensive function. Caches by
  *(function source + arguments)* and persists to `./.cash` across processes.

## See what cash did — required for headless / agent runs

Make the **second line** `%cash_badge print`. The default badge is an HTML widget
that papermill / nbconvert / headless kernels **strip** — without this you are
blind. The text badge prints under each cell:

```
[Cash] EXECUTED (0.01s, saved 0.42s)
  CACHED: df = load()      EXECUTED: x = f(df)      NOT CACHED: cheap = 1 + 1
```

`CACHED` = served from cache · `EXECUTED` = ran · `NOT CACHED` = ran but not
stored (too cheap, a side effect, or `# @cash:no-cache`). Check the running total
with `%cash_stats` ("Net time saved" — honest, and it will report a **loss** if
you cached something too cheap).

## The six rules that keep you correct

1. **Cache pure + expensive only.** Same code + same inputs → restored. Editing
   the code or an upstream input invalidates automatically.
2. **Cross-process persistence has a ~0.1 s floor.** A result whose compute was
   faster than that stays in RAM only — instant on a repeat *in the same process*,
   but recomputed after a kernel restart or a fresh `python run.py`. Force disk
   with `# @cash:persist`.
3. **Non-determinism is FROZEN, not blocked.** An unseeded random draw is cached
   and replayed forever. **Seed it:** `np.random.seed(0)` for the global functions;
   for a generator pass the seed — `np.random.default_rng(42)` — because seeding
   the global does *not* affect a `default_rng()` draw. Use `# @cash:no-cache` for
   a value that must be fresh every run (timestamps, IDs).
4. **ML: wrap-and-return, never a bare `.fit()`.** `model.fit(X, y)` on its own
   line isn't cached (it mutates in place and returns nothing). Do:
   ```python
   @cash.cache(assume_safe=True)   # fit()'s discarded return trips the purity check; safe here
   def train(X, y):
       m = RandomForestClassifier(random_state=42)
       m.fit(X, y)
       return m
   ```
5. **Side effects run on the FIRST call only**, then the return value replays. A
   purity warning on discarded calls (prints, logging, `.fit()`) is advisory — the
   result is still correct; `assume_safe=True` silences it once you've audited.
6. **Verify with `f.explain(*args)`** (→ `[HIT]` / `[MISS]` + reason) or
   `%cash_stats` — both read through to the real cache. Prefer these over
   `f.cache_info()` in a notebook: its hit/miss counters live on the wrapper
   object, which is recreated every time you re-run the function's `@cash.cache`
   definition cell, so they reset during iteration and mislead.

## Statement annotations

Put the comment **directly above** the statement, no blank line, lowercase `@cash:`:

```
# @cash:no-cache      never cache (timestamps, side effects, values that must be fresh)
# @cash:persist       force-cache a cheap value that must survive a kernel restart
# @cash:ttl=300       expire after N seconds (integer only)
# @cash:allow-random  acknowledge a frozen unseeded draw (silences the warning only)
```

Default: **annotate nothing** — the cost model decides what's worth caching.

## Don't bother caching

Sub-10 ms statements (cache overhead dominates — `%cash_stats` will say "cash cost
you X ms"), pure I/O / network-polling scripts, and cells that are *meant* to
differ each run.

## Silent-failure gotchas (no error is raised)

- **Unhashable argument** (a lock, socket, open file, or a bare `self`) → caching
  silently does nothing. `f.explain(...)` shows the key couldn't be built; fix with
  `cash.register_hasher(SomeType, lambda x: ...)` or pass a hashable identifier.
- **Annotation typo, wrong case, or a blank line above the statement** → the
  directive silently no-ops. Confirm the badge actually changed.
- **Write cache-safe cells:** rebind, don't mutate an object from an *earlier* cell
  (`df = df.assign(c=...)`, not `df['c'] = ...`); seed RNG in the same cell as the
  draw; pass state in and out of functions instead of mutating globals. Full list:
  [Known limitations](https://cash-lib.readthedocs.io/en/latest/known-limitations/).
