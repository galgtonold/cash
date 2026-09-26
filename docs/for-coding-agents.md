# For coding agents

cash caches slow Python results so notebooks and scripts re-run fast. Use it
**only** on slow, deterministic work. This page is also what `cash.help()`
returns.

## Two paths

- **Script / module:** put `@cash.cache` on a slow function. A call is keyed
  on its arguments, the function's source and the source of the helpers it
  calls. Every result is written to disk (`.cash/` in the project), so a new
  process gets it back.
- **Notebook:** the first cell is:

  ```
  import cash
  %cash_on
  %cash_badge print
  ```

  To change a setting, add `cash.configure(...)` before `%cash_on`. Nothing
  that does work goes in this cell: statements in the `%cash_on` cell are
  never cached.
  Put imports, paths and constants in the next cell. Each statement below is
  keyed on its code, its input variables and the files it reads; editing a
  cell or an input re-runs what depends on it.

## Headless runs (papermill, nbconvert, CI)

- The default badge is an HTML widget that headless runs strip.
  `%cash_badge print` prints a text badge instead.
- Cash cannot find the `.ipynb` in a headless run. It warns
  `NOTEBOOK-NOT-FOUND` and turns off upstream tracking: a variable defined in
  the `%cash_on` cell then makes every statement that reads it `NOT CACHED`.
  That is why the first cell does no work.

The text badge, one line per statement:

```text title="Output"
[Cash] CACHED (2 restored, 1 ran; 0.01s, saved 0.15s) - 1 not cached
  CACHED: df = pd.read_csv('sales.csv')  (saved 0.07s)
  CACHED: summary = df.groupby('region').sum()  (saved 0.08s)
  NOT CACHED: summary.to_csv('summary.csv')  (0.00s) - Side effect: summary.to_csv() (file_write)
```

`CACHED` = restored · `EXECUTED` = ran (a first run adds `-> RAM` or
`-> RAM+DISK`, where it was stored) · `NOT CACHED` = ran, not stored, with the
reason · `SKIPPED` = nothing to do. `%cash_stats` shows net time saved, and
reports a loss when caching cost time. `cash inspect` in a terminal lists what
is on disk.

**Read the labels, not your prints.** A restored notebook statement replays
the output it printed, so a print marker shows up either way. A cached
`@cash.cache` call prints nothing: the body does not run.

## Rules

1. **Cache pure, slow work only.** Same code and inputs give the stored result.
2. **Notebook statements under 0.1 s stay in RAM**: gone after a kernel
   restart. Under 10 ms they are not cached at all. Force disk with
   `# @cash:persist`. Decorated results always go to disk.
3. **Randomness is frozen, not blocked.** An unseeded draw is stored and
   replayed. Seed it: `np.random.seed(0)`, or `np.random.default_rng(42)` for
   a generator (seeding the global does not affect `default_rng()`). Use
   `# @cash:no-cache` for values that must be fresh (timestamps, IDs).
4. **ML: return the model, never a bare `.fit()`.** `model.fit(X, y)` alone
   changes the model in place and returns nothing to cache. Do:
   ```python
   # fit()'s discarded return trips the purity check; safe here
   @cash.cache(assume_safe=True)
   def train(X, y):
       m = RandomForestClassifier(random_state=42)
       m.fit(X, y)
       return m
   ```
5. **Side effects run on the first call only.** A purity warning is advisory.
   Put `# @cash:assume-safe` on each audited line; `assume_safe=True` waives the
   whole function, including anything added later.
6. **Verify with `f.explain(*args)`** (`[HIT]` / `[MISS]` and why) or
   `%cash_stats`. In a notebook, `f.cache_info()` resets whenever the defining
   cell re-runs.

## Statement annotations (notebook)

The comment goes directly above the statement, no blank line, lowercase:

```
# @cash:no-cache        never cache (timestamps, side effects, IDs)
# @cash:persist         keep a cheap value across a restart (disk)
# @cash:ttl=300         expire after N seconds (integer)
# @cash:allow-random    hide the unseeded-draw warning (still frozen)
# @cash:no-cache-calls  stop caching the slow call inside a statement
# @cash:cache-fit       cache a bare model.fit(X, y) (off by default)
```

Call-level caching is on by default, so a callee with side effects the
analyzer cannot see skips them on a hit. Use `no-cache-calls` when unsure.
Default: annotate nothing.

## Gotchas

- **Unhashable argument** (a lock, socket, open file): cash warns
  `KEY-UNHASHABLE-ARG` and the call runs uncached. Register a hasher with
  `cash.register_hasher(SomeType, lambda x: ...)` or pass a hashable ID.
- **Wrong case or a blank line above an annotation**: it does nothing, with no
  error (a misspelled name warns `ANNOT-UNKNOWN-DIRECTIVE`). Check the badge.
- **Cache-safe cells:** rebind instead of changing an object from an earlier
  cell (`df = df.assign(c=...)`, not `df['c'] = ...`); seed in the same cell
  as the draw; pass state in and out of functions. Full list:
  [Writing cache-safe cells](https://cash-lib.readthedocs.io/en/latest/known-limitations/).
  For `@cash.cache`:
  [Known limitations of `@cash.cache`](https://cash-lib.readthedocs.io/en/latest/decorator-limitations/).

## Related

- [The `@cash.cache` guide](https://cash-lib.readthedocs.io/en/latest/decorator/):
  every parameter, and how to see what cash did.
- [Annotations](https://cash-lib.readthedocs.io/en/latest/annotations/): every
  `# @cash:` comment and what it changes.
- [Warnings](https://cash-lib.readthedocs.io/en/latest/warnings/): every
  warning code, what it means and the fix.
