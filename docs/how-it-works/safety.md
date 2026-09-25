# Knowing when *not* to cache

!!! info "Applies to: both paths"
    Anyone who wants to know why Cash warned about a function or refused to cache a statement.

Some results should not simply be replayed: the code changed a value in place,
or it did something besides computing a value, such as writing a file or
sending a request. The two engines handle this differently.

=== "Decorator"

    The decorator caches and warns. A function that writes a file or sends a
    request is cached like any other, with an
    [`IMPURE-SIDE-EFFECTS`](../warnings.md#impure-side-effects) warning, and a
    hit skips the effect. Network reads get a
    [`KEY-NETWORK-READ`](../warnings.md#key-network-read) advisory that `ttl=`
    silences. A body that picks code from a run-time value (`eval`, `exec`,
    `getattr(obj, name)()`, `importlib.import_module`) raises
    `CashImpureFunctionError`. [How `@cash.cache`
    decides](decorator-path.md#side-effects) has the details, and the
    [decorator guide](../decorator.md#side-effects) covers `assume_safe`.

=== "Notebook"

    The notebook refuses. A statement that changes a variable it does not
    produce, or that writes files, sends requests, reads the clock or asks for
    input, is not cached: it runs every time, and the badge says why. The rest
    of this page explains how Cash decides. The full list of what is cached,
    refused or keyed is in the [notebook guide](../notebook_caching_api.md#what-gets-cached).

Unseeded randomness is the exception on both paths: it is cached, and a cached
draw keeps its first value.

## In a notebook

### The mutation problem

A stored value is a snapshot. If a statement changes a value in place, the
snapshot and the live object drift apart:

<!-- test:skip reason="illustrative pseudo-code (Cell 1/Cell 2 separators)" -->
```python
# Cell 1
data = [1, 2, 3]      # stored: [1, 2, 3]

# Cell 2
data.append(4)        # data is now [1, 2, 3, 4]; the stored copy is not
```

<!-- claim: cash/analysis/mutations.py:MutationVisitor @7c19f318, cash/analysis/mutations.py:MUTATING_METHODS @245ce55b broad="the table lists every pattern the visitor detects" -->
What decides the verdict is whether the statement also *produces* the variable
it changed. If it does, the change is a new version of that variable: it is
stored, its lineage advances, and the statement caches. If it changes some
other variable, there is nothing to store the new version under, so the
statement is not cached.

| Pattern | Example | Verdict |
|---------|---------|---------|
| Augmented assignment | `total += 1` | Cached: `total` is the statement's output |
| Item or attribute store | `d['k'] = v`, `obj.attr = v` | Cached: `d` or `obj` is the output |
| pandas in place | `df.dropna(inplace=True)` | Cached: `df` is the output |
| Known mutating method | `data.append(4)`, `d.update(o)`, `lst.pop()` | Not cached |
| Other method call that changed its object | `bus.on(fn)`, `model.fit(X, y)` | Not cached |
| `del` of an item | `del d['k']` | Not cached |
| numpy `out=` | `np.add(a, 1, out=a)` | Not cached |

<!-- claim: cash/analysis/mutations.py:selfref_inplace_write_vars @c8102f53 -->
The "Cached" rows hold only when the variable was made in the same cell. A
variable that comes into the cell from above and is changed in place is reset
to its value at the start of the cell, so the statement that changes it runs
again. See
[mutating an object created in an earlier cell](../known-limitations.md#mutating-an-object-created-in-an-earlier-cell).

#### Method calls

<!-- claim: cash/analysis/mutation_effects.py:classify_receivers @704f9e6f, cash/analysis/mutations.py:KNOWN_PURE_METHODS @b44508ae, cash/analysis/mutations.py:chain_is_pure @96104373 -->
A method call has no assignment target, so Cash classifies its object:

- **Not a change.** A call on a module (`np.mean(x)`, `time.sleep(1)`), a call
  that only writes a file from the object (`df.to_csv(path)`), and methods on
  the known-pure list (`head`, `describe`, `mean`, `groupby`, `plot`, ...).
  A chain that passes through a pure method is pure, because what follows acts
  on the new object: `df[mask].groupby("hour").size()` leaves `df` alone. A
  module setting such as `plt.style.use(...)` or `pd.set_option(...)` counts
  as changing the module, so it is replayed after a restart.
- **Always a change.** Any call on a live matplotlib `Figure` or `Axes`, or
  passing one to a call (`df.plot(ax=ax)`, `draw(axes[0], df)`), draws on it.
  A call that fits an estimator (`fit`, `fit_transform`, `fit_predict`, ...)
  changes the estimator, even in `X = vec.fit_transform(texts)`. These
  statements run every time.
- **Observed.** Anything else is fingerprinted before and after the call. A
  DataFrame, array or large collection that can only be sampled cannot be
  proved unchanged, so it counts as changed. Each variable passed by name to a
  bare call (`im.add_qc(df)`) is fingerprinted in full the same way.

A changed object gets a new lineage from the statement, so everything
downstream of it misses, and the statement itself is not cached. Restoring it
would rebind the name to a copy, while every other reference to the object
kept the unchanged original.

Inside a loop or `if`/`try` body, the loop or branch as a whole owns the
changes its body makes. Drawing on a `Figure`/`Axes` and fitting an estimator
still run every time there.

An accumulator loop (`out = []`, then `for e in it: out.append(slow(e))`) does
not cache as a statement, but the expensive call inside it does, so a re-run
reuses every element already computed. `out = [slow(e) for e in it]` assigns
its result and caches as a whole.

<!-- claim: cash/analysis/annotations.py:CacheAnnotation.cache_fit == False -->
A bare `model.fit(X, y)` runs every time. `# @cash:cache-fit` on the line
stores the fitted state and restores it into the existing estimator. To cache
training reliably, put it in a function that returns the model and move it
into a module; see
[moving to a module](../tutorials/feature-guides/production-transition.md).

### Side effects

<!-- claim: cash/effects.py:METHOD_VERBS @49934ce1, cash/effects.py:is_open_write_mode @fa37e14b, cash/effects.py:MODULE_CALLS @7f115c19 -->
Cash spots side effects from the statement's source, without running it. So:

- `open(p, "w")` counts, but `open(p, mode)` does not, because the mode is
  only known at run time.
- Write methods are matched by name on any object: `obj.save(x)` counts even
  on an object Cash knows nothing about, while `obj.write_thing(x)` does not.
  `rename` and `replace` are left off the list, because `str.replace` would
  match them.
- Reads are not side effects: `requests.get(url)`, `open(p)` for reading and a
  literal `SELECT` query cache normally. Writing to the console
  (`sys.stdout.write`) counts as a `print`, not a file.
- A call to a function of yours whose body writes a file counts as a write
  (`save(fig, "chart.png")`), whether it is defined in the notebook or in a
  module of your project (`helpers.save(fig, "chart.png")`), and so does one
  that writes through another of your functions. A function of an installed
  package is not looked into, and neither is a method called on an object
  (`report.build()`): put `# @cash:no-cache` on such a statement.

<!-- claim: cash/analysis/code_analyzer.py:_forbidden_call @8d78391d, cash/notebook/lineage_formula.py:statement_environment_component @d70a1c80 -->
A statement that reads the clock (`time.time()`, `datetime.now()`), makes a
fresh id (`uuid.uuid4()`) or asks for input (`input()`, `getpass.getpass()`)
runs every time too. A statement that reads an environment variable by name
(`os.getenv("TENANT")`) or `os.getcwd()` is cached, with a digest of the value
in its key. A read inside a function the statement calls is not seen.

When a side effect is harmless to skip, put `# @cash:assume-safe` on the
statement: it is cached, and a hit skips the call. It waives side effects
only; a change in place, the clock and `input()` still make the statement run.

<!-- claim: cash/analysis/file_effects.py:statement_write_repeatability @5af2a939 -->
A statement that wrote a file is re-run when a later statement reads that file
and it is out of date. Cash tells repeatable writes (`to_parquet`, `savefig`,
which replace the file) from appending ones (`open(p, "a")`,
`to_csv(p, mode="a")`), so rebuilding does not duplicate rows in an
append-mode log.

### Unseeded randomness

<!-- claim: cash/tracking/randomness/detect.py:RANDOM_FUNCTIONS @5801a3eb, cash/tracking/randomness/detect.py:RandomnessDetector.is_seeded @9ff99734 -->
Cash spots draws from `random`, `numpy.random`, `torch` and
`tensorflow.random`. An unseeded draw is cached with a
[`RANDOM-UNSEEDED`](../warnings.md#random-unseeded) warning, and the first
value is kept on every re-run. Once a module is seeded with `seed()`, its later
draws are reproducible and do not warn. A generator object
(`rng = np.random.default_rng()`) counts as seeded only if it was built with a
seed; seeding `np.random` says nothing about it.

<!-- claim: cash/tracking/randomness/state.py:capture_rng_state @421bfe05 -->
A draw too cheap to cache is held too when it comes from the `random`,
`numpy.random` or `torch` module stream: before a re-run, Cash rewinds those
streams to where the cell started, so the draw repeats. Cash does not rewind a
generator held in a variable, so a cheap draw from `rng` changes on every run.

<!-- claim: cash/notebook/statement/randomness.py:StatementRandomness.warn_unseeded @79868eb7 -->
`# @cash:allow-random` silences the warning and changes nothing else.
`# @cash:no-cache` on the statement draws fresh on every run, with no warning.
The badge marks the row `seed`, `random` (a seeded draw) or `unseeded`. A cached fit (`# @cash:cache-fit`) of an estimator with
`random_state=None` warns the same way.

### Values that cannot survive a round trip

<!-- claim: cash/notebook/statement/derivation_edges.py:is_uncacheable_alias @2e425a0f, cash/analysis/cacheability_decision.py:identity_coupled_reason @77bfb1cc -->
Two kinds of value are refused after the statement runs, because restoring a
copy would break them: a view of another variable (a numpy slice, a pandas
`groupby` object), which would come back detached from its base, and a
matplotlib `Figure`/`Axes`, which would come back detached from pyplot.

<div class="cash-cacheability-checker" markdown="0">
  <table>
    <thead><tr><th>Statement</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td><code>df = pd.read_csv('data.csv')</code></td><td>Cached — the file is tracked</td></tr>
      <tr><td><code>result = df.groupby('k').sum()</code></td><td>Cached — nothing is changed in place</td></tr>
      <tr><td><code>total += 1</code></td><td>Cached — the change is the statement's own output</td></tr>
      <tr><td><code>data.append(4)</code></td><td>Not cached — changes a variable it does not produce</td></tr>
      <tr><td><code>del lookup['stale']</code></td><td>Not cached — a deletion with no output</td></tr>
      <tr><td><code>x = np.random.randn(100)</code></td><td>Cached + warning — unseeded</td></tr>
      <tr><td><code>model.fit(X, y)</code></td><td>Not cached unless <code>@cash:cache-fit</code></td></tr>
      <tr><td><code>df.to_parquet('out.pq')</code></td><td>Not cached — writes a file</td></tr>
      <tr><td><code>r = requests.post(url, json=payload)</code></td><td>Not cached — sends a request</td></tr>
      <tr><td><code>r = session.post(url, json=payload)</code></td><td>Not cached — the same request through a client object</td></tr>
      <tr><td><code>r = requests.get(url)</code></td><td>Cached — a read</td></tr>
      <tr><td><code>tenant = os.getenv('TENANT')</code></td><td>Cached — the value is part of the key</td></tr>
    </tbody>
  </table>
</div>
