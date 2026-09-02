# Knowing when *not* to cache

Cash's first rule is **never serve a wrong answer**. When it cannot prove a
statement is safe to replay, it re-executes instead of guessing. This page
walks the three things Cash watches for — mutations, side effects, and
unseeded randomness — and shows the verdict it reaches for real snippets.

Two of the three can refuse a cache lookup outright. Randomness never does:
an unseeded draw is cached *by design*, and the badge says so.

## The mutation problem

A cached value is a *snapshot*. If a statement mutates a value in place, the
snapshot and the live object can drift apart:

<!-- test:skip reason="illustrative pseudo-code (Cell 1/Cell 2 separators)" -->
```python
# Cell 1
data = [1, 2, 3]      # cached snapshot: [1, 2, 3]

# Cell 2
data.append(4)        # data is now [1, 2, 3, 4] — but the snapshot still says [1, 2, 3]
```

<!-- claim: cash/notebook/cacheability.py:_MutationVisitor @0eefc402, cash/notebook/cacheability.py:StatementAnalysis.skip_reasons @0d07d966 broad="the claim is about the visitor's whole set of visit_* patterns, not one of them" -->
Cash answers two questions about every statement, in that order:

1. **Does it mutate something?** — a pure-AST scan (`analyze_statement`), plus
   a runtime check for method calls the AST cannot classify.
2. **Does it also *produce* the thing it mutated?** — i.e. does the statement
   appear, to the dependency analyzer, to be the definition of that variable?

The second question is what decides the verdict. A mutation Cash can attribute
to the statement's own output is a new *version* of that variable: the value is
captured, the variable's lineage advances, and the statement caches normally. A
mutation of some *other* variable has nowhere to hang that new version, so the
statement is refused and re-executes every run.

<!-- claim: cash/notebook/cacheability.py:MUTATING_METHODS @a555babd, cash/notebook/cacheability.py:PANDAS_INPLACE_METHODS @5345187b, cash/notebook/cacheability.py:_MutationVisitor @0eefc402 broad="the table enumerates every pattern the visitor detects; a new visit_* method is a missing row" -->
| Pattern | Example | How it's detected | Verdict |
|---------|---------|-------------------|---------|
| Augmented assignment | `total += 1` | `ast.AugAssign` node | **Cached** — `total` is the statement's output |
| Subscript store | `d['k'] = v`, `arr[0] = 1` | `ast.Assign` with an `ast.Subscript` target | **Cached** — the base is the output |
| Attribute store | `obj.attr = v` | `ast.Assign` with an `ast.Attribute` target | **Cached** — the base is the output |
| Pandas in-place | `df.dropna(inplace=True)` | `inplace=True` keyword | **Cached** — the receiver is the output |
| Known mutating method | `data.append(4)`, `lst.pop()`, `d.update(o)` | name in `MUTATING_METHODS` | **Not cached** |
| Any other method call on a live object | `bus.on(fn)`, `model.fit(X, y)` | runtime content observation (below) | **Not cached** when it mutated |
| `del` on a subscript | `del d['k']` | `ast.Delete` with an `ast.Subscript` | **Not cached** |
| NumPy `out=` target | `np.add(a, 1, out=a)` | `out=` keyword | **Not cached** |

The split looks arbitrary until you write the two forms side by side.
`d['k'] = v` has a *store target*, so the analyzer already lists `d` among the
statement's outputs; `d.update(o)` is a bare expression with no target at all.
The first can be re-derived from the statement that made it; the second cannot.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker.check_and_reexecute @20807ce5, cash/notebook/cacheability.py:selfref_inplace_write_vars @5fcce56c -->
!!! note "…but only when the base was made in the same cell"
    The **Cached** verdicts above are this classifier's per-statement decision.
    A separate rule sits on top, in the upstream checker: a variable the cell
    receives as an *input* and then writes in place is reset to its cell-entry
    value before the cell runs, so the writing statement re-executes rather than
    restoring. A variable created in the same cell is not an input, so nothing
    resets it and the statement caches.

    So `df['x'] = …` caches when `df` was built in the same cell and re-runs
    when `df` came from upstream — measurable as a 1.0 s recompute versus a
    0.01 s restore for the identical statement. See
    [Known limitations](../known-limitations.md#mutating-an-object-created-in-an-earlier-cell).

### Method calls: what the AST can't see

`data.append(4)` is easy — `append` is on the known-mutating list. But
`bus.on(handler)` or `tracker.record(x)` could do anything, and a method call
has no store target to give the receiver a fresh lineage. So Cash classifies
method-call receivers in tiers, in this order:

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._classify_method_mutations @4528ef14, cash/notebook/cacheability.py:KNOWN_PURE_METHODS @6ebfef8a, cash/notebook/cacheability.py:RECEIVER_READONLY_WRITE_METHODS @d0495812, cash/notebook/statement/processor.py:StatementProcessor._receiver_observable @81f477db -->

- **Excluded outright.** A module receiver is a plain function call, not a
  mutation: `np.foo()`, `time.sleep()`, `plt.title()`. So is a receiver-pure
  writer — `df.to_csv(path)` *reads* the frame and writes a file, so it must
  never bump `df`'s lineage — and so is anything on the known-pure list
  (`head`, `describe`, `value_counts`, `plot`, …).
- **Always mutating.** A method call on a live matplotlib `Figure`/`Axes` draws
  on it whatever it returns. This tier is tested *before* the known-pure list,
  which is what makes `ax.hist(...)` behave exactly like `ax.bar(...)` even
  though `hist` is itself a known-pure name on any other receiver.
- **Observed.** Everything else is content-hashed before and after the call. If
  the content changed, the receiver mutated. Receivers that can only be
  *sampled* rather than hashed whole (DataFrames, Series, ndarrays, collections
  over 200 elements) can't be proved unchanged, so they are assumed to mutate.

When a receiver is classified as mutating, Cash does two things: it adds the
receiver to the statement's outputs — so the receiver's lineage advances from
this statement's source, and downstream consumers see a changed input — and it
**skips the cache** for the statement, so the mutated object is never
round-tripped through serialization.

The observed verdict is recorded per statement source and read back by the
upstream simulation, which replays cells without executing them and therefore
cannot observe anything itself. Both sides compute the bumped lineage from the
same source text, so a simulated restore and a live run agree.

One exception: statements inside a loop or `if`/`try` body are *not* classified
this way. The simulation treats a control structure as a single unit, so
bumping a body statement's receiver from a per-statement source would desync
the two. The control structure owns its body's mutation lineage instead.

??? question "Why skip the cache if the lineage is already bumped?"
    The lineage bump and the cache skip answer different questions. Bumping
    tells *downstream* cells that `data` changed, so a consumer cached against
    the old `data` misses. Skipping is about `data` itself: a hit would rebind
    the name to a **deserialized copy**, and every other reference to the
    original object — an alias, an attribute on some other object, an entry in
    a list — would keep pointing at the un-mutated original. Re-running
    `data.append(4)` costs microseconds; getting object identity wrong costs
    a silently wrong notebook.

    This does mean the accumulator *statement* itself (`out = []` then
    `for e in it: out.append(slow(e))`) never caches — there is no output to
    cache, only a mutation. The loop is not a total loss, though: by default
    Cash caches the expensive **call inside** it (`slow(e)`, not the `append`
    around it), so re-running the loop reuses every element it has already
    computed. See [Call-level
    caching](../annotations.md#call-level-caching-default-and-cashno-cache-calls-alias-nocachecalls).

    Cash says so in the badge and points at the rewrite that caches the loop
    as a whole: `out = [slow(e) for e in it]` assigns its result, so it has an
    output and caches like any other statement.

### A bare `model.fit(X, y)`

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._estimator_fit_receivers @8efc6b4a, cash/notebook/annotations.py:CacheAnnotation.cache_fit == False -->
A bare fit is a method-call mutation of its receiver, so it takes the default
path above: **skip-cache, re-execute every run**. That is net-neutral — a fit
that would keep missing cannot cost more than it saves — and it avoids the
identity trap, where a cache hit rebinds `model` and leaves `backup = model`
pointing at the pre-fit object.

Caching the fit is available behind an opt-in annotation:

<!-- test:skip reason="requires sklearn and a live notebook kernel" -->
```python
# @cash:cache-fit
model.fit(X_train, y_train)
```

With the annotation, the fitted state is cached and restored **in place** onto
the existing estimator. Without it, nothing about a fit is cached. For reliable
ML caching, wrap training in a function that *returns* the model and decorate it
with `@cash.cache` — see [The decorator path](decorator-path.md).

## Side effects

Some statements don't just compute a value — they *do something to the world*.
Replaying them from cache would skip the action (a file never gets written, a
request never gets sent). Cash's side-effect analysis flags these statements as
**uncacheable** so they always run:

<!-- claim: cash/notebook/cacheability.py:_IO_SIDE_EFFECT_FUNCTIONS @7c59797e, cash/notebook/cacheability.py:_SideEffectVisitor @6c2e4302 broad="the table enumerates every call shape the visitor flags; a new branch is a missing row" -->
| Pattern | Examples | Why it's unsafe to replay |
|---------|----------|---------------------------|
| File writes | `open('f', 'w')`, `df.to_csv()`, `df.to_parquet()`, `Path(p).write_text()` | The file wouldn't be written on a cache hit |
| Serializing writers | `json.dump()`, `pickle.dump()`, `np.save()`, `fig.savefig()` | The artifact wouldn't be produced |
| Filesystem changes | `os.remove()`, `shutil.move()`, `os.mkdir()` | The change to disk wouldn't happen |
| System calls | `os.system()`, `subprocess.run()` | The process wouldn't run |
| Network writes | `requests.post()`, `requests.put()`, `requests.delete()`, `requests.patch()` | The request wouldn't be sent |
| Database writes | `df.to_sql()` | The rows wouldn't reach the database |

Read-style calls are deliberately **not** treated as side effects:
`requests.get()`, `urllib` fetches, and `open(...)` in read mode are safe to
cache, exactly like reading a CSV. Only the verbs that *change* the world are
flagged.

<!-- claim: cash/notebook/cacheability.py:_WRITE_METHODS @d59e035c, cash/notebook/cacheability.py:_WRITE_MODES @44a74dfd, cash/notebook/cacheability.py:_is_open_write_mode @ca4d33fa, cash/notebook/cacheability.py:_IO_SIDE_EFFECT_FUNCTIONS @7c59797e -->
Detection is by call shape, so it works without importing anything, with two
consequences worth knowing. A bare `open(...)` counts only when its mode
argument is **statically** a write mode: `open(p, 'w')` is flagged, and
`open(p, mode)` is not, because the analyzer never runs the code to find out
what `mode` holds. And the write-method names are a **fixed list** matched on
any receiver — not a `to_*` / `write_*` wildcard. `obj.save(x)` is flagged even
on a receiver Cash knows nothing about, while `obj.write_thing(x)` and
`obj.to_widget(x)` are not flagged at all. The list stops where names start
colliding: `rename`, `replace` and `touch` are deliberately absent, because
`str.replace` would otherwise flag half a notebook.

<!-- claim: cash/notebook/cacheability.py:statement_write_repeatability @795a3382, cash/notebook/cacheability.py:_REPLACING_WRITE_METHODS @ff293068, cash/notebook/cacheability.py:_is_append_mode_call @3caac057 -->
Being uncacheable is not the end of the story for a writer. Because a file
write has no variable edge, nothing in the lineage graph would ever re-run one,
so Cash separately records which statements wrote which paths and re-fires a
stale writer when a downstream statement reads its output. It also classifies
how safe a write is to repeat — an `open(p, 'a')` or `df.to_csv(p, mode='a')`
*accumulates*, so re-firing it duplicates data, while `to_parquet` and `savefig`
truncate and land the same bytes. That distinction is what keeps reconstruction
from corrupting an append-mode audit log.

## Unseeded randomness

Random calls are *deterministic only if seeded*. Cash's `RandomnessDetector`
finds unseeded draws and **warns** — the statement is still cached, and the
first result is simply frozen:

<!-- claim: cash/notebook/randomness.py:RANDOM_FUNCTIONS @928168d0, cash/notebook/randomness.py:SEED_FUNCTIONS @39b1ffc1 -->
| Module | Tracked functions |
|--------|-------------------|
| `random` | `random()`, `randint()`, `choice()`, `shuffle()`, `sample()`, `uniform()`, … |
| `numpy.random` | `rand()`, `randn()`, `randint()`, `choice()`, `normal()`, `integers()`, … |
| `torch` | `rand()`, `randn()`, `randint()`, `randperm()`, `normal()`, … |
| `tensorflow.random` | `uniform()`, `normal()`, `truncated_normal()`, `shuffle()`, … |

<!-- claim: cash/notebook/randomness.py:RandomnessDetector @befaee4a broad="the claim is about the detector having exactly two channels, which is a property of the class", cash/notebook/randomness.py:RandomnessDetector.is_seeded @9ff99734, cash/notebook/randomness.py:RNG_CARRIER_CONSTRUCTORS @3248b870 -->
Two channels feed it, because there are two ways to be random. **Module
globals** (`np.random.rand()`) are reproducible if the *module* was seeded, so
the detector tracks `seed()` calls across the session: once a module is seeded,
later draws from it are treated as deterministic and no warning fires.
**Carriers** (`rng = np.random.default_rng()`) are reproducible if the *object*
was constructed with a seed, which only the source can say — `default_rng()`
and `default_rng(42)` produce indistinguishable objects. A carrier draw is
therefore never filtered through the module seed ledger: seeding
`np.random` two cells up says nothing about an independent `Generator`.

Freezing is deliberate, and it is the part most worth understanding: the value
is frozen whether or not the cache holds it, because Cash rewinds the RNG so a
re-run consumes the same stream position. So Cash announces it twice — once at
compute time ("cached results may not be reproducible") and again at restore
time, with a different claim, because by then the number on screen *is*
definitively a replay rather than a fresh draw.

The badge carries the same information as a text pill on the statement row:

<!-- claim: cash/notebook/badge_renderer/renderers/html.py:_rng_pill @7b72eb6c, cash/notebook/statement/processor.py:StatementProcessor._stamp_random_effect @8cd2ee49 -->
| Pill | Meaning |
|------|---------|
| `seed` | The statement sets an RNG seed |
| `random` | The statement draws, from a seeded (reproducible) source |
| `unseeded` | The statement draws unseeded — the cached value is a frozen replay |

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._warn_unseeded_randomness @adb698f3, cash/notebook/statement/restore.py:StatementRestorer.restore_from_cache @0388af0f -->
To silence the warning deliberately, annotate the statement with
`@cash:allow-random` (see [Annotations](../annotations.md)). That is *advisory
only* — it suppresses the message and changes no caching decision. To actually
redraw on every run, use `@cash:no-cache`, which switches off both the cache and
the RNG rewind.

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._warn_unseeded_estimator_fit @5c44fa11, cash/notebook/statement/processor.py:StatementProcessor._unseeded_estimator_fits @104fd0be -->
One hazard the AST cannot see: an sklearn-style `estimator.fit()` draws its
randomness inside compiled code, with no Python call to scan. When a fit is
cached (under `# @cash:cache-fit`) and the estimator has `random_state=None`,
Cash checks the live estimator and warns through the same channel.

## From watching to deciding

<!-- claim: cash/notebook/cacheability_decision.py:decide_cacheability @894ac130 -->
The findings above are merged into a single verdict per statement by
`decide_cacheability`. It has five reason-sources and the first one that
triggers wins:

```python
import ast

from cash.notebook.cacheability import analyze_statement
from cash.notebook.cacheability_decision import decide_cacheability

code = "df.to_parquet('out.pq')"
tree = ast.parse(code)

cacheable, reasons = decide_cacheability(
    code=code,
    tree=tree,
    inputs={"df"},
    outputs=set(),               # this statement assigns nothing
    annotation=None,             # 1. @cash:no-cache
    analysis=analyze_statement(code, tree),   # 4. mutations + side effects
    user_ns={"df": object()},
    variable_lineage={"df": "abc123"},        # 5. inputs missing lineage
    is_stateful_call=lambda name: False,      # 3. @stateful calls
    scan_forbidden=lambda code, ns, tree: [], # 2. forbidden calls (input(), ...)
)

assert cacheable is False
assert reasons == ["Side effect: df.to_parquet() (file_write)"]
```

Note the `outputs` argument: it is what turns "this statement mutates `df`"
into "this statement *produces* `df`". Pass `outputs={"df"}` for a statement
like `df.dropna(inplace=True)` and the mutation stops being a reason at all.

<!-- claim: cash/notebook/statement/derivation_edges.py:is_uncacheable_alias @2e425a0f, cash/notebook/cacheability_decision.py:identity_coupled_reason @77bfb1cc -->
Two more refusals are decided *after* execution, because they are properties of
the value rather than the source: a live-alias object (a NumPy view, a pandas
`groupby` ref-holder) would be decoupled from its base by a round trip, and an
identity-coupled matplotlib `Figure`/`Axes` would be detached from pyplot's
current-figure registry.

So the decision is simple and conservative: **an unattributable mutation or a
side effect → always re-run; unseeded randomness → cache but say so; otherwise
→ cache normally.** Try it on real snippets below.

<div class="cash-cacheability-checker" markdown="0">
  <table>
    <thead><tr><th>Statement</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td><code>df = pd.read_csv('data.csv')</code></td><td>Cached — the file is tracked as a dependency</td></tr>
      <tr><td><code>result = df.groupby('k').sum()</code></td><td>Cached — pure transformation</td></tr>
      <tr><td><code>total += 1</code></td><td>Cached — the mutation is the statement's own output</td></tr>
      <tr><td><code>data.append(4)</code></td><td>Not cached — in-place mutation of a variable this statement doesn't produce</td></tr>
      <tr><td><code>del lookup['stale']</code></td><td>Not cached — deletion with nothing to attribute it to</td></tr>
      <tr><td><code>x = np.random.randn(100)</code></td><td>Cached + warning — unseeded randomness</td></tr>
      <tr><td><code>model.fit(X, y)</code></td><td>Not cached by default — opt in with <code>@cash:cache-fit</code></td></tr>
      <tr><td><code>df.to_parquet('out.pq')</code></td><td>Not cached — file-write side effect</td></tr>
      <tr><td><code>r = requests.post(url, json=payload)</code></td><td>Not cached — network side effect</td></tr>
    </tbody>
  </table>
</div>

Cash also exposes these verdicts at runtime: `@cash:no-cache` forces a
statement to never cache, and the decorator path has matching **purity
markers** for functions — see [The decorator path](decorator-path.md).

<!-- claim: cash/core.py:Cash._surface_purity @8f789834, cash/purity_analyzer.py:ISSUE_UNTRACKABLE_DEP @67696b30 -->
The decorator takes one verdict further than the notebook path: a `@cash.cache`
function whose body resolves a dependency from a **runtime value** cash can't
track — `eval`/`exec`/`compile`, dynamic dispatch via `getattr(obj, name)()`, or
`importlib.import_module` — **raises `CashImpureFunctionError` by default**
(caching correctness can't be guaranteed), rather than merely warning. Pass
`@cash.cache(assume_safe=True)` to accept the risk. See
[the decorator's purity gates](../decorator.md).
