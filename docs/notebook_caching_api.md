# `%cash_on` — notebook caching guide

This page is the cohesive walkthrough of notebook caching: how `%cash_on`
turns a normal Jupyter session into a cached one, what "statement-level"
actually buys you, and the behaviors that are unique to running inside a live
kernel.

It's the notebook-side twin of the [decorator guide](decorator.md). For the
full magic-command reference (all 20 magics, every flag) see
[Magic commands](magics.md); for per-statement control comments see
[Annotations](annotations.md); for the programmatic entry points (writing
tooling around cash) see the [notebook API reference](api/notebook.md).

---

## When to use it

Turn on `%cash_on` whenever you're iterating in a notebook — exploring data,
tuning a model, building a pipeline cell by cell. The payoff grows with:

- **Expensive upstream cells** you re-run constantly while editing something
  downstream (a 30 s `read_csv` + join you don't want to pay on every tweak).
- **Long dependency chains** where a kernel restart would otherwise mean
  re-running everything from the top.
- **Loops over cases** (tickers, files, hyperparameters) where you change one
  case and want the rest to stay put.

It's less useful for a notebook of sub-second cells with no restarts — cash's
per-cell bookkeeping can outweigh what it saves. `%cash_stats` reports **net**
time saved (gross minus cash's own overhead) precisely so you can tell.

For caching individual functions in a module or script instead, reach for the
[`@cash.cache` decorator](decorator.md). The two share one engine and
[interoperate](#the-two-paths-meet) — a decorated function called in a cell
shows its hits on that cell's badge.

---

## The minimum

```python
import cash

%cash_on
```

That's the whole setup. `import cash` auto-registers the magics, so
`%load_ext cash` is **not** required, and there's no config file or decorator
to add. Every cell you run from here on is cached automatically.

<!-- test:skip reason="illustrative — references missing large_dataset.csv" -->
```python
import pandas as pd

df = pd.read_csv('large_dataset.csv')   # first run: executes and caches
result = df.groupby('category').sum()
```

Re-run the cell and both statements restore from cache instead of recomputing.
You see that happen on the **badge** above the cell's output — `EXECUTED` (ochre)
on the first run, `RESTORED` (green) on the second:

<iframe class="cash-badge" src="/_badges/status_computed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<iframe class="cash-badge" src="/_badges/status_restored.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

`%cash_off` disables auto-caching again; `%%cash` caches a single cell
explicitly when auto-caching is off. Both are covered in
[Magic commands](magics.md).

!!! note "Cross-process persistence has a compute floor"
    Only results whose computation took **longer than ~0.1 s** are written to
    disk. A cheaper result is still cached in RAM — down to a ~10 ms "too cheap
    to cache" floor, below which nothing is stored at all — so a repeat run *in
    the same kernel* is instant, but a fresh kernel recomputes it. That's the right
    tradeoff — the expensive cells worth caching survive a restart, trivial ones
    don't waste disk. Force it with [`# @cash:persist`](annotations.md#cashpersist)
    (or `%cash_persist on` for the whole session) when a fast result must
    survive a restart. See the [cost model](cost-model.md).

---

## The badge tells you what happened, and why

Caching that works silently is caching you can't trust. Every cell cash touches
gets a badge above its output — one row per statement, with the status, the time
saved, and, when something recomputed, **the reason**. It is the primary way you
work with cash: you don't guess whether a hit happened, you read it.

A mixed cell — upstream input restored, current statement computed fresh:

<iframe class="cash-badge" src="/_badges/status_mixed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

The valuable part is the *why*. When a statement re-runs, the badge names the
cause rather than leaving you to wonder — here, a function the statement calls
was edited:

<iframe class="cash-badge" src="/_badges/miss_function_source_changed.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

Other reasons it will name: an input's lineage changed, a file on disk changed, a
module was reloaded, or the statement was deliberately **not cached** (too cheap,
a side effect, an explicit `# @cash:no-cache`).

**→ [Reading the Cash badge](badges.md)** is the full reference: every status and
chip, the anatomy of a row, and a walkthrough of the common cache-miss and
not-cached situations. It's the page to keep open while you get used to cash.

Switch modes with [`%cash_badge`](magics.md#cash_badge) — `html` (default),
`print` for a plain-text summary (nbconvert, CI, agents), or `off`.

---

## Statement-level, not cell-level

This is what sets notebook caching apart from every "cache the cell" tool: cash
caches each **statement** in a cell independently, keyed on the *source* of that
statement and the lineage of its inputs.

<!-- test:skip reason="illustrative — references missing data.csv" -->
```python
df     = pd.read_csv('data.csv')       # statement 1
daily  = df.resample('D').mean()       # statement 2
result = daily.rolling(7).mean()       # statement 3
```

Edit statement 3 and only it re-runs — statements 1 and 2 stay cached. A plain
cell cache would recompute all three, because the cell's text changed.

Because the key is the *source*, not the cell text, cash follows your functions:
edit a function's body — or a helper it calls, transitively within the module —
and the statements that use it recompute. It's not blind text matching.

### Loops and branches decompose too

A `for` loop caches **each iteration separately**, keyed on the loop variable,
so changing one case leaves the rest cached:

<!-- test:skip reason="illustrative — references undefined fetch_and_model/prices" -->
```python
for ticker in ["AAPL", "MSFT", "GOOG"]:
    prices[ticker] = fetch_and_model(ticker)   # each iteration cached on its own
```

`if`/`elif`/`else` and `try`/`except` bodies are decomposed per branch, so only
the branch that ran is cached. `while` and `with` blocks (and a `for` containing
`break`/`continue`) run as a **single** cache unit — they have no enumerable
iteration space to key on. The full mechanism is in
[The notebook path](how-it-works/notebook-path.md#fine-grained-caching-loops-and-branches).

### In-place mutation runs fresh

One deliberate exception to statement-level caching: a **top-level** statement
that mutates an object **created in an earlier cell** (`df['x'] = ...` on an
upstream `df`, `lst.append(...)` on an upstream list) re-runs rather than
replaying a snapshot. Cash tracks the mutation so everything downstream stays
correct — it just doesn't cache the mutating statement itself. The *same*
mutation on an object built in the same cell caches normally, and inside a loop
body iterations are cached whole, mutations included. See
[Staying correct](how-it-works/invalidation.md) and
[Known limitations](known-limitations.md#mutating-an-object-created-in-an-earlier-cell).

### Controlling individual statements

Per-statement `# @cash:` comments override the defaults — force-cache a cheap
statement, opt one out entirely, set a TTL, or acknowledge unseeded randomness:

<!-- test:skip reason="illustrative — references datetime without import; shows annotation placement" -->
```python
# @cash:no-cache
now = datetime.utcnow()      # always fresh, never stored
```

The five directives and their scoping rules (including how a directive on a loop
header cascades to the whole body) are documented in [Annotations](annotations.md).

---

## Consumable inputs and isolated re-runs

Some objects are **drained in place** by reading them: a generator, a
`queue.Queue`, an open file handle. Re-running *only* the cell that drains one
would otherwise read the leftovers of its own previous run — a drained queue
gives `got=[]`, an exhausted generator totals `0` — where **Run all** re-runs
the producer first and gives the real answer.

Cash detects this and re-executes the producer, so an isolated re-run matches
**Run all**:

<!-- test:skip reason="illustrative — spans multiple notebook cells with a live kernel" -->
```python
# Cell 1
q = Queue()
for i in range(3):
    q.put(i)

# Cell 2 — re-running this alone still prints got=[0, 1, 2]
got = []
while not q.empty():
    got.append(q.get())
print(f"got={got}")
```

The check compares the object's drain position (generator state, queue
`qsize()`, file offset) against a baseline recorded at the cell's *entry*, so it
self-disables on **Run all** — where the producer has already handed the cell
the same state — and on a cell's first run, which has no baseline. It's also
scoped to inputs the cell actually consumes: a reporting read like
`n = q.qsize()` leaves the producer alone.

Deep-copyable iterators (`map`, `zip`, `enumerate`, `io.StringIO`,
`iter(range(6))`) are snapshotted fresh by the cache and restore correctly, so
they are **not** flagged and their producers are never re-run. Opaque
`itertools` cursors (`cycle`, `chain`, `tee`) expose no observable position and
are deliberately left alone.

---

## Output replay and display suppression

A cache hit replays the statement's captured `stdout`, `stderr`, and rich
outputs, so a restored cell looks exactly like one that just ran.

A trailing `;` still suppresses output on a **cached** re-run. `ast.unparse`
drops the semicolon, so cash recovers it from the raw cell source and
re-attaches it to the statement — the suppression rides through both the cache
key and the execution path, and nothing is displayed *or* captured:

<!-- test:skip reason="illustrative — display suppression requires a live IPython kernel" -->
```python
df.head();   # no repr on the first run, and none on a cached re-run either
```

Because the `;` is part of the statement code, `df.head()` and `df.head();` are
distinct cache entries.

---

## Top-level `await` cells

Cells using Jupyter's top-level `await` are cached like any other cell — no
opt-in, no separate magic. Awaited cells get the same lineage tracking, upstream
reset, **and** result caching as synchronous ones:

<!-- test:skip reason="illustrative — top-level await requires a live IPython kernel" -->
```python
data = await fetch_from_api(url)   # cached: lineage, reset, and the result
```

On a cache hit the restore returns before the coroutine is ever built, so an
unchanged re-run skips the `await` entirely rather than re-issuing the request.
See [`%%cash`](magics.md#cash-cell) for the mechanism.

---

## Surviving a kernel restart

The cache lives on disk, not just in memory, so notebook state survives a
restart. Run a cell that needs `df` after restarting and cash restores `df`
straight from cache instead of replaying the chain that built it — a deep
pipeline comes back in the time it takes to deserialize, not the time it took to
compute.

This is the payoff `%store` and `diskcache` can't match: they persist values
too, but don't know whether a stored value is still *valid*. Cash proves
freshness through lineage before it restores, and re-runs the upstream cells it
can't prove — see [Picking up after a kernel
restart](how-it-works/notebook-path.md#picking-up-after-a-kernel-restart) and
[Staying correct](how-it-works/invalidation.md).

---

## File changes are tracked automatically

Cash intercepts file reads (`pd.read_csv`, `np.load`, `open`, `joblib.load`, …)
and records each file's fingerprint — change the file on disk and the statements
that read it recompute, no annotation needed. This works the same in the
[decorator path](decorator.md#file-reads-are-tracked-automatically),
where you can also name files explicitly with `file_depends_on=` (auto-tracking
fingerprints file *content*; `file_depends_on=` keys on the file **mtime**).

---

## The two paths meet

Call a [`@cash.cache`](decorator.md)-decorated function inside a cell and its
hits and misses show up on that cell's badge alongside the statement rows — same
engine, either way. This is the natural bridge from notebook exploration to a
reusable module: prototype with `%cash_on`, then lift the stable pieces into
decorated functions without giving up caching.

---

## Inspecting and managing a session

These are all magics — the full reference, with every flag, is in
[Magic commands](magics.md):

| Want to… | Magic |
|---|---|
| See session-wide hits, misses, and **net** time saved | [`%cash_stats`](magics.md#cash_stats) |
| Trace how a variable was computed | [`%cash_provenance`](magics.md#cash_provenance) |
| Watch a local module for source changes | [`%cash_track`](magics.md#cash_track) |
| Snapshot a cache to a file (bug report, archive) | [`%cash_export`](magics.md#cash_export) / [`%cash_import`](magics.md#cash_import) — for team sharing prefer a [shared backend](tutorials/feature-guides/sharing-caches.md) |
| Audit or repair cache integrity | [`%cash_verify`](magics.md#cash_verify) / [`%cash_repair`](magics.md#cash_repair) |
| Print the quick-reference card | [`%cash_help`](magics.md#cash_help) |

For programmatic access — reading statement metrics or lineage from tooling
rather than a magic — see the [notebook API reference](api/notebook.md) and
[`%cash_status`](magics.md#cash_status).

---

## Configuration

`%cash_on` takes only an optional `ttl=N`. To point cash at a different backend
or cache directory, call `cash.configure(...)` *before* enabling the magic — the
magics use whatever configuration is current:

<!-- test:skip reason="illustrative — Redis backend requires a running server" -->
```python
import cash

cash.configure(cache_dir=".my_cache", backend="redis")   # applies to all caching from here
%cash_on
```

(You can also construct a `Cash(...)` instance explicitly and hand it a backend
object — see [Configuration](getting-started/configuration.md).)

Optional backends (SQLite, Redis, S3) install via extras
(`pip install "cash-lib[redis]"`, `[s3]`, `[all]`). Every setting is also
bindable via a `CASH_*` env var or a TOML file — see the
[Configuration reference](getting-started/configuration.md).

!!! note "Cache errors never break your cell"
    If cash hits an error while caching a statement — an unpicklable result, a
    corrupted entry — it falls back to running the statement normally and the
    cell still produces the right output. The badge names what wasn't cached and
    why; caching is an optimization, never a correctness dependency. (Cash *can*
    still stop loudly on byte-identical duplicate cells it can't disambiguate —
    see [Known limitations](known-limitations.md#ambiguouscellerror).)

---

## Do's and don'ts

**Do**

- ✅ Lean on it for expensive upstream cells and long chains — that's where the
  restart and partial-recompute payoff is largest.
- ✅ Enable [`%cash_debug on`](magics.md#cash_debug) when a cache decision
  surprises you.
- ✅ Use [`# @cash:no-cache`](annotations.md#cashno-cache-alias-nocache) for
  statements with side effects that must run every time (API writes, `datetime.now()`).
- ✅ Seed your RNG (`np.random.seed(...)`) for reproducible cached draws, or
  acknowledge the freeze with [`# @cash:allow-random`](annotations.md#cashallow-random-alias-allowrandom).

**Don't**

- ❌ Don't expect an unseeded random draw to vary once cached — the first value
  is frozen and replayed. `allow-random` silences the warning but does *not*
  stop the caching; `# @cash:no-cache` is what forces a fresh draw.
- ❌ Don't put byte-identical code in two different cells — where the
  environment exposes no resolvable cell ID this raises `AmbiguousCellError`
  ([Known limitations](known-limitations.md#ambiguouscellerror)). A distinguishing
  comment is enough to separate them.
- ❌ Don't cache reads from `stdin` or closures over mutable state you expect to
  keep changing.

---

## Where to go next

- [Magic commands](magics.md) — the full reference for all 20 `%cash_*` magics.
- [Annotations](annotations.md) — every `# @cash:` directive and its scoping.
- [Reading the Cash badge](badges.md) — the visual vocabulary in full.
- [`@cash.cache` decorator](decorator.md) — the script/module path.
- [Known limitations](known-limitations.md) — the one page to read before writing cached notebooks.
- [How Cash works](how-it-works/overview.md) — the architecture tour, start to finish.
- [The notebook path](how-it-works/notebook-path.md) — how a cell runs, end to end.
- [Staying correct: invalidation](how-it-works/invalidation.md) — how cash proves a cached value is still valid.
- [Notebook API reference](api/notebook.md) — programmatic entry points for tooling.
- [Purity decorators](tutorials/feature-guides/purity-decorators.md) — `@stateful` to stop a statement from caching (and where `@pure` actually applies).
