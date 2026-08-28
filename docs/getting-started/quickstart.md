# Quick Start

Cash works two ways. Pick the one that matches how you write code — each track
below is self-contained.

<div class="cash-doc-grid" markdown="0">
  <a class="cash-cta-card primary" href="#in-a-notebook">
    <h3>In a notebook →</h3>
    <p><code>%cash_on</code> and every cell caches automatically — statement-level,
    with a badge above each cell's output. For interactive work in Jupyter, Colab, or VS Code.</p>
  </a>
  <a class="cash-cta-card" href="#in-a-script">
    <h3>In a script →</h3>
    <p><code>@cash.cache</code> on a function — caches by its arguments and its own
    source code. For modules, pipelines, and batch jobs.</p>
  </a>
</div>

Most people start in a notebook, but the two share one engine — the same lineage,
hashing, and backends underneath — so nothing is second-class.

---

## In a notebook

!!! tip "Prefer to just try it?"
    Run cash in your browser with no install —
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb) — then come back here to wire it into your own notebooks.

### Step 1: Enable Cash

```python { .nb-cell }
import cash

# Enable auto-caching for all subsequent cells
%cash_on
```

That's the whole setup — no decorators, no config file. (`import cash` auto-registers
the magics, so `%load_ext cash` is not required.)

!!! tip "Skip the boilerplate"
    Run `cash autoload on` once and every new IPython/Jupyter kernel starts with
    `import cash` plus `%cash_on` already executed. See the [CLI reference](../cli.md).

### Step 2: Write code normally

<!-- test:skip reason="illustrative — references missing large_dataset.csv" -->
```python { .nb-cell }
import pandas as pd

df = pd.read_csv('large_dataset.csv')   # cached after the first run
print(f"Loaded {len(df)} rows")
```

### Step 3: Re-run — it's instant

Run the cell again and the `read_csv` result loads from cache. The badge above the
cell flips from `EXECUTED` (ochre) to `CACHED` (green):

**First run** — Cash ran the statement and cached the result:

<iframe class="cash-badge" src="/_badges/quickstart_first_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

**Second run** — Cash restored the value instead of recomputing:

<iframe class="cash-badge" src="/_badges/quickstart_second_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

See [Reading the Cash badge](../badges.md) for the full anatomy.

### Change one thing upstream — run only the cell you care about

This is the part a plain cache can't do. Cash answers "what depends on this?" per
**statement**, so it can repair *part* of a cell you never ran.

Suppose Cell 2 mixes expensive work that ignores `THRESHOLD` with a cheap step that
reads it:

<!-- test:skip reason="illustrative — references undefined build_features()/score()" -->
```python { .nb-cell }
# Cell 1
THRESHOLD = 10                          # ← change me to 15

# Cell 2 — one statement ignores THRESHOLD, the other reads it
features = build_features(df)           # expensive (minutes), THRESHOLD-independent
flagged  = score(features, THRESHOLD)   # cheap, THRESHOLD-dependent

# Cell 3
print(f"{len(flagged)} rows flagged")
```

Change `THRESHOLD`, then **run Cell 3 by itself** — not *Run All*, and without
touching Cell 2. Cash walks back from what Cell 3 needs, works out that `flagged`
is stale but `features` is not, and repairs exactly that:

- `build_features(df)` **does not run again** — cash reuses the value it already has,
- `score(features, THRESHOLD)` re-runs, once, with the new threshold,
- Cell 3 prints the new number.

!!! warning "If cash has no live reader, save first — it falls back to the file on disk"
    This is the one thing that will make the walk-back above appear not to work.
    With no live reader, cash reads the cells it didn't execute from the **saved
    `.ipynb`**, not from your editor's in-memory buffer. Change `THRESHOLD` to 15,
    run Cell 3 without saving, and cash still reads `10` — so it concludes nothing
    upstream changed, the repair never fires, and Cell 3 prints the *old* number
    while your screen shows the new one.

    **Three editors have a live reader and don't need the save:** Google Colab
    (cells come from the frontend), JupyterLab (`pip install cash-lib` also
    installs an extension that pushes your live cells to the kernel before each
    run), and VS Code with hot exit on (cash reads its unsaved-state backup).
    Everywhere else — and on JupyterLab for the *first* run after a kernel
    restart, including a cold `Run All` — **press `Ctrl+S` (`Cmd+S` on macOS)
    after editing an upstream cell.** JupyterLab's autosave runs on a timer, so a
    quick edit-then-run lands inside that window.

    Nothing is corrupted when it happens: you get the value your kernel actually
    holds, which is the same thing plain Jupyter would give you. What you lose is
    the safety net — cash's upstream check is only ever as current as the cells it
    read. cash tells you which case you are in: it prints *"cash cannot see
    unsaved edits here"* once per session, and only when reading the saved file.
    Full detail in
    [editing without saving](../known-limitations.md#editing-without-saving).

You get the same result you'd get from *Run All*, at the cost of the one cheap
statement that actually changed. That's the difference between caching *cells* and
tracking lineage *between statements*: the unit of repair is the statement, and cash
will reach into a cell you didn't run to fix just the part that went stale.

Nothing to declare — no decorators on Cell 2, no dependency graph. And the same
walk-back works from cold: jump straight to Cell 3 after a kernel restart and cash
restores or re-runs the upstream statements it needs.

**Cell 3's badge after the change.** The `Upstream` section lists what cash had to
repair in cells you didn't run. `features = build_features(df)` isn't there — it
never needed touching:

<iframe class="cash-badge" src="/_badges/quickstart_partial_upstream.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

### It follows your functions and loops

Cash keys on the **source** of what you run, not the text of a cell:

- Edit a function's body — or a helper it calls — and the statements that use it
  recompute (the badge marks it **changed**). It's not blind text matching.
- In a loop, each iteration is cached on its own — add a case and the ones you
  already ran still restore:

<!-- test:skip reason="illustrative — references undefined fetch_and_model/prices" -->
```python { .nb-cell }
for ticker in ["AAPL", "MSFT", "GOOG"]:
    prices[ticker] = fetch_and_model(ticker)   # each iteration cached separately
```

!!! warning "Collect with an assignment or a store, not `.append()`"
    A subscript store like the one above caches the **statement** per iteration.
    A `results.append(...)` statement does **not** — cash has no snapshot that
    would reproduce an append, so that statement always re-executes.

    That no longer means the *work* is repeated, though: by default cash also
    caches the expensive **call inside** the statement (`fetch_and_model(ticker)`
    here, not the `append` around it), so a `.append()` loop still skips
    redoing the slow part on a re-run — only the cheap append itself happens
    again. `# @cash:no-cache-calls` turns that off, if you need the call to
    genuinely re-run too. Prefer a comprehension
    (`results = [fetch_and_model(t) for t in tickers]`) or a dict store when
    you can — either one caches the *statement* itself and sidesteps the
    question entirely.

    That guarantee is for loops sized like the one above. A long loop (roughly
    125+ iterations for a one-line body) switches to whole-loop caching, and
    calls inside it never reach the interceptor either — see
    [A long for-append loop can stop caching](../known-limitations.md#a-long-for-append-loop-can-stop-caching).

Two more things worth knowing before you lean on loop caching: a *statement*
that accumulates (`s += f(x)`, not a bare call) has reuse that follows the
*order* of the items — but that historical limitation is exactly what the
default call-level caching above already dissolves for eligible calls that
compute and return (reordering costs nothing, not "just the tail" — a callee
that also writes a global or a file is re-executed instead), and a long loop can switch
to whole-loop caching. Both are measured in
[Known limitations](../known-limitations.md#reordering-a-loops-items-re-runs-the-tail).
See [The notebook path](../how-it-works/notebook-path.md) for how partial hits work.

### Statement-level, not cell-level

Cash caches each **statement**, not the whole cell:

<!-- test:skip reason="illustrative — references missing data.csv" -->
```python { .nb-cell }
df     = pd.read_csv('data.csv')       # statement 1
daily  = df.resample('D').mean()       # statement 2
result = daily.rolling(7).mean()       # statement 3
```

Change statement 3 and only it re-runs — statements 1 and 2 stay cached.

*One exception: a top-level statement that mutates an object **created in an
earlier cell** (`df['x'] = ...` on an upstream `df`, `lst.append(...)` on an
upstream list) runs fresh each time — cash tracks the mutation so everything
downstream stays correct, it just doesn't replay a snapshot. The same mutation on
an object built in the same cell caches normally, and inside a loop iterations are
cached whole, mutations included.* See
[Knowing when to recompute](../how-it-works/invalidation.md).

### File changes are tracked automatically

Cash intercepts file reads (`pd.read_csv`, `np.load`, `open`, `joblib.load`, …) and
records each file's fingerprint — change the file on disk and the statements that
read it recompute, no annotation needed. See the [notebook reference](../notebook_caching_api.md).

### It survives a kernel restart

The cache lives on disk, not just in memory. Restart the kernel, re-run a cell, and
cash restores the value instead of replaying the whole chain — a fresh kernel picks
up where you left off, and each restored value has been checked against its lineage
first, so you are not trusting a stale snapshot.

---

## In a script

Outside notebooks, wrap any function with `@cash.cache`. It caches by the function's
**arguments** and its own **source code**, so an identical call returns instantly and
editing the body invalidates the entry:

<!-- test:skip reason="illustrative — uses a 5s sleep to stand in for real work" -->
```python
import cash

c = cash.Cash(cache_dir='./my_cache')

@c.cache
def expensive_function(x, y):
    import time
    time.sleep(5)          # stand-in for real work
    return x + y

expensive_function(1, 2)   # first call: ~5 seconds
expensive_function(1, 2)   # second call: instant
```

<!-- claim: cash/backends/factory.py:_SMART_PERSIST_COMPUTE_FLOOR_S == 0.1 -->
Results that cost more than ~0.1s persist to disk across processes and restarts;
cheaper ones are kept in memory for the session (see the [cost model](../cost-model.md)).

<!-- claim: cash/core.py:Cash._hash_arg_payload @8be5a896, cash/core.py:Cash._try_builtin_type_hash @9c5166b5 -->
### Pass DataFrames and arrays — they just work

Arguments don't need to be hashable. Cash hashes by **content**, so a DataFrame or
array is fine — and two *content-equal* frames hit the same entry, even if they're
different objects:

<!-- test:skip reason="illustrative — references undefined df/weights" -->
```python
@c.cache
def summarize(df, weights):        # a pandas DataFrame and a numpy array
    return (df * weights).sum()

summarize(df, weights)             # runs
summarize(df.copy(), weights.copy())   # instant — same content, different objects
```

Built-in hashers cover pandas, numpy, polars, PyArrow, modin, and dask; register your
own for custom types with [`register_hasher`](../decorator.md).

### Edit a helper — the cache notices

Cash folds the source of the functions you *call* into the key, **transitively**.
`pipeline` below never mentions `clean` — but editing `clean` still invalidates it:

```python
import cash

c = cash.Cash()

def clean(text):
    return text.strip().lower()

def features(text):
    return len(set(clean(text).split()))

@c.cache
def pipeline(text):
    return features(text)

sample = "  The cat  the HAT  "
print(pipeline(sample))          # 3  -> {'the', 'cat', 'hat'}
print(pipeline(sample))          # 3  -> cache hit; the body never ran
print(pipeline.cache_info())     # {'hits': 1, 'misses': 1, ...}
```

Now edit `clean`, two calls below `pipeline` — stop lower-casing:

<!-- test:skip reason="the docs harness infers expected hits/misses from call sites, which cannot model an invalidation between two identical calls; this fence is executed and its output asserted by tests/docs/test_quickstart_helper_invalidation.py instead" -->
```python
def clean(text):
    return text.strip()

print(pipeline(sample))          # 4  -> {'The', 'cat', 'the', 'HAT'}
print(pipeline.cache_info())     # {'hits': 1, 'misses': 2, ...}
```

The answer changed *and* `misses` went up: cash threw the cached result away
because a function `pipeline` calls indirectly changed. You don't have to
remember which entries a helper edit reaches. Helpers are resolved within the
module; for cross-module dependencies, name them with `depends_on=`.

### File reads are tracked here too

The automatic file tracking from notebook mode applies in scripts as well — no
annotation required:

<!-- test:skip reason="illustrative — references missing data.csv" -->
```python
@c.cache
def load():
    return pd.read_csv("data.csv")   # change data.csv on disk → recomputes
```

For explicit control there's `@c.cache(file_depends_on="data.csv")` and `depends_on=`.

And the two paths meet: **call a `@cash.cache` function inside a notebook and its
hits show up on that cell's badge** — the same engine, either way.

The full decorator surface — `ttl`, `cache_if`, async support, iterator caching, the
purity analyzer, and `explain()` — is in the [decorator guide](../decorator.md).

---

## Configuration (both paths)

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_on @e19353cf -->
`%cash_on` takes only an optional `ttl=N`; to pick a different backend or cache
directory, construct a `Cash(backend=...)` instance before enabling the magic (in a
notebook), or decorate with its `@c.cache` (in a script). Optional backends — SQLite,
Redis, S3 — install via extras (`pip install "cash-lib[redis]"`, `[s3]`, `[all]`) and
work the same for either path. See [Configuration](configuration.md).

## What's next?

- **[Why Cash?](../why-cash.md)** — when Cash helps, and how it compares to other caches.
- **[Notebook Caching](../notebook_caching_api.md)** & **[Decorator guide](../decorator.md)** — the full reference for each path.
- **[Annotations](../annotations.md)** — per-statement control (`# @cash:no-cache`, `ttl`, `persist`, `allow-random`).
- **[Magic commands](../magics.md)** — `%cash_stats`, `%cash_help`, `%cash_debug`, and friends. Run `%cash_help` in any notebook for the list.
- **[For coding agents](../for-coding-agents.md)** — using cash from Claude Code, Copilot, Cursor and friends (text badges, what an agent should read).
- **[Benchmarks](../benchmarks.md)** — measured speedups, and how to reproduce them on your own workload.
- **Tutorials:** [Data science](../tutorials/use-cases/data-science.md) · [LLM API calls](../tutorials/use-cases/llm-api-calls.md) · [Caching class methods](../tutorials/feature-guides/caching-class-methods.md) · [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md).
