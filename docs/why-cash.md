# Why Cash?

You probably aren't shopping for a notebook-caching library — there isn't one
to shop for. But there's a good chance you've already built one in pieces,
without calling it that. **Cash is the library you'd write yourself if you
had a spare month.**

## Does this sound familiar?

<div class="cash-pain-grid" markdown="0">

  <div class="cash-pain-card">
    <span class="cash-pain-icon">🕐</span>
    <span class="cash-pain-title">"Restart and Run All takes 20 minutes."</span>
    <span class="cash-pain-sub">Every small edit pays the full pipeline cost. You stop iterating and start avoiding restarts.</span>
  </div>

  <div class="cash-pain-card">
    <span class="cash-pain-icon">❓</span>
    <span class="cash-pain-title">"You're not sure if <code>df</code> is stale."</span>
    <span class="cash-pain-sub">Did the upstream cell change? Did someone mutate it? You squint at the timestamp and hope.</span>
  </div>

  <div class="cash-pain-card">
    <span class="cash-pain-icon">🥒</span>
    <span class="cash-pain-title">"Your notebook has 12 pickle files."</span>
    <span class="cash-pain-sub"><code>tmp.pkl</code>, <code>df_v3_USE_THIS.pkl</code>. You're afraid to delete any of them.</span>
  </div>

  <div class="cash-pain-card">
    <span class="cash-pain-icon">🌅</span>
    <span class="cash-pain-title">"Your kernel has been running for 4 days."</span>
    <span class="cash-pain-sub">You can't restart — that's where the state lives. One crash and you rebuild from scratch.</span>
  </div>

</div>

If two of those landed, keep reading.

## What happens when you turn cash on

Above each cell, cash shows a **badge** summarising what it did. Watch the same notebook through four states — first run, re-run, restart, then editing an upstream cell:

<iframe class="cash-badge" src="/_badges/why_cash_reel.html" loading="lazy" scrolling="no" height="260" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

Here's the same notebook workflow written three different ways. **The business logic is identical in all three tabs — only the caching scaffolding changes.**

=== "Without cash"

    ```python
    import pickle, os
    from pathlib import Path

    CACHE = Path(".cache")
    CACHE.mkdir(exist_ok=True)

    # Manual: load if fresh, else recompute and dump.
    df_path = CACHE / "df.pkl"
    if df_path.exists() and df_path.stat().st_mtime > Path("large_file.csv").stat().st_mtime:
        with open(df_path, "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv("large_file.csv")
        with open(df_path, "wb") as f:
            pickle.dump(df, f)

    # ... repeat for every expensive variable. Don't forget version stamps.
    result = df.groupby("category").sum()
    ```

=== "With `%store`"

    ```python
    # IPython %store has no granularity, no auto-invalidation:
    # you have to remember to %store after compute and %store -r on restart,
    # and there's no signal if the underlying CSV changed.
    %store -r df
    if 'df' not in dir():
        df = pd.read_csv("large_file.csv")
        %store df

    result = df.groupby("category").sum()
    # ...did you remember to %store result too? Did df change since you last stored?
    ```

=== "With cash"

    ```python { .nb-cell }
    %cash_on

    df = pd.read_csv("large_file.csv")
    result = df.groupby("category").sum()
    ```

What each pain looks like with cash:

- 🕐 **Restart cost** → statement-level caching + automatic restore-after-restart. Re-running a cell hits the cache; restarting the kernel hits disk.
- ❓ **Staleness** → lineage hashes invalidate automatically when any upstream cell changes. The badge tells you what was reused and what was recomputed.
- 🥒 **Pickle sprawl** → no filenames. The cache is keyed by code + inputs, stored in a single managed backend.
- 🌅 **Restart fear** → restart freely. The cache survives the kernel, so the next run is RESTORED, not recomputed.

### How much time would *you* reclaim?

<div class="cash-calculator" markdown="0"></div>

## Why this works

```mermaid
flowchart LR
    subgraph Manual["Manual workflow"]
        A1[Cell 1: load CSV] -->|pickle.dump| P1[(tmp.pkl)]
        A2[Cell 2: transform] -->|pickle.dump| P2[(tmp2.pkl)]
        P1 -.->|"pickle.load (which one?)"| A3[Cell 3: model]
        P2 -.-> A3
        A3 -->|pickle.dump| P3[(model_v3_FINAL.pkl)]
    end
    subgraph Cash["Cash workflow"]
        B1[Cell 1: load CSV] --> C[(cash cache)]
        B2[Cell 2: transform] --> C
        B3[Cell 3: model] --> C
        C -.->|automatic restore| B1
        C -.-> B2
        C -.-> B3
    end
```

*Cash replaces ad-hoc plumbing with a single dependency-aware cache.*

## Is this for you?

<div class="cash-shines-skip-grid" markdown="0">
  <div class="cash-shines-card">
    <h4>Cash shines for</h4>
    <ul>
      <li>Long-running notebook pipelines (data prep, feature engineering, model exploration).</li>
      <li>Iterative analysis with frequent upstream-cell editing.</li>
      <li>Notebooks with expensive file reads (large CSVs, parquet, pickles).</li>
      <li>Mixed-language teams who don't want to learn Make / snakemake / DVC just for caching.</li>
    </ul>
  </div>
  <div class="cash-skip-card">
    <h4>Skip cash if</h4>
    <ul>
      <li>Your notebook is a single cell with no expensive steps.</li>
      <li>You're writing a pure I/O script (API ingestion, network polling) — caching is at the wrong layer.</li>
      <li>You need hard real-time behaviour.</li>
      <li>Your cells <em>intentionally</em> produce different output each run — though see the <a href="../tutorials/feature-guides/purity-decorators/"><code>@stateful</code> decorator</a> for the nuanced case.</li>
    </ul>
  </div>
</div>

## Cash vs. the alternatives you've tried

If you *do* already use a caching tool, here's where cash sits in the landscape. **All ⚠️ cells have a hover tooltip explaining the partial.**

<div class="cash-matrix-filter" markdown="0">
  <input type="text" id="cash-matrix-filter" placeholder="Filter capabilities…" aria-label="Filter capability rows">
</div>

<div class="cash-matrix-table" markdown="1">

| Capability | Manual pickling | `%store` | `lru_cache` | `joblib.Memory` | `jupyter-cache` | `diskcache` | **cash** |
|---|---|---|---|---|---|---|---|
| Statement-level granularity | ❌ | ❌ | ❌ | ❌ | <span title="Whole-notebook: matches on the set of code cells">❌</span> | ❌ | ✅ |
| Automatic invalidation on upstream change | ❌ | ❌ | ❌ | <span title="Only on direct argument change — not on transitive code edits">⚠️</span> | <span title="Invalidates the whole notebook when any code cell changes — not per-statement or transitive">⚠️</span> | ❌ | ✅ |
| File-dependency tracking | <span title="Possible if you write mtime checks yourself">⚠️</span> | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Survives kernel restart | <span title="Yes if you remember to dump; no automatic restore">⚠️</span> | ✅ | ❌ | ✅ | <span title="Persists executed outputs across runs/builds, but restores into a book build — not a live interactive kernel">⚠️</span> | ✅ | ✅ |
| Observable (badges / provenance) | ❌ | ❌ | ❌ | <span title="call_and_shelve prints when verbose; no badges">⚠️</span> | <span title="jcache CLI lists cached notebooks and staging state; no per-cell badges">⚠️</span> | <span title="Hit/miss counts via Cache.stats(); no badges or provenance">⚠️</span> | ✅ |
| Works in plain scripts (non-notebook) | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Mutation detection | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Function-source change detection | ❌ | ❌ | ❌ | <span title="Hashes function source — but only the decorated function, not its callees">⚠️</span> | <span title="Re-executes when a code cell's source changes, but is blind to edits in imported .py modules">⚠️</span> | <span title="memoize keys on the function name + arguments (like lru_cache); editing the body does not invalidate">❌</span> | ✅ |
| Native pandas / numpy / polars / PyArrow hashing | ❌ | ❌ | <span title="Numpy and pandas aren't hashable by default; you'd need a wrapper">⚠️</span> | ✅ | <span title="Caches serialized cell outputs; does not hash input objects">❌</span> | <span title="Stores them fine as cached values, but does not content-hash them for keys">⚠️</span> | ✅ |
| Zero-config to start | ✅ | ✅ | ✅ | <span title="Requires picking a Memory location and decorating each function">⚠️</span> | <span title="Needs Jupyter Book / MyST-NB or the jcache CLI workflow">⚠️</span> | <span title="Pick a cache directory and wrap each function with @memoize">⚠️</span> | ✅ |

</div>

## Common questions

### Correctness

??? question "What if cash returns a stale value?"
    Cash invalidates a cached result whenever the code that produced it
    changes *or* any of its inputs change. Inputs are tracked by lineage
    hash, so a change three cells upstream still propagates. If a cell
    reads a file, cash hashes the file and invalidates if it changes.
    See [Knowing when to recompute](how-it-works/invalidation.md).

??? question "What about in-place mutations like `df['x'] = 0`?"
    Cash uses AST-based mutation detection to flag in-place mutations
    so cached objects can be invalidated correctly.
    See [the mutation pattern in the data-science tutorial](tutorials/use-cases/data-science.md).

### Coverage

??? question "Does it work with pandas / numpy / polars / torch / duckdb?"
    Native built-in hashers cover pandas, numpy, polars, PyArrow, modin,
    and dask. For anything else — torch tensors, duckdb relations,
    custom domain types — register a hasher with `cash.register_hasher`.
    See [API reference](api/cash.md#cash.Cash).

??? question "Notebooks only, or scripts too?"
    Both. The `@cash.cache` decorator works in plain Python scripts.
    The notebook integration (`%cash_on` and statement-level caching)
    is what the rest of this page focuses on, but the underlying engine
    is independent of Jupyter.
    See [API reference](api/index.md).

### Overhead

??? question "How much does cash slow down a cold run?"
    Cash adds ~5–30ms per cached statement on a cold run (lineage
    computation, cache key, write). For most real notebook work the
    overhead is dwarfed by the work being cached. The
    [cost model](cost-model.md) lays out the maths in detail.

??? question "What about disk usage?"
    Cash uses a tiered backend by default: RAM (L1) → file on disk (L2).
    Other backends — SQLite, Redis, S3 — are available for custom tier
    stacks. Eviction rules and a size-budget cap are configurable.
    See [Configuration](getting-started/configuration.md).

### Production readiness

??? question "Is a 0.x release safe for real work?"
    Yes for notebook use; `0.1.1`, the first public release, is backed by
    comprehensive unit + integration test coverage (the test suite is thousands
    of integration tests, many derived from real-world bug reports). Treat it
    like any other library you'd pin a version of — this is a `0.x` release, so
    the API and the cache format may change between minor versions (run
    `%cash_repair --full` after upgrading). The
    [CHANGELOG](https://github.com/galgtonold/cash/blob/main/CHANGELOG.md)
    documents breaking changes between releases.

??? question "How do I force a fresh run?"
    Three escape hatches. (1) `@cash:no-cache` annotation on a single
    statement. (2) `%cash_off` to disable for a whole cell. (3) The
    `cash clear` CLI command. See [annotations](annotations.md) and
    [CLI reference](cli.md).

### Vs. alternatives

??? question "Why not joblib.Memory?"
    `joblib.Memory` caches at the function call level — you wrap each
    function with `@memory.cache` and joblib hashes the arguments. It
    doesn't see notebook-level dependencies (a cell that uses an
    upstream variable, file reads, mutations), so you'd still hit all
    four pains on this page. See [migration guide](migration_guide.md#from-joblibmemory).

??? question "Why not `%store`?"
    IPython's `%store` is a manual save/load primitive — no automatic
    invalidation, no granularity, no signal when an underlying file
    changes. It's a useful primitive for kernel restart, not a caching
    system. The "With `%store`" tab above shows what its limits look
    like in practice.

??? question "Why not `functools.lru_cache`?"
    `lru_cache` keys on hashable function arguments — pandas DataFrames
    and numpy arrays aren't hashable, and even if they were, an `lru_cache`
    miss costs a recompute every time the process restarts. It's not in
    the same category as a persistent, dependency-aware cache.
    See [migration guide](migration_guide.md).

??? question "Why not `jupyter-cache`?"
    `jupyter-cache` (the engine behind Jupyter Book / MyST-NB) caches a
    notebook's executed outputs so a *build* can skip re-running an unchanged
    notebook. It matches at whole-notebook granularity — hashing the code
    cells — so editing any one cell re-executes the entire notebook, and it
    doesn't track data-file reads, imported-module edits, or in-place
    mutations. It's built for reproducible book builds in CI, not the
    interactive edit-one-line-and-re-run loop cash optimizes.

??? question "Why not `diskcache`?"
    `diskcache` is an excellent persistent key→value store — cash can even use
    a disk backend for the same storage job. Its `@memoize` decorator keys on
    the function's name and arguments (like `lru_cache`), so results survive a
    restart, but editing the function body does **not** invalidate the entry,
    and it has no notebook awareness, dependency lineage, file tracking, or
    mutation detection. Think of it as a fast storage layer, not a
    "know-when-to-recompute" layer.

??? question "Why not marimo?"
    [marimo](https://marimo.io) is a *reactive notebook environment* — it models
    the dataflow between cells and re-runs exactly the cells that depend on a
    change, so there's no hidden state or stale output. It also ships its own
    caching (`mo.cache` in memory, `mo.persistent_cache` to disk) that skips a
    wrapped block when neither its code nor its ancestors changed, and survives a
    restart — genuinely dependency-aware, much closer to cash than the key→value
    caches above.

    The difference is what you adopt. marimo is a **replacement** for Jupyter — a
    new runtime and file format you move your work into, with caching you wrap
    around chosen blocks. Cash is **additive**: `%cash_on` inside the Jupyter,
    Colab, or VS Code notebook you already use, caching every statement
    automatically. If you're happy switching environments, marimo's reactivity is
    a strong answer to notebook staleness; if you'd rather stay in Jupyter and
    make re-running cheap, that's cash.

## Try it / Go deeper

<div class="cash-cta-split" markdown="0">
  <a class="cash-cta-card primary" href="../getting-started/quickstart/">
    <h3>Try it now →</h3>
    <p><code>pip install cash-lib</code>, then drop <code>%cash_on</code> in your first cell.</p>
    <p><strong>Quickstart →</strong></p>
  </a>
  <div class="cash-cta-card secondary">
    <h3>Go deeper →</h3>
    <p>Read how lineage tracking, upstream simulation, and the cache key work under the hood.</p>
    <p><a href="../how-it-works/overview/"><strong>How Cash Works</strong></a> · <a href="../cost-model/"><strong>Cost model</strong></a></p>
  </div>
</div>

### See it applied to your workflow

End-to-end recipes for the workflows cash is built for:

- [Data science](tutorials/use-cases/data-science.md) — exploratory notebooks
  with expensive ETL, feature engineering, and model iteration.
- [LLM API calls](tutorials/use-cases/llm-api-calls.md) — memoising expensive
  completions, replaying prompts deterministically across runs.
- [Data engineering](tutorials/use-cases/data-engineering.md) — pipeline-style
  workflows where intermediate frames need to survive kernel restarts.
- [Scientific computing](tutorials/use-cases/scientific-computing.md) —
  simulations, parameter sweeps, and reproducibility against changing code.
