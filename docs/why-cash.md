# Why cash?

!!! info "Applies to: both paths"
    Anyone deciding whether cash fits their scripts or their notebooks.

cash saves the time you spend re-running work whose code and inputs have not
changed. It does not make a first run faster: the first run stores results,
and later runs get them back instead of recomputing them.

## What each path does

| | `@cash.cache` (decorator) | `%cash_on` (notebook) |
|---|---|---|
| What is cached | One function call | Each top-level statement, and slow calls inside it |
| What the key is built from | Arguments, the function's source, the source of the helpers it calls, files it reads | The statement's code, how its input variables were made, files it reads |
| Edit a helper function | Recomputes | Recomputes |
| Change a data file | Recomputes | Recomputes |
| Change an object in place | A call caught changing its arguments is not stored | Detected: statements below see the change |
| Written to disk | Every result | Results that took more than 0.1 s and reload faster than they recompute |
| Side effects | Warned about; they run on the first call only | The statement is not cached |
| Shows what it did | `f.explain()`, `f.cache_info()` | A badge above each cell, `%cash_stats` |
| Needs IPython | No | Yes |

Details: [`@cash.cache` guide](decorator.md) and
[notebook guide](notebook_caching_api.md).

## When it helps, and when to skip it

cash helps when the same slow work runs again:

- **Decorator:** a function called again with the same arguments, in the same
  process or the next run: a pipeline step, a simulation, a paid API call.
- **Notebook:** a notebook you re-run while editing it, or after a kernel
  restart, with slow loads or transforms.

Skip it for:

- Work that takes a few milliseconds. Storing and reloading costs more than
  recomputing.
- Results that must differ on every run, such as timestamps or fresh random
  draws.
- Results that depend on something cash cannot see, such as a table in a
  database, unless you give them a `ttl` or an explicit dependency.

## Compared with other tools

✓ yes, ~ partly (the numbered note under the table says how far), ✗ no.
Click a column header to sort.

<div class="cash-matrix-table" markdown="1">

| Capability | `@cash.cache` | `%cash_on` | Manual pickle | `%store` | `lru_cache` | `joblib.Memory` | `diskcache` | `jupyter-cache` |
|---|---|---|---|---|---|---|---|---|
| Survives a restart | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[1](#matrix-note-1)</sup> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[2](#matrix-note-2)</sup> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[3](#matrix-note-3)</sup> |
| Recomputes when the function or cell changes | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[4](#matrix-note-4)</sup> |
| Recomputes when a called helper changes | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[5](#matrix-note-5)</sup> |
| Recomputes when a data file changes | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[6](#matrix-note-6)</sup> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> |
| Accepts DataFrames and arrays as arguments | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | n/a | n/a | n/a | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-partly" role="img" aria-label="Partly" title="Partly">~</span><sup>[7](#matrix-note-7)</sup> | n/a |
| Caches single statements in a notebook cell | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> |
| Tracks in-place changes between cells | n/a | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> |
| Works outside notebooks | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> |
| No code to add per function | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-no" role="img" aria-label="No" title="No">✗</span> | <span class="cash-mx cash-mx-yes" role="img" aria-label="Yes" title="Yes">✓</span> |

</div>

<ol class="cash-matrix-notes">
<li id="matrix-note-1"><code>%cash_on</code>, survives a restart: only results over 0.1 s that reload faster than they recompute.</li>
<li id="matrix-note-2">Manual pickle, survives a restart: only if you remember to dump and load.</li>
<li id="matrix-note-3"><code>jupyter-cache</code>, survives a restart: stores executed outputs for a book build, not for a live kernel.</li>
<li id="matrix-note-4"><code>jupyter-cache</code>, recomputes when a cell changes: re-runs the whole notebook when any code cell changes.</li>
<li id="matrix-note-5"><code>jupyter-cache</code>, recomputes when a helper changes: only for helpers defined in the notebook's own cells.</li>
<li id="matrix-note-6">Manual pickle, recomputes when a data file changes: only if you write the checks yourself.</li>
<li id="matrix-note-7"><code>diskcache</code>, accepts DataFrames and arrays: stores them, but does not hash them by content for the key.</li>
</ol>

- **`functools.lru_cache`** keeps results in memory for one process and needs
  hashable arguments.
- **`joblib.Memory`** stores results on disk and hashes the decorated
  function's own source, but not the helpers it calls, so editing a helper
  returns the old result.
- **`diskcache`** is a persistent key-value store. Its `memoize` keys on the
  function's name and arguments, so editing the function does not invalidate
  a result.
- **`%store`** saves and loads variables by hand. It never invalidates
  anything.
- **`jupyter-cache`** (behind Jupyter Book and MyST-NB) skips re-executing an
  unchanged notebook in a book build. Any code-cell edit re-runs the whole
  notebook.
- **marimo** is a reactive notebook with its own dependency-aware cache
  (`mo.cache`, `mo.persistent_cache`). Its cache
  [does not key on the source of imported modules](https://docs.marimo.io/api/caching/), and file dependencies are declared by hand
  (`mo.watch.file()`). marimo replaces Jupyter and stores notebooks as `.py`
  files; cash adds to the Jupyter, Colab or VS Code notebook you already have.

To switch from one of these, see [Coming from other caches](migration_guide.md).
Questions are answered in the [FAQ](faq.md).
