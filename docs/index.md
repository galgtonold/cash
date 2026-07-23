# Cash - Smart Caching for Python

> **New to cash?** Read **[Why Cash?](why-cash.md)** first — it answers
> *"is this for my workflow?"* before you spend time on installation.

Cash is a transparent caching library for Python with two primary use cases —
**Jupyter notebook caching** (add `%cash_on` and cell results cache themselves)
and **decorator-based caching** (wrap any function with `@cash.cache`).

## Quick Example

```python { .nb-cell }
import cash
import pandas as pd

# Enable auto-caching for all cells
%cash_on

# First run: executes normally (~5 seconds)
df = pd.read_csv('large_file.csv')
result = df.groupby('category').sum()

# Second run: instant from cache ✅
```

Cash adds a badge as the **first output below each cell**, summarizing what it
did. The example below — restored upstream `df`, restored intermediate
`features`, recomputed `preds` because a new `features` lineage invalidated its
cache — gives you the visual vocabulary at a glance:

<iframe class="cash-badge" src="/_badges/anatomy_hero.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

See [Reading the Cash badge](badges.md) for the full anatomy.

## Key Features

- **Statement-level, not cell-level** — change one line in a 20-line cell and
  only that line and its dependents recompute; the rest stay cached.
- **Dependency-aware** — Cash builds a lineage graph, so touching an upstream
  value re-runs only what transitively depends on it.
- **File-aware** — `pd.read_csv`, `np.load`, `open`, … are intercepted; swap a
  data file and the cells that read it recompute automatically.
- **Mutation-aware** — in-place changes like `df.append(...)` and `+=` are
  detected, so you never read a stale cache.
- **Survives kernel restarts** — the cache lives on disk by default; restart,
  re-run the cell, get the value back instantly.
- **Zero-config** — `%cash_on` and you're done. No decorators, no config file.

See the full feature list and the [comparison against other caching
tools](why-cash.md#cash-vs-the-alternatives-youve-tried) in [Why Cash?](why-cash.md).

!!! danger "The cache is executable — only load caches you trust"
    Cash persists results by **pickling** Python objects (the file, SQLite,
    Redis, and S3 backends all use `pickle`). Unpickling runs arbitrary code,
    so **loading a cache is equivalent to running a Python script from
    whoever produced it.**

    - **Don't** open someone's exported `.cash/` directory, or point Cash at a
      Redis/S3 store that untrusted parties can write to, unless you trust the
      source as much as you'd trust running their code.
    - **Do** treat a shared cache like a shared `.py` file: fine from a
      teammate on your own infrastructure, dangerous from a stranger.

    Your own local `.cash/` directory is as safe as the code that wrote it.
    See [Backends](api/backends.md#security) for the full trust model.

## Install

```bash
pip install cash-lib
```

Optional backends: `pip install "cash-lib[redis]"`, `[s3]`, or `[all]`. See
[Installation](getting-started/installation.md) for the full matrix.

## Documentation

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install Cash and cache your first notebook.

    [Quick Start →](getting-started/quickstart.md)

-   **Feature Guides**

    ---

    Deep dives on purity, backends, hashers, async, iterators, and more.

    [Browse guides →](tutorials/feature-guides/purity-decorators.md)

-   **Use Cases**

    ---

    End-to-end recipes for data science, LLM API calls, data engineering, and
    scientific computing.

    [See use cases →](tutorials/use-cases/data-science.md)

-   **API Reference**

    ---

    Autodoc reference for every public symbol.

    [Open the API →](api/index.md)

-   **How Cash Works**

    ---

    How statement-level caching and the lineage graph work under the hood.

    [Read the internals →](how-it-works/overview.md)

-   **FAQ & Glossary**

    ---

    Quick answers, troubleshooting, and the vocabulary in one place.

    [Open the FAQ →](faq.md)

</div>
