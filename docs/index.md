# Cash - Smart Caching for Python

Cash is the only Python cache that works at **notebook-statement level**: turn it
on with `%cash_on`, change one line in a cell, and only that line and its
dependents re-run — everything else restores from cache, even across a kernel
restart. In the notebook: no decorators, no config, no manual invalidation.
Outside notebooks, the same engine wraps any function as `@cash.cache`.

> **New to cash?** **[Why Cash?](why-cash.md)** answers *"is this for my
> workflow?"* before you spend time on installation.

<video class="cash-hero-video" controls playsinline preload="none"
       poster="https://pub-f7df49dc5f45413aad945c29892e0566.r2.dev/hero-poster.jpg"
       style="width:100%;max-width:900px;border-radius:6px;margin:1.25em 0;display:block;">
  <source src="https://pub-f7df49dc5f45413aad945c29892e0566.r2.dev/hero_web.mp4" type="video/mp4">
  Your browser can't play this video — <a href="https://pub-f7df49dc5f45413aad945c29892e0566.r2.dev/hero_web.mp4">download it</a> instead.
</video>

<p style="text-align:center; margin:0 0 1.75em;">
  <a href="https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open the feature tour in Google Colab" style="display:inline; vertical-align:middle; height:28px;"></a>
  &nbsp;&nbsp;
  <a href="https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open the feature tour in Binder" style="display:inline; vertical-align:middle; height:28px;"></a>
  <br>
  <strong>Take the interactive feature tour in your browser — no install.</strong>
</p>

## Quick Example

<!-- test:skip reason="illustrative — references a missing large_file.csv" -->
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

Cash adds a badge **above each cell's output**, summarizing what it
did. The example below — restored upstream `df`, restored intermediate
`features`, recomputed `preds` because a new `features` lineage invalidated its
cache — gives you the visual vocabulary at a glance:

<iframe class="cash-badge" src="/_badges/anatomy_hero.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

See [Reading the Cash badge](badges.md) for the full anatomy.

## Two ways to use cash

Pick the path that matches how you write code — both ride the same engine:

<div class="cash-doc-grid" markdown="0">
  <a class="cash-cta-card primary" href="getting-started/quickstart/#in-a-notebook">
    <h3>In a notebook →</h3>
    <p>Add <code>%cash_on</code> and every cell caches itself — statement-level,
    with a badge above each cell. For interactive work in Jupyter, Colab, or VS Code.</p>
  </a>
  <a class="cash-cta-card" href="getting-started/quickstart/#in-a-script">
    <h3>In a script →</h3>
    <p>Wrap a function with <code>@cash.cache</code> — it caches by its arguments and
    its own source. For modules, pipelines, and batch jobs.</p>
  </a>
</div>

## Key Features

- **Statement-level, not cell-level** — change one line in a 20-line cell and
  only that line and its dependents recompute; the rest stay cached.
- **Dependency-aware** — Cash builds a lineage graph, so touching an upstream
  value re-runs only what transitively depends on it.
- **File-aware** — `pd.read_csv`, `np.load`, `open`, … are intercepted; swap a
  data file and the cells that read it recompute automatically.
- **Mutation-aware** — in-place changes like `lst.append(...)`, `d[k] = v`, and
  `+=` are detected, so a mutation invalidates what it touched.
- **Survives kernel restarts** — the cache lives on disk by default; restart,
  re-run the cell, get the value back instantly.
- **Zero-config** — `%cash_on` and you're done. No decorators, no config file.

See [Why Cash?](why-cash.md) for the full feature list and the [comparison
against other caching tools](why-cash.md#cash-vs-the-alternatives-youve-tried).

## Install

```bash
pip install cash-lib
```

Optional backends: `pip install "cash-lib[redis]"`, `[s3]`, or `[all]`. See
[Installation](getting-started/installation.md) for the full matrix.

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

## Documentation

<div class="cash-doc-grid" markdown="0">
  <a class="cash-cta-card primary" href="getting-started/quickstart/">
    <h3>Getting Started</h3>
    <p>Install cash and cache your first notebook.</p>
  </a>
  <a class="cash-cta-card" href="tutorials/feature-guides/purity-decorators/">
    <h3>Feature Guides</h3>
    <p>Deep dives on purity, backends, hashers, async, iterators, and more.</p>
  </a>
  <a class="cash-cta-card" href="tutorials/use-cases/data-science/">
    <h3>Use Cases</h3>
    <p>End-to-end recipes for data science, LLM calls, data engineering, and scientific computing.</p>
  </a>
  <a class="cash-cta-card" href="api/">
    <h3>API Reference</h3>
    <p>Autodoc reference for every public symbol.</p>
  </a>
  <a class="cash-cta-card" href="how-it-works/overview/">
    <h3>How Cash Works</h3>
    <p>How statement-level caching and the lineage graph work under the hood.</p>
  </a>
  <a class="cash-cta-card secondary" href="faq/">
    <h3>FAQ &amp; Glossary</h3>
    <p>Quick answers, troubleshooting, and the vocabulary in one place.</p>
  </a>
  <a class="cash-cta-card secondary" href="for-coding-agents/">
    <h3>For Coding Agents</h3>
    <p>Using cash from Claude Code, Copilot, or Cursor — and what an agent should read first.</p>
  </a>
</div>
