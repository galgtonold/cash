# cash: re-run only what changed

!!! info "Applies to: both paths"
    Everyone. Pick the path that matches how you write code.

cash is a Python cache. It stores the results of slow work and hands them back
when the code and the inputs that produced them have not changed. You use it in
one of two ways:

<div class="cash-doc-grid" markdown="0">
  <a class="cash-cta-card" href="getting-started/quickstart-script/">
    <h3>Decorator: <code>@cash.cache</code></h3>
    <p>Wrap a slow function. For scripts, services, pipelines and libraries.</p>
  </a>
  <a class="cash-cta-card" href="getting-started/quickstart-notebook/">
    <h3>Notebook: <code>%cash_on</code></h3>
    <p>Turn it on once and every statement caches itself. For Jupyter, Colab and VS Code.</p>
  </a>
</div>

## Decorator

<!-- test:skip reason="illustrative: a 2-second sleep stands in for real work" -->
```python
import time
import cash

@cash.cache
def slow_square(x):
    time.sleep(2)        # stand-in for real work
    return x * x

slow_square(4)           # runs: 2 seconds
slow_square(4)           # cache hit: instant, also in the next process
```

- The key is the arguments (DataFrames and arrays are hashed by content), the
  function's source, and the source of the helpers it calls.
- Files the function reads are tracked: change the file and the call recomputes.
- Every result is written to disk, so a new process gets it back.
- Works in any Python script; no notebook needed.

[Decorator quick start](getting-started/quickstart-script.md)

## Notebook

Make this the first cell, and work in the cells below it:

```python { .nb-cell }
import cash
%cash_on
```

<!-- test:skip reason="illustrative: sales.csv is not shipped with the docs" -->
```python { .nb-cell }
import pandas as pd

df = pd.read_csv("sales.csv")
summary = df.groupby("region").sum()
```

- Each statement is cached on its own: edit one line and only that line and
  what depends on it re-run.
- Files a statement reads are tracked, and in-place changes such as
  `df["x"] = ...` are detected.
- Expensive results are saved to disk and come back after a kernel restart.
- A badge above each cell's output says what cash restored and what ran.

<iframe class="cash-badge" title="cash badge example: a cell restored from the cache" src="/_badges/anatomy_hero.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<p class="cash-badge-caption">Click a badge to open it.</p>

Try it in your browser with no install:
[Colab](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb) ·
[Binder](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb).

[Notebook quick start](getting-started/quickstart-notebook.md)

## Install

```bash
pip install cash-lib
```

That one install is all cash needs. A few features, such as the Redis
backend, need a package of their own; see
[Installation](getting-started/installation.md).

!!! danger "Only load caches you trust"
    cash stores results with `pickle`, and loading a pickle can run code. Treat
    a cache from someone else like a script from them. Your own `.cash/` folder
    is as safe as the code that wrote it. See the
    [trust model](api/backends.md#security).

## Related

- [Why cash?](why-cash.md): what each path does, and how cash compares with
  other caches.
- [FAQ](faq.md) and [Glossary](glossary.md): short answers and the words the
  docs use.
- [How it works](how-it-works/overview.md): how cash decides what to reuse.
- [For coding agents](for-coding-agents.md): the page an agent should read
  first.
- Look something up: [Settings](getting-started/configuration.md#all-settings),
  [CLI](cli.md), [Warnings](warnings.md), [Magics](magics.md),
  [Annotations](annotations.md), [API](api/index.md).
