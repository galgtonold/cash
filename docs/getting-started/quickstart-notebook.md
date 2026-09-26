# Quick start: notebooks

!!! info "Applies to: notebook"
    Jupyter, JupyterLab, Colab and VS Code notebooks that use `%cash_on`.

Install cash first: `pip install cash-lib` (see [Installation](installation.md)).
To try it without installing, open the
[Colab](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb)
or [Binder](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb)
tour.

## 1. Turn cash on

Make this the first cell:

```python { .nb-cell }
import cash
%cash_on
```

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_on @503eaf12 -->
This is the setup cell: `import cash`, then any `cash.configure(...)` call, then
`%cash_on`. cash starts tracking after this cell, so work you put in it is never
cached. Do your work in the cells below. To skip this cell in every new kernel,
run `cash autoload on` once in a terminal.

## 2. Write your code in the cells below

<!-- test:expect-badge first=EXECUTED -->
```python { .nb-cell }
import pandas as pd

df = pd.read_csv("large_dataset.csv")
```

On the first run the load runs, and the badge above the output says
`EXECUTED`:

<iframe class="cash-badge" title="cash badge example: first run, EXECUTED" src="/_badges/quickstart_first_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

<p class="cash-badge-caption">Click a badge to open it.</p>

## 3. Run it again

Now the badge says `CACHED`: cash restored `df` instead of reading the file
again.

<iframe class="cash-badge" title="cash badge example: second run, CACHED" src="/_badges/quickstart_second_run.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

From here on:

- Edit one line and only that line and the statements that use its result
  re-run. The rest of the cell stays cached.
- Change `large_dataset.csv` on disk and the statements that read it re-run.
- Restart the kernel and run the cell: expensive results come back from disk.

## What's next

- [Notebook guide](../notebook_caching_api.md#what-gets-cached): what cash caches, what it
  refuses, and why.
- [Reading the badge](../badges.md): every status and reason.
- [Controlling caching](../tutorials/feature-guides/controlling-cache-behavior.md):
  `# @cash:` annotations for single statements.
- [Writing cache-safe cells](../known-limitations.md): the patterns cash cannot
  see.
- [Moving to a module](../tutorials/feature-guides/production-transition.md):
  when the notebook code goes to production.
