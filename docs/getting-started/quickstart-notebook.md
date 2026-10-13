# Quick start: notebooks

!!! info "Applies to: notebook"
    Jupyter, JupyterLab, Colab and VS Code notebooks that use `%cash_on`.

Three steps take a notebook from no cache to a cached re-run. Install cash
first: `pip install cash-lib` (see [Installation](installation.md)).
To try it without installing, open the
[Colab](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb)
or [Binder](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb)
tour. To see it first, watch the [30-second video](../index.md#notebook).

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

<!-- claim: cash/tracking/function_tracker.py:FunctionTracker.check_and_reload_changed_modules @9a04e5bd -->
**If you edit your own `.py` modules**, cash reloads an edited module by
itself, so the next cell uses the new code. Plain Jupyter keeps the old code
until you restart the kernel. A notebook that imports a module you edit then
gives different results once cash is off (`%cash_off`, `CASH_DISABLE=1`, or a
colleague without cash). Turn on IPython's autoreload in the cell that
imports the module, and the notebook behaves the same both ways; cash works
with it:

<!-- test:skip reason="illustrative: mylib stands for a module of your own" -->
```python { .nb-cell }
%load_ext autoreload
%autoreload 2
import mylib
```

The two magic lines run on every run and their badge rows say `NOT CACHED`
(`Runs an IPython magic`): that is expected, and so is IPython's own "The
autoreload extension is already loaded" when you run the cell again. With both
on, an edit to `mylib.py` is reloaded once, and the badge shows one
`MODULE RELOADED` row.

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

Two habits keep the slow statements cached:

- **Fit a model in the assignment that binds it**,
  `model = Ridge().fit(X, y)`: a separate `model.fit(X, y)` changes `model`
  in place and runs every time. See
  [Fitting models](../notebook_caching_api.md#fitting-models).
- **Build a new value instead of changing one from an earlier cell**,
  `df = df.assign(c=...)` rather than `df["c"] = ...`: a change in place runs
  every time. See [What gets cached](../notebook_caching_api.md#what-gets-cached).

## Related

- [Notebook guide](../notebook_caching_api.md#what-gets-cached): what cash caches, what it
  refuses, and why.
- [Reading the badge](../badges.md): every status and reason.
- [Controlling caching](../tutorials/feature-guides/controlling-cache-behavior.md):
  `# @cash:` annotations for single statements.
- [Writing cache-safe cells](../known-limitations.md): the patterns cash cannot
  see.
- [Moving to a module](../tutorials/feature-guides/production-transition.md):
  when the notebook code goes to production.
