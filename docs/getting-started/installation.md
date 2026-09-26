# Installation

!!! info "Applies to: both paths"
    Everyone installing cash, for scripts or for notebooks.

<!-- claim: cash/config.py:CashConfig @9e5fc345 broad="the requires-python floor and zero-deps claim are properties of the package, checked against pyproject by test_documented_install_extras_exist" -->
cash needs Python 3.10 or newer and has no required dependencies:

```bash
pip install cash-lib
```

The notebook path also needs IPython or Jupyter. In JupyterLab, the install
adds a small extension (`cash-live-cells`) that sends your unsaved cells to
the kernel, so cash sees your latest edits without a save.

## Optional extras

Install only what you need:

<!-- test:long-lines reason="install line, being rewritten" -->
```bash
pip install "cash-lib[pandas]"      # pandas + pyarrow, for stable DataFrame hashing
pip install "cash-lib[polars]"      # polars support
pip install "cash-lib[notebook]"    # IPython, for %cash_on and the other magics
pip install "cash-lib[memory]"      # psutil, for memory-aware cache budgets
pip install "cash-lib[redis]"       # Redis backend, for a cache shared between machines
pip install "cash-lib[s3]"          # S3 backend (boto3)
pip install "cash-lib[ipynbname]"   # one more way to find the running notebook's file
pip install "cash-lib[viz]"         # matplotlib + ipywidgets, for the cash.show_stats() dashboard
pip install "cash-lib[toml]"        # tomli, to read [tool.cash] in pyproject.toml on Python 3.10
pip install "cash-lib[all]"         # all of the above
```

`[all]` is exactly the extras above. The contributor extras (`dev`, `docs`,
`docs-test`) are not part of it; see [Contributing](../contributing.md).

## Check the install

```python
import cash
print(cash.__version__)
```

## Related

- [Decorator quick start](quickstart-script.md): cache a slow function.
- [Notebook quick start](quickstart-notebook.md): cache a notebook statement
  by statement.
- [Configuration](configuration.md): where the cache goes and how big it may
  grow.
