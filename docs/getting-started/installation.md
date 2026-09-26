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

Install only what you need, in quotes so the shell leaves the brackets alone:

```bash
pip install "cash-lib[pandas]"
pip install "cash-lib[pandas,notebook]"   # several at once
```

| Extra | What it adds | You need it when |
|---|---|---|
| `cash-lib[pandas]` | pandas and pyarrow | you want stable DataFrame hashing |
| `cash-lib[polars]` | polars | you use polars |
| `cash-lib[notebook]` | IPython | you use `%cash_on` and the other magics |
| `cash-lib[memory]` | psutil | you want memory-aware cache budgets |
| `cash-lib[redis]` | the Redis client | you share a cache between machines through Redis |
| `cash-lib[s3]` | boto3 | you store the cache in S3 |
| `cash-lib[ipynbname]` | ipynbname | you want one more way to find the running notebook's file |
| `cash-lib[viz]` | matplotlib and ipywidgets | you use the `cash.show_stats()` dashboard |
| `cash-lib[toml]` | tomli, on Python 3.10 only | you set `[tool.cash]` in `pyproject.toml` on Python 3.10 |
| `cash-lib[all]` | all of the above | you want everything |

`[all]` is exactly the extras above. The contributor extras (`dev`, `docs`,
`docs-test`) are not part of it; see [Contributing](../contributing.md).

## Check the install

```python
import cash
print(cash.__version__)
```

Then follow the quick start for your path:
[decorator](quickstart-script.md) or [notebook](quickstart-notebook.md).
