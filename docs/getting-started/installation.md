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
# several at once:
pip install "cash-lib[pandas,notebook]"
```

<!-- test_documented_install_extras_exist checks these names: cash-lib[pandas,polars,notebook,memory,redis,s3,ipynbname,viz,toml,all] -->

| Extra | What it adds | You need it when |
|---|---|---|
| `pandas` | pandas and pyarrow | you want stable DataFrame hashing |
| `polars` | polars | you use polars |
| `notebook` | IPython | you use `%cash_on` and the other magics |
| `memory` | psutil | you want memory-aware cache budgets |
| `redis` | the Redis client | you share a cache between machines through Redis |
| `s3` | boto3 | you store the cache in S3 |
| `ipynbname` | ipynbname | you want one more way to find the running notebook's file |
| `viz` | matplotlib and ipywidgets | you use the `cash.show_stats()` dashboard |
| `toml` | tomli, on Python 3.10 only | you set `[tool.cash]` in `pyproject.toml` on Python 3.10 |
| `all` | all of the above | you want everything |

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
