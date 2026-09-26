# Installation

!!! info "Applies to: both paths"
    Everyone installing cash, for scripts or for notebooks.

<!-- claim: cash/config.py:CashConfig @9e5fc345 broad="the requires-python floor and the one-install claim are properties of the package, checked against the docs by test_docs_install_cash_without_extras" -->
cash needs Python 3.10 or newer. One install gives you everything cash needs:

```bash
pip install cash-lib
```

The notebook path also needs IPython or Jupyter. In JupyterLab, the install
adds a small extension (`cash-live-cells`) that sends your unsaved cells to
the kernel, so cash sees your latest edits without a save.

## Packages for specific features

cash works with the libraries you already use, such as pandas and polars,
and does not install them. A few features need a package of their own:

| Feature | Install |
| --- | --- |
| Redis backend | `pip install redis` |
| S3 backend | `pip install boto3` |
| Tracking `s3://` or `gs://` reads | `pip install s3fs` or `pip install gcsfs` |
| Storing pandas DataFrames as Parquet (otherwise they are pickled) | `pip install pyarrow` |
| The `cash.show_stats()` dashboard in Jupyter | `pip install ipywidgets matplotlib` |

When a backend or a remote read needs a package you do not have, the error
names the package to install.

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
