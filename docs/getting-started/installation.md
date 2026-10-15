# Installation

!!! info "Applies to: both paths"
    Everyone installing cash, for scripts or for notebooks.

<!-- claim: cash/config/schema.py:CashConfig @78378872 broad="the requires-python floor and the one-install claim are properties of the package, checked against the docs by test_docs_install_cash_with_no_extra_but_fast" -->
cash needs Python 3.10 or newer. One install gives you everything cash needs:

```bash
pip install cash-lib
```

cash does not install Jupyter or IPython: the notebook path runs in the
kernel you already use. In JupyterLab, the install adds a small extension (`cash-live-cells`) that sends your unsaved cells to
the kernel, so cash sees your latest edits without a save.

<!-- claim: cash/bulk_digest.py:_scheme @93bd31ed broad="the [fast] extra and the per-algorithm tags, checked by test_big_buffers_hash_with_blake3_when_it_is_installed" -->
## Faster hashing of big data

cash reads every byte of a big array, column or input file to tell whether
it changed. The `fast` extra adds the `blake3` package, and cash then hashes
that data with BLAKE3 instead of SHA-256:

```bash
pip install "cash-lib[fast]"
```

On a CPU without SHA instructions, BLAKE3 hashes a 100 MiB array in about a
tenth of the time; on one with them (most recent x86 and ARM chips) the gain
is smaller. Nothing else changes, and nothing needs configuring.

Entries written by an install with `blake3` and one without are keyed
apart, so a cache folder shared between the two only misses for the other's
big inputs and computes them again; it never serves one install's entry to
the other. Give every machine that shares a cache (a team's S3 or Redis
backend, CI and laptops) the same install to keep their hits.

<!-- claim: cash/_native.py:_load @1ea87886, cash/__main__.py:cmd_info @3e16686c, cash/notebook/ipython/magics.py:CashMagics.cash_status @a494ae38 -->
## The C fast path

On CPython 3.12 or newer, pip installs a wheel with a small C extension for
Linux (x86-64 and ARM64), macOS (Intel and Apple silicon) and Windows
(x86-64). It tells cash which values have provably not changed since it
last hashed them, so cash skips hashing them again. Nothing needs
configuring.

Everywhere else, on Python 3.10 and 3.11, other interpreters or other
platforms, pip installs the pure-Python wheel and cash hashes those values
as it always does. Results, cache keys and cache folders are the same with
and without the extension, so installs of both kinds share a cache. When pip
builds cash from source, it compiles the extension if it finds a C compiler
and installs without it if it does not; the install never fails over it.

To see whether the fast path is on, and why not when it is off:

=== "Decorator"

    Run `cash info` in a terminal. The `Fast path` line says `on`, or `off`
    with the reason.

=== "Notebook"

    Run `%cash_status` in a cell. Its `fast_path` entry says `on`, or `off`
    with the reason.

Set `CASH_PURE_PYTHON=1` in the environment to run without the extension.
Set while installing from source, it builds cash without the extension.

## Packages for specific features

cash works with the libraries you already use, such as pandas and polars,
and does not install them. A few features need a package of their own:

| Feature | Package |
| --- | --- |
| Redis backend | `redis` |
| S3 backend | `boto3` |
| Tracking `s3://` or `gs://` reads | `s3fs` or `gcsfs` |
| The `cash.show_stats()` dashboard in Jupyter | `ipywidgets` and `matplotlib` |

Install the one you need with pip, for example the Redis client:

```bash
pip install redis
```

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
