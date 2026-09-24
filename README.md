<!-- Built from docs/_brand/readme-header.html by scripts/build_brand_assets.py.
     Absolute raw URLs, not relative paths: PyPI renders this same README and
     does not resolve repo-relative image paths. PyPI's sanitizer also strips
     <picture>/<source>, which is why the light variant is the inner <img> --
     it survives as the fallback there, while GitHub honours the dark source. -->
<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="https://raw.githubusercontent.com/galgtonold/cash/main/docs/_brand/readme-header-dark.png">
  <img alt="Cash — only recompute what actually changed. Statement-level caching in Jupyter, @cash.cache anywhere else."
       src="https://raw.githubusercontent.com/galgtonold/cash/main/docs/_brand/readme-header-light.png">
</picture>

[![PyPI version](https://img.shields.io/pypi/v/cash-lib.svg)](https://pypi.org/project/cash-lib/)
[![Downloads](https://img.shields.io/pepy/dt/cash-lib.svg)](https://pepy.tech/project/cash-lib)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/cash-lib/)
[![CI](https://github.com/galgtonold/cash/actions/workflows/ci.yml/badge.svg)](https://github.com/galgtonold/cash/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/cash-lib/badge/?version=latest)](https://cash-lib.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#status)


cash is a Python cache that re-runs only what changed. It stores the results
of slow work and hands them back when the code and the inputs that produced
them have not changed, including the helper functions your code calls and the
data files it reads.

You use it in one of two ways:

- **In scripts, services and libraries:** decorate a slow function with
  `@cash.cache`.
- **In a notebook:** put `import cash` and `%cash_on` in the first cell, and
  every statement below it caches itself.

**What a restore costs:** a 100 MB DataFrame comes back from disk in **70 ms** on the machine the [benchmark table](https://cash-lib.readthedocs.io/en/latest/benchmarks/) was measured on. Your hardware and disk change that number, so the benchmarks page shows how to measure your own. cash quotes this rather than a speedup, because the speedup depends on how long your code takes to compute.

**[Read the docs →](https://cash-lib.readthedocs.io/en/latest/)**

## Install

```bash
pip install cash-lib
```

Extras for pandas, polars, Redis, S3 and more: see [installation](https://cash-lib.readthedocs.io/en/latest/getting-started/installation/).

## Decorator

```python
import cash
import pandas as pd

def clean(frame):                 # a plain function, not decorated
    return frame.dropna()

@cash.cache
def features(path):
    return clean(pd.read_csv(path))["feature_a"].mean()

features("large_dataset.csv")     # runs
features("large_dataset.csv")     # restored, and again in the next process
```

- The key is the arguments (DataFrames and arrays are hashed by content), the
  function's source and the source of the helpers it calls. Edit `clean` and
  the next call recomputes.
- Files the function reads are tracked: change `large_dataset.csv` and the
  next call recomputes.
- Every result is written to disk, so a new process gets it back.
- Side effects (HTTP calls, file writes) are flagged, because they run on the
  first call only.

[Decorator quick start](https://cash-lib.readthedocs.io/en/latest/getting-started/quickstart-script/) · [`@cash.cache` guide](https://cash-lib.readthedocs.io/en/latest/decorator/)

## Notebook

**Cell 1**, on its own:

```python
import cash
%cash_on
```

**Cell 2** and below, your code:

```python
import pandas as pd
df = pd.read_csv("large_dataset.csv")
summary = df.describe()
```

Re-run it, and cash puts a badge above the cell's output saying what it did:

<a href="https://cash-lib.readthedocs.io/en/latest/badges/">
  <img width="480" alt="A Cash badge below a notebook cell: the pd.read_csv row restored from cache, the df.describe row recomputed"
       src="https://raw.githubusercontent.com/galgtonold/cash/main/docs/_badges/status_mixed.png">
</a>

Here the file had not changed, so the load was restored; `describe()` had been
edited, so only that line ran.

- Each statement is cached on its own: edit one line and only that line and
  what depends on it re-run.
- Files a statement reads are tracked, and in-place changes such as
  `df["x"] = ...` or `items.append(...)` are detected.
- Expensive results are saved to disk and come back after a kernel restart.

A 90-second demo of cash in a real notebook:

https://github.com/user-attachments/assets/3f376660-aeb5-4794-89cc-532a04f82f32

Or try it in your browser with no install:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb)

[Notebook quick start](https://cash-lib.readthedocs.io/en/latest/getting-started/quickstart-notebook/) · [Reading the badge](https://cash-lib.readthedocs.io/en/latest/badges/)

## Learn more

- [Why cash?](https://cash-lib.readthedocs.io/en/latest/why-cash/): what each path does, and a [comparison](https://cash-lib.readthedocs.io/en/latest/why-cash/#compared-with-other-tools) with joblib, lru_cache, diskcache and others
- [FAQ](https://cash-lib.readthedocs.io/en/latest/faq/) · [Command-line interface](https://cash-lib.readthedocs.io/en/latest/cli/) · [Configuration](https://cash-lib.readthedocs.io/en/latest/getting-started/configuration/)
- [Writing cache-safe cells](https://cash-lib.readthedocs.io/en/latest/known-limitations/) · [Benchmarks](https://cash-lib.readthedocs.io/en/latest/benchmarks/) · [Versioning](https://cash-lib.readthedocs.io/en/latest/versioning/)

## Status

**Beta.** This is a `0.x` release: a minor release can change the API. cash clears a cache written in an older format by itself. See [versioning](https://cash-lib.readthedocs.io/en/latest/versioning/) and the [known limitations](https://cash-lib.readthedocs.io/en/latest/known-limitations/).

[How cash is tested](https://cash-lib.readthedocs.io/en/latest/how-it-works/testing/): <!-- docnum:tests_total -->~10,750<!-- /docnum --> tests across <!-- docnum:platforms -->15<!-- /docnum --> platform combinations, documentation whose code is executed and whose prose is pinned to the source, and what that still misses.

Bug reports are welcome: use the badge's "Report a bug" button, the links at the end of `%cash_help`, or open an [issue](https://github.com/galgtonold/cash/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are [tagged on the tracker](https://github.com/galgtonold/cash/issues?q=label%3A%22good+first+issue%22).

## License

MIT — see [LICENSE](LICENSE).
