# Cash — stop re-running notebooks that haven't changed

[![PyPI version](https://img.shields.io/pypi/v/cash-lib.svg)](https://pypi.org/project/cash-lib/)
[![Downloads](https://img.shields.io/pypi/dm/cash-lib.svg)](https://pypi.org/project/cash-lib/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/cash-lib/)
[![CI](https://github.com/galgtonold/cash/actions/workflows/ci.yml/badge.svg)](https://github.com/galgtonold/cash/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/cash-lib/badge/?version=latest)](https://cash-lib.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#status)

> **Two lines. Zero config. Restart-and-run-all in seconds instead of minutes.**

```python
import cash
%cash_on
```

That's it. The next time you re-run the notebook, every statement that hasn't changed is **restored from cache** instead of recomputed.

📺 **Watch the 90-second demo** — Cash caching a real notebook, end to end:

https://github.com/user-attachments/assets/3f376660-aeb5-4794-89cc-532a04f82f32

**Or try it live in your browser** — no install:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash_colab.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash_binder.ipynb)

**[Read the docs →](https://cash-lib.readthedocs.io/en/latest/)**

---

## Why Cash is different

Most caching tools cache *cells* or *function calls*. Cash caches **statements** — the individual lines inside a cell — and tracks the dependency graph between them.

- **Statement-level, not cell-level.** Change one line in a 20-line cell → that line and its dependents recompute. The other 19 stay cached.
- **Dependency-aware.** Cash builds a lineage graph. Touch `config` → only cells that read `config` (transitively) re-run.
- **File-aware.** Cash intercepts `pd.read_csv`, `np.load`, `open`, etc. Replace `data.csv` → dependent cells recompute automatically.
- **Mutation-aware.** `df.append(...)` and `+=` are detected, so you don't get stale reads.
- **Survives kernel restarts.** The cache lives on disk by default. Restart, run the cell, get the value back instantly.
- **Zero-config.** `%cash_on` and you're done. No decorators, no config file.

Cash saves time on **re-runs** — restoring an unchanged result instead of recomputing it, not speeding the first execution up. The more a statement costs to compute relative to the size of its result, the more a restore saves; `%cash_stats` reports your actual numbers, and says so plainly when caching cost you time. See the [benchmarks](https://cash-lib.readthedocs.io/en/latest/benchmarks/) for how that plays out on real workloads.

---

## Install

```bash
pip install cash-lib
```

Optional extras: `pip install "cash-lib[all]"` (pandas, polars, redis, s3, …). See [installation](https://cash-lib.readthedocs.io/en/latest/getting-started/installation/).

## Quick start

**Cell 1** — turn it on:

```python
import cash
%cash_on
```

**Cell 2** — your normal code:

```python
import pandas as pd
df = pd.read_csv("large_dataset.csv")   # tracked: file change → recompute
summary = df.describe()
```

Re-run the notebook:

- ✅ Nothing changed → restored from cache instantly.
- 🔄 `large_dataset.csv` changed → the cells that read it recompute.
- ⚡ Only the analysis changed → the load stays cached, only the analysis recomputes.

Cash shows a badge above each cell summarizing what it did — see [Reading the Cash badge](https://cash-lib.readthedocs.io/en/latest/badges/).

## Beyond notebooks

`@cash.cache` caches any Python function across processes, with the same dependency- and file-awareness. Impure functions (LLM calls, HTTP, file writes) are flagged by default, since their side effects only run on the first call.

```python
import cash

@cash.cache
def expensive(x):
    return x ** 2 + sum(range(x))
```

Full walkthrough in [the decorator guide](https://cash-lib.readthedocs.io/en/latest/decorator/).

---

## Learn more

- [Documentation home](https://cash-lib.readthedocs.io/en/latest/) · [Why Cash?](https://cash-lib.readthedocs.io/en/latest/why-cash/) · [Quick start](https://cash-lib.readthedocs.io/en/latest/getting-started/quickstart/)
- [Reading the badge](https://cash-lib.readthedocs.io/en/latest/badges/) · [Magic commands](https://cash-lib.readthedocs.io/en/latest/magics/) · [Annotations](https://cash-lib.readthedocs.io/en/latest/annotations/)
- [The decorator](https://cash-lib.readthedocs.io/en/latest/decorator/) · [Backends](https://cash-lib.readthedocs.io/en/latest/api/backends/) · [Command-line interface](https://cash-lib.readthedocs.io/en/latest/cli/)
- [Known limitations](https://cash-lib.readthedocs.io/en/latest/known-limitations/) · [Benchmarks](https://cash-lib.readthedocs.io/en/latest/benchmarks/) · [Versioning & compatibility](https://cash-lib.readthedocs.io/en/latest/versioning/)

## Status

**Beta.** `0.2.0` is the current release. The public API is stabilizing, but this is a `0.x` release — the cache format may still change between minor versions, so run `%cash_repair --full` after upgrading (see [versioning & compatibility](https://cash-lib.readthedocs.io/en/latest/versioning/)). The [known limitations](https://cash-lib.readthedocs.io/en/latest/known-limitations/) are documented honestly.

Bug reports welcome: the badge has a "Report a bug" button, `%cash_feedback` prints how, or open an [issue](https://github.com/galgtonold/cash/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are [tagged on the tracker](https://github.com/galgtonold/cash/issues?q=label%3A%22good+first+issue%22).

## License

MIT — see [LICENSE](LICENSE).
