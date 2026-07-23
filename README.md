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

That's the install. The next time you re-run the notebook, every statement that hasn't changed is restored from cache.

<!-- Demo video is hosted as a GitHub issue attachment (user-attachments CDN), so the
     binary stays out of the repo. GitHub auto-renders a bare attachment URL on its own
     line as an inline <video> player. Source upload: issue "Demo video — README asset".
     Note: this renders on GitHub only — PyPI/npm strip <video>; add a poster fallback there. -->

📺 **Watch the 90-second demo** — Cash caching a real notebook, end to end:

https://github.com/user-attachments/assets/6cd99ff1-68fe-4e8f-bf60-4572569cd34c

<!-- R2-hosted copy (awaiting link) — enable by replacing R2_MP4_URL and deleting these markers.
     Renders where the GitHub-attachment player is stripped (e.g. some mirrors).
<video src="R2_MP4_URL" controls muted loop playsinline width="100%"></video>
-->

> See [Reading the Cash badge](docs/badges.md) for a walkthrough of every part of the badge and the most common questions it answers.

**Or try it live in your browser** — no install, ~30 seconds:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/galgtonold/cash/blob/main/examples/try_cash.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/galgtonold/cash/main?labpath=examples/try_cash.ipynb)

---

## What you get

| Without Cash | With Cash |
|---|---|
| Edit one line → re-run → every cell above it recomputes | Edit one line → re-run → only that line and its dependents recompute |
| Kernel restart → rebuild all state by hand | Cache survives restarts; re-running rehydrates what's worth persisting |
| Cloud notebook bills for every re-run | Bills drop with cache hit rate |
| Sharing preprocessed data = "rerun the pipeline" | Export the cache, teammate imports → instant results |

### How much faster, honestly

**It depends on the ratio of compute cost to result size**, and the range is wide. Numbers below are measured by independent testers on real workloads, not projections:

| Workload | Measured |
|---|---|
| An expensive loop body (backtest windows, per-entity API calls) | **~190x** on the loop |
| Model training via `@cash.cache` | **~9–11x** |
| Monte Carlo / restart-and-re-run | **~4–5.5x** |
| Big-frame pandas ETL, written the way people naturally write pandas | **~1.2x** |
| The same ETL, restructured cache-friendly | **~1.4–1.6x** |

**The loop number depends on how the loop is written.** Assign the result (`out = [f(e) for e in items]`) and it caches at any length. Append into a list inside the loop and a long, multi-statement body can stop caching entirely. The badge says when that happens; [known-limitations](docs/known-limitations.md#a-long-for-append-loop-can-stop-caching) explains the conditions and what to change.

**Run #1 is slower** — around 1.3x on a big-frame ETL. Cash pays to fill the cache before it can pay you back; the win is on iteration and restart, not the first execution.

The pattern: Cash wins big when a statement is **expensive to compute and cheap to store** (a long fit, a slow API call, a heavy loop). It wins little, or loses, when the result is a multi-hundred-MB frame that is nearly as slow to load from disk as to recompute — Cash's cost model tries to detect this and decline to persist, and testers who overrode it with `# @cash:persist` measured it was right to.

`%cash_stats` will tell you which case you're in, and it will say so plainly when Cash cost you time.

> Full methodology and how to reproduce these numbers: [docs/benchmarks.md](docs/benchmarks.md).

---

## Why Cash is different

Most caching tools cache *cells* or *function calls*. Cash caches **statements** — the individual lines inside a cell — and tracks the dependency graph between them.

- **Statement-level, not cell-level.** Change one line in a 20-line cell → that line and its dependents recompute. The other 19 stay cached.
- **Dependency-aware.** Cash builds a lineage graph. Touch `config` → only cells that read `config` (transitively) re-run.
- **File-aware.** Cash intercepts `pd.read_csv`, `np.load`, `open`, etc. Replace `data.csv` → dependent cells recompute automatically. No hand-rolled `if mtime_changed` guards.
- **Mutation-aware.** `df.append(...)` and `+=` are detected so you don't get stale reads.
- **Survives kernel restarts.** Cache lives on disk by default. Restart, run the cell, get the value back instantly.
- **Zero-config.** `%cash_on` and you're done. No decorators, no config file, no setup.

---

## Install

```bash
pip install cash-lib
```

Optional extras:

```bash
pip install "cash-lib[pandas]"   # pandas + pyarrow for stable DataFrame hashing
pip install "cash-lib[all]"      # everything: pandas, polars, redis, s3, viz, ...
```

## Quick start

**Cell 1** — turn it on:

```python
import cash
%cash_on
```

(`import cash` auto-registers the magics. `%load_ext cash` also works if you prefer the IPython convention.)

**Cell 2** — your normal data-loading code:

```python
import pandas as pd
df = pd.read_csv("large_dataset.csv")   # tracked: file change → recompute
```

**Cell 3** — your normal analysis:

```python
summary = df.describe()
print(summary)
```

Re-run the notebook:

- ✅ Nothing changed → both cells restore from cache instantly.
- 🔄 `large_dataset.csv` was modified → Cell 2 and Cell 3 recompute.
- ⚡ Only Cell 3 changed → Cell 2 stays cached, Cell 3 recomputes.

## Function caching outside notebooks

<!-- test:skip reason="ends with cache_clear() which resets stats and breaks inferred hit/miss claims" -->
```python
import cash

@cash.cache
def expensive(x):
    return x ** 2 + sum(range(x))

expensive(1_000_000)            # first call: computed
expensive(1_000_000)            # cache hit
expensive.cache_info()
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5,
#  'total_time_saved': 0.045, 'warnings': []}

# Diagnose why a call hits or misses, without invoking the function:
expensive.explain(1_000_000)
# [HIT] __main__.expensive — hit
#   cache_key: ...
#   cached_at: 1779637032.79, execution_time_saved: 0.045

expensive.cache_clear()
```

> **In a notebook, use `explain()` rather than `cache_info()`.** The
> `cache_info()` counters live on the wrapper object, and cash may rebuild the
> cell that defines your function — producing a fresh wrapper with fresh
> counters. It can therefore read `{'hits': 0, 'misses': 0}` even while caching
> is working and saving you minutes. `explain()` and `%cash_stats` read through
> to the real cache, so they tell you the truth in either environment.

Decorate impure functions (LLM calls, HTTP fetches, file writes) and
Cash warns by default that the side effect will only run on the first
call. Pass `assume_safe=True` to silence after auditing, or
`strict=True` to make CI fail when impure code gets cached. Full
walkthrough in [docs/decorator.md](docs/decorator.md).

---

## Always-on caching across notebooks

Tired of typing `import cash` + `%cash_on` at the top of every notebook? Install a one-time IPython startup hook:

```bash
cash autoload on            # every new kernel auto-imports cash AND runs %cash_on
cash autoload on --mode=available   # only imports cash; you still call %cash_on per notebook
cash autoload off           # remove the hook
```

`on` (active mode, default) makes cash transparent: open any notebook, caching is already on. The badge still appears per cell, and `%cash_off` opts a single session out. `--mode=available` is the lighter option — just makes `import cash` happen for you so `%cash_on` and `@cash.cache` are reachable without an explicit import.

Under the hood the command writes a single file to `~/.ipython/profile_default/startup/00-cash.py`. It refuses to clobber an existing file there without `--force`, and `cash autoload off` refuses to remove a file it didn't write — so your own startup scripts are safe.

Full CLI reference: [docs/cli.md](docs/cli.md).

---

## Most-used magic commands

| Command | What it does |
|---|---|
| `%cash_on` | Turn caching on for every cell |
| `%cash_off` | Turn it off |
| `%cash_help` | Print a quick-reference card |
| `%cash_status` | Stats for the last cell |
| `%cash_stats` | Session totals (hits, misses, time saved) |
| `%cash_repair` | Fix corrupted cache; `--full` clears everything |
| `%cash_feedback` | How to report bugs / give feedback |

Full list: [docs/api/](docs/api/index.md).

## Backends

`TieredBackend` (RAM L1 + disk L2) is the default and the right choice for almost everyone. Other options if you need them: `InMemory`, `File`, `SQLite`, `Redis`, `S3`. See [docs/api/backends.md](docs/api/backends.md).

## Where does the cache live?

Disk cache: **`./.cash/`** next to your notebook (whatever the kernel's working directory is). One sub-file per cached statement, named by its cache-key SHA. Override with the `CASH_CACHE_DIR` environment variable, or pass `cache_dir=` when constructing a `Cash()` instance.

The default `TieredBackend` is **smart about what reaches disk**: cells that took **< 100 ms** stay RAM-only (disk I/O alone would cost more than rerunning). Above that floor, a fitted cost model promotes a result to disk when recomputing it would cost more than restoring it — so an expensive frame is persisted even when it's large, while a huge-but-cheap result that would be slower to reload than to recompute stays in RAM. Force-persist any cell with a `# @cash:persist` annotation when you know better than the heuristic.

To wipe the cache: delete `.cash/` or run `%cash_repair --full`.

---

## Status

**Beta.** `0.1.1` is the first public release. The public API is stabilizing, but this is a `0.x` release — the cache format may still change between minor versions, so run `%cash_repair --full` after upgrading (see the [versioning & compatibility policy](docs/versioning.md)). The [known limitations](docs/known-limitations.md) are documented honestly.

We want bug reports. The badge has a "Report a bug" button; or run `%cash_feedback`; or open an issue on [GitHub](https://github.com/galgtonold/cash/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are tagged on the [issue tracker](https://github.com/galgtonold/cash/issues?q=label%3A%22good+first+issue%22).

## License

MIT — see [LICENSE](LICENSE).
