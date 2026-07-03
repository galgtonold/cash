# Cash — stop re-running notebooks that haven't changed

[![PyPI version](https://img.shields.io/pypi/v/cash-lib.svg)](https://pypi.org/project/cash-lib/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/cash-lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](#status)

> ⚠️ **Work in progress — not released yet.** Cash is under active development and has not been published to PyPI. The install instructions below won't work yet, and APIs and the cache format may change without notice. Watch the repo for the first release.

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

> See [Reading the Cash badge](docs/badges.md) for a walkthrough of every part of the badge and the most common questions it answers.

---

## What you get

| Without Cash | With Cash |
|---|---|
| Edit one line → re-run notebook → wait 3 minutes for the CSV to reload | Edit one line → re-run → ~5 seconds, only the changed line recomputes |
| 10 runs/day × 3 min each = **30 min wasted** | 10 runs/day × ~10 sec each = **2 min total** |
| Cloud notebook bills for every re-run | Cache survives kernel restarts; bills drop with cache hit rate |
| Sharing preprocessed data = "rerun the pipeline" | Export the cache, teammate imports → instant results |

For a team of 10, that's roughly **$75K/year in recovered productivity**, on top of cloud compute savings.

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

The default `TieredBackend` is **smart about what reaches disk**: cells that took **< 100 ms** stay RAM-only (disk I/O alone would cost more than rerunning), while 100 ms – 1 s cells with small results (< 64 KB) always persist so a fresh kernel can restore them. For heavier intermediates, the `smart_persistence_threshold` config knob (default 1 s) decides. Force-persist any cell with a `# @cash:persist` annotation when you know better than the heuristic.

To wipe the cache: delete `.cash/` or run `%cash_repair --full`.

---

## Status

**Pre-release / work in progress.** Cash has not been published to PyPI yet. The public API is stabilizing, but anything — including the cache format — may break before the first release; assume `%cash_repair --full` after upgrading.

This is the first public release. We want bug reports. The badge has a "Report a bug" button; or run `%cash_feedback`; or open an issue on [GitHub](https://github.com/galgtonold/cash/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Good first issues are tagged on the [issue tracker](https://github.com/galgtonold/cash/issues?q=label%3A%22good+first+issue%22).

## License

MIT — see [LICENSE](LICENSE).
