# Notebook guide

!!! info "Applies to: notebook"
    Jupyter, JupyterLab, VS Code and Colab notebooks. For scripts and modules, see
    the [decorator guide](decorator.md).

With `%cash_on`, cash caches every statement you run in a notebook. When you run
a cell again, statements whose code and inputs have not changed come back from
the cache; the rest run. Results survive a kernel restart. This page covers how
to turn it on, what is and is not cached, and where the cache lives.

## Turn it on

The first cell is the setup cell:

```python { .nb-cell }
import cash
%cash_on
```

It holds `import cash`, then any `cash.configure(...)` call, then `%cash_on`.
Nothing in the `%cash_on` cell is cached, so load data and do work in the cells
below it. `import cash` registers the magics. `%cash_off` switches caching off
again. With `CASH_DISABLE=1` set, `%cash_on` says so and does nothing.

To change a setting, put it in `pyproject.toml` or an environment variable, or
call `cash.configure(...)` between the two setup lines:

<!-- test:skip reason="illustrative: shows the first cell with a setting" -->
```python { .nb-cell }
import cash
cash.configure(min_cache_savings_pct=0.5)
%cash_on
```

`%cash_on ttl=3600` makes every stored result expire after an hour. All settings
are in [Configuration](getting-started/configuration.md).

## What cash caches

Cash caches each **statement** on its own, keyed on the statement's code and on
where its inputs came from. Edit one statement and only it, and what reads its
result, run again:

<!-- test:skip reason="illustrative: references data.csv" -->
```python { .nb-cell }
df = pd.read_csv("data.csv")       # cached
daily = df.resample("D").mean()    # cached
result = daily.rolling(7).mean()   # edit: only this runs again
```

- **Functions you call are tracked.** Edit a function, or a helper it calls, even
  in your own `.py` module, and the statements that use it run again.
- **Files you read are tracked.** Change a file read through `pd.read_csv`,
  `np.load`, `open` or another [tracked reader](how-it-works/invalidation.md#what-counts-as-a-change),
  and the statements that read it run again.
- **Loops are cached per iteration**, keyed on the loop variable, so changing one
  item leaves the others cached. `if` and `try` blocks are cached per branch. A
  `while` or `with` block, or a `for` with `break`, `continue` or `else:`, runs
  as one unit.
- **Calls inside a statement are cached too.** In `out.append(compute(x))` the
  append runs every time, but `compute(x)` comes from the cache.
- **Printed output is replayed** on a hit, along with rich output. A trailing
  `;` still hides it.
- **Top-level `await` cells** are cached like any other; a hit skips the `await`.

Every cell shows a [badge](badges.md) with one row per statement:

<iframe class="cash-badge" src="/_badges/status_restored.html" loading="lazy" scrolling="no" height="40" style="width:100%;border:0;display:block;margin:8px 0;"></iframe>

## What gets cached

Some statements always run, because a cache hit would skip something that has to
happen or would freeze a value that has to change. Others are cached although
they talk to the outside world.

<!-- claim: cash/analysis/file_effects.py:NOTEBOOK_POLICY @5ffd29f3 -->
| A statement that... | What cash does |
|---|---|
| writes a file (`open(p, "w")`, `df.to_csv`, `fig.savefig`), directly or through a function you wrote (in the notebook or your own module) | runs every time |
| sends something over the network (`requests.post`, `session.post`, `client.publish`, `s3.upload_file`) | runs every time |
| writes to a database (`INSERT`, `commit`, `df.to_sql`) | runs every time |
| starts a process (`subprocess.run`, `os.system`) | runs every time |
| reads the clock (`datetime.now()`, `time.time()`, `uuid4()`) or asks for `input()` | runs every time |
| draws on the current pyplot figure (`plt.plot`, `plt.show`) | runs every time |
| calls a function marked [`@stateful`](tutorials/feature-guides/controlling-cache-behavior.md#stateful-helpers) | runs every time |
| changes an object made in an earlier cell (`df["c"] = ...`, `lst.append(...)`) | runs every time |
| takes under 10 ms | runs every time: too cheap to store |
| reads over the network (`requests.get`, `session.get`) | cached |
| reads a database (`SELECT`, `pd.read_sql`) | cached |
| prints | cached; the output is replayed on a hit |
| reads a file | cached; runs again when the file's contents change |
| reads an environment variable (`os.getenv`) | cached; the value is part of the key, so a new value runs it again |

Only effects written in the statement count, plus file writes in your own
functions. A clock read inside a function you call is not seen; see
[Writing cache-safe cells](known-limitations.md#a-helper-that-reads-the-clock-is-frozen).
Unseeded random draws are cached and replayed; see
[Randomness](known-limitations.md#randomness).

Three directives change these rules for one statement:

- [`# @cash:no-cache`](annotations.md#cashno-cache): run it every time. Use it for
  what cash cannot see, such as a GET whose answer must be fresh.
- [`# @cash:assume-safe`](annotations.md#cashassume-safe): cache it despite a side
  effect that is harmless to skip, such as a POST that only runs a search.
- [`# @cash:persist`](annotations.md#cashpersist): store it on disk even though it
  was quick to compute.

[Controlling caching](tutorials/feature-guides/controlling-cache-behavior.md)
walks through them. [Knowing when not to cache](how-it-works/safety.md) explains
the reasoning.

## After a restart

The cache is on disk, so a fresh kernel picks up where you left off. Run a cell
after a restart and cash restores its inputs from the cache instead of running
the cells that built them, after checking that the code and files behind them
have not changed. Results that took less than 0.1 s to compute are kept in
memory only and run again after a restart.
[Restarts and persistence](tutorials/feature-guides/smart-persistence.md) has the
details, and the [cost model](cost-model.md) has the thresholds.

## Where the cache is

The cache is a `.cash` folder in the kernel's working directory, which is
usually the notebook's folder. Notebooks in the same folder share it. Set
`CASH_CACHE_DIR` or `[tool.cash] cache_dir` to put it elsewhere. The folder holds
a `.gitignore`, so git ignores it.

<!-- claim: cash/backends/adaptive_caps.py:DISK_FLOOR, cash/backends/adaptive_caps.py:DISK_CEILING, cash/backends/adaptive_caps.py:RAM_FLOOR, cash/backends/adaptive_caps.py:RAM_CEILING -->
By default the disk cache may use a quarter of the free disk space, at least
8 GiB and at most 100 GiB, and the in-memory cache a fifth of your RAM, between
512 MiB and 4 GiB. When either is full, cash removes the entries least worth
keeping. See [Where your cache lives](how-it-works/storage.md#where-the-cache-folder-is).

## Inspect, clear and share

From a terminal in the notebook's folder, or from a cell with a leading `!`:

```bash
cash info          # where the cache is and how big
cash inspect       # entries by size, with the time each saves
cash clear --all   # delete the whole cache
```

After `cash clear`, restart the kernel so it does not keep using values it has in
memory. `cash clear --function NAME` and `--entry ID` remove less. All commands
are in the [CLI reference](cli.md).

To let colleagues or CI reuse your results, see
[Sharing a cache](tutorials/feature-guides/sharing-caches.md).

## Running a notebook headless

Under papermill, nbconvert or CI:

- Add `%cash_badge print` to the first cell, after `%cash_on`, for a
  plain-text badge that reads well in logs.
- Expect one `NOTEBOOK-NOT-FOUND` warning. With no Jupyter server there is no
  notebook file to read, so cash does not re-check earlier cells. A
  top-to-bottom run does not need that. In JupyterLab or VS Code the warning
  means the kernel is stale: restart it.
- Point `CASH_CACHE_DIR` at a folder your CI keeps between runs, or start clean
  with `cash clear --all`.

## Where to go next

- [Reading the badge](badges.md): every status and reason.
- [Controlling caching](tutorials/feature-guides/controlling-cache-behavior.md)
  and [Annotations](annotations.md): the `# @cash:` directives and `@stateful`.
- [Writing cache-safe cells](known-limitations.md): patterns cash cannot follow.
- [Debugging](tutorials/feature-guides/debugging-and-monitoring.md): when a cell
  does not do what you expect.
- [Magic commands](magics.md): the full magic reference.
- [Moving to a module](tutorials/feature-guides/production-transition.md): from a
  notebook to `@cash.cache`.
- [The notebook path](how-it-works/notebook-path.md): how a cell runs, step by
  step.
