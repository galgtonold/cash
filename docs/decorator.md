---
search:
  boost: 2
---

# The `@cash.cache` guide

!!! info "Applies to: decorator"
    Scripts, services and libraries that use `@cash.cache`.

Put `@cash.cache` on a slow function and cash stores its result. The next call
with the same arguments, in this run or a later one, returns the stored result
instead of running the body. This page covers where results go, how to see what
cash did, what makes an entry go stale, the parameters, and the limits.

Good candidates are file loads, ETL steps, model fits, simulations and paid
API calls. A function that takes microseconds gains nothing: building the key
costs more than the body.

## The minimum

```python
import cash

@cash.cache
def slow_square(n):
    return sum(i * i for i in range(n))

slow_square(1_000_000)   # first call: runs the body, stores the result
slow_square(1_000_000)   # cache hit: returns the stored result
```

That is all the setup there is. A few rules hold for every cached function:

<!-- claim: cash/decorator/store.py:ResultStore.refusal @cf16cee7, cash/decorator/store.py:ResultStore.store @e289b59d, cash/decorator/store.py:ResultStore.restore_identity @2e52815d -->
- **Exceptions are never cached.** If the body raises, nothing is stored and the
  exception reaches you as usual. The next call runs the body again.
- **A hit does not replay output.** Anything the body printed or logged appears
  only on the call that ran it.
- **A hit returns a copy.** The value is rebuilt from the stored bytes, so
  writing into a result you got from a hit never changes what the next caller
  gets. The copy keeps what the result was: a read-only numpy array is
  read-only again, and a module global or closure variable the function
  names, compared by identity (a sentinel such as `MISSING = object()` in
  `return d.get(key, MISSING)`), comes back as that very object, so
  `is MISSING` holds on a hit. That holds for the result itself and for such
  an object held inside it, in a list, tuple or dict or in an attribute of
  an object of your own class (`[d.get(k, MISSING) for k in keys]`). An object whose class has been
  defined again since it was stored (`importlib.reload` of its module) comes
  back as an instance of the class its module names now, as it does when
  read from disk.

To configure your own instance instead of the shared default, create a `Cash`
and use its `cache` method:

```python
from cash import Cash

app = Cash(cache_dir="./my_app_cache")

@app.cache
def slow_cube(n):
    return sum(i * i * i for i in range(n))

slow_cube(1000)   # first call: runs the body
slow_cube(1000)   # cache hit, from ./my_app_cache
```

`Cash(...)` takes `cache_dir`, `backend` or `backends`, `use_locking`, `debug`,
`verbose`, `register_magic`, `config_path`, and any configuration field as a
keyword (`summary=True`, `max_cache_size="5GB"`, `disable=True`). `backend`
takes a backend instance or a backend type name (`backend="sqlite"`). See the
[`Cash` reference](api/cash.md) and [Configuration](getting-started/configuration.md).

## Where results are stored

<!-- claim: cash/backends/persistence_policy.py:PersistencePolicy.decide @e3aa6741, cash/backends/tiered_backend.py:TieredBackend.set @8aecb084 -->
**Every result is written to disk.** However cheap the call was, a decorated
result goes to the RAM tier and to the disk tier, so the next process finds it.
There are two exceptions, and each stays in RAM for this process with a
warning: a value too big for every disk tier's size cap
([`CACHE-VALUE-TOO-BIG`](warnings.md#cache-value-too-big)), and the result of
a function that calls a cached function whose `dynamic_depends_on=` source
cannot be pickled
([`STORE-UNTRACKED-SOURCE`](warnings.md#store-untracked-source)).

### Cache folder

<!-- claim: cash/_location.py:_own_project_anchor @38fe0ec3, cash/config/resolve.py:_anchor_cache_dir @edf1f957 -->
The cache folder (the cache directory) is `.cash` at your project root. So
`python /srv/etl/run.py` uses the same cache whether you, cron or a CI step
started it, from any directory. cash finds the folder from the running script:

- **Your project root** is the first directory above the script that holds a
  `setup.py`, a `setup.cfg`, a `.git`, a `pyproject.toml` that declares a
  project (a `[project]`, `[build-system]`, `[tool.poetry]` or `[tool.cash]`
  table), or a `.cash` cache folder cash already made (below your home
  folder).
- **A script with no project above it** caches next to itself.
- **An installed tool run from outside any project** caches per user, in the
  platform's cache folder.

[Where your cache lives](how-it-works/storage.md#where-the-cache-folder-is)
has the full rule.

<!-- claim: cash/_location.py:announce_new_cache_dir @5260de8d -->
The first time cash makes a `.cash` folder there, it says where, once
([`CACHE-DIR-NEW`](warnings.md#cache-dir-new)):

```text title="Output"
run.py:9: CashWarning: [CACHE-DIR-NEW] new cache at .cash (no project found above the script). Fix: set CASH_CACHE_DIR or [tool.cash] cache_dir to cache elsewhere. (https://cash-lib.readthedocs.io/en/latest/warnings/#cache-dir-new)
```

The first run in a new folder starts with an empty cache, so every call
misses; this line tells you that is why.

To choose the folder yourself, highest priority first:

| How | Relative to | Good for |
|---|---|---|
| `Cash(cache_dir="…")` | the current directory | an app that owns its cache |
| `CASH_CACHE_DIR=…` | the current directory | cron, containers, CI |
| `cache_dir` under `[tool.cash]` in `pyproject.toml` | that file's folder | a setting shared through the repo |

`CASH_CACHE_DIR=/var/cache/myapp python run.py` needs no code change, and an
absolute path leaves no doubt about where entries go.

### Cache size limit

<!-- claim: cash/config/schema.py:CashConfig.max_cache_size == None, cash/backends/adaptive_caps.py:adaptive_disk_cap @0d13d1d2 -->
The disk cap is automatic. By default the disk tier may use a quarter of
the room on its volume, between 8 GiB and 100 GiB but never more than 80% of
that room, and the RAM tier a fifth of memory. `cash info` prints both numbers:

```text title="Output"
Max size:   RAM 3.1 GiB (auto), disk 8.0 GiB (auto)
```

When the disk tier is full, cash evicts the entries worth least per byte
first: cheap to recompute, large, and rarely read. To pin the disk cap, set
`max_cache_size` (or `CASH_MAX_CACHE_SIZE`) to a number of bytes or a size
such as `"20GB"`.

<!-- claim: cash/backends/file_eviction.py:FileEvictor.ensure_size_scanned @adb3043c, cash/backends/file_eviction.py:FileEvictor.report_eviction @0a7cf82f -->
With `CASH_VERBOSE=1` or `CASH_DEBUG=1` (or an INFO level you set on the `cash`
logger), the first result a run writes to disk logs the folder and the cap, and
the first time the cap makes cash remove entries it logs how much it removed:

```text title="Output"
cash.storage: caching in /srv/proj/.cash, up to 26.0 GiB (a quarter of the free disk space; set max_cache_size to change it)
```

### Clearing the cache

<!-- claim: cash/decorator/wrappers.py:Wrappers._cache_clear.cache_clear @9691cbcc, cash/__main__.py:cmd_clear @fd92c8a3, cash/decorator/maintenance.py:Maintenance.delete_function_entries @e215cf03 -->
Pick the narrowest tool that does the job:

| To remove | Run |
|---|---|
| One function's entries, from the code | `slow_square.cache_clear()` |
| One function's entries, from a shell | `cash clear --function slow_square` |
| One entry | `cash clear --entry ID` (ids from `cash inspect --function NAME`) |
| Entries whose `ttl` has run out | `cash clear --expired`, or `cash.cleanup()` in code |
| Everything | `cash clear --all` |

A clear reaches processes that are still running, whichever of these made it:
within about a second they stop serving what was cleared.

An entry file that
another process holds open (on Windows) cannot be removed; `cache_clear()`
then warns ([`CACHE-CLEAR-INCOMPLETE`](warnings.md#cache-clear-incomplete)),
and the CLI prints the entries it could not remove. Plain `cash clear` with no
option prints help and exits with an error.

Your test suite runs from the same project, so it reads and writes this same
cache. Give it its own; see [Testing your code](tutorials/feature-guides/testing-your-code.md#isolating-the-suites-cache).

## Seeing what cash did

A script shows nothing by default. Use these to check that caching works,
and to answer "why did it miss?" when a call ran that you expected to be a
cache hit. Each miss names its reason: new arguments, a changed file, a
helper you edited, an expired `ttl`.

<!-- claim: cash/decorator/run_summary.py:RunSummary.text @d64f5e55, cash/decorator/run_summary.py:RunSummary.reasons @e23c40a0, cash/decorator/run_summary.py:RunSummary.print_at_exit @63645a47 -->
**A summary at exit.** `CASH_SUMMARY=1` prints one table to stderr when the
process ends: hits and misses per function, the time saved, and why calls
missed. Here, after `prices.csv` was edited and the global `THRESHOLD` changed:

```bash
CASH_SUMMARY=1 python model.py
```

```text title="Output"
cash: 4 of 7 calls restored, 0.3s saved
  cache: /srv/proj/.cash, up to 26.0 GiB (a quarter of the free disk space)
  model.load_prices  1 hit,    1 miss      0.3s saved
      missed: 1 file changed
  model.build_grid   3 hits,   0 misses    0.0s saved
  model.score        0 hits,   2 misses    -
      missed: 2 code or state changed
      code or state changed (2x): global THRESHOLD changed
```

When a run you expected to be warm was not, check the `cache:` line first: it
is the folder this run actually used. The summary also lists results that were
computed but not stored. It prints on any normal exit or uncaught exception,
not when the process is killed. `summary=True` in code or config does the same.

<!-- claim: cash/decorator/run_summary.py:_head @089b4b31, cash/decorator/run_summary.py:_cash_seconds @16ac9ccb -->
When cash's own work is noticeable, the first line adds what it cost, and
each function's row its share:

```text title="Output"
cash: 0 of 33 calls restored, 0.0s saved -- and 5.3s spent by cash on keys, lookups and stores, a net loss of 5.3s
```

*Saved* is the time the hits would have taken to compute. *Spent by cash* is
cash's own work: building keys, checking the files an entry read, looking
entries up and storing results. When it is larger, the line ends with
`a net loss`. On a first run that is expected: nothing is stored yet, so
nothing can be saved, and storing is paid once for each new call. The next
run shows what the cache is worth.

<!-- claim: cash/decorator/explain.py:describe_state_change @58a52664, cash/decorator/explain.py:MissHistory.absent_entry_reason @27204896 -->
**One line per call.** `CASH_DEBUG=1` logs every call to stderr, with cash's
other debug records. `CASH_VERBOSE=1` gives only the call lines:

```text title="Output"
cash.calls: MISS model.build_grid  [4cc0d96b86b7]  no entry yet: the first call with these arguments in this process, and no earlier run stored one  (ran 0.05s)
cash.calls: HIT  model.build_grid  [4cc0d96b86b7]  (saved 0.05s)
cash.calls: MISS model.build_grid  [3bb6d830f0b9]  new arguments: called with arguments not seen on the last call  (ran 0.05s)
cash.calls: MISS model.score  [53bb9553d5ad]  code or state changed: the function's code, a helper it calls, or a value it reads changed since an earlier run stored it -- global THRESHOLD changed  (ran 0.20s)
cash.calls: RAISE model.load_prices  ValueError: no rows for 2026-09-10; nothing stored  (ran 0.21s)
```

The id in brackets is the one `cash inspect --function` lists and
`cash clear --entry` takes. After `--`, a miss names what changed: `its own
source changed`, `helper model._rank changed`, `environment variable TENANT
changed`, and so on. This works across runs too, including `ttl expired` and
entries the size cap evicted (`evicted to make room`).

<!-- claim: cash/decorator/reporting.py:describe_call @b8e1d0da -->
A hit whose lookup took a noticeable share of what it saved says so:

```text title="Output"
cash.calls: HIT  model.load_stores  [8d1c0e2a4b7f]  (saved 0.02s; the lookup took 0.04s; a net loss)
```

Reading the entry back cost more than running the body. A function that
always shows this is too quick to be worth caching: take the decorator off,
or cache a larger step that calls it.
[`CACHE-NET-LOSS`](warnings.md#cache-net-loss) warns when such losses add up
to more than two seconds.

<!-- claim: cash/_log.py:_StandDownWhenTheAppLogs.filter @1f08254a -->
If your program configures `logging`, these lines go to your handlers in your
format instead of stderr.

<!-- claim: cash/diagnostics.py:warn_diagnostic @c9245473, cash/config/schema.py:CashConfig.warning_detail == "short" -->
**Warnings.** cash's warnings are Python warnings, not log records, and each
prints as one line: the code, what happened, the fix, and a link to its entry
on [Warnings](warnings.md). The `file:line` in front is the call in your code
that set it off; a `line N` inside the message is a line of the cached
function's own file. When a second function gets the same code, its line
says only `also on <name>`. `f.cache_info()["warnings"]` keeps each one's
full text, with every finding listed, and `CASH_WARNING_DETAIL=full` prints
that instead. `logging.captureWarnings(True)` routes warnings to your
handlers.

**One function, from code.** `f.explain(*args)` tells you whether the next call
with those arguments would hit, and why, without running anything.
`f.cache_info()` returns the counters for this process. Both are described
under [Methods on a cached function](#methods-on-a-cached-function).

**What is on disk.** `cash inspect` lists functions, entry counts, sizes and
last use; `cash inspect --function NAME` lists one function's entries. See the
[CLI reference](cli.md).

## What invalidates an entry

With a bare `@cash.cache`, a call recomputes when anything in the first list
changed. The second list is what cash does not see, and what to do about it.

<!-- claim: cash/dependency_state.py:DependencyStateHasher.compute @8e272f43, cash/decorator/runtime.py:CallRunner._analyze_dependencies @6f5bcbac, cash/decorator/globals_fold.py:GlobalsFold.fold_read_globals @8944d903, cash/decorator/code_args.py:CodeArgs.fold_code_args @96467b11 -->
<div class="grid cards" markdown>

-   **Tracked for you: a change recomputes**

    ---

    - The **arguments**, by content and type. Equal values share an entry.
      With `ignore=` or `key=`, only what
      [you leave in](#leaving-arguments-out-of-the-key).
    - The function's **own code**. Comments, docstrings and formatting are
      ignored, unless the code reads a docstring (`f.__doc__`,
      `inspect.getdoc(tool)`): then the docstrings it reaches count.
    - The code of every **helper it calls**, transitively, in your project
      (an editable `pip install -e` counts) or in the package the cached
      function itself is in.
    - **Module globals** read by the function or its helpers, parameter
      defaults, and captured variables, also those of a helper a factory
      made (`score = make_scorer(weights)` over an array or a model). A
      global naming a library class or a builtin (`DTYPE = np.float32`,
      `AGG = sum`) counts by what it names.
    - The **classes it uses**: their code, their class attributes, and the
      module globals their methods read, inherited ones included.
    - Another **cached function** it reaches, directly, through a helper or
      an import in the body, on any `Cash` instance, or passes on
      (`pool.map(inner, xs)`), with what that function reads.
    - **Your class or function passed as an argument** or held in an argument,
      a global, another module's table (`steps.STEPS`) or a class attribute,
      also inside a library object (a transformer in an sklearn
      pipeline), and what that code reads, its environment reads included:
      a handler in a dispatch table, a step in a list (a `lambda` too), a
      property or dunder method of a settings object. Also code your project
      installs on a pandas or polars class: an accessor, a namespace, a
      function patched onto `pd.DataFrame`, whether the frame is an
      argument, sits inside one or is made in the body; on another
      library's class, when the value is an argument of that class. A table of plain
      functions is keyed once per version of it: a hit on a table of 1000
      handlers costs about 3 ms, not a re-analysis of each handler.
    - A **file** read by a
      [tracked reader](tutorials/feature-guides/custom-file-sources.md#whats-automatically-tracked),
      by content, or declared with `file_depends_on=`. Also a file it looked
      for and did not find, once it appears, and one it found without
      reading, once it is gone. A folder it lists (`glob.glob`,
      `os.listdir`, `Path.glob`, `Path.iterdir`, `os.walk`, ...), once a
      file in it is added or removed.
    - An **environment variable** read by literal name (`os.getenv("TENANT")`,
      `"DEBUG" in os.environ`) and the working directory (`os.getcwd()`,
      `Path.cwd()`, `os.path.abspath(p)`). Also what standard-library helpers
      read for you: `os.path.expandvars("$DATA_DIR/x")`, `expanduser`,
      `Path.home()`, `tempfile.gettempdir()`, `shutil.which` (`PATH`).
    - The **command line**, when the function or a helper it calls reads
      it: `sys.argv`, and an `argparse` parser's `parse_args()` called
      without a list of its own. Flags your `main()` parses and passes on
      count only as those arguments.
    - Sources named in `depends_on=` or `dynamic_depends_on=`, and an elapsed
      `ttl`.

-   **Not tracked: what to do**

    ---

    - **Library code** (`site-packages`, a `pip install --target` folder
      outside your project, the standard library), your team's own package
      too when it is installed without `-e`. The version is not
      in the key: see
      [Code in installed packages](decorator-limitations.md#code-in-installed-packages).
    - What a **server or database** returns: set `ttl=`
      ([`KEY-NETWORK-READ`](warnings.md#key-network-read)).
    - The **clock**, a random UUID, or the whole environment
      (`os.environ.copy()`): pass the value as an argument
      ([`KEY-AMBIENT-READ`](warnings.md#key-ambient-read)).
    - **Code picked at run time** from a dict built in the body: name it with
      `depends_on=`. A `getattr(mod, name)()`, `importlib.import_module` or
      `eval` runs uncached
      ([`KEY-UNTRACKABLE-DEP`](warnings.md#key-untrackable-dep)).
    - A file read by a reader cash does not know: use `file_depends_on=`.
    - The decorator's own parameters (`ttl`, `cache_if`, `strict`, ...):
      changing them keeps entries. `ignore=` and `key=` are the exception:
      they decide what the key reads, so changing them recomputes.

</div>

File reads need no annotation. `pd.read_csv`, `open()`, `np.load` and the other
tracked readers record each file with the entry, and every lookup checks that
its content still matches. A touch that leaves the bytes alone still hits.

To see why a particular call missed, use `explain()`. For how the key is built,
see [The decorator path](how-it-works/decorator-path.md). To keep an edit to a
loader or a shared settings class from rerunning everything, see
[Shaping code so an edit reruns less](#shaping-code-so-an-edit-reruns-less).

## Shaping code so an edit reruns less

A cached function is keyed by the **code** of everything it calls, and by the
**content** of what it is given. The two recipes below move what a function
only needs as data out of the first group and into the second.

### Pass data in, don't load inside { #pass-data-in }

<!-- claim: cash/dependency_state.py:DependencyStateHasher.compute @8e272f43 -->
Here `signals` loads its own input:

<!-- test:skip reason="the layout the next fence replaces; it defines the same names" -->
```python
import cash

from ticks import load_ticks   # load_ticks calls _parse on each line

@cash.cache
def signals(window):
    ticks = load_ticks(1000)
    return [p for _, p in ticks[::window]]

@cash.cache
def backtest(window, fee):
    sig = signals(window)
    return round(sum(sig) - fee * len(sig), 2)

for w in (5, 10, 20):
    print(backtest(w, fee=0.1))
```

`load_ticks` and `_parse` are helpers `signals` calls, so their code is in its
key, and `signals` is in `backtest`'s. A harmless fix in `_parse` (a
`line.strip()`), or a new optional parameter on `load_ticks`, reruns every
`signals` and every `backtest` call, although the ticks come out the same.
Cash cannot know that without running the new code. Caching `load_ticks` too
does not help while `signals` still calls it: then `signals` is keyed by the
code of the cached function it calls, and that changed.

Hand the data over instead:

```python
import cash

def _parse(line):
    ts, price = line.split(",")
    return int(ts), float(price)

def load_ticks(n):
    return [_parse(f"{i},{100 + i % 7}") for i in range(n)]

@cash.cache
def signals(ticks, window):
    return [p for _, p in ticks[::window]]

@cash.cache
def backtest(sig, fee):
    return round(sum(sig) - fee * len(sig), 2)

ticks = load_ticks(1000)
print(backtest(signals(ticks, 5), fee=0.1))
print(backtest(signals(ticks, 10), fee=0.1))
print(backtest(signals(ticks, 20), fee=0.1))
```

Now `load_ticks` runs on every run, outside the cache, and `signals` is keyed
by the ticks it gets. After an edit to `_parse` or `load_ticks`, the ticks
hash as before and every cached call hits; a change that does alter the ticks
reruns what depends on them, as it should. With `load_ticks` taking 0.5 s,
each `signals` call 3 s and each `backtest` 1 s:

| Step | `signals` loads the ticks | ticks passed in |
|---|---|---|
| First run | 14.4 s | 16.4 s |
| Rerun, nothing edited | 0.0 s | 0.6 s |
| `_parse` gets `line.strip()` | 14.1 s, everything reruns | 0.6 s |
| `load_ticks` gets an unused `tz=None` | 14.3 s, everything reruns | 0.5 s |
| A price in the data changes | | 13.4 s, new results |

The rerun pays for `load_ticks` itself. Decorate it with `@cash.cache` as well
to skip that too: a rerun with nothing edited then takes 0.0 s, and after the
`_parse` fix 0.8 s, since `load_ticks` runs once, returns the same ticks and
everything after it hits. The ticks are hashed on every call they are passed
to; for a large frame see [`frozen=` and large arguments](#frozen-and-large-arguments).

After an edit like this, `f.explain(...)` names the helper that changed and
the call whose result `f` could be passed instead
([Methods on a cached function](#methods-on-a-cached-function)).

<!-- claim: cash/tracking/file_dep_snapshot.py:snapshot_is_fresh @959623d2 -->
Hand an intermediate result on the same way, as a value, rather than writing
it to a file and reading it back: a file written again can get new bytes
although its data is the same (an Excel file records when it was saved), and
whatever read it then runs again after every write. Write the file once, as
the last step.

### Split parameter objects { #split-parameter-objects }

<!-- claim: cash/decorator/code_args.py:CodeArgs.fold_code_args @96467b11, cash/decorator/runtime.py:KeyBuilder.build @3514afa3 -->
An object passed in is keyed by all of its content and by the code of its
class. With one `Params` holding both the signal settings and the trading
costs, a new fee value, or a new cost field, reruns every signal search,
although `signals` never reads a cost. Give each function only the fields it
uses:

```python
from dataclasses import dataclass

import cash

@dataclass(frozen=True)
class SignalParams:
    window: int = 10

@dataclass(frozen=True)
class CostParams:
    fee: float = 0.1
    slippage: float = 0.0

@cash.cache
def search_signals(prices, sp: SignalParams):
    return prices[::sp.window]

@cash.cache
def run_backtest(sig, cp: CostParams):
    return round(sum(sig) - (cp.fee + cp.slippage) * len(sig), 2)

prices = [100 + i % 7 for i in range(1000)]
sp = SignalParams(window=10)
sig = search_signals(prices, sp)        # first call: runs the body
run_backtest(sig, CostParams(fee=0.1))  # first call: runs the body

sig = search_signals(prices, sp)        # cache hit
run_backtest(sig, CostParams(fee=0.2))  # a new fee: runs the body
```

A cost change now reruns only `run_backtest`, and so does an edit to
`CostParams`, a new field included. Leaving the cost fields out with `key=`
instead (`key=lambda prices, p: (prices, p.window)`) keeps a new fee value
from rerunning the search, but not a new field: the code of the class an
argument carries stays in the key whatever the key function returns
([Leaving arguments out of the key](#leaving-arguments-out-of-the-key)).

## Parameters

<!-- claim: cash/core.py:Cash.cache @470582df -->
All parameters are keyword-only and optional:

| Parameter | Default | What it does |
|---|---|---|
| `ttl=` | `None`: no expiry | Seconds an entry stays valid, or a `datetime.timedelta` |
| `cache_if=` | `None`: store every result | Predicate `(result) -> bool`. A falsy answer returns the result without storing it |
| `depends_on=` | `None` | A callable or `DataSource` object, or a list of them, to add to the key. Anything else (a path: use `file_depends_on=`) raises `TypeError` |
| `file_depends_on=` | `None` | A path or list of paths, tracked by content as if the body read them. A directory covers every file under it (through linked sub-folders too), a glob pattern its matches |
| `dynamic_depends_on=` | `None` | A callable (or list) that gets the call's arguments and returns `DataSource` objects |
| `frozen=` | `False` | Promise that nothing modifies the result after it is returned, so a cached function receiving it skips hashing it |
| `strict=` | `False` | Raise `CashImpureFunctionError` on any purity finding. For CI |
| `assume_safe=` | `False` | Silence purity findings and cache anyway. For some lines only, see [Side effects](#side-effects) |
| `allow_random=` | `False` | Silence the unseeded-randomness warning |
| `chunk_max_items=`, `chunk_max_bytes=` | 1,000,000 items, 1 GB | Chunk size for iterator results |
| `ignore=` | `None`: every argument is in the key | Names of parameters to leave out of the key. Annotating a parameter `cash.Ignore[...]` does the same. See [Leaving arguments out of the key](#leaving-arguments-out-of-the-key) |
| `key=` | `None`: every argument is in the key | A function that gets the call's arguments and returns what stands for them in the key. See [Leaving arguments out of the key](#leaving-arguments-out-of-the-key) |

Checks and rules that hold for all of them:

- `strict=True` with `assume_safe=True` raises `ValueError`, and so does a
  negative, NaN or infinite `ttl=`. A `ttl=` that is not a number (a `"300"`
  read from an environment variable) raises `TypeError`. Both happen when
  the function is decorated.
- An `ignore=` name that is not a parameter of the function raises
  `ValueError`, and `key=` together with an ignored parameter raises
  `TypeError`, also when the function is decorated.
- Changing a parameter keeps the entries already stored, except `ignore=`
  and `key=` (and a `cash.Ignore` annotation): they decide what the key
  reads, so adding, removing or changing one recomputes, as does adding,
  removing or changing a declared dependency.
- Locking is not a decorator parameter: it is `Cash(use_locking=True)`, see
  [Threads and processes](tutorials/feature-guides/thread-safety.md).

### `ttl=`

```python
@cash.cache(ttl=300)   # five minutes
def rates():
    return requests.get("https://api.example.com/rates").json()
```

<!-- claim: cash/decorator/backend_slot.py:BackendSlot.entry_ttl @4d5b2997, cash/core.py:Cash.cleanup @c2aa13fc -->
After the ttl, the next call recomputes and replaces the entry.

- **The current ttl applies.** An entry is judged by the decorator's `ttl=`
  as it stands now: shortening it expires older entries at once, and
  lengthening it keeps them longer. `cash.cleanup()` judges them the same way
  while the function is decorated in that process.
- **A caller refreshes as often.** A cached function that reaches one with a
  ttl, whether it calls it by name, is passed it as an argument, or finds it
  in a dict, list or closure, uses the shorter of the two ttls.
- **A default for every function** comes from configuration: set
  `default_ttl` on the disk tier; see
  [Deploying](tutorials/feature-guides/deploying.md#a-default-lifetime-for-every-entry).
- **Expired entries stay on disk** until they are called again or you run
  `cash clear --expired`.
- `ttl=0` recomputes on every call. An entry whose stored ttl cannot be read
  as a number counts as expired.

### `file_depends_on=` and `depends_on=`

```python
@cash.cache(file_depends_on="config.yaml")
def parse_config():
    return yaml.safe_load(open("config.yaml"))
```

<!-- claim: cash/decorator/file_deps.py:FileDeps.track_declared_files @eabb186b -->
Use `file_depends_on=` for a file the body reads in a way cash cannot see (a C
library, a subprocess). It is checked by content, like a tracked read. A URL is
treated as a missing local file, so for `s3://` or `https://` data pass
`depends_on=[RemoteFileDataSource(url)]` instead
([Remote objects](tutorials/feature-guides/custom-file-sources.md#remote-objects-tracked-by-the-stores-own-validator)).

`depends_on=` names things cash cannot follow on its own: a function picked at
run time, a library function whose code you want in the key, or a
`DataSource` for a database table or API version. A plain function in the list
is keyed by its source:

```python
def score(user):             # not decorated
    return user.visits * 2

@cash.cache(depends_on=[score])
def leaderboard():
    return sorted(db.query("SELECT * FROM users"), key=score)
```

`dynamic_depends_on=` builds the dependency from the call's arguments, for
example one file per tenant. If the resolver raises or returns something that is
not a `DataSource`, the call runs uncached with
[`KEY-DYNAMIC-DEP-FAILED`](warnings.md#key-dynamic-dep-failed). A cached
function that calls this one depends on the same sources: its entry keeps
each source with its token and asks it again on every lookup
([Cached functions that call it](tutorials/feature-guides/dynamic-dependencies.md#cached-functions-that-call-it)).
See [Dynamic dependencies](tutorials/feature-guides/dynamic-dependencies.md).

### `cache_if=`

```python
@cash.cache(cache_if=lambda r: r is not None)
def lookup(key):
    return cache_backend.get_or_none(key)
```

<!-- claim: cash/decorator/store.py:ResultStore.refusal @cf16cee7 -->
The predicate runs after the body returns. It decides what is **written**, not
what is served: a `None` stored before you added the predicate is still
returned. Clear the function after adding or tightening one.

If the predicate
raises, the result is returned and not stored, with a warning. For an iterator
result larger than one chunk the predicate cannot run
([`CACHE-IF-BYPASSED`](warnings.md#cache-if-bypassed)).

### Leaving arguments out of the key

Every argument is part of the key, so a `verbose=` flag, a logger or an
`n_jobs=` setting splits the cache: `fit(data, verbose=True)` misses after
`fit(data)` ran. Three ways to say which arguments decide the result:

<!-- claim: cash/decorator/arg_key.py:arg_key_spec @648484e3, cash/decorator/arg_key.py:keyed_arguments @d9fd1022 -->
**`ignore=`** names the parameters to leave out. It works on any function,
including one you do not own, and the names are checked against the
signature when the function is decorated:

```python
import cash

@cash.cache(ignore=["verbose"])
def fit(data, verbose=False):
    return sum(data) / len(data)

fit([1, 2, 3])                 # first call: runs the body
fit([1, 2, 3], verbose=True)   # cache hit: verbose is not in the key
```

For a library function: `fill = cash.cache(ignore=["width"])(textwrap.fill)`.
A misspelt name raises at once:
`ignore= names no parameter 'verbsoe' in fit(data, verbose)`. Under a
decorator that fills a parameter itself (`f(LOG, *args, **kwargs)`, click's
`pass_obj`), the call binds to the parameters the caller passes, and the
filled one is never in the key, ignored or not.

Adding `ignore=` to a function that already has entries starts new ones:
the old entries were keyed with the parameter in, so each call runs once
more, and `explain()` says "it now ignores parameter ...".

<!-- claim: cash/decorator/arg_key.py:annotated_ignores @3c698cef -->
**`cash.Ignore`** says the same on the parameter itself. A type checker sees
the type inside, `bool` here; `Annotated[bool, cash.Ignore]` is the same
marker. String annotations (`from __future__ import annotations`) are read
too. The list and the annotation add up to one set, and `*args` or
`**kwargs` can be left out as a whole:

```python
import logging

import cash

log = logging.getLogger(__name__)

@cash.cache
def load(path, debug: cash.Ignore[bool] = False):
    if debug:
        log.warning("loading %s", path)
    return path.upper()

load("a.csv")               # first call: runs the body
load("a.csv", debug=True)   # cache hit: nothing is logged
```

<!-- claim: cash/decorator/runtime.py:KeyBuilder.key_arguments @316e84b5, cash/decorator/runtime.py:KeyBuilder._fold_key_function @125f0613 -->
**`key=`** takes a function that gets each call's arguments, bound to the
signature with the defaults filled in, so `f(2, "3")`, `f(2, unit="3")` and
`f(x=2, unit="3")` reach it alike. What it returns (a tuple, a string, a
frame, an object with `__cash_key__`) is hashed like an argument and stands
for all of the arguments in the key. Use it to make two spellings of one
input key alike:

```python
import cash

def by_unit(x, unit=1):
    return (x, int(unit))

@cash.cache(key=by_unit)
def scale(x, unit=1):
    return x * int(unit)

scale(2, "3")   # first call: runs the body
scale(2, 3)     # cache hit: by_unit gives both calls the key (2, 3)
```

`self` reaches the key function like any other argument, and it works the
same for `async def` functions and generators. `key=` and ignored parameters
cannot be combined: leave those arguments out of what the key function
returns.

<!-- claim: cash/decorator/runtime.py:KeyBuilder.build @3514afa3 -->
Only the arguments' value part of the key changes. The function's code, its
helpers, the globals and files it reads, `depends_on=`, the random seed and
the rest stay in the key as before, and so does the code the arguments carry:
with `key=lambda p: (p.fast,)`, an edit to a method of `p`'s class re-keys the
call, and two functions passed in run as two calls whatever the key function
returns for them. A parameter left out with `ignore=` is left out with its
code. The key function's own code is in the key
too, with the helpers it calls and what it holds: the values a key function
made by a factory captured (`key=by("id")`), or the state of a callable
object (`key=Pick("id")`). Editing any of them re-keys every call.

!!! warning "The key is a promise"
    `key=` and `ignore=` promise that calls with equal keys return equal
    results and have the same side effects. Cash cannot check it. Leave out an
    argument that changes the result and cash serves one call's result to
    another without noticing: the same trade as `assume_safe=True`. A hit also
    does not run the body, so with an ignored `debug` flag the debug prints of
    a hit do not happen.

<!-- claim: cash/decorator/arg_key.py:key_function_impurities @784f8639, cash/core.py:Cash._check_key_function @c5ba8c38 -->
The key function must compute the key from its arguments alone. One that
reads a file, the clock, a random number, the network or the environment
gets [`KEY-FUNCTION-IMPURE`](warnings.md#key-function-impure)
(`strict=True` raises instead, `assume_safe=True` silences it). One that
raises runs the call uncached with
[`KEY-FUNCTION-RAISED`](warnings.md#key-function-raised), and so does one
that reads an iterator argument, which leaves the body an emptied iterator
([`KEY-ITERATOR-CONSUMED`](warnings.md#key-iterator-consumed)).

<!-- claim: cash/decorator/explain.py:Explainer._note_matched_by @1d34c690, cash/decorator/runtime.py:KeyBuilder.call_args_hash @18fb6b48 -->
A hit whose arguments differ from those of the call that stored the entry
is not reported, since that is the point. To see it,
`f.explain(*args).details["matched_by"]` says when a hit was matched by
`key=`. Under `ignore=` it is on every hit, since cash never hashes the
ignored arguments and cannot tell whether they differ.

<!-- claim: cash/decorator/purity_checks.py:PurityChecks._checked_arguments @51380d20 -->
**Ignored arguments are not checked for in-place changes.** On a miss cash
normally checks that the body did not change an argument in place (a body
that sorts the list it was handed is not stored). Ignoring a parameter is
your statement that it does not matter to the call, like `assume_safe=True`:
cash takes your word, so a scratch buffer left out with `ignore=` costs a
miss nothing, and a body that changes it in place is cached all the same.
`key=` is different: it says how to tell calls apart, not that an argument
does not matter, so every argument is still checked, and every miss hashes
each one in full before the body, and again after it unless every argument
is plain data that cash checks by identity instead
([`IMPURE-OBSERVED-EFFECTS`](warnings.md#impure-observed-effects)).
[`CACHE-NET-LOSS`](warnings.md#cache-net-loss) says so when that is what a
miss costs.

### `allow_random=`

<!-- claim: cash/decorator/rng.py:RngWatch.warn_unseeded_randomness @37dc56c2 -->
When the body draws from an unseeded random generator, cash warns
([`RANDOM-UNSEEDED`](warnings.md#random-unseeded)): the first draw is stored and
every later call gets the same "random" value. The fix is a generator seeded
from an argument, `rng = np.random.default_rng(seed)`.

Pass
`allow_random=True` only when a frozen draw is what you want. It silences the
warning and still caches.

A seed of the global stream set by the caller counts: after
`np.random.seed(s)` or `random.seed(s)` (at module level, or in the function
that calls this one), a function seen drawing from that stream is keyed by
where the stream stands, so each seed and each draw after it gets its own
entry. A seed set before the function is decorated is not seen. A function
whose own body seeds the stream is not keyed by where the caller left it.

<!-- claim: cash/decorator/rng.py:replay_rng_state @42b65738, cash/decorator/runtime.py:CallRunner.finish_miss @164e7c5c -->
The same holds for a generator in a module global that the body draws from
(`rng = np.random.default_rng(42)` at module level, `rng.normal()` inside): the
key includes where `rng` stands, and a hit moves `rng` on to where the computed
call left it, so a loop of calls draws what it would without the cache. A
generator passed in as an argument (`boot(x, rng)`) is keyed and moved on the
same way, and drawing from it is not reported as changing the argument. A
generator only a closure holds cannot be found again by a later run, so a call
that draws from one is not stored.

Don't call the global `np.random.seed()` inside a
cached function: a hit skips the reseed, so later draws differ between a hit and
a miss.

### `frozen=` and large arguments

<!-- claim: cash/decorator/frozen.py:FrozenResults.audit @2571147a, cash/decorator/frozen.py:FrozenResults.warn_has_no_effect @46f8683e -->
An argument is keyed by its content at the time of the call, so a big array or
frame is hashed on every call it is passed to. When a result comes from another
cached function and nothing changes it afterwards, say so on the producer:

```python
@cash.cache(frozen=True)
def train(data):
    # nothing downstream modifies the model
    return fit_model(data)

@cash.cache
def score(model, batch):
    # keys `model` by the call that made it
    return model.predict(batch)
```

The consumer then keys the model by the call that produced it, in microseconds.
A pandas frame or numpy array is hashed once, when the producer stores it. The
hash is kept with the entry, so the process that computed the result and every
process that restores it later (each worker of a pool, say) key it without
reading it again. A frozen numpy array comes
back read-only, and
[`KEY-FROZEN-MUTATED`](warnings.md#key-frozen-mutated) names the producer if a
frozen result was changed after all. [`CACHE-NET-LOSS`](warnings.md#cache-net-loss)
warns when hashing an argument costs more than the cache saves.

<!-- claim: cash/decorator/store.py:ResultStore.attach_lineage @f47f5d75, cash/decorator/frozen.py:FrozenResults.container_hash @cfd38330 -->
`frozen=True` covers the very object the function returns, in the process
that called it:

| What the consumer gets | Keyed by |
|---|---|
| The returned array or pandas frame | its content, hashed once per process |
| The returned model, object, list, tuple or dict, passed on whole | the call that produced it |
| A value taken out of a returned container (`graph, nodes = build()`, `parts["counts"]`) | its content, as if the function were not frozen |
| A list or dict you build from frozen results (`[load(a), load(b)]`) | its content, as if the function were not frozen |
| A result a pool worker returned to the parent process | its content: the parent holds a copy |

So pass a returned container on whole, or return one value per frozen
function, and build a container you pass on inside a frozen function of its
own. `frozen=` has no effect together with `ttl=`: such a result is keyed by
its content.

<!-- claim: cash/decorator/frozen.py:FrozenResults.audit @2571147a, cash/decorator/frozen.py:FrozenResults.remember_array @59fe65cc -->
A frozen numpy array comes back read-only, so a write raises. A frozen pandas
frame or any other object stays writable, and writing into it breaks the
promise: the calls after the write can be served results computed for its
old content. cash checks a frozen result only now and then;
[`KEY-FROZEN-MUTATED`](warnings.md#key-frozen-mutated) names the producer
when a check finds a change. Modify a copy (`df = df.copy()`).

In a notebook too: `model = train(data)` in one cell and `score(model, batch)`
in the next key `model` by the call that made it, as a script does, so the
notebook and a script share `score`'s entries. That holds after a restart,
when the cell restores `model` from the cache instead of calling `train`. Once
a cell changes `model` (`model.set_params(...)`), it is keyed as a changed
value.

<!-- claim: cash/decorator/frozen.py:FrozenResults.remember_container @1e8af0c7, cash/decorator/store.py:ResultStore.attach_lineage @f47f5d75 -->
**Big inputs: path in, data passed on.** [Pass data in](#pass-data-in) and
`frozen=` work together. Give the file path to a cached loader marked
`frozen=True`, parse inside it, and pass what it returns to the functions
that use it:

```python
import cash

@cash.cache(frozen=True)
def load_prices(path):
    with open(path) as f:
        return [float(line) for line in f]

@cash.cache
def mean_price(prices):
    return sum(prices) / len(prices)

with open("prices.txt", "w") as f:
    f.write("1.0\n2.0\n4.5\n")

prices = load_prices("prices.txt")  # first call: parses the file
mean_price(prices)    # first call: runs the body
mean_price(prices)    # cache hit; the list is not read again
```

The loader is keyed by the path, its code and the file's content, and
`mean_price` by the call that made `prices`. When the big value is put
together outside any cached function (a `pd.concat` of cached per-file
results), do that in a cached `frozen=True` function too, so it has a
producer. The one trade-off: a value keyed by its producer call reruns its
consumers when you edit the loader's code, even if the loader returns the
same value; a frame or array, keyed by its content, does not. In a chain of
frozen functions that pass lists or objects on (`feat(denoise(load(path)))`),
each result is keyed by the call before it, so an edit to `load` reruns the
whole chain.

For an object of your own class that holds the big data, a
[`__cash_key__`](tutorials/feature-guides/custom-hashers.md#cash-key) method
that returns a version or id keys it without reading the data.

A list of plain objects or dataclasses whose fields hold only values (numbers,
strings, dates) is keyed by its class and one read of all their fields, not
object by object. So is an object whose attributes hold JSON-like data, such as
a `self.data` of records that a cached method reads. An object held twice in
one call's arguments still keys apart from two equal copies.

## Side effects

A hit returns the stored value without running the body. Anything else the body
did (a file written, a request sent, a line printed) does not happen again. On
the first call, cash reads the function and its helpers and reports what a hit
would skip or get wrong:

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @c395765d, cash/analysis/purity_policy.py:DECORATOR_POLICY @44b8bc03, cash/analysis/purity_report.py:ISSUE_UNTRACKABLE_DEP == "untrackable_dep" -->
| The body... | cash |
|---|---|
| Writes, posts, prints to stdout, or changes state outside the function | Warns ([`IMPURE-SIDE-EFFECTS`](warnings.md#impure-side-effects)) and caches |
| Reads the network or a database (`requests.get`, `pd.read_sql`) | Warns ([`KEY-NETWORK-READ`](warnings.md#key-network-read)) and caches. `ttl=` answers it and silences the warning |
| Reads the clock, a random UUID, or an environment variable by computed name | Warns ([`KEY-AMBIENT-READ`](warnings.md#key-ambient-read)) and caches the first value |
| Reads an environment variable by literal name, the working directory, or the command line (`sys.argv`, `parser.parse_args()`) | Puts the value in the key. No warning |
| Uses `eval`/`exec`, `importlib`, or `getattr(obj, name)()` with a computed name, or calls a function `exec`/`eval` built from a string (rules or formulas loaded from a file) | Warns ([`KEY-UNTRACKABLE-DEP`](warnings.md#key-untrackable-dep)) and runs uncached, because edits to that code can't be tracked. `strict=True` raises `CashImpureFunctionError` |

**When writing a file is part of the job** (a report, a thumbnail, a saved
model), a hit skips the write. Split the function: cache the computation, and
write the file in an uncached function, so the file is written on every run
while the slow part comes from the cache:

```python
import cash

@cash.cache
def totals(rows):
    return {"n": len(rows), "sum": sum(rows)}

def write_report(t, path):
    with open(path, "w") as f:
        f.write(f"{t['n']} rows, sum {t['sum']}\n")

rows = [1, 2, 3]
write_report(totals(rows), "report.txt")  # first call: runs totals
write_report(totals(rows), "report.txt")  # cache hit; written again
```

<!-- claim: cash/analysis/purity_visitor.py:PurityVisitor._works_on_its_own_object @220ea3bf, cash/analysis/purity_visitor.py:PurityVisitor.visit_Expr @84080202 -->
**Fitting a model.** A model the body builds and fits is local to the call,
so `m = LinearRegression(); m.fit(X, y); return m` and
`return LinearRegression().fit(X, y)` both cache with no warning and need
no `assume_safe`.
When the model comes from a helper or a factory such as `make_pipeline(...)`,
cash cannot tell that `fit` changes only that new object, and reports the
discarded return of `p.fit(X, y)` as
[`IMPURE-SIDE-EFFECTS`](warnings.md#impure-side-effects). Chain the fit
instead, as in `return make_pipeline(...).fit(X, y)`, or put
`# @cash:assume-safe` on the fit line.

Logging calls are not side effects for this purpose. A special method or a
property of an object of your class that the body reads by a global name
(`BAG['k']`, `BAG.value`) is read as a helper: a global it changes is reported
as one the body changes.

<!-- claim: cash/effect_observer.py:EffectObserver @2a3f8781 broad="the observed-effect contract is the class as a whole", cash/decorator/purity_checks.py:PurityChecks.report_observed_effects @2d8839cd, cash/effect_observer.py:EffectObserver.record_write @659fb0d2 -->
cash also **watches the first call**. Library code is not read, so a
`session.post` or an SDK request is invisible to the analysis above.

While a miss runs, cash records file writes (an `os.open` to write included),
outbound connections and subprocesses, and
warns once about any it had not already reported (a write into a scratch
directory the call made and removed again is not one, nor is a temporary file
the call removed again, nor is a write into
the interpreter's or numba's own cache)
([`IMPURE-OBSERVED-EFFECTS`](warnings.md#impure-observed-effects)). A call to an
LLM or HTTP SDK shows up this way, as a network read
([`KEY-NETWORK-READ`](warnings.md#key-network-read)): `ttl=` silences it, and
under `strict=True` it raises unless a `ttl=` is set.

Two observations stop the result from being stored, because a hit could not
reproduce them: the call changed an argument in place, or it called a
`unittest.mock` object.

<!-- claim: cash/decorator/runtime.py:KeyBuilder._hands_a_state_changing_callable @d02c3941 -->
A call that **hands the function one of your functions that changes state**
when it is called runs uncached, every time, as plain Python runs it: in
`run(record, 3)` where `record` appends to a global list, a hit would skip the
append. The same goes for a bound method that changes a global, one inside a
`functools.partial`, one in a list or object that the body calls into, and an
object of your class whose special methods or properties change a global (a
timer's `__exit__` the body enters with `with cm:`, a `__getitem__`, a
property). cash warns once
([`IMPURE-HANDED-CALLABLE`](warnings.md#impure-handed-callable)). A function
that changes nothing is keyed by its code and cached as usual, and
`assume_safe=True` caches such calls again.

**Accepting an effect.** Once you have checked that skipping the effect on a hit
is fine, put a comment on the line the warning names:

```python
@cash.cache
def fetch_user(uid):
    url = f"https://api.example.com/users/{uid}"
    return requests.get(url).json()   # @cash:assume-safe
```

<!-- claim: cash/analysis/annotations.py:audited_lines @5aa0a42f -->
The comment covers that statement only (put it on the line the warning names,
on the opening line of a call that spans lines, or on the line above), so code
added later is still checked. In a method chain over several lines, the warning
names the line of the method it is about (`.to_csv(path)`), and the comment
works there too.

On the `def` line it covers findings about the whole body. In a helper it
covers every caller of that helper.

<!-- claim: cash/analysis/annotations.py:assume_safe_block_lines @8b9e6cfa, cash/effect_observer.py:EffectObserver.record_effect @dce8a421, cash/source_norm.py:drop_waiver_blocks @358fa0bc -->
**Several lines at once.** Wrap them in `with cash.assume_safe():`. It waives
what the comment waives, on every line inside the block. While the block runs,
it also waives the effects cash observes, including those of helpers it calls.
It is code, so your editor completes it and a typo fails on the first run. Like
the comment, it is not part of the key: adding or removing it keeps the stored
results.

```python
import cash

@cash.cache
def summarize(rows):
    total = sum(rows)
    with cash.assume_safe():
        print(f"{len(rows)} rows")
        print(f"total {total}")
    return total

summarize([1, 2, 3])  # prints on this first call only
```

`assume_safe=True` silences the whole function instead, including code added
after your review, so prefer the comment or the block.

**In CI**, `strict=True` turns every finding into `CashImpureFunctionError`, so
caching a side-effecting function fails the build. It honours
`# @cash:assume-safe` comments and `with cash.assume_safe():` blocks, and a
network read passes once the function has a `ttl=`.

To tell cash about a helper it cannot judge, mark it with `@cash.pure` or
`@cash.stateful`; see [Purity markers](tutorials/feature-guides/purity-decorators.md).

## Methods on a cached function

<!-- claim: cash/decorator/wrappers.py:Wrappers._cache_info.cache_info @b2155947 -->
**`f.cache_info()`** returns this process's counters:

```python
@cash.cache
def double(x):
    return x * 2

double(1); double(1); double(2)
print(double.cache_info())
```

```text title="Output"
{'hits': 1, 'misses': 2, 'hit_rate': 0.333...,
 'total_time_saved': 1e-05,
 'miss_reasons': {'no entry yet': 1, 'new arguments': 1},
 'changed': {},
 'warnings': []}
```

`changed` counts the `code or state changed` misses by what changed
(`{'helper ticks._parse changed': 3}`). `warnings` holds the last 20 cash warnings for the function, even when a
warnings filter hid them. The counters belong to the wrapper, so they start at
zero in each process.

<!-- claim: cash/decorator/explain.py:Explainer.explain @e18309b7, cash/decorator/explain.py:Explainer._absent_entry @28c80c9d, cash/decorator/explain.py:callee_to_pass_in @eadc3715 -->
**`f.explain(*args, **kwargs)`** says whether that call would hit, and why. It
does not run the function, change the counters or write anything:

```python
print(double.explain(5))
double(5)
print(double.explain(5))
```

```text title="Output"
[MISS] model.double - no_entry
  cache_dir: /srv/proj/.cash
  cache_key: model.double:8ff9a351...::abc50414...
  entry_id: 58872c3e0a1e
  why: new arguments: called with arguments not seen on the last call
[HIT] model.double - hit
  cache_dir: /srv/proj/.cash
  ...
  execution_time_saved: 3.5e-06
```

`reason` is one of `hit`, `no_entry`, `ttl_expired`, `file_changed`,
`key_uncomputable` or `disabled`. For `file_changed`, `details["changed_files"]`
names each changed file. After an edit, `details["changed"]` names what
changed, in a new process too, and when that is code the function calls,
`details["hint"]` says which call to move out
([Pass data in](#pass-data-in)):

```text title="Output"
[MISS] run_dev.signals - no_entry
  ...
  changed: helper ticks._parse changed
  hint: If signals only needs what load_ticks returns, call load_ticks outside and pass its result in as an argument: then signals reruns only when that result changes (https://cash-lib.readthedocs.io/en/latest/decorator/#pass-data-in).
```

See
[`CacheExplanation`](api/cash.md#cash.CacheExplanation).

<!-- claim: cash/decorator/explain.py:check_explain_arguments @f3526db8 -->
For a cached method, explain through the class and pass the instance:
`Model.score.explain(m, 2)` explains `m.score(2)`. `m.score.explain(2)` cannot
pass the instance, and raises `TypeError`, as do arguments the function
cannot be called with.

**`f.cache_clear()`** deletes the function's entries from every tier, including
iterator chunks, and resets its counters and warning log.

**`f.__wrapped__`** is the undecorated function. Call it to bypass the cache,
for example in a test.

## Common questions

<!-- claim: cash/decorator/runtime.py:KeyBuilder.build @3514afa3, cash/decorator/arg_hashing.py:ArgHasher.arg_hash @59bed08e -->
**Do new command-line flags make every cached function miss?** No. Only a
function that reads `sys.argv` or calls `parse_args()` itself, or through a
helper, is keyed by the command line. One that gets the parsed values as
arguments is keyed by those values alone.

**Is a path argument part of the key?** Yes, as text, like any argument, and
the file the function reads through it is tracked by its content as well.
With an absolute path, a copy of the project in another folder passes
another path, so each call runs once there. A relative path keeps the key,
but names a file in the working directory
([Relative paths](decorator-limitations.md#relative-paths-and-working-directories)).

**Pass a big input in, or load it by path?** Both: a cached `frozen=True`
loader takes the path, and what it returns is passed on. See
[`frozen=` and large arguments](#frozen-and-large-arguments).

**Does `frozen=True` cover a tuple I unpack?** Only the tuple passed on
whole. A value unpacked from it is keyed by its content, as if the function
were not frozen.

**Is a fitted model I pass to another cached function keyed by its content
or by the call that made it?** By its content, read in full on every call,
unless the function that returned it is `frozen=True`: then by that call,
in a script and through a notebook variable alike.

**A miss says `new arguments`, but I made this call before.** The reason
names what differs from the nearest earlier call cash knows of. An argument
that another cached function returned differs once that function reruns
after an edit: its result is a new value, or, from a `frozen=True` function,
a new call. Look one step up: the function that made the argument says
`code or state changed`.

<!-- claim: cash/code_digest.py:callable_identity @38daa9e0 -->
**Can I cache code that calls numba?** Yes. A function compiled with
`@njit` or `@jit` is keyed like any helper, by the Python source it compiles
and its compile options (`fastmath`, `parallel`, ...), so editing a kernel
reruns the cached functions that call it. Put `@cash.cache` on a plain
Python function that calls the kernels, or on top of `@njit` itself.
`@njit(cache=True)` is fine: numba's own cache files are not reported as
side effects. A miss's run time, and so the time a hit saves, includes any
compiling the call did.

<!-- claim: cash/tracking/reader_patches.py:_patch_multiprocessing_pool @c3389dba -->
**Can I use cached functions in a process pool?** Yes: map the cached
function itself, or a plain function that calls cached ones. Workers share
the cache folder with the parent. See
[Threads and processes](tutorials/feature-guides/thread-safety.md#across-processes-pool-processpoolexecutor-joblib).

## Known limitations

A few kinds of function need a change on your side.
[Known limitations of `@cash.cache`](decorator-limitations.md) gives each case
with its fix:

- <a id="arguments-cash-cannot-hash"></a>[Arguments cash cannot hash](decorator-limitations.md#arguments-cash-cannot-hash):
  a lock, an open file, a connection.
- [An argument that does not change the result](decorator-limitations.md#an-argument-that-does-not-change-the-result):
  a logger or a `verbose=` flag.
- <a id="methods-and-self"></a>[Methods and `self`](decorator-limitations.md#methods-and-self)
- <a id="code-you-pass-as-an-argument"></a>[Code you pass as an argument](decorator-limitations.md#code-you-pass-as-an-argument)
- <a id="reads-cash-cannot-see"></a>[Reads cash cannot see](decorator-limitations.md#reads-cash-cannot-see):
  C extensions, subprocesses, threads you start.
- [An edit that keeps the size and timestamps](decorator-limitations.md#an-edit-that-keeps-the-size-and-timestamps)
  (Windows)
- <a id="relative-paths-and-working-directories"></a>[Relative paths and working directories](decorator-limitations.md#relative-paths-and-working-directories)
- <a id="results-cash-refuses-to-store"></a>[Results cash refuses to store](decorator-limitations.md#results-cash-refuses-to-store):
  figures, unpicklable values, calls that change their arguments.
- <a id="generators-and-async-functions"></a>[Generators and async functions](decorator-limitations.md#generators-and-async-functions)
- <a id="edits-inside-a-running-process"></a>[Edits inside a running process](decorator-limitations.md#edits-inside-a-running-process)
- <a id="using-a-decorated-function-in-a-notebook"></a>[Using a decorated function in a notebook](decorator-limitations.md#using-a-decorated-function-in-a-notebook)

## Related

- [Known limitations of `@cash.cache`](decorator-limitations.md): the cases
  that need a change on your side.
- [Deploying](tutorials/feature-guides/deploying.md): services, workers, CI,
  shared caches, libraries.
- [Testing your code](tutorials/feature-guides/testing-your-code.md): keep the
  cache from passing tests for you.
- [Threads and processes](tutorials/feature-guides/thread-safety.md): thread
  pools, process pools, joblib and locking.
- [File dependencies](tutorials/feature-guides/custom-file-sources.md): which
  file reads are tracked, and how to name the others.
- [Dynamic dependencies](tutorials/feature-guides/dynamic-dependencies.md): a
  dependency chosen by the arguments.
- [Configuration](getting-started/configuration.md): every setting, and where
  settings come from.
- [The decorator path](how-it-works/decorator-path.md): how the key is built.
