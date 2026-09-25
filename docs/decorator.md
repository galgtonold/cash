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

slow_square(1_000_000)   # first call: runs the body and stores the result
slow_square(1_000_000)   # cache hit: returns the stored result
```

That is all the setup there is. A few rules hold for every cached function:

<!-- claim: cash/decorator/store.py:ResultStore.refusal @50f5a969, cash/backends/serialization.py:get_serializer @76cf2c1b -->
- **Exceptions are never cached.** If the body raises, nothing is stored and the
  exception reaches you as usual. The next call runs the body again.
- **A hit does not replay output.** Anything the body printed or logged appears
  only on the call that ran it.
- **A hit returns a copy.** The value is rebuilt from the stored bytes, so
  writing into a result you got from a hit never changes what the next caller
  gets.

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

<!-- claim: cash/backends/persistence_policy.py:PersistencePolicy.decide @2270c6c5, cash/backends/tiered_backend.py:TieredBackend.set @4e1a354c -->
**Every result is written to disk.** However cheap the call was, a decorated
result goes to the RAM tier and to the disk tier, so the next process finds it.
The one exception is a value too big for every disk tier's size cap: it stays in
RAM for this process, and
[`CACHE-VALUE-TOO-BIG`](warnings.md#cache-value-too-big) says so.

<!-- claim: cash/_location.py:project_anchor @46e903a7, cash/config.py:_anchor_cache_dir @fb2bceec -->
**The folder is `.cash` at your project root.** Cash starts at the running
script and walks up to the first directory that holds a `setup.py`, a
`setup.cfg`, a `.git`, or a `pyproject.toml` that declares a project (a
`[project]`, `[build-system]`, `[tool.poetry]` or `[tool.cash]` table). So
`python /srv/etl/run.py` uses the same cache whether you, cron or a CI step
started it, from any directory. A script with no project above it caches next
to itself. An installed tool run from outside any project caches per user, in
the platform's cache folder. [Where your cache lives](how-it-works/storage.md#where-the-cache-folder-is)
has the full rule.

To choose the folder yourself, highest priority first:

| How | Relative to | Good for |
|---|---|---|
| `Cash(cache_dir="…")` | the current directory | an app that owns its cache |
| `CASH_CACHE_DIR=…` | the current directory | cron, containers, CI |
| `cache_dir` under `[tool.cash]` in `pyproject.toml` | that file's folder | a setting shared through the repo |

`CASH_CACHE_DIR=/var/cache/myapp python run.py` needs no code change, and an
absolute path leaves no doubt about where entries go.

<!-- claim: cash/config.py:CashConfig.max_cache_size == None -->
**The disk cap is automatic.** By default the disk tier may use a quarter of
the room on its volume, and the RAM tier a fifth of memory. `cash info` prints
both numbers (`Max size: auto -- disk 8.0 GiB, RAM 3.1 GiB`). When the disk
tier is full, cash evicts the entries worth least per byte first: cheap to
recompute, large, and rarely read. Set `max_cache_size` (or
`CASH_MAX_CACHE_SIZE`) to a number of bytes or a size such as `"20GB"` to pin
the disk cap.

<!-- claim: cash/backends/file_eviction.py:FileEvictor.ensure_size_scanned @adb3043c, cash/backends/file_eviction.py:FileEvictor.report_eviction @aa4b93be -->
With `CASH_VERBOSE=1` or `CASH_DEBUG=1`, the first result a run writes to disk
logs the folder and the cap, and the first time the cap makes cash remove
entries it logs how much it removed:

```text
cash.storage: caching in /srv/proj/.cash, up to 26.0 GiB (a quarter of the free disk space; set max_cache_size to change it)
```

<!-- claim: cash/core.py:Cash._wrap_with_stats.cache_clear @0e137c19, cash/__main__.py:cmd_clear @a08b9044, cash/core.py:Cash._delete_backend_entries @b7c16174 -->
**Clearing.** Pick the narrowest tool that does the job:

| To remove | Run |
|---|---|
| One function's entries, from the code | `slow_square.cache_clear()` |
| One function's entries, from a shell | `cash clear --function slow_square` |
| One entry | `cash clear --entry ID` (ids from `cash inspect --function NAME`) |
| Entries whose `ttl` has run out | `cash clear --expired`, or `cash.cleanup()` in code |
| Everything | `cash clear --all` |

A clear reaches processes that are still running, whichever of these made it:
within about a second they stop serving what was cleared. An entry file that
another process holds open (on Windows) cannot be removed; `cache_clear()`
then warns ([`CACHE-CLEAR-INCOMPLETE`](warnings.md#cache-clear-incomplete)),
and the CLI prints the entries it could not remove. Plain `cash clear` with no
option prints help and exits with an error.

Your test suite runs from the same project, so it reads and writes this same
cache. Give it its own; see [Testing your code](tutorials/feature-guides/testing-your-code.md#isolating-the-suites-cache).

## Seeing what cash did

A script shows nothing by default. Use these to check that caching works.

<!-- claim: cash/core.py:Cash.run_summary @1b06ce83, cash/core.py:Cash._summary_reasons @30c139d9, cash/core.py:Cash._print_run_summary @f2a46f9f -->
**A summary at exit.** `CASH_SUMMARY=1` prints one table to stderr when the
process ends: hits and misses per function, the time saved, and why calls
missed. Here, after `prices.csv` was edited and the global `THRESHOLD` changed:

```bash
CASH_SUMMARY=1 python model.py
```

```text
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

<!-- claim: cash/decorator/explain.py:describe_state_change @7b3bcda1, cash/decorator/explain.py:MissHistory.absent_entry_reason @de955740 -->
**One line per call.** `CASH_DEBUG=1` logs every call to stderr, with cash's
other debug records. `CASH_VERBOSE=1` gives only the call lines:

```text
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

<!-- claim: cash/_log.py:_StandDownWhenTheAppLogs.filter @1f08254a -->
If your program configures `logging`, these lines go to your handlers in your
format instead of stderr. Cash's warnings are Python warnings, not log records;
`logging.captureWarnings(True)` routes them to your handlers too.

**One function, from code.** `f.explain(*args)` tells you whether the next call
with those arguments would hit, and why, without running anything.
`f.cache_info()` returns the counters for this process. Both are described
under [Methods on a cached function](#methods-on-a-cached-function).

**What is on disk.** `cash inspect` lists functions, entry counts, sizes and
last use; `cash inspect --function NAME` lists one function's entries. See the
[CLI reference](cli.md).

## What invalidates an entry

With a bare `@cash.cache`, a call recomputes when any input below changed. The
left column is tracked for you. The right column is not, and says what to do.

<!-- claim: cash/dependency_state.py:DependencyStateHasher.compute @3825a447, cash/decorator/runtime.py:CallRunner._analyze_dependencies @6f5bcbac, cash/decorator/globals_fold.py:GlobalsFold.fold_read_globals @34ac7e63, cash/decorator/code_args.py:CodeArgs.fold_code_args @196f393c -->
| Tracked: a change recomputes | Not tracked: what to do |
|---|---|
| The **arguments**, by content and type. Equal values share an entry | **Library code** (`site-packages`, the standard library). Pin versions |
| The function's **own code**. Comments, docstrings and formatting are ignored | What a **server or database** returns. Set `ttl=` ([`KEY-NETWORK-READ`](warnings.md#key-network-read)) |
| The code of every **helper it calls**, transitively, in your project or your own installed package | The **clock** or a random UUID. Pass the value as an argument ([`KEY-AMBIENT-READ`](warnings.md#key-ambient-read)) |
| **Module globals** read by the function or its helpers, parameter defaults, and captured variables | **Code picked at run time** (`getattr(mod, name)()`, a dict built in the body). Name it with `depends_on=` |
| Another **cached function** it calls or passes on (`pool.map(inner, xs)`) | A file read by a reader cash does not know. Use `file_depends_on=` |
| **Your class or function passed as an argument** or held in an argument or global, also inside a library object (a transformer in an sklearn pipeline), and what that code reads | The decorator's own parameters (`ttl`, `cache_if`, `strict`, ...). Changing them keeps entries |
| A **file** read by a [tracked reader](tutorials/feature-guides/custom-file-sources.md#whats-automatically-tracked), by content, or declared with `file_depends_on=`. Also a file it looked for and did not find, once it appears | |
| An **environment variable** read by literal name (`os.getenv("TENANT")`) and the working directory | |
| Sources named in `depends_on=` or `dynamic_depends_on=`, and an elapsed `ttl` | |

File reads need no annotation. `pd.read_csv`, `open()`, `np.load` and the other
tracked readers record each file with the entry, and every lookup checks that
its content still matches. A touch that leaves the bytes alone still hits.

To see why a particular call missed, use `explain()`. For how the key is built,
see [The decorator path](how-it-works/decorator-path.md).

## Parameters

<!-- claim: cash/core.py:Cash.cache @2d082328 -->
All parameters are keyword-only and optional:

| Parameter | What it does |
|---|---|
| `ttl=` | Seconds an entry stays valid, or a `datetime.timedelta`. `None` (default): no expiry |
| `cache_if=` | Predicate `(result) -> bool`. A falsy answer returns the result without storing it |
| `depends_on=` | A callable or `DataSource` object, or a list of them, to add to the key. Anything else (a path: use `file_depends_on=`) raises `TypeError` |
| `file_depends_on=` | A path or list of paths, tracked by content as if the body read them |
| `dynamic_depends_on=` | A callable (or list) that gets the call's arguments and returns `DataSource` objects |
| `frozen=` | Promise that nothing modifies the result after it is returned, so a cached function receiving it skips hashing it |
| `strict=` | Raise `CashImpureFunctionError` on any purity finding. For CI |
| `assume_safe=` | Silence purity findings and cache anyway |
| `allow_random=` | Silence the unseeded-randomness warning |
| `chunk_max_items=`, `chunk_max_bytes=` | Chunk size for iterator results (default 1,000,000 items, 1 GB) |

`strict=True` with `assume_safe=True` raises `ValueError`, and so does a
negative, NaN or infinite `ttl=`; a `ttl=` that is not a number (a `"300"`
read from an environment variable) raises `TypeError`, when the function is
decorated. Changing a parameter
keeps the entries already stored; adding, removing or changing a declared
dependency recomputes. Locking is not a decorator parameter: it is
`Cash(use_locking=True)`, see [Threads and processes](tutorials/feature-guides/thread-safety.md).

### `ttl=`

```python
@cash.cache(ttl=300)   # five minutes
def rates():
    return requests.get("https://api.example.com/rates").json()
```

<!-- claim: cash/decorator/backend_slot.py:BackendSlot.entry_ttl @672c5d75, cash/core.py:Cash.cleanup @b561dc3e -->
After the ttl, the next call recomputes and replaces the entry. An entry keeps
the ttl it was written with, and the decorator's current ttl applies too: the
shorter one wins. So lengthening `ttl=60` to `ttl=3600` does not rescue entries
written under 60 seconds. To give every function a lifetime from
configuration, set `default_ttl` on the disk tier; see
[Deploying](tutorials/feature-guides/deploying.md#a-default-lifetime-for-every-entry).
An expired entry stays on disk until it is called again or you run
`cash clear --expired`. `ttl=0` recomputes on every call. An entry whose
stored ttl cannot be read as a number counts as expired.

### `file_depends_on=` and `depends_on=`

```python
@cash.cache(file_depends_on="config.yaml")
def parse_config():
    return yaml.safe_load(open("config.yaml"))
```

<!-- claim: cash/decorator/file_deps.py:FileDeps.track_declared_files @4027a947 -->
Use `file_depends_on=` for a file the body reads in a way cash cannot see (a C
library, a subprocess). It is checked by content, like a tracked read. A URL is
treated as a missing local file, so for `s3://` or `https://` data pass
`depends_on=[RemoteFileDataSource(url)]` instead
([Remote objects](tutorials/feature-guides/custom-file-sources.md#remote-objects-tracked-by-the-stores-own-validator)).

`depends_on=` names things cash cannot follow on its own: a function picked at
run time, a library function whose version you want in the key, or a
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
[`KEY-DYNAMIC-DEP-FAILED`](warnings.md#key-dynamic-dep-failed). See
[Dynamic dependencies](tutorials/feature-guides/dynamic-dependencies.md).

### `cache_if=`

```python
@cash.cache(cache_if=lambda r: r is not None)
def lookup(key):
    return cache_backend.get_or_none(key)
```

<!-- claim: cash/decorator/store.py:ResultStore.refusal @50f5a969 -->
The predicate runs after the body returns. It decides what is **written**, not
what is served: a `None` stored before you added the predicate is still
returned. Clear the function after adding or tightening one. If the predicate
raises, the result is returned and not stored, with a warning. For an iterator
result larger than one chunk the predicate cannot run
([`CACHE-IF-BYPASSED`](warnings.md#cache-if-bypassed)).

### `allow_random=`

<!-- claim: cash/decorator/rng.py:RngWatch.warn_unseeded_randomness @52f9e356 -->
When the body draws from an unseeded random generator, cash warns
([`RANDOM-UNSEEDED`](warnings.md#random-unseeded)): the first draw is stored and
every later call gets the same "random" value. The fix is a generator seeded
from an argument, `rng = np.random.default_rng(seed)`. Pass
`allow_random=True` only when a frozen draw is what you want. It silences the
warning and still caches. Don't call the global `np.random.seed()` inside a
cached function: a hit skips the reseed, so later draws differ between a hit and
a miss.

### `frozen=` and large arguments

<!-- claim: cash/decorator/frozen.py:FrozenResults.audit @0786c6c6, cash/decorator/frozen.py:FrozenResults.warn_has_no_effect @46f8683e -->
An argument is keyed by its content at the time of the call, so a big array or
frame is hashed on every call it is passed to. When a result comes from another
cached function and nothing changes it afterwards, say so on the producer:

```python
@cash.cache(frozen=True)
def train(data):
    return fit_model(data)          # nothing downstream modifies the model

@cash.cache
def score(model, batch):            # keys `model` by the call that made it
    return model.predict(batch)
```

The consumer then keys the model by the call that produced it, in microseconds.
A frozen numpy array comes back read-only, and
[`KEY-FROZEN-MUTATED`](warnings.md#key-frozen-mutated) names the producer if a
frozen result was changed after all. [`CACHE-NET-LOSS`](warnings.md#cache-net-loss)
warns when hashing an argument costs more than the cache saves. For big inputs,
key the cached functions by **file path** and parse inside them.

## Side effects

A hit returns the stored value without running the body. Anything else the body
did (a file written, a request sent, a line printed) does not happen again. On
the first call, cash reads the function and its helpers and reports what a hit
would skip or get wrong:

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @21132aa4, cash/analysis/purity_analyzer.py:DECORATOR_POLICY @44b8bc03, cash/analysis/purity_analyzer.py:ISSUE_UNTRACKABLE_DEP == "untrackable_dep" -->
| The body... | Cash |
|---|---|
| Writes, posts, prints to stdout, or changes state outside the function | Warns ([`IMPURE-SIDE-EFFECTS`](warnings.md#impure-side-effects)) and caches |
| Reads the network or a database (`requests.get`, `pd.read_sql`) | Warns ([`KEY-NETWORK-READ`](warnings.md#key-network-read)) and caches. `ttl=` answers it and silences the warning |
| Reads the clock, a random UUID, or an environment variable by computed name | Warns ([`KEY-AMBIENT-READ`](warnings.md#key-ambient-read)) and caches the first value |
| Reads an environment variable by literal name, or the working directory | Puts the value in the key. No warning |
| Uses `eval`/`exec`, `importlib`, or `getattr(obj, name)()` with a computed name | Raises `CashImpureFunctionError`, because edits to that code can't be tracked |

Logging calls are not side effects for this purpose.

<!-- claim: cash/effect_observer.py:EffectObserver @45e537a1 broad="the observed-effect contract is the class as a whole", cash/decorator/purity_checks.py:PurityChecks.report_observed_effects @5af70afb -->
Cash also **watches the first call**. Library code is not read, so a
`session.post` or an SDK request is invisible to the analysis above. While a
miss runs, cash records file writes, outbound connections and subprocesses, and
warns once about any it had not already reported
([`IMPURE-OBSERVED-EFFECTS`](warnings.md#impure-observed-effects)). A call to an
LLM or HTTP SDK shows up this way. Two observations stop the result from being
stored, because a hit could not reproduce them: the call changed an argument
in place, or it called a `unittest.mock` object.

**Accepting an effect.** Once you have checked that skipping the effect on a hit
is fine, put a comment on the line the warning names:

```python
@cash.cache
def fetch_user(uid):
    return requests.get(f"https://api.example.com/users/{uid}").json()   # @cash:assume-safe
```

<!-- claim: cash/analysis/annotations.py:audited_lines @da3b0e65 -->
The comment covers that statement only (put it on the opening line of a call
that spans lines, or on the line above), so code added later is still checked.
On the `def` line it covers findings about the whole body. In a helper it
covers every caller of that helper. `assume_safe=True` silences the whole
function instead, including code added after your review, so prefer the
comment.

**In CI**, `strict=True` turns every finding into `CashImpureFunctionError`, so
caching a side-effecting function fails the build. It honours
`# @cash:assume-safe` comments, and a network read passes once the function has
a `ttl=`.

To tell cash about a helper it cannot judge, mark it with `@cash.pure` or
`@cash.stateful`; see [Purity markers](tutorials/feature-guides/purity-decorators.md).

## Methods on a cached function

<!-- claim: cash/core.py:Cash._wrap_with_stats.cache_info @9b54927a -->
**`f.cache_info()`** returns this process's counters:

```python
@cash.cache
def double(x):
    return x * 2

double(1); double(1); double(2)
double.cache_info()
# {'hits': 1, 'misses': 2, 'hit_rate': 0.333..., 'total_time_saved': 1e-05,
#  'miss_reasons': {'no entry yet': 1, 'new arguments': 1}, 'warnings': []}
```

`warnings` holds the last 20 cash warnings for the function, even when a
warnings filter hid them. The counters belong to the wrapper, so they start at
zero in each process.

<!-- claim: cash/decorator/explain.py:Explainer.explain @7736721e -->
**`f.explain(*args, **kwargs)`** says whether that call would hit, and why. It
does not run the function, change the counters or write anything:

```python
double.explain(5)
# [MISS] model.double - no_entry
#   cache_dir: /srv/proj/.cash
#   cache_key: model.double:8ff9a351...::abc50414...
#   entry_id: 58872c3e0a1e
#   why: new arguments: called with arguments not seen on the last call
double(5)
double.explain(5)
# [HIT] model.double - hit
#   cache_dir: /srv/proj/.cash
#   ...
#   execution_time_saved: 3.5e-06
```

`reason` is one of `hit`, `no_entry`, `ttl_expired`, `file_changed`,
`key_uncomputable` or `disabled`. For `file_changed`, `details["changed_files"]`
names each changed file. See
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

## Known limitations

### Arguments cash cannot hash

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.hash_payload @c4f48efb -->
An argument that cannot be pickled (a lock, an open file, a live connection, a
closure) cannot be keyed. The call runs uncached and warns
[`KEY-UNHASHABLE-ARG`](warnings.md#key-unhashable-arg); any other failure while
building the key does the same. Pass a plain value that identifies the object
instead, or register a hasher that returns a **stable** identifying value, such
as a database URL:

```python
import hashlib
import cash

class Store:
    def __init__(self, url):
        self.url = url

cash.register_hasher(Store, lambda s: hashlib.sha256(s.url.encode()).hexdigest())
```

Never hash by `id()`: ids repeat across processes, so a later run could get
another object's entry. See [Custom hashers](tutorials/feature-guides/custom-hashers.md).

### Methods and `self`

`self` is an argument like any other, hashed by its state: two instances with
equal attributes share entries. An unpicklable attribute makes every call
uncached. See [Class methods](tutorials/feature-guides/caching-class-methods.md).

### Code you pass as an argument

A class or function of yours passed as an argument is keyed by its code and by
what that code reads, so editing it recomputes. Three cases are not covered:

- **Library classes and functions** are keyed by name, not code. Pin versions.
- **A closure or `lambda`** can't be pickled, so the call runs uncached. Pass a
  module-level function and give it the captured value as an argument.
- **An implementation picked at run time** (`HANDLERS[name]()` on a dict built
  inside the body, a plugin registry) is not reached. Name the candidates:
  `@cash.cache(depends_on=[FastPath, ExactPath])`.

A marker class you pass but whose code never affects the result can be excluded
with `@cash.opaque`; see [Purity markers](tutorials/feature-guides/purity-decorators.md#cashopaque-leave-a-class-out-of-the-key).

### Reads cash cannot see

Cash does not record a file opened by a C extension, `os.open`, a subprocess,
a `threading.Thread` you start, or a `multiprocessing.Pool` worker. Reads in a
`ThreadPoolExecutor` or `ProcessPoolExecutor` the function starts are recorded.
A polars `LazyFrame` argument from `scan_csv` is keyed by its path, not the
file's content; collect it first. Name any file cash misses with
`file_depends_on=`.

### Relative paths and working directories

`open("data.csv")` from two working directories reads two different files under
one key. Each switch recomputes and replaces the other directory's entry. Build
paths from the project root or pass absolute paths.

### Results cash refuses to store

- A matplotlib `Figure` or `Axes` is never cached
  ([`CACHE-IDENTITY-COUPLED`](warnings.md#cache-identity-coupled)): a restored
  copy would become pyplot's "current figure" and `plt.savefig()` would save a
  blank image. Cache the data and draw from it.
- A result that can't be pickled is not stored
  ([`STORE-FAILED`](warnings.md#store-failed)), unless the function is
  `frozen=True`.
- A call that changed its argument in place is not stored, so it runs every
  time. Return a modified copy instead (`rows = sorted(rows)`).

### Generators and async functions

A generator result is stored in chunks as you consume it, and nothing is stored
if you stop early; an infinite generator never finishes, so it never caches. See
[Iterators](tutorials/feature-guides/iterator-caching.md). `async def` functions
cache like sync ones; async generators are returned undecorated with a warning.
See [Async functions](tutorials/feature-guides/async-caching.md).

### Edits inside a running process

Cash reads a helper's source once per process, so an edit between two calls
of one long-running process is seen only by the next process. If a source file
changes on disk after the process started (a deploy), cash keys by the code
that is running and warns [`KEY-SOURCE-CHANGED`](warnings.md#key-source-changed).

### Using a decorated function in a notebook

Decorated functions work in a notebook, but re-running the cell that defines
one creates a new wrapper with new counters, so `cache_info()` may read zero.
Use `explain()` there. The notebook's own caching is covered in the
[Notebook guide](notebook_caching_api.md).

## Next

- [Deploying](tutorials/feature-guides/deploying.md): services, workers, CI, shared caches, libraries.
- [Testing your code](tutorials/feature-guides/testing-your-code.md): keep the cache from passing tests for you.
- [File dependencies](tutorials/feature-guides/custom-file-sources.md) and [Dynamic dependencies](tutorials/feature-guides/dynamic-dependencies.md).
- [Choosing a backend](tutorials/feature-guides/choosing-a-backend.md) and [Configuration](getting-started/configuration.md).
- [The decorator path](how-it-works/decorator-path.md): how the key is built.
