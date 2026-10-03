# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] - 2026-10-04

This release takes the hidden limits out of the key and makes a cache hit
cheap, and lets you choose what a call's arguments contribute to it. Every walk that decides what reaches a key used to stop at a fixed
depth or count, and keyed what lay beyond by name or not at all; each now
follows everything, and a value or helper graph cash cannot finish keying runs
uncached with a warning instead of being keyed on part of itself. A class can
say what identifies its instances with `__cash_key__`, DataFrames are now
pickled instead of written as Parquet, `key=` and `ignore=` let a cached
function leave arguments such as a verbose flag out of its key, and a repeat call on a big pandas frame
no longer reads the whole frame. Around that, another sweep for stale answers
found more code and data the key never saw: a class's own data, a docstring
the code reads, a seed set by the caller, a helper reached as `module.f`.
311 commits in all.

A few names and one setting are gone, so read **Breaking** first. Some cached
entries miss once after you upgrade, but the cache folder is kept — see
**Upgrading** at the end.

### Breaking

- **`file_hash_full_max_bytes` is removed.** Every tracked file is now hashed
  in full whatever its size (see **Changed**), so the setting did nothing.
  `Cash(file_hash_full_max_bytes=...)` now raises as for any name that is not
  a setting, and a config file that sets it warns `CONFIG-UNKNOWN-KEY`.
- **`ParquetSerializer` and `get_serializer` are removed** from
  `cash.backends`: DataFrames are pickled like every other result. Use
  `PickleSerializer` where you named a serializer.
- **`cash.config` no longer re-exports its names.** Import `CashConfig`,
  `get_config` and `create_default_config` from `cash`, and `TierConfig` from
  `cash.config.schema`.

### Added

- **Choose what the arguments contribute to the key.** Every argument used to
  go into the key, so a `verbose` flag, a logger, or `"1"` against `1` split
  the cache, and wrapping a library function with such a parameter needed a
  hand-written wrapper. `@cash.cache(ignore=["verbose"])` leaves parameters out
  (a name that is not a parameter raises when the function is decorated), and
  so does annotating one as `cash.Ignore[bool]` (or `Annotated[bool,
  cash.Ignore]`); the two combine. `@cash.cache(key=fn)` hands `fn` the call's
  arguments with defaults filled in, and its return value stands for them. The
  key function's own code is part of the key, so editing it clears the
  entries; one that reads a file, the clock, randomness, the network or the
  environment warns `KEY-FUNCTION-IMPURE`, and one that raises runs the call
  uncached with `KEY-FUNCTION-RAISED`. `explain()` says when a hit was matched
  this way. Decorator only.
- **`__cash_key__` for objects that hold big data.** An object holding several
  large DataFrames was pickled and hashed whole on every call that took it,
  including as `self` in a cached method: about 0.7 s a call for eight 40 MB
  frames, even for a method returning a row count. A class can now define
  `__cash_key__(self)` returning what identifies an instance (a version, a path
  and mtime); the key uses it wherever the object sits, the call drops to about
  2 ms, and the entry is found again after a restart. A hasher registered for
  the type still wins. The first time an object is keyed this way in a process,
  a background check compares its content with what the same key stood for
  before and warns `KEY-STALE-CASH-KEY` on a mismatch; `check_cash_keys=False`
  turns the check off.
- **`RANDOM-UNSEEDED` for library draws.** `train_test_split(X)`,
  `KFold(shuffle=True)`, `SGDClassifier()`, `make_classification()` and
  `df.sample(3)` in a cached function froze the first split, fit or sample
  without a word, because the draw happens in library code. A library call that
  takes `random_state=None` and is called without one, or `.sample(...)` without
  `random_state` or `seed`, now warns and names the call.
- **New warnings for calls that run uncached**, each with its own page:
  `KEY-HELPERS-UNWALKABLE` (the helpers a call reaches cannot all be found or
  never end), `KEY-UNHASHABLE-CAPTURE` (a closure reads a captured object cash
  cannot hash), `CACHE-RETURNS-AWAITABLE` (a plain wrapper around an
  `async def` returns a coroutine; put `@cash.cache` on the `async def`) and,
  in notebooks, `NOTEBOOK-ANALYSIS-FAILED` (a safety check crashed, so the
  statement runs every time).

### Changed

- **DataFrames are pickled, not written as Parquet.** Parquet was 5 to 40
  times slower to write and 2 to 20 times slower to read than pickle protocol 5,
  and changed frames it could not store exactly: list and dict cells, UUIDs,
  `datetime64[s]`, integer axis names, `attrs`, and subclasses such as a
  GeoDataFrame. Every result is now stored with pickle protocol 5, so a hit
  hands back the frame the call returned. Parquet files were smaller for columns
  with few distinct values; if disk space matters more than speed, the
  `compress` setting recovers most of that. pyarrow is no longer used for
  storage.
- **No hidden depth or count limits on what reaches a key.** The helper walk
  stopped six calls down, the search for code in an argument eight containers
  down, the walk of a value 50 levels down, nested code four or eight levels
  down, and several other walks at fixed limits; what lay beyond was keyed by
  name or left out, and editing it served the old result. Which helpers fell
  past the cut even depended on `PYTHONHASHSEED`, so deep helper graphs never
  hit after a restart. Every walk now follows everything, in a fixed order, and
  ends cycles by what it has seen. The value walk is iterative, so a deeply
  nested value no longer overflows the stack and kills the process. When a call
  cannot be keyed whole — helpers that cannot all be found or that never end,
  a value deeper than pickle can follow — it runs uncached with a warning
  (`KEY-HELPERS-UNWALKABLE`, or `KEY-UNHASHABLE-ARG`, which now says the
  argument is nested too deeply instead of asking for a hasher). The notebook's
  checks for file writes, kept state and the globals a call reads follow every
  function a call reaches too, and the check for a matplotlib Figure in a result
  looks at every item at every depth.
- **Every tracked file is hashed in full.** A file over 256 MiB was hashed from
  three samples with its timestamps standing in for the rest, so a same-size
  edit between the samples with its modification time put back was served from
  the cache. Content now decides at any size. A full hash costs about one
  second per GiB, paid when the body reads the file or when its size or
  timestamps moved; a settled, unchanged file is still one `stat`.
- **Objects holding big data are keyed part by part.** Without `__cash_key__`,
  an object holding a pandas frame, or an array, frame or table of 1 MiB or
  more (sparse matrices included), is opened up and each part keyed on its own,
  so its frames go through the same fast check as a frame passed directly.
  Objects holding only small arrays, such as a fitted random forest, are still
  pickled whole, which is faster for them.
- **Cache hits cost less.**
  - A repeat call on a big pandas frame no longer reads the frame when the frame
    came from a numpy array, `read_csv` or a groupby, or shares its data with
    another frame (`df.assign`, `df[cols]`, `df.reset_index()`): 150–190 ms a
    hit on a 1M × 20 frame is now well under a millisecond.
  - A call with only numbers, strings, bytes or `None` builds its key about 30
    times faster; module constants are hashed once while they hold the same
    object; a list of 20,000 JSON-like records keys in under 100 ms instead of
    850 ms; dataclass methods and metadata lookups are no longer redone on every
    hit.
  - A file read soon after it was written is no longer hashed in full on every
    hit for the life of the entry once it has settled.
  - A polars result is served from RAM instead of being read back from disk on
    every hit in the same process, and a RAM hit on a pandas frame no longer
    scans its object columns each time.
- **Storing and starting cost less.** Large arrays and frames are written and
  read without being copied again and again (a store took about 5× `np.save`,
  a disk hit about 5× `np.load`). A plain script no longer imports IPython,
  and on Linux and Windows no longer imports psutil (175 ms → 125 ms for a
  one-function script). The first call after a restart reads each source file
  once (about 340 ms → 180 ms on a 3000-line module).
- **Notebooks cost less.** A loop that reads a folder of files
  (`for f in files: d = pd.read_csv(f)`) runs as one unit and still depends on
  every file it read: 1000 CSVs took 10.6 s cold and 14.1 s after a restart,
  now about 2.4 s. Notebook statement writes stay in the background, and the
  cost records behind `%cash_stats` are no longer rewritten after every cell.

### Fixed

**Wrong or stale answers in decorated functions.**

- **Data the key never saw.** A user class's own data — a module global read by
  an inherited method, property or `__init__`, a class attribute set at run
  time, `cfg.Cfg.RATE`, the fields of a namedtuple or dataclass made at run
  time — was served stale after it changed (a namedtuple whose fields were
  reordered even swapped values on a hit). Also keyed now: a module constant
  read through a package path (`pkg.conf.RATE`), a constant read by a lambda
  default, data set on a function (`scale.k = 1`), a docstring the code reads
  (`f.__doc__`, a tool description built from it), environment variables read
  through `os.path.expandvars`, `expanduser`, `Path.home()`,
  `tempfile.gettempdir()` and `shutil.which`, and a clock helper called as
  `clocks.now()` or `self.stamp()`.
- **Code the key never saw.** A helper handed on as `helper.g`
  (`df.apply(features.row)`, `map(helper.g, xs)`), or looked up by a
  written-out name (`sys.modules["helper"].g`, `attrgetter("g")`), a function
  inside an object inside a global, user code in a numpy object array or an
  object column, the state of a callable instance passed to `cash.cache` or
  captured by a factory, and what a callback passed as an argument reads. Each
  edit was served the old result; each is now keyed. `cash.cache(Scaler(2))`
  also finds its entries in a new process.
- **Cached functions calling cached functions.** A cached function reached
  through a helper, a static or class method or an import inside the body, one
  passed as an argument, held in a list or captured by a factory, one cached by
  another `Cash` instance, and one swapped by `importlib.reload` did not carry
  its edits, globals or TTL into the caller's key.
- **Arguments that shared an entry.** `functools.partial` bound values keyed by
  their `repr` (numpy elides the middle of a long array), a set inside a
  partial, deque or iterator keyed in a per-process order, a list shared by two
  arguments keyed like two equal lists, numpy scalars next to an empty list
  (which dropped `*args` from the key), and two lambdas on one line differing
  only in a comprehension or inner lambda. A config written with literals and
  the equal one parsed from JSON now share a key.
- **A caller's seed is part of the key.** `np.random.seed(s)` at module level or
  in an outer function used to serve every seed the first seed's draw; a
  function seen drawing is now keyed by where the stream stands. A function
  that seeds itself is not, so it still hits. An optional generator written
  `rng or np.random.default_rng()` or with `if ... else` now warns
  `RANDOM-UNSEEDED`.
- **A hit moves a generator on.** A cached function drawing from a generator
  held in a global (`rng = np.random.default_rng(42)`) or handed in as an
  argument (`boot(x, rng)`) left it where it was on a hit, so the caller's
  next draws came from the wrong place in the stream. A miss now records where
  the call moved the generator and a hit moves it on the same way. A generator
  passed in no longer needs `assume_safe` to be cached, and is no longer
  reported as an argument mutation.
- **What can't be keyed runs uncached.** A closure over an object cash cannot
  hash (a settings object holding a lock) and a passed function whose default
  cannot be keyed were left out of the key without a word; they now run
  uncached with `KEY-UNHASHABLE-CAPTURE` and `KEY-UNHASHABLE-DEFAULT`.
- **Calls that never cached now do.** Any function reaching a dataclass or an
  annotated class on Python 3.14; every scipy sparse argument; any function
  calling `pl.col(...)`; a function logging through a module logger in Jupyter
  (every call missed with "global log changed"); a function that writes, reads
  back and removes a scratch file or unzips into a temporary directory; and
  keying an object held in a global on Python 3.10.
- A library writing into a big argument is caught on every miss, not only for
  arguments that hash quickly, so the warm run no longer skips the change.

**What a hit hands back.**

- A nested cache hit brings only the file reads of the entry it served: after
  `fit("r1")` and `fit("r2")` ran, `scen("r2")` no longer recomputes when
  `r1`'s file changes.
- A cached generator replays the items it yielded, even if the caller edited
  them, and passes on its return value (what `yield from` evaluates to).
- A read-only numpy array comes back read-only, and a module-level sentinel
  (`MISSING = object()`) comes back as itself, so `is MISSING` holds on a hit.
- Lists, dicts and arrays inside a DataFrame's object column are copied on a
  RAM hit, so appending to one no longer changes the next call's result.
- A result no longer gains `_cash_lineage_*` attributes in its `__dict__`, so
  `vars()`, `==` on a `SimpleNamespace` and an object you passed in and got back
  stay as they were.
- A Figure or Axes anywhere inside a result is refused, so pyplot's current
  figure is never swapped for the cache's copy.

**Files.**

- `file_depends_on=` a directory or a glob tracks what it holds: an edited,
  added or removed file recomputes. A glob used to be recorded as a missing
  file of that literal name.
- Reads that were not recorded: package data read through `pkgutil.get_data`
  or `importlib.resources`, sizes from `os.lstat` and `os.scandir` entries, a
  path whose `os.stat`, `getsize` or `listdir` failed because it was missing
  (it is now an input that recomputes once it appears), the `-wal` file of a
  SQLite database in WAL mode, and every file a memoised loader read (more than
  sixteen used to be dropped).
- A metadata or existence check records a path by one rule, whichever spelling
  the code used.
- A package first imported inside a cached call or with notebook tracking on
  can open its own data files again.
- With `cache_dir=".cash"`, a write to `.cash_exports/` is reported as the
  function's effect again, and on Windows a differently cased spelling of the
  cache folder is not.

**Notebooks.**

- **Re-seeding brings back the right draws.** After `rng =
  np.random.default_rng(7)` was re-run, a slow cell drawing from `rng` was
  served the draws a moved generator gave, which no top-to-bottom run gives,
  and Restart & Run All could do the same. A statement that draws now moves
  the generator on, also through a helper that reads it as a global. Running a
  later draw cell on its own first runs or restores the draws above it, even
  after a restart and even cheap ones cash did not store, so it shows what a
  top-to-bottom run shows; plain Jupyter would draw on from where the
  generator stands. A helper call that moves a generator it reaches is no
  longer cached per call.
- **Cells that clear and redraw a folder of files.** An unrelated cell, even
  `x = 41 + 1`, re-ran a chart cell that empties its folder with `unlink`,
  `os.remove` or an `os.listdir` loop, deleting its files, sometimes without
  drawing them again. Clearing a folder now counts as writing it and listing
  one as reading it, so only cells that read the folder re-run the writer, and
  a reader of a file in the folder re-runs after it. A replay clears the
  folder first, so a chart dropped from the cell or a file put there from
  outside no longer survives.
- **A deleted temporary is rebuilt.** After a repair left
  `m = clean.merge(...)`, `m['hour'] = ...`, `del m` partly re-run, the next
  edit of `clean` re-ran the column write on the old `m` and skipped the
  merge, so the aggregates below came from the old cleaning while the badge
  said the chain was refreshed.
- **A broken module fails the cell.** Editing a project module into one that
  no longer imports (a syntax error, say) gave an empty error, and the next
  cell ran the module's old code as if the edit had loaded. Every cell now
  fails, showing the error with file and line, until the module loads again;
  a cell of only `%cash` magics still runs.
- **Values are hashed whole.** The value hash read a frame's first five rows,
  an array's first 100 elements and a long list's ends, so an edit past them
  read as no change: a statement below was served the old result, and a call
  that rewrote row 500 of its argument was skipped on a hit.
- A closure that keeps state is never restored, however deep it sits in a
  result, and a value whose storability check fails is not stored.
- A statement whose safety check crashes runs every time instead of being
  judged pure.
- The notebook and the decorator give one answer to which globals a function
  changes, following helpers at any depth: `x = f(data)` no longer runs every
  time because a loop variable looked like a global, and a hit no longer skips
  a helper's append to a global.
- A restart no longer skips statements that write into an object they did not
  copy (`b = a.astype(float, copy=False); b += 1`), so `a` keeps its new value.
- Editing a function that a module passed whole (`run(helpers)`) reaches, a
  `%reset` that deletes no variable, a reloaded module whose function landed on
  a freed address, and a loop body's bare method call (`buf.write(...)`) each
  left a later statement on the old value; each is now seen.
- A user function named like a builtin (`sorted`, `next`) and marked
  `@cash.stateful` runs every time; another cell calling a class-decorated
  function is counted; a write through `np.asarray(arg)` is reported; and an
  installed copy of a function no longer shares the project copy's verdict.
- `# @cash:no-cache` on a statement ending in `;` is kept when cash re-runs it,
  and `# @cash: assume-safe` is read by one parser on both paths.
- A cell with top-level `await` runs uncached when the upstream check fails,
  as other cells do, instead of failing.
- A statement's time no longer includes cash's own setup, so `x = 1 + 1` as a
  kernel's first statement is not stored.
- A value nested too deeply to pickle no longer ends the cell with an internal
  error.

**Warnings.**

- Work on an object the function made (`h = hashlib.sha256(); h.update(b)`,
  `m = LinearRegression(); m.fit(X, y)`, `d = deque(xs); d.popleft()`) is no
  longer reported as a side effect.
- Reading `sys.stdin` is reported like `input()`.
- An attribute that shares a module global's name (`b.lock`) no longer warns
  `KEY-UNHASHABLE-GLOBAL` about that global.
- Once one cached function returned `time.time()`, every cached callee in the
  process was reported as a clock read; and a library function such as
  `os.path.isdir` no longer adds `<frozen genericpath>` as a file dependency.

**Storage, backends and the CLI.**

- A failed lookup no longer fails the call: a Redis server that is down, or an
  entry naming a class that is gone, reads as a miss in every backend.
- The RAM tier is locked against use from several threads, which could raise
  `KeyError` or "dictionary changed size" out of your call; a process forked
  while cash was writing no longer hangs.
- A wrong or unwritable `cache_dir` with the SQLite tier warns
  `CACHE-DIR-UNWRITABLE` and carries on, instead of making `Cash()` or
  `%cash_on` raise.
- One TTL rule in every backend, the decorator and `cash clear --expired`: a
  bare file backend no longer serves entries cleanup deletes.
- A cache without a RAM tier no longer reports a value written to disk as
  "RAM only, gone at the next restart".
- `explain()` reports a miss for a generator entry whose chunk is gone, as the
  next call does.
- `cash clear --entry`, `--function` or `--tool` with an empty value exits with
  an error instead of clearing every entry.
- `cash info` shows the cap each tier is built with, and a cap you set as you
  wrote it (`2GB`, not `1.9 GiB`); `cash inspect` totals only entry files.
- The cache explorer no longer writes its display fields into stored entries,
  and escapes names and source in its HTML view.
- On a console that cannot print emoji, every status marker turns into its
  ASCII stand-in instead of vanishing.

### Upgrading

**The cache folder is kept.** The storage format version did not change: new
entries are written beside the old ones, and entries 0.12 wrote are still read.
An older version of cash reads a large array or frame stored by 0.13 as a miss.

**Some entries miss once**, where the key now holds more than it did:

- functions that build classes, and functions reading a global list that holds
  one inner list twice;
- calls taking an object that holds a pandas frame or an array, frame or table
  of 1 MiB or more (it is now keyed part by part);
- arguments holding one string, date or `Decimal` object in several places,
  which now key like the same values written out separately;
- functions reaching a user class with class-level data, a cached function, or
  a callback, whose key now folds what those read;
- cached `functools.partial`s and callable instances, now named by content and
  by class;
- values more than 50 levels deep, and `dict` or `list` subclasses with
  attributes of their own;
- an entry whose input file over 256 MiB was hashed from samples;
- notebook calls and statements keyed by the value hash, which now reads whole
  values;
- a DataFrame 0.12 stored as Parquet, which is read as a miss and stored again
  as a pickle.

Then:

- Remove `file_hash_full_max_bytes` from your settings, and replace
  `ParquetSerializer`, `get_serializer` and imports from `cash.config` as listed
  under **Breaking**.
- If your cache holds many DataFrames with low-variety columns and disk space
  matters, turn on `compress`.

## [0.12.0] - 2026-09-27

This release makes cash smaller to learn and harder to fool. There is one plain
`pip install cash-lib`, one name and one import path for each public thing, and
eleven fewer magics: the ones that repeated what the badge, `%cash_stats`,
`explain()` or `cash clear` already do. Around that, more sweeps went looking
for a cached call that returns a stale answer — an environment variable, a file
checked only by size or existence, a dict's order, a pandas category, a helper
marked `@cash.pure`, a closure, a compiled extension — and each one found is now
part of the key. The notebook and the decorator now judge side effects from the
same list, and the documentation was rewritten around those two paths. 817
commits in all.

Some names and settings are gone, so read **Breaking** first. Every cached entry
misses once after you upgrade — see **Upgrading** at the end.

### Breaking

- **One install, no extras.** `pip install cash-lib` now installs everything
  cash itself needs: psutil, and tomli on Python 3.10. The extras (`[pandas]`,
  `[polars]`, `[notebook]`, `[redis]`, `[s3]`, `[viz]`, `[memory]`, `[toml]`,
  `[ipynbname]`, `[cloudpickle]`, `[sqlite]`, `[all]`) are gone; pip warns about
  an extra that does not exist and installs cash alone. Install what you use
  yourself: `pyarrow` to store DataFrames as Parquet, `redis` or `boto3` for
  those backends, `s3fs` or `gcsfs` for tracked `s3://` and `gs://` reads,
  `ipywidgets` and `matplotlib` for the `show_stats()` dashboard. When a feature
  needs a package that is missing, the error names it (`pip install redis`).
  ipynbname is no longer used: cash finds the notebook itself.
- **Removed magics**, and what to use instead:
  - `%cash_verify`, `%cash_repair`: a damaged entry is already discarded when it
    is read. For a clean start, `cash clear --all` and restart the kernel.
  - `%cash_export`, `%cash_import`, `%cash_diff`: share a cache through a shared
    backend (one `cache_dir`, Redis or S3).
  - `%cash_benchmark`: time the cell with `%%time` under `%cash_off`, and read
    the badge and `%cash_stats` under `%cash_on`.
  - `%cash_audit`: `%cash_provenance` records the same for every variable.
  - `%cash_log`: `%cash_debug on` prints the same records in the cell.
  - `%cash_track`: modules of your project are tracked as soon as a cell imports
    them. Watching an installed package for edits has no replacement.
  - `%cash_feedback`: its two links now end `%cash_help`.
  - The `%%cash` cell magic: use `%cash_on`, and `# @cash:ttl=N` in place of its
    `ttl=N` line.
  - The `collab` and `inspect` topics of `%cash_help`: `%cash_help <magic>`
    now prints any magic's full help (`%cash_help badge`, `%cash_help debug`).
- **One spelling per directive.** `# @cash:nocache`, `allowrandom`, `cachefit`
  and `nocachecalls` are now only `no-cache`, `allow-random`, `cache-fit` and
  `no-cache-calls`. `# @cash:cache-calls` is gone: calls are cached by default.
  An old spelling is not silently ignored — it warns
  `ANNOT-UNKNOWN-DIRECTIVE` and names the directive you meant.
- **One name and one import path for each public thing.**
  - `cash.mark_pure(f)` and `cash.mark_stateful(f)`: use `cash.pure(f)` and
    `cash.stateful(f)`, which mark `f` and return it unwrapped.
  - `cash.mark_opaque(cls)`: use `cash.opaque(cls)`, as a decorator or as a
    call on a class you do not own. It no longer sets `__cash_opaque__` on the
    class, so it works on classes that refuse new attributes.
  - `cash.experimental` is gone. Import `CacheExplorer` from `cash.ui.explorer`
    (or call `Cash().explorer()`), `DependencyGraph` from `cash.graph` and
    `AnalyticsManager` from `cash.analytics`. `cash.ui` re-exports nothing.
  - `CacheDebugger` and `visualize_notebook` are removed: their hit and miss
    answers never matched the real key. The badge and `f.explain()` answer "why
    did this miss?" correctly.
  - `CascadingBackend` is removed: use `TieredBackend`. `Cash(backends=[...])`
    now builds a `TieredBackend`, with the same caps and rules as the default.
  - `CacheExpiredError` is removed. Nothing raised it: an expired entry is a
    miss.
  - `cash.analyze_function_purity` is removed, with no replacement.
  - `from cash import *` no longer imports `cache`, `show_stats`,
    `register_hasher` or `help` (it replaced the builtin `help` and built the
    default cache as a side effect). Use `cash.cache` and so on.
  - Modules that were never public API moved: `cash.purity_analyzer` and
    `cash.purity_flow` are under `cash.analysis`; `cash.notebook.purity` is
    gone (import `pure`, `stateful`, `is_pure`, `is_stateful` from `cash`);
    `cash.notebook` no longer re-exports `CashMagics`, `StatementProcessor`,
    `UpstreamChecker` or `CodeAnalyzer`; `cash.utils` and `cash.logging` are
    private.
- **For custom backends and data sources.**
  - A `DataSource` implements `get_id()` and `state_token()`; `has_changed()`
    and `update_state()` are gone.
  - `cash.backends` no longer exports `CacheMetadata`. The metadata a backend
    sees is a plain dict, and `EntryMetadata` (in `cash.backends`) documents its
    keys. A custom backend subclasses `CacheBackend`.
  - `CloudPickleSerializer` and `cash.backends.lazy` (`LazyProxy`) are removed;
    cash never used them.
  - `TieredBackend(min_persist_compute_s=..., min_persist_savings_pct=...)` is
    now `TieredBackend(policy=PersistencePolicy(...))`, from
    `cash.backends.persistence_policy`.
- **Settings.**
  - `smart_persistence` (`CASH_SMART_PERSISTENCE`) is removed. There is one
    rule for what reaches disk, and a `TieredBackend` you build yourself uses
    the same 0.1 s floor as the default one instead of 1.0 s.
  - `"tiered"` is no longer a tier type in a `tiers` list;
    `backend = "tiered"` still names the RAM + disk stack.
  - A config file's settings must sit under `[cash]` (or `[tool.cash]` in
    `pyproject.toml`). A file with only top-level keys is not read, and
    `CONFIG-INVALID` says so.
  - Settings given in code are checked, and a bad one raises instead of being
    dropped: `Cash(...)` refuses a keyword that is not a setting (`Cash(ttl=60)`
    used to be accepted and ignored), negative sizes, counts and intervals, and
    an empty `cache_dir`; `@cash.cache(ttl=...)` takes `None`, a number of
    seconds or a `timedelta`; `depends_on=` takes callables and `DataSource`s
    (a path belongs in `file_depends_on=`); `configure()` refuses to change the
    tiers of a backend you passed in. From a file or an environment variable, a
    bad value is reported as `CONFIG-INVALID` and the default applies. A
    `CASH_*` variable set to the empty string counts as unset.
  - `create_default_config()` no longer overwrites an existing file; pass
    `force=True`.

### Added

- **`with cash.assume_safe():`** waives purity findings for the lines inside
  it — between `# @cash:assume-safe` on one line and `assume_safe=True` for a
  whole function. It is code, so a typo fails on the first run instead of
  waiving nothing, and adding or removing it keeps your stored results. It works
  at the top level of a notebook cell too. `# @cash:assume-safe` now also works
  on a notebook statement, where it lets cash cache a call it would refuse by
  name (a `session.post` that only runs a search, say).
- **Environment variables are inputs.** A cached function or notebook statement
  reading `os.environ["TENANT"]` used to serve the first tenant's answer to
  every other one. A named environment variable and the working directory are
  now folded into the key on both paths — including `"X" in os.environ`,
  `Path.cwd()` and `abspath` of a relative path — and a miss names the variable
  that changed. Only the value's digest is kept, never the value.
- **How big the disk cache may grow, said once.** `%cash_on` prints the disk
  cap, the folder and the rule that set it ("a quarter of the free disk space",
  "set by max_cache_size"); scripts log it at the first disk write and the exit
  summary names it. The first time the cap removes entries, cash says how much
  went. A miss on an entry the cap removed now reads "evicted to make room", and
  recomputing one that took two seconds or more warns `CACHE-EVICTED-RECOMPUTE`.
- **`KEY-NETWORK-READ`.** A GET or a database query in a decorated function is
  an input the key cannot see, so cash now advises `ttl=` for it instead of
  calling it a side effect; a `ttl=` silences it, and `strict=True` raises
  unless one is set. A connection your code opens through a client object
  (`requests.Session().get`) counts too.
- **Rewording a docstring or running a formatter keeps the cache.** Quote style,
  trailing commas, added parentheses and docstrings no longer change a key, as
  comments already did not.
- **An `analytics` setting** (`CASH_ANALYTICS=0`) turns off the notebook's
  usage history; nothing is written when it is off.
- `ttl=` accepts a `datetime.timedelta`, `depends_on=` a single callable and
  `file_depends_on=` a single path.

### Changed

- **Editing a `@cash.pure` or `@cash.stateful` helper invalidates its callers.**
  A marked helper was left out of the key, so its edit kept serving the old
  result. A marker says what a helper may do, not what it computes: it is now
  keyed like any other helper.
- **A statement around a cached call is stored only for its own work.**
  `b = shifted(a) + 1` used to keep `shifted(a)`'s result twice, as the call's
  entry and inside the statement's value. The statement is now priced and stored
  for what it does beyond the call, and `b = shifted(a)` stores a small
  reference to the call's entry instead of a copy.
- **A call whose result a method is called on is cached.** In
  `b = shifted(a).sum()`, `shifted(a)` used to run again on every run.
- **A registered hasher keys the value on its own.** `cash.register_hasher` now
  applies wherever the value sits in an argument — inside a list or a dict, not
  only at the top level — and cash no longer searches the value for code. The
  documented way to keep a logger out of the key used to reach every logger and
  handler in the process and warn `KEY-OPAQUE-CALLABLE`.
- **Side effects are judged the same way on both paths.** The notebook and the
  decorator read one list of what writes, reads the clock or talks to the
  network. In a notebook these statements now run every time instead of being
  restored: `input()` and `getpass`, `os.popen` and the other process
  starters, database writes (`INSERT`, `commit`), a POST through a
  client object (`session.post`, `s3.upload_file`) or spelled
  `requests.request("POST", ...)`, more file writers (`shutil.copyfile`,
  `Path.touch`, `Path.unlink`, `os.chmod`), and clock reads such as
  `time.strftime` and `pd.Timestamp.now()`. A decorated function is now warned
  about the same calls, and about pyplot drawing; `re.compile`, `gzip.open` and
  `df.eval` are no longer reported as I/O.
- **Tables, dicts and objects are keyed by all they hold.** Two values that
  differed used to share an entry: pandas columns of the same values over
  different categories, a column holding `1` and one holding `'1'`, two slices
  of one PyArrow table, a polars column of Python objects, equal dicts in a
  different order (which changes `pd.DataFrame(d)`'s columns), an ndarray or
  `str` subclass carrying a set, a grid built as `[[0] * 3] * 3`, and two classes
  named `Config` in two modules. The notebook's loop and call keys now use the
  decorator's hashers, so an `int64` and an `Int64` series no longer match.
- **Notebook usage history moved** from `~/.cash/analytics.db` to the per-user
  cache folder (`~/.cache/cash` on Linux, `~/Library/Caches/cash` on macOS,
  `%LOCALAPPDATA%\cash` on Windows), and `show_stats()` shows this session's
  events.
- **Cash spends less on itself.** A hit on 200,000 numpy floats in a list takes
  18 ms instead of 1.3 s; string columns key five to eight times faster; cash's
  I/O wrappers are installed only while a cached call or `%cash_on` needs them
  and keep the real function's signature and docstring for `help()`;
  `%cash_status` counts entries without reading them; a long-running process no
  longer grows a call log, a lock per key or a `Cash` that can never be freed.
- Byte sizes read the same everywhere, in binary units labelled as such
  (`2.0 GiB`, not `2 GB`).
- Links in warnings and in the config template point at the current docs.
- **The documentation was rewritten** around the decorator and the notebook,
  each with its own tab and limitations page, with dark mode and pages that fit
  a phone.

### Fixed

**Wrong or stale answers in decorated functions.**

- **Code the key never saw.** An object a factory-built helper captured, the
  class a classmethod helper is bound to, a function or model held in a data
  global (inside an sklearn pipeline too), the arguments of a `partial` or the
  state of a callable object in a global, a function from a C or Cython
  extension built in your project, a function with no source file (`python -c`,
  a heredoc), two closures from one factory capturing different modules, and a
  script's cached callee run as `python pipeline.py`. Each edit was served the
  old result; each is now keyed.
- **A redefined function is never served the old version's result.** A new
  definition could inherit a dead one's identity and hit on its entries.
- A `functools.wraps` wrapper with parameters of its own binds a call to them,
  not to the inner function's.
- A part of the key that cannot be built makes the call run uncached with a
  warning, instead of being left out of the key.
- `explain()` builds the key a real call builds, and `m.score.explain(2)` on a
  method raises and names the spelling that works,
  `Model.score.explain(m, 2)`.

**Files, remote data and pools.**

- **Reads that were not recorded.** `os.path.getsize` and `getmtime` (the
  "newest export" idiom), an existence check that answers yes, a reader bound
  by name (`from pyarrow.parquet import read_table`) and pyarrow's reader
  classes, a dataset read from a directory, a glob or a list of paths, a
  SQLite database opened by `file:` URI, a file read while the app imports
  before the first decoration, and what `multiprocessing.Pool` and joblib
  workers read (every scikit-learn `n_jobs=`).
- `file_depends_on=` and `FileDataSource` check the file's content, not its
  mtime, like any tracked read.
- A remote prefix or glob is checked by its listing, so a new partition
  invalidates; a read with `storage_options` is checked against the store they
  name; remote checks use the proxy settings in effect at the time.
- On Windows, a path longer than 260 characters and a drive letter written as
  `D://` are kept as local files, and a dependency cash cannot check is never
  counted as fresh.
- A file rewritten with the same size within two seconds is caught when cells
  run without history (from an agent or an extension).

**Notebooks.**

- **cash finds the notebook** when the kernel runs in its own virtualenv or
  conda env, on JupyterHub, and behind an HTTP proxy. Upstream tracking used to
  stay off for the whole session.
- **Top-level `await` works on IPython 9.16 and later**; every such cell failed
  or hung.
- **A virtualenv inside the project folder** (`python -m venv .`) no longer
  makes every local module look installed, so editing one invalidates its
  cells.
- **Module edits are seen**: a submodule imported as `from pkg import helpers`
  hits on the second run, a same-size edit saved within a second of the import
  is not hidden by its `.pyc`, and a comment added to a module no longer re-runs
  the work built on it.
- A `@cash.stateful` function or a file writer called through a module of your
  project (`helpers.announce(...)`, `wr.write_csv(...)`) runs every time, as the
  same function defined in the notebook did.
- `# @cash:no-cache`: statements reading its output see the new value, a draw
  under it continues the live random stream, and the directive at the end of a
  line works like one on its own line.
- Cached control flow behaves like Python: `try`/`except`/`finally` runs
  `finally` and matches handlers as Python does, a `for` loop's `else` runs, an
  error in an `if` condition keeps its type, and a loop that fails part-way
  stops and shows your error.
- **Restart & Run All re-runs less.** Several statements keyed differently in
  each kernel and recomputed after every restart: `M = enc.fit_transform(data)`,
  calls into libraries whose names clash with a variable (`u`, `random`), seeded
  draws, loops that learned to split, a `def` in a cell with a magic, and code
  run under coverage.
- **A function that keeps state in its closure** (a `nonlocal` counter, a list
  it appends to) is no longer served the previous call's answer. A cached call
  to it restores what it changed, or runs every time when cash cannot restore
  it, and a statement around it re-runs instead of skipping the call.
- One stall while cash times the cheap calls in a comprehension no longer
  keeps them cached for the rest of the statement.
- An open file or a generator returned by a call inside a statement is never
  stored and handed back drained.
- Code a frontend runs silently (a variable explorer) is no longer treated as a
  cell.
- `cash.configure(cache_dir=...)` and `configure(persist_all=...)` reach a
  running notebook, and `call_cost_floor_seconds` is honoured.

**Badges, magics and warnings.**

- The badge's EXECUTED time leaves out cash's own work, so it agrees with the
  saved time; "step x/N" counts only statements that run; and quotes in a value
  can no longer break out of the badge's HTML.
- `%cash_provenance` and `%cash_badge` use the badge's words (EXECUTED, CACHED);
  `%cash_status` and `%cash_provenance` refuse arguments they do not know;
  `%cash_debug` keeps a log path's case and turns fully off.
- `RANDOM-UNSEEDED` says what happens to a cheap draw (it is drawn again) and is
  not given for a `no-cache` statement; a fit that is not cached is not called
  frozen.
- Warnings: two different warnings about one function are both shown; a warning
  shown once per cache is recorded only when a filter let it through, so
  `filterwarnings("error")` in CI fails every run, not just the first;
  `KEY-DEPENDS-ON-OPAQUE` fires; library C code (a lock, a stream) no longer
  warns `KEY-OPAQUE-CALLABLE`; one slow first call no longer triggers
  `CACHE-NET-LOSS`; every warning carries its `.code`.
- `show_stats()` prints its table in a script, the cache explorer's preview
  works, and the nbconvert strip preprocessor removes only cash's own output.
- `cash.help()` no longer tells coding agents four wrong things.

**Storage, configuration and the CLI.**

- `f.cache_clear()` reaches other running processes, retries a file Windows
  holds open and warns `CACHE-CLEAR-INCOMPLETE` about what survives, and resets
  every counter.
- Cached generators can be advanced from any thread, two streams of one call no
  longer share chunks, and a quick generator reaches disk by the same rule as
  any result.
- A result one backend cannot pickle no longer fails the call; a forked worker
  no longer hangs on the parent's in-flight write; another process's new entry
  is never read with the old entry's metadata; every tier expires an entry by
  one `ttl` rule, and a bad stored `ttl` reads as expired.
- A DataFrame comes back from disk with its `RangeIndex`, and one Parquet
  cannot return as it was is pickled instead.
- A small RAM tier keeps its entries under memory pressure, and a failed memory
  reading never fails a write. `shutdown_write_timeout=0` no longer warns about
  writes that landed.
- Settings: a relative `cache_dir` is resolved once, where it was given; a
  `cache_dir` too long for Windows paths warns at once; a failed `configure()`
  leaves everything as it was and takes settings the way `Cash(...)` does;
  `configure(summary=...)` works both ways; every tier setting is used or
  reported; `configure()` and `disabled()` reach pool workers; `disabled()`
  blocks in two threads unwind correctly; `Cash(backend="sqlite")` builds that
  backend.
- `cash info`, `inspect` and `clear` find the cache where the kernel or a tier's
  own `cache_dir` puts it, `cash clear` removes every file cash writes, names a
  file another process holds instead of printing a traceback, and can be re-run
  after it stops part-way. `--function`, `--entry` and `--expired` on a SQLite
  cache are refused rather than reporting nothing cleared. Names the console
  cannot encode no longer crash the CLI or `show_stats()`.

### Upgrading

**Every cached entry misses once.** Keys changed in several ways — formatting
and docstrings are ignored, and tables, dicts and objects are hashed more fully
— so the first run after upgrading recomputes everything and writes new entries
beside the old ones. From then on, reformatting or rewording a docstring keeps
the cache. `cash clear --all` removes the old entries if you would rather not
carry both.

Then:

- Replace the removed names, magics and directive spellings listed under
  **Breaking**. An old directive spelling now warns rather than being ignored.
- Drop the extras from your install line: `pip install cash-lib` plus the
  libraries you use (for example `pip install cash-lib pandas pyarrow`).
- Move the settings of a config file that has only top-level keys under
  `[cash]`.
- Old cache layouts are no longer migrated: an entry written without a checksum
  is recomputed, and a SQLite database at its old location (where the cache
  folder path itself was the database file) is not picked up. A loop's learned
  split point is measured again once.
- Usage history starts over in the new location. The old `~/.cash/analytics.db`
  is no longer read and can be deleted.

## [0.11.0] - 2026-09-22

Eight rounds of user testing on the notebook path — five projects each, run by
testers who had not seen the code — plus several sweeps spent deliberately
attacking the decorator. They found wrong answers, stale charts, repairs that
gave up, and a great deal of time cash was spending on itself. This release is
those fixes — 236 commits of them — and the last round found no wrong value in
any of its five projects, the first round for which that is true.

Nothing here breaks an API. Some cached statements will recompute once after you
upgrade — see **Upgrading** at the end.

### Added

- **`cash info`, and a cache you can look at.** `cash info` says what the cache
  holds and what it weighs; `cash inspect` lists entries with the time each saves
  beside the space it takes, and names the function behind a call entry.
  `%cash_stats` now says up front that its numbers are this kernel's.
- **`cash.disabled()` and `CASH_DISABLE`.** Turn caching off for a block and get
  back exactly what you had, or run a whole program uncached — the measurement a
  "did cash help?" question actually needs, and the way to test code that uses
  cash.
- **Why a call missed.** The miss reason survives a process boundary and shows up
  in the summary, per call, and in `explain()`. In a notebook, a cached call that
  re-ran names the input that moved rather than reporting "0/6 hit" and leaving
  you to guess.
- **A net saving you can quote after a restart.** `%cash_stats` keeps what each
  computation cost beside the cache and credits the least it was ever measured to
  cost, so a Restart & Run All reports a number instead of a range straddling
  zero. A cache built on another machine still vouches for nothing and still
  prints the range.
- **A statement depends on the symbols it reads, not on the whole module.**
  Editing one helper in a shared `utils.py` no longer invalidates everything that
  imported anything from it.
- **A damaged entry is a miss.** A truncated or corrupt entry recomputes instead
  of raising — or, worse, being trusted.
- **`CACHE-RESULT-SHARED`** names a result the caller still holds: the shape where
  mutating what you got back changes what the cache serves next.

### Fixed

**Wrong or stale answers.**

- **A library call that changes its argument in place is replayed after a
  restart.** The scanpy shape — `sc.pp.calculate_qc_metrics(adata, inplace=True)`
  — restored `adata` from its constructor and never re-ran the call, so the cell
  below stopped with `KeyError`. `heapq.heapify(xs)` is the silent form: no error,
  the wrong answer. The same call is no longer restored as a no-op mid-session
  either, and the decorator no longer stores a call that changed an argument it
  was given.
- **The runtime and the upstream simulation agree on every lineage.** The two
  engines derived a statement's inputs differently in a dozen places — imports,
  two calls on one line, a loop's accumulator, a `def` consuming the globals its
  body reads, an attribute a callee uses. Each disagreement was a statement that
  re-ran forever, or one served against state the simulation had mis-modelled.
- **An edited helper module reaches the cells below its caller.** A reload
  compiles from source rather than stale bytecode, an aliased module is still the
  module it is, what a reload computed restores in the next kernel, and the repair
  rebuilds the whole chain built from the edit rather than its last link.
- **After a restart, the repair finishes the job.** Imports are re-run and never
  stored; a re-run import gets the `sys.path` setup above it; a call to a notebook
  function, what a loop built, a frame rebuilt in steps, and a writer whose files
  are already on disk all behave as the kernel that made them would.
- **Consumables are never served.** A statement producing an open file handle or a
  generator is not cached, and the decorator keeps a returned file handle a file
  handle.
- **Files cash could not see.** `Path.stat()`, `Path.glob` on 3.10 and 3.13,
  pyarrow readers, a reader called by keyword, a file read once and memoised, a
  file read in a thread pool or a process pool the call started, and a file that
  changes *while* the call runs — each was a dependency that went unrecorded, or a
  cache that kept serving after the file moved.
- **Reads that are not your data no longer key your work.** The interpreter's own
  files, package metadata, what a library reads while importing, and a file
  belonging to another installed package — a time zone `zoneinfo` loads once per
  process was the last of these, and it made the first restart after an edit re-run
  an entire pipeline.
- **Hashing.** A namedtuple row (so `itertuples` loops restore), a frame with a
  text column (by content, not by address), an F-like strided array against its
  F-contiguous copy, a cyclic argument holding a set (reported instead of hanging),
  and a mutated cached result keyed by what it holds.
- **The decorator's key.** Several attack sweeps closed the ways a call could be
  served a result built from code or data the key never saw: globals reached
  through a callable or a default, constants imported inside a function, a module
  in a closure, `itemgetter`-type globals and partials, classes named in type
  annotations, a helper edited on disk after import, a patched helper, a mocked
  library function (now run uncached rather than storing the fake), and a function
  run by joblib's or a spawned pool's workers.

**Repairs and charts.**

- **A chart is judged by what was drawn into it.** A figure's own artists —
  `sc = ax.scatter(...)`, handed to `fig.colorbar(sc)` — are part of the figure,
  not part of the data, so a chart with a colorbar is no longer reported stale
  whenever an unrelated repair asks. A chart already on disk is not drawn again
  after a restart, a helper handed an `Axes` in a loop draws on every run, and a
  replay redraws a figure with all of its history, only when it must.
- **A stale export is named.** When an upstream edit changes what a writer would
  write, the badge says which file is now out of date instead of leaving it to be
  discovered later — judged by what that writer recorded when it wrote, so the
  notice fires on a real change rather than on a lineage that merely moved.
- **A cell that reads files through a list of paths leaves unrelated writers
  alone.** A read whose path is only known at run time used to rule nothing out,
  so a new cell could silently redraw a chart above it.
- **A display expression does not change the frame it shows.**
  `df[mask].groupby('hour').size()` counted as an in-place mutation of `df`, and
  everything built from `df` re-ran.

**Badges, warnings and numbers.**

- The text badge headlines a mostly restored cell as CACHED with its counts, lists
  an upstream step once, follows the notebook's order, shows what a re-run upstream
  step printed, counts a loop's trips rather than its statement runs, says ERROR in
  the header when a statement raised, and names what kept changing an unstable key.
- A `def` is no longer refused for a clock its body reads when called, and a clock
  read inside a helper no longer keys the whole module.
- `CACHE-NOT-WORTH-BYTES` fires once per cell naming its statements rather than
  once per statement; several warnings that were simply wrong were removed; and a
  warning or traceback quotes the line you wrote.
- **`%cash_stats` counts cash's own time inside a statement as overhead**, not as
  your compute. A paired run measured 370 s of slowdown where the badge reported
  210 s.

### Changed

- **Cache weight and eviction.** Both tiers evict by value per byte rather than by
  age, a statement's superseded versions are pruned by what they are worth, a value
  built from cached calls is stored once, and a value that cannot earn the disk it
  occupies is not written. A value the RAM cap cannot hold is refused rather than
  stored and immediately dropped.
- **Cash spends less on itself.** A long cheap loop runs as one unit instead of
  snapshotting every iteration; a folder read, a crowded directory, a shared input
  file and a file's fingerprint are each paid for once per cell rather than per
  statement or per file; a frame is shared under copy-on-write instead of
  deep-copied; plain data is keyed and sized without walking it in Python.
- **More calls are cached.** Calls inside a comprehension, inside a loop cached as
  one unit, inside `dict(...)` or a class call, with unpacked arguments, and an
  expensive call handed to another call (`weights(fit(x, y), cap)` caches the `fit`
  too). A call can be keyed on what it receives rather than on where its arguments
  came from, so the same work reached from two places is computed once.
- **A restore that would cost more than recomputing is not served as a hit.**

### Upgrading

Your cache is not invalidated: the on-disk format is unchanged. But several fixes
change *which* key a statement gets — mutation classification, what counts as a
file the statement read, and calls that are now cached separately — so the first
run after upgrading recomputes some statements and writes new entries beside the
old ones. `%cash_repair --full` (notebook) or `cash clear` settles it in one step
if you would rather not carry both.

Cash also keeps two small files of its own beside the cache
(`_compute_baselines.json`, `_loop_split.json`). They are bookkeeping, not entries,
and `cash clear` removes them with everything else.

## [0.10.0] - 2026-09-09

Two independent testing rounds, five projects each, on the decorator and script
path. They found eight ways to get a wrong answer and two ways to lose a job;
this release is those fixes. It is a minor bump because one of them moves where
your cache lives.

### Breaking

- **The cache belongs to the project, not to the directory you launched from.**
  The default cache directory and `[tool.cash]` discovery used to resolve from
  `os.getcwd()`, so running the same script from a cron entry, a CI step or a
  colleague's terminal silently discarded the whole cache and built a second one
  beside you — measured at `6 of 6 restored` dropping to `0 of 6`, a fresh 232 MB
  `.cash`, no warning, indistinguishable from a cold run. One tester's job wrote
  a `.cash` at the drive root. The documented escape hatch was discovered by
  walking up from the cwd too, so it was ignored in exactly the case that needed
  it.

  Both now resolve from a **project anchor**: the first directory above the
  *running script* holding a `pyproject.toml`, `setup.py`, `setup.cfg` or `.git`.
  Relative paths follow whoever wrote them — the default against that anchor, a
  config file's against its own directory, an env var or `cache_dir=` argument
  against your cwd, because you typed those where you are standing. Interactive
  sessions (a notebook, a REPL, `python -c`) have no running script and keep
  using the cwd exactly as before.

  **What you will see:** if your cache sat wherever you happened to run from,
  one cold run and a [`CACHE-DIR-MOVED`](https://cash-lib.readthedocs.io/en/stable/warnings/#cache-dir-moved)
  notice naming both directories. Silently relocating a 500 MB cache would be
  the same bad surprise pointing the other way. `cash.configure(cache_dir=...)`
  or `CASH_CACHE_DIR` pins it wherever you want.

### Fixed

**Wrong answers.** Each of these could serve you a cached result that no longer
matched your code or your data.

- **A global that only a cached callee reads now reaches the caller's key.** A
  helper that read `THRESHOLD` and was itself decorated contributed nothing to
  its caller's key, so editing the constant left every aggregate above it stale.
- **Reading a global through a method call, or a dunder, counts as reading it.**
  `CONFIG.get("mode")` and `len(TABLE)` were invisible to the key while
  `CONFIG["mode"]` was not.
- **A captured callable decides the result, so it reaches the key.** The
  strategy-factory shape — `make_scorer(weight_fn)` returning a closure — keyed
  identically for every `weight_fn`, so the second strategy got the first one's
  numbers.
- **An ndarray's memory layout is part of what it is.** A C-ordered and an
  F-ordered array holding equal values shared one entry, and a layout-sensitive
  callee was served the other one's result: `np.ravel(x, order="A")` returned
  `[0, 1, 2, …]` for an input whose true answer is `[0, 4, 8, 1, …]`.
- **A file that was looked for and was not there is a dependency.**
  `if os.path.exists("local_overrides.toml")` recorded nothing at all, so an
  entry computed *without* the optional file looked valid everywhere — one
  directory's answer served in another, silently.
- **A same-size edit to a large file with the mtime restored no longer slips
  through.** `cp -p`, `rsync -a`, `tar -x` and any script that saves and puts
  back timestamps defeated the sampled-hash backstop. Sampled files now carry
  the inode change time as well (POSIX), and the full-hash threshold moved to
  64 MiB so an ordinary CSV or parquet is hashed whole on every platform.
- **Concurrent first calls agree on one cache key.** Two threads entering a
  function whose source had never been analyzed could compute different keys for
  the same call, so one of them wrote an entry nobody would ever read again.
- **The locked re-read validates what the unlocked one validates.** With
  `use_locking=True`, the double-checked read inside the lock was a hand-written
  copy that had fallen behind — it served entries the ordinary path rejects,
  including a chunked iterator with a missing chunk (10 items unlocked, 3
  locked). Both paths now share one helper.
- **A variable mutated inside a `for` loop takes its identity from its inputs.**
  An upstream edit did not propagate through a cell that assigns columns in a
  loop; the stale value survived a kernel restart and an in-order run-all, and
  `%cash_verify` called every entry healthy.
- **Reconstruction runs the statements it depends on.** A forced fill executed
  `ax.plot(sub[...])` without the `sub = mm[...]` four statements above it, and
  the resulting `NameError` surfaced on a completely unrelated cell — five cells
  blocked at once in the reporter's session.
- **A figure that was rebuilt but never drawn is never flushed.** Asking for an
  unrelated downstream cell after a kernel restart could overwrite a saved chart
  with a blank one — no error, no badge, and a restart cannot undo it.
- **Pool workers share their parent's cache.** A spawned worker resolved its own
  anchor from `sys.argv[0]` and cached into the wrong place.

**Jobs that did not finish, or cached nothing.**

- **An unwritable cache directory no longer outlives the job.** Pointed at a
  directory it could read but not write, a process ran to completion, printed
  its result, and then never exited — measured at 28.7 s writable against still
  running at 150 s, 200 s, and eleven minutes. Background writes now run on
  daemon threads with a bounded shutdown wait (`shutdown_write_timeout`,
  60 s), and expiry says so via `CACHE-WRITE-ABANDONED`.
- **A cache directory that cannot be used at all no longer kills the job.** It
  used to raise straight out of `get()`, so "cash cannot cache" became "your job
  does not run", exit 1, before the caller's own work had started. The file tier
  now turns itself off, loudly (`CACHE-DIR-UNWRITABLE`), and the process carries
  on computing.
- **A cache cap bounds the cache; it does not switch it off.** A single value
  larger than *half* the disk cap was refused, so `CASH_MAX_CACHE_SIZE=500MB` on
  a 263 MB working set cached **nothing at all** — three of four stages
  recomputed nightly and the directory held 29 KB. The threshold is the whole
  cap now, measured on serialized bytes, so what fits is stored and LRU eviction
  does the rest.
- **The RAM tier is sized from the memory this process may actually use** — it
  now reads a cgroup limit, so a container no longer scales its cache to the
  host's RAM. `cash info` prints the two caps it resolved to; a tester spent a
  round reading a growing RSS as a leak when it was a 4 GiB cap doing its job.
- **The badge's progress counter no longer names a statement that has not
  started** — `(2/6)` while nothing was running, and `(7/6)` on the last one.

### Added

- **`KEY-AMBIENT-READ`** — a warning of its own for a cached body that reads the
  clock, the environment, the working directory or a fresh `uuid4()`. These are
  hidden *inputs*, not side effects: the first call's value is what every later
  call gets back, in this process and in every process after it.
- **`CACHE-FRESHNESS-COST`** — says so when proving a cached result fresh cost a
  serious share of what it saved, with both numbers. Local file checking had no
  such guard; only remote sources did.
- **An upstream `NameError` names whose fault it is** — whether the statement
  that defines the name has genuinely never run, or cash failed to schedule a
  producer that the notebook does contain.
- **`cash.run_summary()` says which cache directory it used**, which is the
  first question to ask when a warm run comes back cold.
- **`file_hash_full_max_bytes`** and **`shutdown_write_timeout`** config fields.

### Changed

- **File digests are memoized per process.** Freshness is checked once per
  cached call and file dependencies propagate, so a pipeline over fifty inputs
  re-read and re-hashed all fifty on every hit. Now the second and later checks
  of an untouched file cost a `stat`: 50 × 2 MiB went from 79.6 ms to 2.7 ms per
  hit, 50 × 8 MiB from 313 ms to 2.8 ms.
- **`file_hash_full_max_bytes` defaults to 64 MiB**, up from 8 MiB. That is what
  the memo bought: a full hash is now paid once per file per process rather than
  on every check, so covering the ordinary data file is affordable. A first
  check of a 64 MiB input costs about 46 ms; `CACHE-FRESHNESS-COST` names the
  number if a workload makes that a bad trade, and lowering the setting restores
  sampling.
- **`cash info` prints the caps that actually resolved** — `auto -- disk 16.4
  GiB, RAM 4.0 GiB` — instead of `auto (scaled per tier)`.

## [0.9.3] - 2026-09-08

Follow-up to 0.9.2: one badge said the wrong thing, and the live demo is
rebuilt.

### Fixed

- **A cell of imports or magics reported `BYPASSED · cash stepped aside`.**
  0.9.2 decided a cell had been bypassed by looking for a badge with no rows in
  it — and a cell that only imports things, or only runs a magic, has no rows
  either. `%cash_on` is not valid Python, so Cash cannot parse that cell, hands
  it to IPython and moves on. That is the ordinary path for a magic, not a
  failure, but the badge announced it as one — including on the opening cell of
  the demo tour.

  `BYPASSED` is now carried explicitly and set at one place: the internal-error
  path that also emits `NOTEBOOK-BAILOUT`. Nothing else can reach it. The three
  states a badge with no rows can be in are now distinct — a cell with nothing
  to cache reads `EXECUTED`, a cell still running reads `PROCESSING`, and only a
  genuine bail-out reads `BYPASSED`.

### Changed

- **The live feature tour is rebuilt around a Monte Carlo stress test.** One
  shared simulation of the market, three volatility scenarios stressed against
  it, one summary — the same dependency lattice as before, in **two functions
  and ten lines** instead of four hand-written model fits and around two
  hundred.

  The work is real: two million price paths walked over 252 trading days, half
  a billion random draws. Deliberately not a `sleep` standing in for slow work,
  because a demo built on sleeps argues against itself.

  It is also sized from measurements taken **on Binder** rather than on a
  developer's machine, which the previous version was not — a cold run there
  took five to ten minutes. It now takes about twenty seconds, and a warm one
  about two.

## [0.9.2] - 2026-09-08

The badge stops telling you a cell has finished while it is still running.

### Fixed

- **Every cell opened by claiming it had already executed.** Cash draws a
  placeholder badge before a cell's first statement runs. That placeholder had
  no rows and no step information, and the renderer had no state meaning
  "running" — so it fell through to the default and drew `EXECUTED · 0.00s`.

  On a fast cell you never saw it: the real badge replaced it within
  milliseconds. On a slow one it stayed for the whole run, so the cell looked
  finished — in no time, with nothing in it — while it was still working. The
  slower and more interesting the cell, the more wrong the badge. It now reads
  `PROCESSING`.

  Measured at one such render per cell — twelve in a twelve-cell notebook —
  so this affected every cell of every run, not an edge case.

- **A badge with nothing in it no longer reports success.** The same default
  was reached from the paths where cash hits an internal problem, steps aside
  and hands your cell to IPython to run uncached. Those now read `BYPASSED ·
  cash stepped aside`, which is what actually happened. Deliberately not
  `NOT CACHED`: that already means a statement that ran under cash and was not
  stored, which is a different thing.

### Added

- **`NOTEBOOK-BAILOUT`** — when cash steps aside because of an internal error,
  it now says so, and names the exception, next to the cell that lost its
  caching. Previously that went to a logger nothing in a notebook surfaces, so
  the only trace was a cell that appeared to do nothing. Your code still runs
  and the result is still correct; what you lose is the caching, and now you
  are told. Documented at
  [warnings#notebook-bailout](https://cash-lib.readthedocs.io/en/stable/warnings/#notebook-bailout).

### Changed

- `BadgeStatus` gains a `RUNNING` member. Cash used to signal a running cell by
  passing `"RUNNING"` and having the renderer infer it from the presence of
  step information, which is exactly the inference that failed above.

## [0.9.1] - 2026-09-07

One crash worth upgrading for, one loop that was doing its work twice, and a
rebuilt live demo.

### Fixed

- **A cell could die under Jupyter with `AttributeError:
  'CapturingDisplayPublisher' object has no attribute 'set_parent'`.** To
  record what a cell displays, Cash swaps IPython's display publisher for a
  capturing one — and that swap is process-wide. Meanwhile the kernel drives
  the publisher itself on the shell channel, so any message that arrived while
  a capture was open reached an object missing the methods the kernel expects.
  The capturing publisher now forwards the real one's kernel-facing API, so the
  swap is invisible to the kernel.

  Worth upgrading for even if you have not seen the traceback. The failure
  lands as an error on a cell whose work then silently did not happen, and the
  capture is open while Cash runs your statements — so the window is ordinary
  cell execution, not some rare path.

### Performance

- **A loop target is no longer hashed twice per iteration.** Cash pays a full
  hash of each value a loop binds, deliberately — a sampled hash once collided
  two iterations onto one cache entry and returned a wrong result. It was
  paying that cost twice: once when binding the value, then again when building
  the iteration's cache context, which did not know the digest had just been
  computed. Sharing it cut a cached re-run of a loop over five 200k-row frames
  from 0.50s to 0.294s, and the hashing itself from 328ms to 161ms.

  The hash is shared, never weakened — loops over large values keep the same
  cache keys and the same collision guarantee.

### Changed

- **The live feature tour is rebuilt around work that caching can actually
  save.** The old tour held 1.38s of cacheable compute, 82% of it in a single
  cell, while the rest sat in the 50–130ms band where Cash's own bookkeeping
  costs about what the work does; its million-row frame took 130ms to build and
  99ms to hash, so caching it roughly broke even.

  It now runs a model bake-off — a shared k-means feature stage feeding a
  lasso, gradient-boosted stumps and a small neural net into a leaderboard —
  around 9s of genuine compute over intermediates that hash in single-digit
  milliseconds. Because those pieces form a lattice rather than a fan-out, the
  tour can finally show the thing it is about: change one model's
  hyperparameter and only that model recomputes, while its siblings and the
  shared stage stay cached. Edit the helper all three share and all three
  invalidate, while the stage that does not call it does not.

## [0.9.0] - 2026-09-07

A release about Cash explaining itself. Every warning now carries an
identifier, one line of what to do, and a link to a page that says whether you
should care — prompted by a reader who hit these warnings in a script, read
them, and still could not tell what they meant.

### Upgrading

- **Every warning's message text changed shape.** If you match on warning text
  — `pytest.warns(match=...)`, a log filter, a regex — that match will break.
  Branch on the code instead, which is stable where prose is not:

  <!-- test:skip reason="illustrative fragment: w is a caught warning the entry does not create" -->
  ```python
  if getattr(w.message, "code", None) == "CACHE-THRASH":
      ...
  ```

  Three notebook-side warnings are raised in a way that cannot carry the
  attribute; the rendered text always starts `[CODE] `, and
  [the warnings page](https://cash-lib.readthedocs.io/en/stable/warnings/)
  gives a helper that reads either.
- **Annotating a cell-defined function now invalidates it.** A
  `# @cash:assume-safe` inside a function defined in a `%cash_on` cell used to
  do nothing at all; now it is honoured, which also makes it part of that
  function's identity. Adding one is a one-time miss, and that is correct — the
  function is treated differently than it was before.

### Added

- **Every warning carries a diagnostic code, a fix, and a link.** 34 codes
  across the `CashWarning` hierarchy, each with a section in the new
  [warnings reference](https://cash-lib.readthedocs.io/en/stable/warnings/)
  covering what happened, why it matters, what to do — and, deliberately,
  **when it is safe to ignore**, because several of these are informational and
  nothing used to say so. A warning now looks like:

  ```
  foo.py:12: CashCacheIneffectiveWarning: [CACHE-THRASH] the cache is full at
  its 500 MB cap and is evicting entries within a couple of writes of storing
  them.
    Fix: raise max_cache_size, or cache fewer values.
    https://cash-lib.readthedocs.io/en/stable/warnings/#cache-thrash
  ```

- **Warnings carry `.code` for handlers**, so you can branch on the diagnostic
  rather than on wording that is free to change.
- **The badge shows a multi-line statement the way you wrote it**, across its
  own lines, instead of collapsing it to a first-line summary.

### Fixed

- **`# @cash:assume-safe` now works inside a notebook cell.** A per-line purity
  waiver on a function defined in a `%cash_on` cell was silently ignored: cells
  are compiled from `ast.unparse` output, which strips comments, so the analyzer
  read the function back and found no waiver. Annotated functions are now
  compiled from your own source. This was visible in the demo tour, which
  displayed the annotation next to the warning it was supposed to silence.
- **`use_locking=True` could serve a truncated iterator.** A chunked entry whose
  manifest outlived one of its chunks was served short — three of ten items, or
  none at all when the missing chunk was the first — with no recompute, no error
  and no warning. The integrity check that turns a broken manifest into a miss
  now runs on the locked read path too, not only the default one.
- **A trailing `;` stopped suppressing output when the line held a non-ASCII
  character.** `df[df.city == "Zürich"];` echoed its repr: the column offset is
  a UTF-8 byte offset and was being read as a character index.
- **Warnings point at your line, not Cash's.** Several pointed inside
  `core.py`, which is the one question a code and a link cannot answer for you.
  The blamed frame is now resolved when the warning fires rather than from
  hand-tuned constants, so it is right at any call depth.
- **`cache_if` guidance was backwards.** The warning told you to *lower* the
  chunk thresholds; `cache_if` only runs when the whole result fits in one
  chunk, so lowering them guarantees the bypass it warns about.

### Changed

- **Warning message text is rewritten across the board** (see **Upgrading**).
  No warning was removed and none fires under new conditions; the wording and
  the blamed source line changed. Warning text was never a documented
  interface — branch on `.code` if you need something stable.
- **Executed notebooks are substantially smaller and the badge is quieter on
  the wire.** The stylesheet is minified once at import, and progress badges
  render on the trailing edge — armed when a statement starts and published only
  if it is still running — so fast statements publish nothing. On the demo tour
  that cut the saved notebook by close to 40% and roughly halved the number of
  badge updates sent to the frontend. A slow statement still shows *itself*
  rather than the previous one.
- **The Binder tour is sized for Binder's CPU**, and its expensive step is now a
  real computation rather than a large pile of data. Its two slowest steps each
  came down by roughly an order of magnitude.

## [0.8.0] - 2026-09-03

A correctness release, focused on the two things a cache can get quietly wrong:
a side effect that runs once and is then skipped forever, and a result stored
under a key that does not describe it.

### Upgrading

Two behaviour changes are worth knowing before you upgrade; neither needs an
edit unless it applies to you.

- **A generator you abandon now caches nothing.** Cash used to drain a cached
  generator eagerly, so taking two items still left a complete entry behind. It
  now streams, which means only what you consumed was ever produced — there is
  no complete result to store, and the next call recomputes. If you relied on a
  partial read populating the cache, consume the generator fully (`list(...)`).
- **New purity warnings fire on code that was silent before**, and
  `getattr(mod, "exec")(...)` now raises at decoration time, like the bare
  builtin it reaches. If a warning names something you have already audited, put
  `# @cash:assume-safe` on that line rather than reaching for `assume_safe=True`.

Functions that take a dataclass or a pydantic model as an argument recompute
once, because their keys now include the field definitions.

### Added

- **Side effects inside libraries are caught.** A missed effect is a hidden
  correctness bug: it runs on the first call and is silently skipped on every
  hit. Measured against 24 planted effects, the analyzer caught 21 — and every
  miss was inside an installed library, all sharing one rule: a library effect
  was noticed only when its return value was *discarded*. So `requests.get(...)`
  warned but `SESSION.get(...)` did not, and a `DB.execute("INSERT ...")` whose
  cursor was used did not. That heuristic inverts exactly where the stakes are
  highest, because an HTTP POST or a DB write is normally called *for* its
  response. Two answers now: the write-method list gains the verbs that mean a
  client changed something remote (`post`/`put`/`patch`, `execute`/`commit`/
  `rollback`, `upload`/`put_object`/`publish`), and cash *watches* the body of a
  miss for effects it can see wherever the code lives — a file opened for
  writing, an outbound connection, a spawned process. The observer never blocks
  and never raises, even under `strict=True`: by the time an effect is observed
  the function has already run, so raising would discard a correct result to
  report something it could not have prevented.

- **A call that mutates the caller's own state is reported.** Side effects are
  not only I/O. Sorting a list in place, setting a key on a config dict, bumping
  a field — each happens once and then stops, and the caller's state quietly
  goes out of date. Cash already hashes arguments to build the key, so a miss
  re-runs that hash afterwards and compares; a difference means the call changed
  what it was handed. Caught 12 of 13 planted mutations. A re-hash over 50 ms
  retires the check for that function, so the repeat cost is dropped rather than
  taxing every later miss.

- **Code chosen at runtime is noticed.** A dispatch table caches correctly, but
  it cannot notice that you edited the function it dispatches to. Across 20
  indirect-invocation shapes cash saw 8; it now sees 18. `TABLE[k]()`,
  `globals()[n]()` and a parameter handed to `map()` warn and name
  `depends_on=[...]`; `getattr(mod, "exec")(...)` raises. Two vectors stay
  silent on purpose — `obj.fn()`, which is statically indistinguishable from a
  method call, and `operator.methodcaller(name)(obj)`.

- **`# @cash:assume-safe` waives one statement instead of the whole function.**
  `assume_safe=True` is a permanent, unscoped waiver: audit the call you meant
  to audit, add an unrelated `session.post(...)` three months later, and nothing
  says a word. A waiver written next to the statement re-arms itself for free —
  new code arrives unannotated, so it is reported, and the scope of the
  exemption is visible in the diff that grants it. Put it on the line or
  directly above it; on a `def` line it waives the function-scoped findings.
  Honoured under `strict=True`, which makes strict mode usable. `assume_safe=True`
  is unchanged and remains the right tool when the audit really is
  function-wide.

- **A thrashing cache is told when large results are the reason.** The
  ineffective-cache warning said the cache was churning but not why, so the only
  remedy it offered was raising the cap — the expensive answer when five 3 GB
  frames are what fills it. The message now names the size that is filling the
  cache, how many of those fit, and points at caching the summary instead. Sized
  by the byte-weighted median rather than the mean, because real caches are
  mixtures: on 3000 × 2 KB plus one 64 MB entry the mean reports 3437 entries
  fitting, where that single entry is 91% of the cache.

### Fixed

- **A cached generator returned nothing on a cross-process hit.** Chunks were
  written with an execution time of zero, which put them under the
  smart-persistence floor, so they stayed in RAM while the manifest went to
  disk. A fresh process found the manifest, looked for the first chunk, did not
  find it — and a missing chunk terminates iteration. The call reported a hit
  and returned an empty iterator. Caching a generator had never worked across
  processes, which is the only reason to cache one. Chunks now carry the
  producer's time, and a manifest whose chunks are missing is a miss.

- **A dataclass used as an output spec served a stale answer.** Field
  descriptions written with `field(metadata=...)` are prompt text, not
  documentation, and `Field.metadata` is a `mappingproxy` that cannot be
  pickled — so the fold caught the error, folded nothing, and dropped it in
  silence. Rewording a description left the key unchanged. Adding a field and
  editing the docstring both invalidated already, which is why it survived: the
  two edits anyone tries first both behaved.

- **A pydantic spec was tracked but never cached.** Pydantic v2 compiles
  `__pydantic_core_schema__`, `__pydantic_serializer__` and
  `__pydantic_validator__` onto every model, and their digest differs in every
  process — so the key moved on every run. Correct answers, no reuse, and
  invisible to every correctness test there is, because recomputing is always
  right. The compiled trio is skipped and `model_fields` folded in its place, so
  a reworded description recomputes and a comment does not.

- **A global read by a helper was reported as mutated — and dropped from the
  key.** The pre-call hash came from the helper's module while the after-check
  re-read it from the decorated function's globals, found nothing, and saw a
  difference. On a three-document pipeline the warm run made 3 provider calls
  where it should have made 0.

- **Three dynamic-dispatch warnings named hazards that do not exist.** A
  module-level `TABLE[k]()` and a callable passed as an argument are both hashed
  and both invalidate correctly. A warning that fires on the most common
  spelling of a correct pattern is how a project teaches people to filter its
  warnings out.

- **An adaptive cache cap no longer shrinks because the cache filled.** The cap
  was sized from free space, which excludes what the cache has already written,
  so it fell as the cache grew and the cache was then over a cap its own
  contents had caused. Simulated on a 48 GiB volume it settles into a two-cycle
  that discards ~13% of the cache every other session. The cache's own size is
  added back in, so growing by N bytes no longer moves the number.

- **Eviction stops shredding small entries to free space they cannot free.**
  Oldest-first is size-blind: with a few huge entries and many tiny ones it
  chews through the tiny ones freeing almost nothing per delete and reaches the
  huge one anyway. On 3000 × 2 KB plus one 64 MB entry needing 22.7 MB freed, it
  deleted 3001 entries to free 69.9 MB — the entire cache — where deleting one
  frees 64 MB. Entries below 0.1% of the cap are now ranked separately, so
  crumbs still go first when they can close the gap and go last only when they
  provably cannot.

- **A read protects an entry that is already queued for eviction.** The ranking
  is a snapshot drained over many passes, so an entry could be read after it was
  queued and still be evicted. On the shape that prompted it — a baseline whose
  cached upstream is reused by later variants — the baseline's survival goes
  from 1-in-3 to 3-in-3.

- **A burst of small writes is ranked by write order, not by directory order.**
  Eviction ranks on modification time, and a filesystem records those in steps:
  on ext4 the step is about 0.57 ms while a burst of small writes is faster, so
  a whole cache-full of entries can share one timestamp — measured at 12 entries
  across 2 distinct values, against 5 on NTFS. Sorting is stable, so everything
  in that tie kept directory order, which is a hash of the filename. The cache
  evicted entries it had just written while older ones sat untouched, and then
  reported itself as thrashing because it had. Ties now break on the write
  counter, so eviction is exact rather than approximate everywhere: on Linux the
  evicted entry went from as recent as 1 write old to never newer than 12.

- **`from cash.backends import SQLiteBackend` raised `ImportError`** while every
  sibling backend resolved.

### Changed

- **Caching a generator no longer changes when the caller sees items.** The miss
  path used to drain the whole thing before returning anything, so a streamed
  response arrived in one lump after the full latency — measured on a token
  stream, 494 ms to the first item uncached against 2444 ms cached. Items now
  pass through as they are produced and the entry is committed at exhaustion:
  515 ms to the first item, and the hit still replays instantly. See
  **Upgrading** for what this costs.

- **Purity warnings recommend the annotation, not just the flag.** The
  line-anchored messages name `# @cash:assume-safe` first and `assume_safe=True`
  as the whole-function fallback, with the trade stated. The observed-effect
  warning is left alone: it reports what the first call did from inside library
  code, so it carries no line number, and pointing it at a per-line annotation
  would be advice the reader cannot follow.

- **Docs.** Three full passes over all 58 published pages, reading each against
  the code that decides it. Twenty-one places where a reader following the docs
  would have been wrong — among them `# @cash:persist` recommended to script
  users, where it is a notebook-only directive that leaves zero entries on disk;
  a text-badge transcript that `cash.help()` returns verbatim, showing a shape
  the renderer cannot emit; and `cash clear`'s exit code. Every figure the docs
  quote about the repository is now derived from source by
  `scripts/doc_numbers.py` and gated, after all of them drifted at once.


## [0.7.0] - 2026-08-30

A storage-layer release. Cache entries changed shape on disk, which makes this
a one-time rebuild for every existing cache — see **Changed** below. Nothing to
run by hand: cash notices the old format on first use, clears it, and the work
recomputes once.

### Upgrading

Every cached result recomputes once on the first run after upgrading. Four
things contribute, and all of them are automatic:

- the on-disk entry format is now one file per entry;
- `SQLiteBackend`'s table declares its columns in a different order;
- a function defined in a script run directly is keyed by the script rather
  than by `__main__`, so it now matches the same function imported;
- numeric literals key by value rather than by spelling.

`register_hasher` for a type cash already hashes now raises unless you pass
`override=True`. It previously accepted the call and silently ignored it, so
code that looked like it had installed a custom hasher had not.

### Added

- **[Caching over a grid](docs/tutorials/feature-guides/caching-over-a-grid.md)**
  — a guide for the question "I only made the axis finer, why did it recompute
  everything?". Covers the case that is already free (going back to a
  resolution you ran before), why refining is different, the precondition for
  reuse, two recipes, and when not to bother. Every number in it comes from
  `benchmarks/bench_grid_refinement.py`.

- **`cash inspect` groups by function, and `cash clear --function` drops one.**
  Someone short on disk had two options: keep the whole cache or delete it.
  `inspect` now leads with a per-function table sorted by size — the question
  that sends people there is "what is filling my disk?" — and `--function NAME`
  drills in or clears. An unambiguous trailing segment is enough, so
  `--function work` finds `__main__.work`, and `notebook` selects the
  notebook statements without their brackets. Each entry row shows what it
  *saves* and how often it has been used, not just its size — a 900-byte entry
  worth 41 seconds and a 5 MB one worth 0.2 seconds are the same size problem
  and opposite keep decisions. `cash clear --entry ID` drops a single entry by
  any unambiguous id prefix. `cash clear` with no arguments prints its help
  instead of naming two of its three options in one line.

- **`summary=True` prints a per-function hit/miss table when a process exits.**
  A notebook says per statement whether it ran or restored; a script said
  nothing, so a user who wanted to know added `print` calls to each branch. Off
  by default, and reachable four ways from one config field —
  `cash.configure(summary=True)`, `Cash(summary=True)`, `CASH_SUMMARY=1`, or a
  TOML key. The env var is the one that needs no edit to the script you are
  already running.

### Changed

- **One file per cache entry, not two — and every cache rebuilds once.** An
  entry used to be a `.meta` and a `.data`; it is now a single `.entry` file
  carrying a small header, the metadata, and the value. The on-disk format
  version is bumped, so an existing cache directory is cleared automatically on
  first use and the work recomputes once. Nothing to run by hand.

  A write was four filesystem operations per file, done twice — create a temp
  file (121 µs), write it (133 µs), rename it into place (156 µs), stat it
  (21 µs) — of which about two thirds is namespace churn rather than data.
  Halving that, and skipping the temp-and-rename entirely for an entry that
  does not exist yet, takes a 512-byte write at 100k entries from **1.70 ms to
  0.59 ms** and halves the number of files on disk.

  The split existed for a good reason and the new layout keeps it: reading an
  entry's metadata must not cost deserializing its value. A length-prefixed
  header means a metadata read stops after the metadata — measured flat at
  ~0.05 ms from a 512-byte entry to a **one-gigabyte** one, against 10.4 ms to
  read that entry's value. Recording an access rewrites only the header, so a
  session that reads a 100 MB frame no longer rewrites 100 MB on the flush
  timer. Entries cost ~76 bytes more each for the header and the slack that
  makes in-place updates possible.

- **`SQLiteBackend` stores the payload column last.** SQLite lays a row out in
  declaration order and spills the overflow onto a page chain, so reading any
  column walks past everything declared before it — and the payload was
  declared first. Reading a 16 MB entry's metadata went from **35.9 ms to
  0.091 ms**, and is now flat in entry size. A table in the old order is
  rebuilt on open, which discards that cache's entries; they recompute on
  demand.

- **`S3Backend` stores one object per entry.** It was two, so every read, write
  and delete cost two requests, and the write had to order them and clean up
  after a partial failure. One object is one request, and it either lands or it
  does not. Reading an entry's metadata is now a ranged GET of the first 8 KB
  rather than a full download: against a 4 MB entry that is **8 KB transferred
  where it was 4,194,457**, and both the request and the bytes are billed.

  Objects written by an earlier version use the old names and are ignored
  rather than migrated — nothing runs against your bucket. They keep occupying
  storage until `clear()` or a lifecycle rule removes them.

### Performance

- **Opening a cache no longer scales with its size.** The byte total for the
  eviction cap was computed by walking the whole directory on the first cache
  operation of every process — 308 ms at 100k entries, ~3.1 µs each. Only
  eviction ever reads that total, so a run that just reads — a kernel restart
  replaying from cache — was paying for a number it never looked at. It now
  happens on the first *write*, on the background write thread: **308 ms →
  1.1 ms**, flat.

- **Eviction ranks from a directory walk instead of reading every entry.** It
  opened and unpickled each one to sort by last access — 44 µs each, ~0.9 s at
  20k entries and ~4.4 s at 100k, on the write thread with every queued write
  behind it — and then held all that metadata in memory, roughly 1.7 KB an
  entry. A file's modification time already tracks its last access, so the same
  ranking now costs **1.6 µs an entry** (32 ms at 20k, 180 ms at 100k) and
  holds none of it.

- **Reading an entry's metadata no longer reads its value.** Every remote
  backend inherited an implementation that performed a full `get()` and threw
  the value away, so asking when an entry was last accessed downloaded the
  whole object: 4,194,457 bytes on S3 and 4,194,460 on Redis for a 4 MB entry,
  to return about 150 bytes of answer.

- **A finished cache write is no longer remembered forever.** The async write
  registry kept one future per distinct key for the life of the backend, and
  `wait_all` walks that registry — so it was O(every key ever written) rather
  than O(in flight). Failed writes are still retained; they are the only record
  that a write was discarded.

### Fixed

- **Eviction could hang the write thread permanently.** Eviction runs on the
  single background write worker, and evicting a key waits for that key's
  pending write — so a write queued behind the one currently running waited for
  a task that could not start until it returned. The write thread stopped, and
  the flush that makes results durable stopped with it. Reachable without
  contriving anything: an entry's recorded access time is still its previous
  write's, so re-computing a long-idle entry while the cache sits near its cap
  makes it the obvious eviction candidate. Eviction now skips any entry with a
  write in flight, which is also right on the merits — it is the newest thing
  in the cache, not the oldest.

- **A failed write no longer deletes the entry it failed to replace.** Writing
  through a temp file and renaming exists to guarantee that a write which dies
  halfway leaves whatever was cached before exactly where it was. The
  error path then deleted the destination to clear a "partial write" — correct
  when an entry was two files, actively harmful when it is one written through
  a temp: the destination was either the previous entry, untouched, or absent.
  So a full disk or a denied replace destroyed a value that was still good.

- **Eviction discarded more than it needed to.** It worked down a shortfall
  measured in whole entries while crediting each eviction with only the
  payload's size, so every eviction looked smaller than it was and the loop
  kept going. Invisible while the per-entry overhead was small.

- **A polars `LazyFrame` argument no longer collides with a different one.**
  It was identified by `explain()` — the human-readable *query plan* — and two
  frames over different in-memory data print identically
  (`DF ["x"]; PROJECT */1 COLUMNS`), so the second call was served the first's
  cached result. A wrong answer, reachable in three lines. `LazyFrame` is now
  identified by `serialize()`, which carries the plan *and* the data it closes
  over, and is byte-identical across processes so persisted entries still hit.

  **Still open, and documented rather than hidden:** a plan that reads from a
  source (`scan_csv`, `scan_parquet`, …) serializes the *path*, not the file's
  contents, so editing the file in place does not invalidate the entry.

- **Running a script and importing it now share one cache.** A function defined
  in a script run directly was keyed under `__main__`, and the same function
  imported as a module was keyed under its real name — two entries for one
  function, and a guaranteed miss when the same code was reached both ways.

- **Re-spelling a number no longer throws away the cache.** `0.5` and `0.50`,
  `1e3` and `1000`, `1_000_000` and `1000000` are the same value and now share
  a cache key. Base prefixes are handled correctly (`0x1e3` is 483, not 1000).

- **`show_stats()` in a script prints something useful** rather than silently
  doing nothing outside a notebook.

## [0.6.0] - 2026-08-28

The theme is **making cash's own advice work.**

Two of cash's warnings named a remedy that could not be taken. The purity
warning told you to add `assume_safe=True` — and doing so discarded the cache
it was warning about. The effectiveness warning correctly diagnosed a large
argument being hashed on every call, then told you to register a cheaper
hasher, which for a numpy array or a dataframe was silently ignored. Both are
fixed here, and the second grew a way to actually do the thing it asks for.

### Breaking

- **`register_hasher` now raises `ValueError`** when the type is one cash
  content-hashes itself — numpy arrays, pandas / polars / PyArrow / modin
  frames, dask collections — and `override=True` was not passed. Cash's own
  hashers run first, so such a registration could never have run: it is dead
  code the caller meant to be live, and it is refused at setup rather than
  stored inert.

  Code registering a redundant hasher for one of those types will raise on
  upgrade. That registration was already doing nothing, so the fix is to
  delete the line — cash hashes those types correctly on its own — or to pass
  `override=True`. Your own subclass of one of them is unaffected: the check
  matches on module prefix, so a subclass defined in your code is dispatched
  by its own module and has always been yours to hash.

### Added

- **`register_hasher(..., override=True)`** — take precedence over cash's own
  content hashers, and over a notebook value's lineage hash.

  A 10000×10000 float64 array is 800 MB; content-hashing it costs ~306 ms per
  call, so a loop of 100 cached calls spent 31 seconds building keys and
  nothing else. With an overriding hasher the same loop is instant.

  What it costs is not hidden: an overriding hasher *becomes* the identity of
  the value, so any two values it hashes alike share one cache entry and the
  second call gets the first one's result. That is a wrong answer, not a slow
  one — right when you hold a version or content id the value does not carry,
  wrong for `lambda a: a[0, 0]`.

  It is allowed because the workaround cash used to recommend — wrap the array
  in a thin type and hash a version field on the wrapper — carries exactly the
  same risk, since the wrapper's hasher reads no more of the array than yours
  does, while costing a signature change through the whole call chain.

### Fixed

- **Changing a `@cash.cache` argument no longer throws away the cache.**
  `inspect.getsource` returns the `@...` lines along with the function, so
  every argument passed to the decorator landed in the function's source
  digest and therefore in its cache key. Adding `assume_safe=True` — which
  `CashImpurityWarning` tells you to add — recomputed everything, on exactly
  the expensive functions that warning fires for. So did `ttl=`, and so did an
  empty `()`, which is how you can tell this was never a decision about
  semantics.

  Cash's own `@....cache` decorator is now excluded from the identity digest.
  The arguments that must still invalidate are unaffected, because none of
  them travelled through the decorator's text: `depends_on`,
  `dynamic_depends_on` and `file_depends_on` reach the key as dependency-graph
  edges, and `ttl` is enforced against entry metadata at read time. Decorators
  that are not cash's own are still hashed in full — `@inject(db=prod)` can
  change what a function returns, and nothing in the source says it does not.

  Existing decorator-path entries are keyed on the old digest and recompute
  once.

### Changed

- `CashCacheIneffectiveWarning` for a net-loss function now names
  `override=True`, so the remedy it points at is reachable for the numpy and
  pandas arguments it fires on most.

## [0.5.0] - 2026-08-27

The theme is **cash telling you when it is not helping you.**

A cache that quietly does nothing looks exactly like a cache that is working.
Everything here is about closing that gap: saying when a write was thrown away,
naming the input that forced a re-run, and — new in this release — speaking up
when caching a function costs more than the function does.

Plus two fixes for a failed cache write, which until now could raise an
exception out of your own call.

### Added

- **`@cash.cache` warns when caching costs more than it saves.** Pass a large
  DataFrame to a function that does something cheap with it and cash hashes the
  whole frame to build a cache key — measured at 390ms of hashing to avoid 11ms
  of work, on *every* call, because the fast path that skips re-hashing an
  unchanged object only applies to notebook-tracked values. That is a 34x
  slowdown, and cash used to say nothing about it.

  It now warns once, naming the numbers and a fix that keeps the caching:
  register a cheaper hasher for the type. It does **not** stop caching — you
  asked for the decorator, and deciding otherwise is not its job.

  Deliberately hard to trigger: it needs seconds of genuinely accumulated loss,
  and the overhead has to exceed even the *largest* compute it has seen. A
  function that is usually fast but occasionally slow is worth caching, and
  saying otherwise would be a confident wrong answer. Filter it with
  `warnings.filterwarnings("ignore", category=cash.CashCacheIneffectiveWarning)`.

- **The badge names the input that forced a statement to re-run.** Previously a
  re-executed row said only that it ran. It now says *why* — `input changed: df`
  — for the most common reason a notebook statement recomputes.

- **The badge says when a cache write was discarded.** A discarded write is not
  a miss; it is a hit that never got the chance to exist, so no other counter
  can show it. Now a row says so directly.

### Fixed

- **A failed cache write no longer raises into your code.** If the backend could
  not store a result — antivirus holding a file, a full disk, a disconnected
  network drive — the exception surfaced out of *your* function call. On a cold
  cache it did so after the value had already been computed, so the work was
  done and thrown away. Compute now succeeds and the failure is reported as a
  warning.

- **A failed write no longer breaks that cache key for the rest of the session.**
  The failure was re-raised on every later lookup of the same key, *including
  after the underlying condition had cleared* — one transient lock made a cached
  function uncallable until you restarted the kernel.

- **The badge no longer blames an input the statement writes itself.** In a chain
  like `df = df[mask]`, every statement writing `df` shared one record, so each
  compared itself against a different statement's — and reported `input changed:
  df` on a run where nothing had changed. Measured on a real notebook: 17 of 21
  such attributions were wrong. A wrong reason is worse than none, so cash now
  stays quiet rather than guessing.

- **Loop-body rows keep their reason.** Every attribution the runtime worked out
  for a statement inside a loop was dropped before it reached the badge — and
  loops are where the expensive work lives.

- **Windows: a loop-split verdict is no longer silently lost.** Same
  cause as the 0.4.1 write fix — Windows refuses to replace a file while any
  handle has it open. The verdict survived in memory but vanished from disk, so
  the next session did not split the loop, keyed it differently, and recomputed
  it.

## [0.4.1] - 2026-08-24

A fix release, and the fix is one you could not have seen: **on Windows, cash
was quietly not caching some of what it computed.**

### Fixed

- **Windows: cache writes were silently discarded.** Replacing a file is atomic
  on both platforms, but Windows refuses the operation while any handle still
  has the destination open, where POSIX simply swaps the directory entry and
  lets the reader finish. Cash expects concurrent readers by design and writes
  on a background thread, so that collision was routine rather than
  exceptional. Every occurrence threw the entry away and recomputed the work on
  the next run.

  Nothing raised, no test went red, and the only report was a log line written
  at kernel shutdown -- so the symptom was never an error message. It was
  "cash doesn't seem to save me much." Writes now wait out a briefly-locked
  destination instead of giving up on it.

  If you are on Windows, this is the release to upgrade to. Nothing on Linux or
  macOS was affected.

### Added

- **`%cash_stats` reports discarded cache writes.** If a write ever failed, the
  summary now names the count and the cause instead of leaving you to infer it
  from savings that never arrive. A discarded write is not a cache miss -- it is
  a hit that never got the chance to exist, so none of the other counters can
  show it. `%cash_stats json` carries the same figure as `discarded_writes`.

  A `reset` deliberately does not clear them: a counter is something you may
  choose to forget, an unresolved fault is not, and those entries are still
  missing from disk afterwards.

## [0.4.0] - 2026-08-21

The theme is **cash noticing more of what you changed**. Editing a class you
pass as an argument, a class your class builds, or a constant a helper reads
all invalidate now — each of them previously returned the old answer. Editing a
*comment* no longer invalidates anything.

Also: unsaved notebook cells are visible in JupyterLab and VS Code, and cash is
much cheaper to import and to call.

### Added

- **JupyterLab: unsaved cell edits are seen without saving.** A prebuilt
  labextension pushes the editor's current cell text to the kernel, so upstream
  tracking compares what you are *looking at* rather than what was last written
  to disk. It ships inside the wheel — no separate `pip install`, no
  `jupyter labextension install`. Disable it from the Extension Manager if you
  would rather it did not run.

- **VS Code: the same, read from the hot-exit backup.** Where JupyterLab pushes,
  VS Code is polled: cash locates the notebook's hot-exit backup and reads the
  unsaved cells out of it, and only trusts a backup that pairs with the file on
  disk.

- **Cash tells you when it cannot trust the saved notebook.** If the file on disk
  is provably older than what the kernel ran, the badge says so instead of
  quietly tracking stale text — and says once per session when freshness cannot
  be verified at all (no live server, papermill, nbconvert).

- **`cash.mark_opaque(T)` and `@cash.opaque`** opt a type out of code-identity
  hashing, for when a third-party class churns your keys and you would rather
  pin the dependency than fold its code.

### Changed

- **The badge uses one word per state.** Rows used to say `RESTORED` /
  `COMPUTED` while the header above them said `CACHED` / `EXECUTED` for the same
  state. Everything now reads **`CACHED`**, **`EXECUTED`**, or **`NOT CACHED`**.

  **Migration:** if you assert on badge text in your own tests, update those
  strings. Note that `CACHED` is a substring of `NOT CACHED`, so a bare
  `"CACHED" in output` check will match an uncacheable row.

- **Comments and formatting no longer invalidate a cached result.** Code
  identity is taken over a normalized form of the source, so adding a comment,
  inserting a blank line, or running a formatter keeps your cache. `# @cash:`
  directives and docstrings still count — they change behaviour.

- **Much cheaper to import and to call.** `import cash` went from ~10.3s to
  ~0.3s: IPython, redis, asyncio and pandas were all imported eagerly to answer
  questions that `sys.modules` answers for free. The first cached call in a
  process went from ~730ms to ~6ms, and the first `%cash_on` cell from ~700ms to
  ~170ms, for the same reason. A cache hit on a function with helpers is ~3x
  cheaper (helper source was re-read and re-tokenized on every call), and
  restoring a large list of scalars no longer deep-copies element by element.

### Fixed

- **Code reached through a cached function's arguments** (a class, a function, or
  an instance's class) now contributes to the cache key, so editing it
  invalidates rather than returning the previously cached value. `args_hash`
  pickles its arguments and pickle serializes a class or function **by
  reference**, so editing a passed schema class used to hit forever — and hand
  back the stale class object.

- **Code reached through *that* code.** Reachability is transitive: if a cached
  function builds an `A`, and `A`'s `field(default_factory=lambda: B())`
  constructs a `B`, editing `B` invalidates too. Names the code *loads* are
  followed; type annotations are not, since they never run. The walk is static,
  so a class chosen at runtime — pulled from a dict, assigned during execution —
  is still invisible and needs `depends_on=`.

- **Globals a helper reads.** A cached function whose *helper* reads a module
  constant now invalidates when that constant changes. Only the cached
  function's own reads counted before, so a helper returning `CONFIG` served the
  old answer indefinitely. Accumulators a helper writes stay excluded, as they
  already were for the cached function itself.

- **Helpers whose source cannot be read** (defined by `exec`, in a REPL, or from
  a file that moved) contributed *nothing* to the key, so any edit to one went
  unnoticed. They are digested from their compiled form instead.

**Migration for all four:** these change the cache key, so affected entries
recompute **once** on upgrade — and once more on a Python-version change, since
the digest is bytecode (measured: the same class body hashes differently under
3.10 and 3.11). One-time recomputes, never wrong answers.

## [0.3.0] - 2026-08-07

One headline change: cash now caches the **expensive call inside a statement**,
automatically. Statements that could never be cached before — the ones where the
work sits inside a call and the statement itself is a mutation — are now fast on
a re-run, with no annotation.

### Changed

**Call-level caching is on by default.** This changes what cash does to code you
have already written, so it is worth understanding before you upgrade.

Previously the unit of caching was the whole statement. That is the wrong unit in
both directions:

<!-- test:skip reason="illustrative: two schematic statement shapes, no compute() to call" -->
```python
out.append(compute(x))     # skip-cached: `.append` is a mutation -> zero reuse, ever
s += compute(x)            # cached, but keyed on the running total -> a reorder re-ran the tail
```

`compute(x)` is now cached in its own right, so the first shape reuses work it
never could before, and reordering the second no longer re-runs everything after
the first change. `for e in items: out.append(f(e))` — the most ordinary way to
run something slow over a list — went from caching nothing to caching per item.

The trade-off is real and worth stating plainly: statements cash previously
declined to cache for an unrelated reason are now purity-judged for the first
time, automatically. If a callee has side effects the analyzer cannot see, a
cached call skips them without being asked. Opt out per statement or per cell
with `# @cash:no-cache-calls`. See *Annotations* → *Call-level caching*.

`# @cash:cache-calls` still parses and now does nothing; it is no longer needed.

### Added

- **Loops that are cheap per iteration but long are handled properly.** Cash
  measures a short head of the loop and, when per-iteration bookkeeping would
  cost more than it saves, stores the tail as one unit instead. A 20,000-iteration
  loop of sub-millisecond work is ~46x faster warm; previously it fell between two
  policies and cached nothing useful.
- **The badge shows which parts of a statement were cached**, so an intercepted
  call's hits and time saved are visible next to hand-decorated ones and tagged
  `[intercepted]`.
- **Three cost thresholds are configurable at runtime** — `call_cost_floor_seconds`,
  `loop_split_max_iter_seconds` and `loop_split_min_remaining_seconds` — via every
  layer (`configure()`, `CASH_*`, TOML). Documented in *Configuration*.
- `# @cash:ttl=` and `# @cash:persist` now reach the calls inside a statement, not
  just the statement. Both used to stop at the statement boundary, which meant
  that in the shape where only the call is cached, the annotation acted on
  nothing.

### Fixed

**Wrong values**

- **A global or captured variable passed to a call now invalidates the cache.**
  `sum(G)`, `len(G)`, `helper(G)` and `model.predict(X_test)` all put their
  argument beyond the tracker, so changing it left `@cash.cache` serving a stale
  result forever — with `.explain()` reporting `[HIT]`. The canonical shape was an
  `X_train`/`X_test` pair read as free variables by a decorated training function:
  changing the split ratio left the old predictions in place. Closure captures had
  the identical bug and are fixed the same way.
- **An argument whose content cannot be hashed is treated as changed, not
  unchanged.** A callee mutating such an argument was cached and its mutation
  silently skipped, on every Python before 3.14.
- **A cached callee's writes to globals are no longer dropped**, including from
  inside a loop body, and are replayed in the right order when the loop's items
  are reordered.
- **`# @cash:ttl=5m` no longer parses as five seconds.** A unit suffix was
  silently truncated to its leading digits — a 60x error whose only symptom was a
  cache that kept missing. Malformed TTLs are now rejected loudly and name
  themselves.
- **`@cash.cache` on a function returning a matplotlib `Figure` no longer hijacks
  pyplot's current figure.** Caching one made the cache's private copy current, so
  a later `plt.savefig()` wrote a figure you never drew on — on the first call,
  silently. Such results are now refused, with a warning explaining why.

**Correctness of behaviour**

- A statement's own `ttl` governs the calls inside it, so `# @cash:ttl=0` really
  does re-fetch.
- Cash's own cache reads are no longer recorded as your file dependencies.
- A cache entry this environment cannot read is treated as absent rather than
  raising, so a cache written with an optional dependency present stays usable
  without it.
- Re-executing a seeded draw now schedules the definitions it reads along with it,
  instead of raising `UpstreamStateError`.
- `%cash_on`'s notebook-not-found message no longer tells JupyterLab users to
  change a VS Code setting.

### Documentation

Three documented claims were found to disagree with the code and corrected: the
`CashConfig` field table was missing three fields, malformed-TTL handling was
described as silent when it warns, and a callee that writes a global was
described as re-executed on a loop reorder when it is in fact served with its
write replayed.

## [0.2.0] - 2026-07-29

Two features: cash can now track objects in remote storage the way it has always
tracked local files, and `# @cash:cache-calls` caches the expensive call inside a
statement for the two shapes statement-level caching structurally cannot help
with.

### Added

**Remote objects are tracked like local files**

- `RemoteFileDataSource` tracks an `s3://`, `gs://` or `http(s)://` object by the
  validator its store already maintains — ETag, version id, GCS generation — so
  editing the object invalidates the cache. It costs one metadata request, and a
  hit skips the download entirely. Because the token comes from the store rather
  than the local filesystem, it is **identical on every machine**, so a cache
  shared between machines actually travels — which a path and an mtime can never
  do. `http(s)://` needs no extra dependency; other schemes resolve through
  `fsspec`.
- **Remote reads are tracked automatically, on by default.** Before this,
  `pd.read_parquet("s3://bucket/key")` inside a `@cash.cache` function recorded
  *no* dependency at all — the URL was mangled through local-path resolution and
  dropped — so the function kept hitting after the object changed. Tracking
  usually *reduces* network traffic: it only engages for code that already reads
  from the network, and it trades a metadata request for a transfer that may be
  hundreds of megabytes.
- The same now holds for **notebook statements**, not just decorated functions. A
  statement reading a remote object re-validates it on lookup, and its downstream
  consumers invalidate through lineage. Previously the read contributed nothing
  and the statement hit forever.
- A **version-pinned URL** (`?versionId=`, `#generation=`) is recognised as
  immutable and costs no request at all — the pin *is* the token. Immutability is
  never inferred from a path shape, whose failure mode would be the worst
  available: never invalidating, silently, forever.
- **Freshness checks report their own cost.** A remote check is a network round
  trip that lands on the *hit* path — exactly where the badge reports a saving
  and nothing used to report what establishing it cost. The badge's overhead
  breakdown now carries a `remote` line with the source count, and cash warns
  once when validation stops being a good trade: either relative (it cost more
  than half the compute it saved) or absolute (seconds of metadata requests is
  unusable in a notebook even when it is net-positive on paper).
- New config `remote_revalidate_max_age_seconds` (default `0` — revalidate on
  every hit, the only setting that cannot serve stale data). It exists for
  auto-tracked reads, where you never construct the source and so have nowhere to
  put a per-source `max_age`. Raising it trades correctness for latency for the
  window's duration.
- **Failure is closed.** An unreachable store yields a never-before-seen token so
  the call recomputes, rather than a constant token that would let the second
  failure serve what the first one stored. Warns once per URL and failure kind. A
  missing `fsspec` is raised rather than hidden behind a silent forever-recompute.
  Cash also warns when size is the only validator a store offers, since a
  same-size edit would be invisible.

**`# @cash:cache-calls` — cache the expensive call, not the statement**

- An opt-in directive for the two shapes statement-level caching cannot help
  with:
  - `out.append(compute(x))` was skip-cached because the append is a mutation, so
    it re-ran in full every time;
  - `s += compute(x)` cached, but keyed on the running prefix, so reordering the
    input re-ran the whole tail.

  In both, the expensive thing is the call and the cheap thing is the wrapper
  around it. For the append this is also *more* correct than before — the
  mutation genuinely happens on every run instead of being surrendered.
- Interception happens **in place at the call node**, never by hoisting into a
  temporary, so short-circuiting (`f() or g()`), ternaries, comprehension scopes
  and `*`/`**` unpacking all keep working — Python still decides whether the call
  is reached.
- Intercepted calls are **labelled on the badge** (`compute() [via
  @cash:cache-calls]: 2/3 cached`) in both the HTML and text renderers, so you
  can confirm the directive engaged. Hits are credited in `%cash_stats` as usual.
- When the directive **cannot apply** — a call that reads the statement's own
  target *is* the fold — cash raises `CashCacheIneffectiveWarning` naming the
  statement and the rule, once per statement, instead of silently caching
  nothing.
- Bound methods are passed through deliberately. Caching a method puts `self` in
  the key, which needs the author's judgement (an unpicklable receiver silently
  fails to cache; a heavy `self.df` is pickled on every call) — see the
  caching-class-methods guide.

### Fixed

- **Kernel pseudo-filesystem reads (`/proc`, `/sys`, `/dev`) are never recorded
  as cache dependencies.** On Linux, cash's own periodic memory check reads
  `/proc/meminfo` *during* a cached call, so the read was attributed to your
  result. That file reports live memory, so its contents change on every read:
  the entry was found on lookup and thrown away as stale, every single time —
  the cache storing, hitting and discarding in a loop, silently, forever. Most
  visible for calls that write several entries at once, such as a chunked
  iterator.
- **`@cash.stateful` is now honoured by `# @cash:cache-calls`.** It is the
  documented way to say "never cache this function", and the statement path
  respected it while the call path did not — so a stateful callee was cached and
  returned a **stale value on the first run**: two calls to a counter in a loop
  gave `[1, 1]` where plain Python gives `[1, 2]`. `# @cash:no-cache` now also
  wins over `cache-calls`, rather than the call being cached anyway.
- **A matplotlib `Figure` from an intercepted call is no longer cached.** The RAM
  tier deep-copies on store and `Figure.__setstate__` re-registers the *copy* as
  pyplot's current figure, so a later bare `plt.savefig()` wrote the cache's
  snapshot instead of the figure you drew — a genuinely wrong PNG, on the first
  run. The statement path already refused this; the call path now refuses it too.
- **Cash's own file-tracking shims are never intercepted.** Its tracking wrappers
  around `open` and `pd.read_csv` are plain functions, so `cache-calls` tried to
  cache one — and `open('audit.log', 'a').write(...)` raised and wrote nothing.
  File-change invalidation was never affected.
- An RNG pill (`seed` / `random` / `UNSEEDED`) on a statement badge no longer
  pushes the timing chip onto a second line.

## [0.1.1] - 2026-07-24

The first public release. `0.1.0` was published to Test PyPI only; an
adversarial testing round against that build found the correctness and
packaging bugs fixed below, so the first release anyone installs from PyPI
is `0.1.1`.

### Fixed

**Randomness correctness**
- **A seed change now invalidates a cached result that depends on it.**
  Editing `np.random.seed(12345)` to `seed(999)` and re-running used to serve
  the value computed under the old seed — silently, with a "restored" badge —
  because a draw hidden inside a called function (an sklearn `fit()` with no
  `random_state`) is only discovered while the statement runs, after its cache
  key was built. That key carried no seed information, so every later run
  rebuilt it and matched the stale entry, and a kernel restart made it certain.
  Cash now declines to store the entry on the run that first discovers the
  draw; the next run keys it correctly. Unseeded draws are unaffected — they
  are still frozen and replayed from the first call.
- Draws hidden inside a called function are now visible to the cache key in
  every engine (runtime and the upstream simulation), so a re-seed above such a
  statement reaches it.
- `np.random.seed(None)` (and bare `seed()`) now warns that cached values below
  it cannot be both fresh and reproducible, and names the two ways out
  (`# @cash:no-cache`, or a fixed-integer seed). Cash cannot make a
  re-randomised stream and a cached value agree, so it says so rather than
  silently serving a value that describes a stream that no longer exists.

**Dependency tracking**
- A `@cash.cache` function that reads a constant through an imported module
  (`import conf; conf.RATE`) now invalidates when that constant changes.
  Previously only `from conf import RATE` was tracked, so the same dependency
  was followed or not depending on the import spelling. One level of recursion
  also covers `conf.get_rate()` whose source is unchanged but whose returned
  constant is not. Standard-library and site-packages modules are excluded.
- A cached **method** now tracks the class-level code it reaches through `self` —
  the methods, property getters, class constants, and `super()` base classes it
  uses — and does so **transitively** (a constant reached only through a helper
  method it calls). Editing any of them invalidates the cached result; before,
  such an edit could be missed and a stale value served.
- A class constant read through the class **name** (`Cfg.LIMIT`) or
  `type(self).LIMIT` is now folded into the key, matching the already-tracked
  `from cfg import LIMIT` spelling.
- A cached function that reads a **pre-built module-level object** (a transformer
  or client constructed once at import and used as data) now tracks that
  object's **class source**, so editing one of its methods invalidates. Before,
  only the object's data was hashed, so a method-body edit was invisible and a
  stale result was served — found by replaying a real sklearn pipeline's git
  history. (Objects you method-call directly or pass as a bare argument are
  unaffected; their called methods are already tracked.)
- A helper **referenced by name but reached through a value** (assigned to a
  local, then called) is now tracked, not just directly-named calls.
- Container **subclasses** (a `namedtuple`, a `dict` subclass) no longer collide
  onto their base type in the cache key: two distinct subtypes with identical
  contents now get distinct entries instead of one shadowing the other.

**Safety defaults**
- By default, `@cash.cache` now **raises** on dependency patterns whose edits it
  cannot track — `getattr(obj, name)()` dynamic dispatch,
  `importlib.import_module(...)`, and `eval`/`exec`/`compile` (including when the
  dynamic result is stashed in a local first). Caching correctness cannot be
  guaranteed for these, so cash refuses rather than risk a silently stale
  result. Opt in with `@cash.cache(assume_safe=True)`, mark an audited callee
  with `@cash.mark_pure`, or refactor to a static call. A statically-named call —
  the common case — is unaffected.

**Notebook caching — plots and figures**
- Re-running a plotting cell no longer shows the previous figure next to the new
  one, in the wrong order. `plt.show()` and pyplot module-level draw/style calls
  (`plt.plot`, `plt.title`, …) mutate global figure state, so they are now always
  re-rendered instead of restored from cache — which also fixes the duplicate
  plots you saw after editing a value above the plot and re-running.
- When cash rebuilds an upstream plot to satisfy a downstream cell, the
  reconstructed figure no longer leaks into that cell as a stray plot.
- A same-size, in-place edit to a large (>8 MiB) data file, landing outside the
  regions cash samples when hashing it, is no longer missed: file freshness now
  also checks the modification time for sampled files, so an edit that kept the
  file's size invalidates the cache instead of serving a stale read.

**Notebook badge**
- A failure while building or displaying the badge can no longer swallow the
  cell's output. The badge is a diagnostic overlay drawn around your statements;
  if it errors, the cell still runs and shows its result. (On Python 3.10/3.11 a
  renderer syntax error used to blank every cell after `%cash_on`; badge
  rendering is now covered across 3.10–3.14.)
- The time a statement shows on first run now matches the time it reports saving
  on restore. The first-run figure was the compute *plus* cash's own
  serialization cost, while "saved" was only the compute; the serialization cost
  now appears in the overhead breakdown instead, which also gained a labelled
  `cache write` line and a hover tooltip on each part.
- A magic indented inside a block (an `if IN_COLAB:` guard around a
  `%pip install`) no longer makes cash treat the whole cell as a syntax error and
  stop dependency-tracking everything that reads from it.
- Third-party import warnings raised while cash locates the notebook no longer
  leak into the cell's output.

**Colab and Jupyter**
- On Colab, running a downstream cell now re-runs the upstream cells it depends
  on. Colab keeps the notebook in Drive rather than as a local file, so cash
  reads the live cell contents through Colab's frontend API to resolve
  dependencies.
- Locating the notebook file no longer crashes in environments where the
  discovery helper raises instead of returning nothing.
- The "save your notebook first" tip is suppressed on Colab, where it does not
  apply.

**Packaging and tooling**
- `cash.help()` no longer crashes on a default Windows console: the guide's
  arrows and dashes are degraded to what a legacy code page can render. This is
  the first call the docs tell coding agents to make.
- The text badge (`%cash_badge print`, the mode meant for headless and agent
  runs) is now ASCII-only. Its emoji were written into the notebook by the
  kernel and then crashed whatever read the notebook back on a legacy code
  page — a traceback instead of a badge, for exactly its intended audience.
- The source distribution no longer bundles stray local virtualenvs (they were
  25 MB of a 36 MB archive; a nested `.gitignore` hid them from `git` but not
  from the build).
- `cash.help()` and `%cash_stats` no longer point at a docs page and a magic
  (`%cash_admin`) that do not exist; both now point at what does.

**Persistence policy**
- Corrected `smart_persistence=False` and the `cash info` output, which
  described a compute-time threshold the cost model stopped consulting; the
  0.1 s smart-persistence floor the default backend actually uses is now
  covered by a test.

### Changed

- Repeated `@cash.cache` calls with the same large, unmutated argument no longer
  re-hash it every time. A cache *hit* on a function taking a multi-million-row
  DataFrame was dominated by hashing that argument to build the lookup key; cash
  now reuses the content hash within a session (guarded by its own mutation
  tracking), so the second call is effectively free. Cache keys are unchanged, so
  entries from earlier calls still match.
- The source distribution now ships only the package and the files needed to
  build it (`pyproject.toml`, `README`, `LICENSE`). Tests, docs, example
  notebooks and benchmarks are no longer bundled in it — they remain on GitHub.

### Removed

- The unused `smart_persistence_threshold` configuration field, which the cost
  model no longer consulted.

### Added

- `cash.help()` gains a coding-agent guide, surfaced through `llms.txt`; the
  "How Cash Works" documentation section; and a warning when `seed(None)` is
  used with downstream caching (see above).
- A live **feature-tour notebook**, launchable in Google Colab or Binder from the
  README and docs, that walks through statement-level caching, cross-cell
  invalidation, and the `@cash.cache` decorator on a small analytics pipeline.

---

## [0.1.0] - 2026-07-22

Test PyPI only — never released to PyPI. See `0.1.1`.

Cash caches expensive work in Jupyter notebooks and Python functions, and
figures out on its own when a cached result is still valid. In a notebook it
works at **statement** level: edit one line, and only what actually depends on
that line recomputes — across kernel restarts, with no manual pickling.

### Added

**Decorator caching**
- `@cash.cache` with automatic dependency tracking — a cached function is
  invalidated when its own source, a helper it calls, or a file it reads
  changes.
- `async def` support: awaited results are cached, including under concurrent
  `asyncio.gather`.
- `cache_if=` predicate to skip storing selected results (e.g. negatives).
- `cash.register_hasher` for arguments that are not hashable by default.

**Notebook caching**
- `%cash_on` enables statement-level caching for the session; `%%cash` caches a
  single cell.
- Upstream simulation works out which earlier statements a cell really needs and
  restores the rest from cache instead of re-running them.
- An interactive badge per cell shows what was restored, what recomputed, and
  **why**.
- Per-statement annotations: `# @cash:no-cache`, `# @cash:persist`,
  `# @cash:cache-fit`.

**Randomness**
- Random draws are tracked. Editing a `seed(...)` invalidates the draws below
  it — including downstream statements that would otherwise have been served
  from cache — and a draw that has to re-run has its seed re-established first,
  so it reproduces the value it had before.
- Draws hidden inside a called function are caught too: cash compares the global
  RNG state across each statement, so a helper that draws internally is seen
  even though the statement spells no `random` call.
- The badge carries a per-statement pill — `seed`, `random`, or `unseeded` —
  and explains a re-run it had to do to restore the stream.
- An **unseeded** draw is flagged, because its cached value is a frozen replay:
  re-running the cell returns the first value again rather than a new one. This
  is by design — it is what makes a notebook reproducible — but it is surfaced
  loudly rather than left implicit. `# @cash:no-cache` opts a statement out and
  makes it draw fresh each time.

**Backends**
- InMemory, File, Redis, S3, and a Tiered backend (the default) with a
  cost-model-driven persistence policy that decides what is worth writing to
  disk.

**File dependency tracking**
- Reads through pandas, numpy and builtins are tracked automatically by content
  hash; `file_depends_on` declares dependencies explicitly.

**Tooling**
- `cash` command-line interface.
- Magics: `%cash_help`, `%cash_status`, `%cash_stats`, `%cash_audit`,
  `%cash_benchmark`, `%cash_persist`, `%cash_debug`, `%cash_badge`,
  `%cash_off`, `%cash_feedback`.
- `cash.CashWarning` and subclasses are exposed at the top level, so a project
  can turn cache-ineffectiveness into a CI failure via
  `warnings.filterwarnings`.
- `cash.help()` prints an orientation summary, and `docs/for-coding-agents.md`
  (surfaced through `llms.txt`) is a single-page reference written for coding
  agents, which do not see the badge a human reads.

### Notes

- **Versioning restarts here.** Development ran through internally-numbered
  versions up to `0.5.0b2`; none of them were ever published. Rather than open
  to the public at a number implying four prior releases nobody could install,
  the first release anyone can `pip install` is `0.1.0`. The earlier entries are
  kept below as a development record.
- This is a `0.x` release: the API may still change between minor versions.

---

## Pre-release development history

**Nothing below this line was ever published to PyPI.** These entries are the
internal development record that led to `0.1.0`, kept for provenance. The
version numbers are historical and do not correspond to anything installable.

### [0.5.0b2] - Unreleased

### Added
- `@cash.cache` now supports `async def` functions. Awaited results are
  cached; auto-file-dep tracking works correctly under concurrent
  `asyncio.gather`. (Async generators emit a `CashCacheIneffectiveWarning`
  and are returned unwrapped — full async-gen caching is planned for
  a later release.)
- `cash.CashWarning`, `cash.CashCacheIneffectiveWarning`,
  `cash.CashCacheStoreFailedWarning` exposed at the top level. Filter
  via standard `warnings.filterwarnings(...)` — e.g. set
  `CashCacheIneffectiveWarning` to `error` in CI to fail the build
  when a deploy introduces an unpicklable arg.
- New tutorial: `docs/caching-class-methods.md` — recipe for caching
  methods on stateful objects (`Loader`, services, database wrappers)
  via `cash.register_hasher`.
- `@cash.cache(cache_if=callable)` — optional predicate that receives
  the function's return value and returns a bool. When false, the
  result is returned to the caller as normal but not stored in the
  cache. Useful for skipping the caching of negative results
  (`cache_if=lambda r: r is not None`). Predicate exceptions are
  caught (debug-logged) and treated as false. Works on both sync and
  async functions.
- `@cash.cache` now caches functions that return one-shot iterators
  (Python generators, `map`/`filter` results, custom iterators). The
  iterator is eagerly materialized into a list, the list is cached,
  and each call returns a fresh iterator over the cached values.
  Generator-specific methods (`.send`, `.throw`) are not supported on
  the cached wrapper. Not suitable for infinite or streaming
  generators — see `docs/caching-class-methods.md` for the trade-off.
- `cash.register_hasher(T, fn)` now hashes `fn`'s source (or
  bytecode) at registration and embeds the hash in the cache key.
  Changing the body of a registered hasher invalidates dependent
  cache entries, even when the new hasher's output coincidentally
  matches.
- `@cash.cache(chunk_max_items=..., chunk_max_bytes=...)` — iterator
  results are now stored in chunks. Defaults are 1M items and 1GB
  bytes; iterators below these thresholds land in a single chunk and
  behave indistinguishably from a list. Larger iterators are split
  across multiple backend keys and the retrieval iterator reads them
  lazily. RAM bounded by chunk size on both write and read. Chunked
  storage is on by default with no opt-in required.
- `f.explain(*args, **kwargs)` — every `@cash.cache`-decorated function
  now exposes an `explain()` method that returns a `CacheExplanation`
  describing whether the next call with those args would hit or miss
  the cache, and *why*. Reasons include `hit`, `key_uncomputable`
  (unhashable arg), `no_entry`, `ttl_expired`, and `file_changed`
  (with the list of changed paths). Pure introspection — never calls
  the function, mutates stats, or writes to the backend. Available on
  async-wrapped functions too. `CacheExplanation` is exported from the
  top-level `cash` package.
- `f.cache_info()` now includes a `warnings` key — a rolling log of
  recent `CashWarning` emissions for that function (capped at the
  last 20). Lets users discover silent misbehavior after the fact
  even when `warnings.simplefilter` swallowed the stderr emission.
  `f.cache_clear()` now also resets this log and forgets dedup marks
  so future misbehavior re-warns.
- **Purity analyzer on the decorator** — `@cash.cache` now AST-walks
  the decorated function body and its module-bounded helpers on
  first call, flagging known-impure calls (`requests.post`,
  `os.system`, file-write methods, `logging.info`, …), scope
  mutations (`global`, `nonlocal`, attribute/subscript assignment),
  explicit dynamism (`eval`/`exec`/`compile`, `getattr(obj, name)()`
  with non-constant `name`, calling a parameter as a function), and
  discarded calls to non-known-pure callees. Surfaced as a one-shot
  `CashImpurityWarning` per `(function, reason)`. Two opt-in modes:
  - `@cash.cache(strict=True)` — raises `CashImpureFunctionError`
    on first call if any issue is found. Also promotes opaque
    callees (no source) to issues. Use in CI to fail builds that
    introduce caching of side-effecting code.
  - `@cash.cache(assume_safe=True)` — silences the warning when
    you've audited the function and know caching is correct (e.g.
    a memoized API call where the side effect is idempotent). The
    analyzer still runs because helper source hashes feed the
    cache key.
  Mutually exclusive — passing both raises `ValueError` at
  decoration time.
- `cash.mark_pure(func)` / `cash.mark_stateful(func)` — module-level
  helpers to annotate third-party callables you've audited. Sets
  the existing `_cash_pure` / `_cash_stateful` attributes the
  analyzer respects. Returns *func* unmodified (no wrapping), so
  it's safe to call on C extensions and callable instances.
- `CashImpurityWarning` (subclass of `CashCacheIneffectiveWarning`)
  and `CashImpureFunctionError` (subclass of `CashError`) exported
  from the top-level `cash` package.
- **Latent-bug fix: helper-source-hash cache invalidation.**
  Previously, editing a plain helper called from a `@cash.cache`d
  function did NOT invalidate the cache — only edits to `@cash.cache`d
  callees were tracked. Now the same analyzer walk captures source
  hashes (and module-resolution paths) of every analyzed user-code
  helper. On every call, helpers are re-resolved from `sys.modules`
  and re-hashed, with current hashes folded into the cache key.
  This catches both cross-process edits (new run = new hash = new
  key) and in-process redefinitions (notebook cell rerun, REPL
  rebind, hot-reload). Per-call overhead is ~5-30μs for typical
  helper counts. The fallback to the recorded snapshot kicks in
  when re-resolution fails (helper deleted/renamed since analysis).
- Purity analyzer now recurses through **closure variables** in
  addition to `__globals__`. A `@cash.cache`d function defined
  inside another function (e.g. a factory pattern) gets its sibling
  helpers analyzed for impurity, scope mutations, and dynamic
  patterns. Closure helpers contribute to the cache-key state hash
  via the analysis-time snapshot (they have no stable
  `sys.modules` path for re-resolution, so per-call invalidation
  defers to the snapshot — which is the right behavior since
  closures are re-created fresh each time the enclosing function
  runs).

### Changed
- Ineffective-cache and store-failure events now emit
  `warnings.warn(...)` instead of `logger.warning(...)`, deduplicated
  per `(category, function, argument type)`. Users who relied on
  silent failure should add `warnings.filterwarnings("ignore",
  category=cash.CashWarning)` to their startup code.
- Three previously-silent failure paths now emit
  `CashCacheIneffectiveWarning` instead of a `logger.debug` /
  `logger.warning` line that nobody read: a `cache_if=` predicate that
  raises (was: silent skip), backend lock acquisition failure (was:
  proceeded unlocked with only a debug log), and a stored entry whose
  metadata fails validation (was: silently treated as miss). Same
  per-`(category, function, reason)` dedup as the existing warnings.
- `FileAccessTracker` now uses `contextvars.ContextVar` for active-
  tracker dispatch. Concurrent `asyncio.gather` and threaded callers
  are correctly isolated. No user-facing API change.
- `cache_if` interaction with iterator-returning functions: predicate
  is honored when the result fits in a single chunk. For multi-chunk
  results, `cache_if` is bypassed and a one-shot
  `CashCacheIneffectiveWarning` fires at the chunk_0 → chunk_1
  transition. To keep gating active on large iterators, lower
  `chunk_max_items` / `chunk_max_bytes` or materialize manually.
- **Expect a one-time recompute after upgrading.** Four fixes in this
  release change how decorator cache keys are computed, so affected
  entries written by an earlier version no longer match and recompute
  once before settling. Nothing is lost — the old entries are simply
  not found. The affected shapes, each narrow rather than universal:
  a cached function that **reads a module global** now folds that
  global's content into its key (CAS-107); a call passing a **dict**
  whose keys are not already in sorted order now canonicalises that
  order (CAS-108, already-sorted dicts keep byte-identical keys); a
  function declaring `depends_on=[plain_function]` now folds the dep's
  source in (CAS-110, the edge previously contributed nothing); and a
  call passing an **object-dtype ndarray** now hashes its content
  rather than raw pointer bytes (CAS-111 — those keys were never
  stable across processes, so they rarely hit in the first place).
  A function matching none of these shapes keeps its existing keys and
  its cache.
- **File-dependency freshness on `@cash.cache` is now decided by content,
  not `(mtime, size)`** (CAS-119), matching the notebook path (CAS-98 /
  CAS-10) so the two subsystems cannot drift. Two verdicts flip: a
  *touched* file with identical bytes no longer forces a recompute, and
  a same-size edit under an indistinguishable mtime is no longer missed
  and served stale. Existing cache entries are **not** invalidated —
  snapshots written before this release carry no content hash and keep
  the old mtime comparison, both for the freshness check and for the
  lineage hash handed to downstream consumers.
- **Decorator caches without an explicit `ttl` now honor the backend's
  `default_ttl`.** Cache metadata moved to frozen dataclasses whose
  `to_dict()` omits unset (`None`) fields, so an entry created without a
  per-call `ttl` no longer writes `ttl=None` into its metadata. Backends
  treat a *missing* `ttl` key as "apply my `default_ttl`", so e.g.
  `FileBackend(default_ttl=3600)` now expires such entries after an hour
  instead of keeping them forever. Previously the producer stamped
  `ttl=None`, leaving the key *present* and suppressing the backend
  default. To keep entries non-expiring on a backend with a
  `default_ttl`, omit `default_ttl` or use a separate backend instance.

### Fixed
- **IPython magics (`%cash_on`, `%%cash`, …) failed to register on
  `%load_ext cash` / `import cash`.** `Cash.register_magic()` imported
  `CashMagics` from a stale module path (left over from the ADR-013
  move into the `cash.notebook.ipython` package), and the resulting
  `ImportError` was swallowed by a guard meant only for "IPython not
  installed" — so auto-load silently registered nothing. The internal
  import now targets `cash.notebook.ipython.magics` and sits outside the
  IPython-availability guard, so a broken path surfaces loudly instead
  of masquerading as a missing dependency.

### Backward compatibility
- v1 iterator cache entries (written by 0.5.0b2 prior to chunked
  storage, with `metadata['materialized_iterator']=True`) continue
  to read correctly via a legacy code path. Old entries are
  eventually replaced by chunked entries on the next compute miss;
  no migration is required.
- The `CachedIterator` class has been renamed to `_ListCachedIterator`
  internally. The old name is kept as a deprecation-friendly alias
  in `cash.core` for one release and will be removed in 0.6.0.

### Not yet supported
- `@cash.cache` on `async def gen(): yield ...` (async generators)
  emits `CashCacheIneffectiveWarning` and returns the function
  unwrapped.
- `use_locking=True` combined with an async function emits
  `CashCacheIneffectiveWarning` and proceeds unlocked.

### Documentation
- **Tutorials section restructured** into two subsections: Feature
  Guides (task-oriented "how do I do X with Cash") and Use Cases
  (domain-driven workflows).
- **10 new feature guides** under `docs/tutorials/feature-guides/`:
  Choosing a Backend, Controlling Cache Behavior, Debugging and
  Monitoring, Custom File Sources, Custom Hashers, Dynamic
  Dependencies, Iterator Caching, Smart Persistence, Async Caching,
  and Thread Safety. Each is grounded in `src/cash/` source citations
  and reflects the *actual* behavior — several guides correct prior
  misconceptions (e.g. `dynamic_depends_on` requires a `DataSource`
  instance, only Redis implements real locking, smart-persistence
  config fields are notebook-only).
- **3 new use case tutorials** under `docs/tutorials/use-cases/`:
  LLM API Calls, Data Engineering, Scientific Computing — each
  tightly focused on where Cash adds value in the domain.
- **Production Transition** slimmed from ~220 to ~140 lines: it now
  covers only the migration story, with decorator mechanics deferred
  to `docs/decorator.md` and `file_depends_on` reframed as an escape
  hatch for non-standard file access.
- **Data Science** use case tightened from ~220 to ~130 lines:
  removed generic file-dependency teaching (now automatic), kept the
  iteration-loop value proposition as the central message.
- **Advanced Configuration** monolithic page split into the three
  focused guides above.
- **Getting Started tutorial** absorbed into
  `docs/getting-started/quickstart.md` to eliminate duplicate
  on-ramps; `cli.md`'s autoload hook and the magic-commands roundup
  now live in the unified quickstart.
- User Guide pages (`decorator.md`, `annotations.md`, `badges.md`,
  `getting-started/configuration.md`, `why-cash.md`,
  `notebook_caching_api.md`, `cost-model.md`, `index.md`) gained
  cross-references to the new tutorials where appropriate.

### [0.5.0b1] - Beta Release

### Added
- **Bug-report button** in the badge header with a budget-aware URL builder that auto-fills a GitHub issue with the failing cell, environment info, and the most recent metrics (without exceeding GitHub's URL length cap).
- **Per-iteration caching for upstream loop re-execution.** When upstream simulation has to re-run a loop, each iteration is now cached individually instead of treating the whole loop as one cache unit. Editing a loop body or extending the iterable only re-runs the affected iterations.
- **Forward-probe skip optimization in upstream simulation.** Before scheduling upstream cells to repair broken variables, Cash now probes the current cell to see whether its disk cache hits would restore the same variables. If so, the upstream re-execution is skipped entirely.
- **File-dependency path fallback.** When a project is moved (e.g. Google Drive path change, repo cloned to a new machine), absolute paths in cache metadata no longer cause full recomputation. `cash.utils.resolve_file_dep_path()` resolves stale paths via CWD-relative basename and suffix matches, and the resolver is wired into all cache-validation paths (`statement_processor`, `magics` restore, `upstream` checks).
- **`uncacheable_reasons` on metrics.** The badge and text-mode output now explain *why* a statement was not cached (`@cash:no-cache annotation`, `Input variable missing lineage`, ...).
- **Storage tier display in COMPUTED badges.** Each computed row now shows where the value landed (`RAM`, `RAM+DISK`, ...) with a hover explanation. Falls back to friendly labels (`- no outputs`, `- trivial`) when storage info is genuinely unavailable.
- **`%cash_help` magic** for a quick-reference command card.
- **`%cash_feedback` magic** that points at the issue tracker and discussions.
- **Welcome message on `%cash_on`** with actionable next steps.
- Expanded documentation with tutorials and use-case guides.

### Changed
- Version bumped to 0.5.0b1 for public beta release.
- Development status updated from Alpha to Beta.
- `[pandas]` / `[all]` / `[dev]` extras now require `pyarrow>=13.0` so DataFrame hashing produces stable, cross-platform results.
- `TieredBackend.set()` now propagates the resolved storage destinations back to the caller's metadata dict (previously the badge couldn't tell where a value landed).
- Status-based labels for upstream auto-execution loop groups in the badge.
- Upstream auto-exec loop groups render with full per-iteration detail in the badge.
- Cached simulation state is restored before the first changed cell regardless of whether the change was a code-hash mismatch or a stale file dependency (previously a stale file dep dropped *all* cached state).

### Fixed
- **Downstream overwrites of loop-produced variables** are now detected by upstream simulation; previously they could mask staleness.
- **Single-unit fallback for small loops with expensive iterations** is no longer triggered, so per-iteration caching stays effective.
- **Progress step lag** in the badge during long-running cells.
- **Bug report `RichOutput` repr** crash fixed.
- Internal `__iteration_context__` / `control_context` comments are stripped from the bug-report URL so reported code matches what the user wrote.
- Drop the inline `_cashBadgeExp` expand/collapse persistence script that caused state drift when cells were re-rendered mid-execution.
- **Windows console emoji crash.** `import cash; %load_ext cash; %cash_on` no longer raises `UnicodeEncodeError` from a vanilla `python.exe` shell on Windows (cp1252). A new `cash.utils.safe_text()` helper passes UTF-8 streams through unchanged and downgrades each emoji to a short ASCII fallback when the active stream cannot encode it (`✅` → `[OK]`, `⚙️` → `[run]`, ...). Jupyter kernels are UTF-8 so notebook users were unaffected.
- **`@cash:no-cache` annotation crash.** The fast path on a cached-skip annotation referenced `metrics` before it was defined, raising `UnboundLocalError` on first hit; the dict is now initialised before the append.
- **Decorator `execution_time` always 0 on Windows.** `@cash.cache` recorded per-call timings with `time.time()`, which has ~16 ms resolution on Windows and produced zero-duration entries that broke the call log; switched to `time.perf_counter()` (nanosecond resolution everywhere).
- **`%cash_benchmark --compare` dropped the `Speedup` line on coarse timers.** Same Windows resolution issue caused `mean_uncached` to round to 0; switched to `perf_counter`, and now print `Speedup: n/a (timings below timer resolution)` instead of silently omitting the line.
- **File-dep cache invalidation missed same-mtime rewrites.** On filesystems with coarse mtime granularity (HFS+/APFS, some ext4 configs) two back-to-back rewrites of the same file produce identical mtimes and the cache stayed valid. `file_dependencies` metadata now records both mtime and size, and all five validation paths check both. Existing on-disk caches load fine — they just lose the size check until they're re-written.

### [0.3.0] - Decorator–Notebook Bridge

### Added

- **Decorator–Notebook Bridge**: `@cash.cache` decorator calls inside notebook cells are now tracked and displayed in badges
  - Call logging via `Cash._log_decorator_call()` with thread-safe append
  - `Cash.drain_decorator_calls()` for atomic retrieval of call events
  - Badge integration showing per-function hit/miss counts with condensed display for many calls
  - Decorator metrics (hits, misses, time saved) visible in `%cash_status` output

- **`cache_info()` and `cache_clear()`**: Per-function introspection on decorated functions
  - `func.cache_info()` returns `{'hits', 'misses', 'hit_rate', 'total_time_saved'}`
  - `func.cache_clear()` clears all cache entries for a function and resets stats
  - `func.__wrapped__` preserved via `functools.wraps`

- **`register_hasher(type_, hasher_fn)`**: Custom type hasher registration
  - Priority chain: `_cash_hash` attr → registered hashers → built-in hashers → pickle
  - Enables caching functions with non-picklable argument types

- **Built-in type hashers**: Native hashing for pandas DataFrame/Series, numpy ndarray, polars DataFrame/Series/LazyFrame, PyArrow Table/RecordBatch, modin DataFrame, dask DataFrame

- **`file_depends_on` parameter**: Shorthand for `@cash.cache(file_depends_on="data.csv")`, equivalent to `depends_on=[FileDataSource("data.csv")]`

- **Automatic import source invalidation**: Local module imports are tracked; changing a helper file invalidates dependent caches with transitive module dependency expansion

- **Opaque call pattern warnings**: Warnings when decorated functions are called with arguments that can't be hashed

- **`cleanup(max_age)` method**: Remove expired cache entries by age or stored TTL

- **`explorer()` method**: Returns `CacheExplorer` instance for interactive cache browsing

- **`register_file_handler()` method**: Extensible file tracking for custom libraries

### Fixed

- **Transitive notebook-level invalidation**: Changing a `@cash.cache` decorated function now correctly invalidates all notebook cells that depend on it, not just direct callers
- **Module-qualified function keys**: `Cash._get_func_key(func)` now uses `f"{func.__module__}.{func.__qualname__}"` to prevent collisions when different modules define functions with the same qualname

### Changed

- Default backend is now `TieredBackend` (InMemory L1 + FileBackend L2) with smart persistence policy
- `_analyzed.discard()` called when function source hash changes, forcing dependency graph rebuild

### [0.2.0] - 2025-02-06

### Added
- **Configuration System** (`cash.config`):
  - Global config file support (`~/.cash/config.toml`)
  - Environment variable support (`CASH_BACKEND`, `CASH_DEBUG`, `CASH_CACHE_DIR`, etc.)
  - `get_config()`, `CashConfig`, `create_default_config()` API
  - Config precedence: env vars > config file > defaults

- **SQLite Backend** (`cash.backends.sqlite_backend`):
  - Single-file cache storage using SQLite
  - WAL mode for concurrent access
  - TTL expiration, LRU eviction, max size limits
  - Thread-safe with entry counting and size tracking

- **FileBackend TTL**:
  - `default_ttl` parameter for automatic cache expiration
  - Per-entry TTL override via metadata
  - Expired entries auto-deleted on access

- **Collaboration Magics**:
  - `%cash_export <file>` - export cache entries to portable file
  - `%cash_import <file>` - import cache entries (with `--merge` mode)
  - `%cash_stats` - session-wide statistics (JSON output, reset)

- **Mutation Detection** (`cash.notebook.mutation_detector`):
  - AST-based detection of in-place mutations (append, extend, etc.)
  - Detection-only mode (does not affect lineage)
  - **Mutation-aware caching**: early detection before cache lookup
  - `get_top_level_mutated_variables()` excludes class/function body internals
  - Pure side-effects on non-output variables correctly marked as uncacheable

- **Purity Declaration System** (`cash.notebook.purity`):
  - `@cash.pure` decorator marks functions as pure (no side effects)
  - `@cash.stateful` decorator marks functions as stateful (always re-execute)
  - `is_pure()` / `is_stateful()` helper functions for checking markers
  - Pure functions skip mutation detection for better performance
  - Stateful functions skip caching entirely to ensure correctness
  - Integrated before skip-optimization to prevent stale @stateful results

- **Function Tracking** (`cash.notebook.function_tracker`):
  - Track function source code changes for cache invalidation
  - Function source hashes included in cache keys and lineage
  - `%cash_track` magic for monitoring imported module files
  - **Hot reload notification**: badge shows "🔄 Function changed" with orange highlight

- **Module Hot Reload**:
  - `%cash_track my_module` to watch for file changes
  - `%cash_track --check` auto-detects and reloads changed modules
  - `.pyc` cache invalidation for reliable reload

- **Structured Logging** (`cash.logging`):
  - JSON formatter for machine-readable log output
  - In-memory log handler with event type filtering
  - `%cash_debug json` for JSON console output
  - `%cash_debug file <path>` for file-based logging
  - `%cash_log` magic to view/filter/clear recent events

- **CLI Tool** (`python -m cash`):
  - `cash version` - show version info
  - `cash info` - show configuration details
  - `cash inspect <notebook>` - show cache statistics
  - `cash clear [dir]` - clear cache directories

- **nbconvert Integration** (`cash.nbconvert`):
  - `CashStripPreprocessor` strips badges, debug output from notebooks
  - Optional magic command stripping for clean exports

- **Documentation**:
  - API reference (`docs/api_reference.md`)
  - Migration guide from lru_cache, joblib, pickle (`docs/migration_guide.md`)
  - Architecture Decision Records (`docs/architecture_decisions.md`)

- **CI/CD**:
  - GitHub Actions CI (Python 3.10-3.13 × Linux/macOS/Windows)
  - PyPI publish workflow (release + Test PyPI)
  - Pre-commit hooks (ruff, file checks)
  - Docker support (Dockerfile + docker-compose.yml)

- **Community**:
  - CODE_OF_CONDUCT.md, SECURITY.md
  - Issue templates (bug report, feature request)
  - Pull request template

- **Badge UX**: Loop iteration grouping with collapsed display, loop variable values shown per iteration

- **Simulation Optimization**: Incremental upstream simulation caching

- **Polars Support**: File tracking for polars read/scan functions

- **CloudPickle Serializer**: Support for lambda functions and closures

- **Error Recovery Magics**:
  - `%cash_verify` - check cache integrity
  - `%cash_repair` - repair corrupted entries

- **Benchmarking**: `%cash_benchmark` magic for performance testing

- **Provenance Tracking** (`cash.notebook.provenance`):
  - `ProvenanceTracker` records variable history with full dependency chain
  - `%cash_provenance` magic: `--all`, `--graph`, `--time`, `--json`, `--clear`
  - Transitive dependency/dependent graph traversal
  - JSON export for external analysis

- **Audit Logging** (`cash.notebook.audit`):
  - `AuditLogger` with `AuditEntry` dataclass for compliance tracking
  - `%cash_audit on/off/show/summary/clear` magic
  - In-memory buffer (max 5000 entries) + optional file output
  - Filter by operation type or variable name

- **Lazy Deserialization** (`cash.backends.lazy`):
  - `LazyProxy` class defers deserialization until value access
  - `FileBackend.get_metadata()` for metadata-only lookups

- **AST Parse Caching**: LRU cache for parsed ASTs in upstream checker

- **Script Caching Demo**: Example showing `@cash.cache` decorator usage in Python scripts

- **Comprehensive Test Suite**: 1474 tests (unit + integration), 81% coverage
  - Library compatibility tests: sklearn, matplotlib, numpy, pandas
  - Data science workflow integration tests (sklearn pipelines, pandas → sklearn, CSV cascades)
  - Type hints for all public API and key internal modules
  - Purity declaration tests (45 tests)
  - Coverage boost tests: serialization, control structures, backends, config, purity, graph, annotations, nbconvert, tiered, SQLite, analysis

- **Documentation Site** (MkDocs Material):
  - Landing page with feature overview and key concepts
  - Getting Started guides: installation, quick start, configuration
  - Contributing guide with development setup and testing
  - API reference, architecture decisions, migration guide

- **Cache Diff** (`%cash_diff`):
  - Compare current session lineage with exported cache file
  - Shows only-in-current, only-in-other, changed, identical variable counts
  - `--vars` flag for variable-level detail
  - Supports both JSON and pickle cache file formats

- **JSON Cache Export** (`%cash_export --json`):
  - Export lineage metadata as JSON for %cash_diff interoperability
  - `%cash_export file.json --json` exports lineage without cache values
  - `--vars` filter works with JSON export

### Changed
- Configuration: `compress` and `debug` parameters now default to `None` (use config)
- Backend creation extracted to `_create_default_backend()` method
- Python version requirement updated to >=3.10
- Package metadata updated with proper classifiers and keywords
- Optional dependencies: `pip install cash-lib[redis]`, `[s3]`, `[all]`, etc.

### Fixed
- 10 test failures from Phase 1.1 cleanup
- `__all__` bug in `cash/__init__.py`
- IPython mock test isolation issues

### [0.1.0-dev] - Initial internal release

> Renamed from `0.1.0` when versioning restarted: `0.1.0` is now the first
> public release above. This entry is the project's original first cut.

### Added
- Core `Cash` class with decorator-based caching
- Jupyter notebook integration via IPython magics (`%cash_on`, `%%cash`)
- Statement-level caching with automatic dependency tracking
- Pluggable backends: InMemory, File, Redis, S3, Tiered/Cascading
- File dependency tracking (pandas, numpy, builtins, etc.)
- Smart persistence policy for tiered caching
- Interactive badge display for cache status
- Upstream dependency detection and re-execution
