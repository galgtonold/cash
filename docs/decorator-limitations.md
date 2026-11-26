# Known limitations of `@cash.cache`

!!! info "Applies to: decorator"
    Scripts, services and libraries that use `@cash.cache`.

Most functions cache correctly with a bare `@cash.cache`. This page lists the
cases that need a change on your side, and what that change is.

## Arguments cash cannot hash

<!-- claim: cash/decorator/arg_hashing.py:ArgHasher.hash_payload @95df02aa -->
An argument that cannot be pickled (a lock, an open file, a live connection)
cannot be keyed. The call runs uncached and warns
[`KEY-UNHASHABLE-ARG`](warnings.md#key-unhashable-arg). Any other failure while
building the key also runs the call uncached, and warns
[`KEY-BUILD-FAILED`](warnings.md#key-build-failed). Pass a plain value that identifies the object
instead. For a class of yours, give it a
[`__cash_key__`](tutorials/feature-guides/custom-hashers.md#cash-key)
method that returns one. For a type you don't own, register a hasher that
returns a **stable** identifying value, such as a database URL:

```python
import hashlib
import cash

class Store:
    def __init__(self, url):
        self.url = url

def hash_store(store):
    return hashlib.sha256(store.url.encode()).hexdigest()

cash.register_hasher(Store, hash_store)
```

Never hash by `id()`: ids repeat across processes, so a later run could get
another object's entry. See [Custom hashers](tutorials/feature-guides/custom-hashers.md).

## A table changed past pandas

<!-- claim: cash/decorator/arg_hashing.py:watch_array_handles @afda31f6, cash/decorator/arg_hashing.py:_watch_callback_handouts @90a4d06c -->
Under pandas copy-on-write, a table passed to a cached function again is not
hashed again while pandas has given it no new data: pandas copies a column
before it writes it. A write that goes past pandas is seen as long as it goes
through memory pandas handed out: `df["x"].array`, what `.values`,
`.to_numpy()` or `np.asarray` returned (a view you made writable again, or
the labels of a numeric index, which `np.asarray` hands out writable), a categorical
column's `.values`, its `.codes` made writable, `pd.Categorical(df["x"],
copy=False)`, and the arrays
pandas passes to your function from `df.apply(f, raw=True)`,
`rolling(...).apply(f, raw=True)`, `expanding().apply(f, raw=True)`, `pipe`
and a groupby's `apply`, `agg` and `transform`. A NumPy view pandas made
somewhere else and you made writable is not seen, and the cached function
returns its old result for the table: write through pandas
(`df.loc[...] = ...`) instead.

## A table served from memory

<!-- claim: cash/backends/frame_sharing.py:_warn_at_write @3bda0831 -->
Within one process, a pandas table a hit serves from memory shares its
memory with the cache's copy, so pandas copies a column before you write
into it. A NumPy array you took from the table before the write
(`v = df["t"].to_numpy()`) then keeps the old values, where plain Python
would show the new ones. cash warns at the write
([`CACHE-SERVED-VIEW`](warnings.md#cache-served-view)). Take the array after
the write, or copy the table first (`df = df.copy()`). A table read back
from disk is your own and behaves as the one your code built.

## An argument that does not change the result

<!-- claim: cash/core.py:Cash.cache @1db754d3 -->
Every argument is part of the key, so a logger, a progress callback or a
`verbose=` flag splits the cache: `fit(data, verbose=True)` misses after
`fit(data)` ran. Leave such a parameter out with `ignore=["verbose"]` or a
`cash.Ignore[bool]` annotation, or say what decides the result with `key=`;
see [Leaving arguments out of the key](decorator.md#leaving-arguments-out-of-the-key).

When the extra argument's work should happen on every call, hits included,
keep it out of the cached function instead. A thin wrapper takes it and calls
a cached core with only the arguments that change the result:

```python
import logging
import cash

@cash.cache
def _fit(data):
    return sum(data) / len(data)

def fit(data, log=None, verbose=False):
    if verbose and log:
        log.info("fitting %d rows", len(data))
    return _fit(data)

fit([1, 2, 3], log=logging.getLogger("a"), verbose=True)
fit([1, 2, 3])   # _fit hits: log and verbose are not in its key
# test:inject: assert _fit.cache_info()["hits"] == 1; _fit.cache_clear()
# test:inject: _fit([1, 2, 3])  # first call after the clear
```

The wrapper's own work runs on every call, so the log line appears on a hit
too, which a logger inside the cached body would not do.

<!-- claim: cash/core.py:Cash.register_hasher @0fc93c81 -->
For an argument **type** that never affects a result, a hasher that returns a
constant does the same without a wrapper. Every logger then counts as the same
value:

```python
import logging
import cash

cash.register_hasher(logging.Logger, lambda log: "any-logger")

@cash.cache
def fit_logged(data, log):
    log.info("fitting %d rows", len(data))
    return sum(data) / len(data)

fit_logged([1, 2, 3], logging.getLogger("a"))   # first call: runs
fit_logged([1, 2, 3], logging.getLogger("b"))   # cache hit
```

The hasher applies to every cached function in the process, so use it only
for a type that is never input data. Don't do this for `bool` or `int`: leave
a `verbose=` flag out with `ignore=`.

## Methods and `self`

`self` is an argument like any other, hashed by its state: two instances with
equal attributes share entries. An unpicklable attribute makes every call
uncached, and a large one (a frame, a model) is read on every call. A
`__cash_key__` method on the class fixes both. See
[Class methods](tutorials/feature-guides/caching-class-methods.md).

## Code you pass as an argument

A class or function of yours passed as an argument is keyed by its code and by
what that code reads, so editing it recomputes. Two cases are not covered:

- **Library classes and functions** are keyed by name, not code. See
  [Code in installed packages](#code-in-installed-packages).
- **A `functools.partial` of a closure or `lambda`** can't be pickled, so the
  call runs uncached. A closure or a `lambda` passed on its own is keyed by its
  code, the globals it reads and the values it captured (`make(3)` and
  `make(4)` key apart). Pass the function itself, with the bound values as
  arguments.

A table of functions you pass in, or an attribute of an argument that holds
one (`router.table[name](x)`), is part of that argument, so the functions in
it are keyed the same way; see
[Dispatch tables and strategy registries](#dispatch-tables).

A marker class you pass but whose code never affects the result can be excluded
with `@cash.opaque`; see [Purity markers](tutorials/feature-guides/purity-decorators.md#cashopaque-leave-a-class-out-of-the-key).

## Dispatch tables and strategy registries {#dispatch-tables}

<!-- claim: cash/analysis/purity_visitor.py:PurityVisitor._table_is_reachable_from_the_key @9018bbca, cash/decorator/global_values.py:GlobalValues.data_callable_identity @e1f4bd8f -->
A cached function that picks a function from a table caches correctly. The
table can be a dict of functions, a registry a `@register` decorator fills, a
dict of classes, or a `.get()` on any of them, and it can live at module level,
be built in the body (`handlers = {"a": fa, "b": fb}; handlers[name](x)`) or be
passed in (`router.table[name](x)`). A module-level table is read as a global,
a table built in the body names its functions there, and a table passed in is
part of an argument; each is keyed by the code of every function in it and the
helpers that code calls, so an edit to any of them recomputes:

```python
import cash

def momentum(prices):
    return [b - a for a, b in zip(prices, prices[1:])]

def mean_reversion(prices):
    mean = sum(prices) / len(prices)
    return [mean - p for p in prices]

STRATEGIES = {"momentum": momentum, "mean_reversion": mean_reversion}

@cash.cache
def signals_from_table(name, prices):
    return STRATEGIES[name](prices)

signals_from_table("momentum", [1, 2, 4])  # first call: runs
signals_from_table("momentum", [1, 2, 4])  # cache hit
```

The cost is that editing `mean_reversion` also recomputes every `momentum`
entry, and every hit hashes the whole table. For invalidation per strategy,
pass the function itself to the cached function, through a thin front that
is not cached:

<!-- test:skip reason="a function passed as an argument is keyed through its module, which the harness's exec namespace is not; this fence is run as a script, across an edit, by tests/docs/test_dispatch_table_recipe.py instead" -->
```python
import cash

@cash.cache
def _signals(strategy, prices):
    return strategy(prices)

def signals(name, prices):
    return _signals(STRATEGIES[name], prices)

signals("momentum", [1, 2, 4])        # runs
signals("mean_reversion", [1, 2, 4])  # runs
signals("momentum", [1, 2, 4])        # _signals hits
```

A function passed as an argument is keyed by its own code and the helpers it
calls, so editing `mean_reversion` keeps the `momentum` entries. A hit costs
about what a direct call to a cached strategy costs (around 0.7 ms with seven
small pandas strategies, against about 2 ms for the table form).

<!-- claim: cash/decorator/purity_checks.py:PurityChecks._warn_untrackable @f9cd7b17 -->
Looking the function up by a name at run time, with `getattr(module, name)()`
or `importlib.import_module(name)`, is not keyed. The
call is cached anyway, on a record of the code it ran that is checked before
each hit; when that record cannot be checked, the function
runs uncached with a [`KEY-UNTRACKABLE-DEP`](warnings.md#key-untrackable-dep)
warning. A table is always keyed.

## Code in installed packages

<!-- claim: cash/install_paths.py:in_own_package @168c5d1e, cash/install_paths.py:is_user_path @34c85fa5, cash/dependency_state.py:DependencyStateHasher.compute @1738c84b -->
cash follows the code a cached function calls, and puts it in the key, when
that code is yours. Yours means:

- a file outside the Python installation and outside any installed package
  (see the `--target` folders below): your project, or a package installed
  with `pip install -e` (an editable install runs from your source folder);
- the package the cached function itself is in, wherever it is installed. A
  cached function in `mytool.jobs` is keyed by the code it calls anywhere in
  `mytool`, even when `mytool` is installed with a plain `pip install`.

<!-- claim: cash/decorator/environment_fold.py:EnvironmentFold._fold_library_versions @9c6400f4 -->
Any other code in `site-packages` is a library, and that includes your team's
internal package when it is installed with a plain `pip install`. cash does not
read a library's code; it puts the library's installed version in the key. So
`pip install -U pandas` recomputes the calls that use pandas, and with the same
versions installed, a call hits as before.

<!-- claim: cash/install_paths.py:installed_outside_project @b6219f35, cash/install_paths.py:_record_tops @c35eade7 -->
A package installed into a plain folder is a library too: `pip install
--target DIR` or `uv pip install --target DIR`, with `DIR` put on the path by
`PYTHONPATH` or a `.pth` file. cash tells it from your code by the record the
installer leaves in `DIR` (a `*.dist-info` folder that lists the package's
files), and only when `DIR` is outside your project and outside the folder of
the script you run. A folder of source on `PYTHONPATH` with no such record,
say a checkout of a shared module, stays your code, and so does a package
installed into a folder inside your project. Its edits recompute.

Two kinds of library count:

- a library the function, its helpers or the cached functions it calls name
  in their code (`np` in `np.mean(x)`, `Ridge` imported from scikit-learn,
  `read_csv` imported from pandas), or whose object a module-level name they
  read holds (`MODEL = joblib.load(...)`);
- the library an argument comes from. `def summarise(df): return
  df.describe()` names no library, and runs pandas' code. The library that
  defines the class of each argument, of `self`, and of the objects they hold
  (a frame in a list, a model in your own object) counts, and so does the
  library of a function or class you pass as a value (`np.mean`).

<!-- claim: cash/decorator/library_versions.py:held_library_versions @9106d491 -->
With each goes every package that library requires, as its install metadata
lists them: a numpy upgrade under pandas recomputes a `pd.read_csv(...)`
result and a call that took a frame. A library's optional extras are left out.

A library the call reaches in neither way is not seen: one held deeper in
module-level data (a dict of models the function reads), or inside a list or
dict that a `frozen=True` function returned. Name a function of it in
`depends_on`, or pass the object as an argument.

<!-- claim: cash/decorator/library_versions.py:_files_digest @87224897 -->
With the version goes a digest of the files the install lists, from the hash
pip or uv writes for each in the `RECORD` file. So a team package rebuilt and
reinstalled under the same number (`pip install .` without a version bump, a
`git+` install of a branch that moved) recomputes, and the same files
reinstalled hit as before. An installer that writes no hashes leaves the
version alone in the key: then clear the affected entries (`f.cache_clear()`
or [`cash clear --function`](cli.md#cash-clear-path-all)), or tell cash what
to watch:

- **While you work on the package**, install it with `pip install -e`. Its
  code is then tracked like your project's.
- **`depends_on=[teamlib.score]`** puts the source of that function, as
  installed, in the key, so a change to `score` recomputes. Only that
  function's own code counts, not the functions it calls. Name each function
  whose change matters.

## Library settings cash cannot check

<!-- claim: cash/decorator/runtime.py:CallRunner._warn_settings_unseen @7feb3be9 -->
cash checks the library settings a call read
(`sklearn.set_config(...)`, `pd.set_option(...)`, `np.seterr(...)`, a
`ContextVar`) before it serves the entry. With `CASH_TRACE=0`,
for an `async` function, or for code a returned iterator runs, it cannot see
those reads: a call made after such a setting changed is served the result
made under the old one, also when the library is reached through an
argument (`df.to_string()`). cash says so once per process, for the first
call stored in such a case
([`KEY-LIBRARY-STATE-UNSEEN`](warnings.md#key-library-state-unseen)). Pass
the setting in as an argument, or clear the function's entries after
changing it. For an `async` function, the other way out is to move the part
that reads the setting into a sync function and cache that one.

## Reads cash cannot see

<!-- claim: cash/tracking/reader_patches.py:_patch_multiprocessing_pool @e9ba11bf, cash/tracking/reader_patches.py:_patch_process_pool_submit @89b877f9, cash/tracking/reader_patches.py:_patch_thread_start @1b59e2a1, cash/tracking/reader_patches.py:_NATIVE_READERS @41501bff -->
cash does not record a file opened by a subprocess, and on macOS one a C
extension opens itself (on Linux and Windows the [native
file hook](known-limitations.md#reads-through-a-loader-cash-cannot-see), on by
default, records those). On macOS a call of `cv2.imread`, `netCDF4.Dataset`,
DuckDB's file readers or GDAL through pyogrio, fiona or `osgeo.gdal` warns
[KEY-NATIVE-READ-UNSEEN](warnings.md#key-native-read-unseen) and runs every
time. Name such a file with `file_depends_on=`. A
`threading.Thread` the function starts on code of yours, a lambda or a nested
function included, reads as part of the call. A thread it starts that is still
running when the call returns keeps the result from being stored, also one it
starts on a library's function (`threading.Timer(0.5, lib.write).start()`)
([STORE-WORK-RUNNING](warnings.md#store-work-running)).

Reads in work the function hands to a pool are recorded: a
`ThreadPoolExecutor`, `ProcessPoolExecutor`, `multiprocessing.Pool` (or
`ThreadPool`), or joblib's default backend (`Parallel(n_jobs=...)`, a
scikit-learn `n_jobs=`). Each task then runs under a tracker of its own in the
worker: about 20 µs a task, which only shows for tiny tasks sent one at a time
(`pool.imap` with the default `chunksize=1`).

<!-- claim: cash/decorator/file_deps.py:FileDeps.track_scanned_files @afeb16b6 -->
A polars `LazyFrame` argument from `scan_csv` or `scan_parquet` names its file
by path, so cash records the files its plan scans as reads of the call, as if
the body had opened them: an edited file recomputes, and so does a new file
that matches a scanned glob. A remote source (`s3://...`) is checked as a
remote read of the call, by the object's ETag or version (a glob or prefix by
its listing), so a replaced or new object recomputes. That check finds the
store through the environment's settings, not the frame's `storage_options=`:
where it cannot read the object's state, the call runs again every time and
warns [REMOTE-STATE-UNREADABLE](warnings.md#remote-state-unreadable).

## An edit that keeps the size and timestamps

<!-- claim: cash/tracking/file_dep_snapshot.py:_unchanged_since_hashed @59019b2b, cash/tracking/written_in_place.py:is_written_in_place @90fefb25 -->
cash checks a file's size and timestamps first and reads its content only when
one of them moved. On Linux and macOS every write moves the inode change time,
so this never misses. On **Windows**, two kinds of edit move nothing: a write
whose modification time is put back afterwards (`os.utime`, `shutil.copystat`,
`robocopy /COPY:T`), and a write through a memory map
(`np.memmap(path, mode="r+")`) made in a process without cash. A file mapped
to write in a process that had imported cash is read at every check from
then on, so those writes are seen.

After such a write, touch the file (`Path(path).touch()`). cash then reads its
content, and if it changed, the calls that read it recompute.

## Relative paths and working directories

`open("data.csv")` from two working directories reads two different files under
one key, and so does a relative `file_depends_on="data.csv"`. Each switch
recomputes and replaces the other directory's entry. Build
paths from the project root or pass absolute paths.

## Results cash refuses to store

- A matplotlib `Figure` or `Axes` is never cached
  ([`CACHE-IDENTITY-COUPLED`](warnings.md#cache-identity-coupled)): a restored
  copy would become pyplot's "current figure" and `plt.savefig()` would save a
  blank image. Cache the data and draw from it.
- A result holding the `id()` of a live object, as a number or in text
  (`f"run-{id(x)}"`, a default `repr` such as `<Q object at 0x7f...>`), is
  not stored ([`STORE-IDENTITY-HELD`](warnings.md#store-identity-held)): a
  hit in another process would hand back the old process's address.
- A result that can't be pickled is not stored
  ([`STORE-FAILED`](warnings.md#store-failed)), unless the function is
  `frozen=True`.
<!-- claim: cash/decorator/runtime.py:CallRunner.arguments_left @487d71cc, cash/decorator/runtime.py:CallRunner._arguments_to_write @e4def210, cash/rewrite_in_place.py:refusal @abbcccec -->
<!-- claim: cash/decorator/runtime.py:CallRunner.arguments_before @50acf221, cash/decorator/arg_hashing.py:ContentKeyed.covers @4d5faf1a -->
- A call that changed its argument in place is stored with the state it left
  the argument in when the argument is a NumPy array (not of objects), a
  pandas frame or Series whose index, columns and dtypes it kept (a Series of
  a NumPy dtype, not pandas' `str` dtype), or a list, dict or set of numbers,
  strings and the like. A hit checks that the
  argument handed in is in the state the entry was stored from, then writes
  the stored state into that very object, so your other names for it and the
  views of an array see the change
  ([`CACHE-ARG-WRITTEN-BACK`](warnings.md#cache-arg-written-back) notes it).
  The cache key already holds such an
  argument by its content, so the first call is stored and the next call
  with the same values is a hit; nothing is hashed a second time for it.
  A hit writes a frame's columns as the call did: into the column's own
  array where the call wrote into it, as a new array where it gave the
  column one, so a name for a column's array (`h = df["a"].array`) sees
  what it sees after a plain run. Which of the two pandas does depends on
  whether another pandas object shares the column under copy-on-write
  (pandas 3), so there a hit is served only on a frame shared the way the
  stored one was; pandas 2 writes into the column however it is shared. cash's own hold on a
  frame it has hashed makes pandas copy a column before writing it: cash
  lets go of the frame before the body of a function seen changing its
  arguments, or whose own source changes this one. A first call that
  changes a frame in a helper runs while cash still holds it, and is
  stored from the next call on.
  (After a call that changed none of its arguments, cash stops looking at
  them before the body: a later call that starts changing one is not
  stored, the calls after it are.)
  When the key holds the argument some other way (a
  [registered hasher](tutorials/feature-guides/custom-hashers.md) for its
  type), the first call seen changing it is not stored: from then on, each
  miss fingerprints it before the body runs. A call keyed with `key=`, one whose
  result shares memory with the argument (`return X[:2]`), one whose changed
  argument is, is held by or shares memory with another argument
  (`f(a, a[::-1])`, `f(a, {"v": a[:2]})`) or a value the code the call ran
  reads (a global, read in a helper, a method or another module of yours),
  one where cash cannot tell (the code looks names up by a string, as
  `globals()["G"]` does, or the call is not recorded, as an `async`
  function's is not), and any other
  argument (a list of lists, a frame given a new column, an object of your
  class changed other than as the next item says) are not stored, so they
  run every time. Return a modified copy instead (`rows = sorted(rows)`).
  <!-- claim: cash/decorator/store.py:ResultStore.restore_identity @255cf26e -->
  A call that fills its `out=` argument and returns it (a body ending
  `np.multiply(x, 2, out=out); return out`) is stored the same way: a hit
  writes the stored state into the `out` you pass and returns that very
  object, so `result is out` holds
  ([`CACHE-ARG-WRITTEN-BACK`](warnings.md#cache-arg-written-back)).
<!-- claim: cash/decorator/attribute_replay.py:AttrVerdict.of @833f9c5b, cash/rewrite_in_place.py:attribute_refusal @0ad90168, cash/rewrite_in_place.py:AttrState.replay @0e4c9439, cash/decorator/attribute_replay.py:find @3277a98d, cash/decorator/runtime.py:_record_held @63feb88b -->
- A call that only binds, rebinds or deletes **attributes of an object it
  was handed** (`est.fit(X)` setting `self.coef_` and returning `self`, a
  resampler's `fit_resample` setting `self.sampling_strategy_`, or a
  `c.cache(est.fit)` whose `est` it changes so) is stored with those
  attributes, and a hit binds them on the object you hand in: it then holds
  the same attributes, in the same order, as after a plain run, and
  `return self` hands back your object. The binding may happen any number
  of calls deep (`resample(sampler, X, y)` calling `sampler.fit_resample`,
  whose helper binds `est.sampling_strategy_`), in code the key names or
  in code a call that picks code by name ran. It also covers an object
  the argument already held as an attribute before the call, one level
  down and no deeper: a NiMARE estimator's fit gives the kernel transformer
  it holds `memory_` and `_shelving` (nilearn's `CacheMixin._fit_cache`),
  and a hit binds them on the kernel transformer your estimator holds,
  the very object a plain run changes, so anything else that holds it sees
  the change too. That object is replayed by the same rules as the
  argument, and only while the argument still holds it under that one
  name after the call (an object it holds under two names, or that is
  also an argument, is not). This holds when:
    - the object is an instance of a class of yours or a library's written
      in Python, with no `__slots__`, no `__setattr__` or `__delattr__` of
      its own, and no property setter or other descriptor that runs code
      when an attribute is set (a setter whose body only binds plain
      attributes, `self._total = value` or `setattr(self, "_t", value)`,
      is fine);
    - the cache key holds the object by all it holds (not a
      `__cash_key__`, a registered hasher or a lineage tag of its own);
    - nothing else changed: no value the object already held was changed in
      place (`self.history.append(x)`, `self.model.partial_fit(...)` on a
      model that is not such an object, `self.cfg["k"] = 1`,
      `self.kernel.cache.append(x)`, or an attribute two levels down,
      `self.kernel.inner.k = 1`), and no other argument either (an array `X`
      written in place still keeps the call from being stored);
    - each value bound is a number, string or similar, one of the call's
      arguments or the object itself, a module global, a class or function,
      the value of another top-level attribute of an object it changes so
      that held it before the call and still holds it after (`self.a =
      self.b`, nibabel's `img._data_cache = np.asanyarray(img._dataobj)` on
      an in-memory image, `self.kernel.memory_ = self.kernel.memory`)
      (a hit binds that very object again, never a copy: for an attribute,
      what your object holds under that name when the hit is served, so
      `img._data_cache is img._dataobj` holds after it as after a plain
      run; an object held under several such names is bound while they all
      still hold one object, and the call runs again when they do not), or
      a new object the call made that nothing outside the call
      holds, nor anything inside it: a value the object held before under a
      name the call rebinds or deletes, something a
      module or class also holds, or a new object holding one of those, or
      holding an argument, keeps the call running every time. Two
      attributes bound to one new object, or an attribute that is also the
      result, come back as one object.
  Anything else runs every time, as before, and the warning says why a hit
  cannot replay it. Sharing cash cannot see stays a limit: an object
  written in C that holds references without telling the garbage collector
  is not looked into.
<!-- claim: cash/decorator/attribute_replay.py:also_changed @0c99ff5f, cash/decorator/runtime.py:CallRunner._source_changes @125fad66 -->
- A call that **writes into a container an argument holds** is not stored,
  and runs every time: a dict, list or array the object keeps as an
  attribute (`data.uns["total"] = ...`, `self.history.append(x)`, a new
  column in a frame it holds, a slot of an array in it). Binding the
  attribute is something a hit can replay; changing what it already held is
  not. Library functions that work in place by default and write their
  output into such containers fall under this. When the call also bound
  attributes, the warning and the miss reason name the change that kept a
  hit from replaying them: the line of your function
  (`self.history.append (line 12)`), or else the container
  (`an in-place change of the list self.history held`). A write into an
  attribute the call itself bound to a new object (`self.null_ = {}`, then
  `self.null_["k"] = v`) changed nothing you had, and is never named. Where the library
  offers a form that returns a new object instead (`copy=True`,
  `inplace=False`), use it, or copy first; the call is then cached:

    <!-- test:expect-warning -->
    ```python
    import copy
    from types import SimpleNamespace as Data  # any data object
    import cash

    @cash.cache
    def tag_in_place(d):
        d.uns["total"] = sum(d.values)  # writes into a dict d holds

    tag_in_place(Data(values=[1, 2], uns={}))  # cache miss, not stored
    tag_in_place(Data(values=[1, 2], uns={}))  # cache miss again

    @cash.cache
    def tag(d):
        out = copy.deepcopy(d)  # or copy=True / inplace=False
        out.uns["total"] = sum(out.values)
        return out

    d = tag(Data(values=[1, 2], uns={}))  # first call: stored
    d = tag(Data(values=[1, 2], uns={}))  # cache hit
    # test:inject: assert d.uns == {"total": 3}
    ```

<!-- claim: cash/backends/set_order.py:restore @d4e04f7d -->
- A set in a result comes back with the same items. With the
  [C extension](getting-started/installation.md#the-c-fast-path) up to CPython
  3.14, it also iterates in the order the stored set did, wherever
  the result holds it. Elsewhere a `for` loop may meet the items in another
  order than in the call that made it, and so may a set of strings in a new
  process, as in plain Python. Use `sorted(s)` where the order matters.
<!-- claim: cash/decorator/runtime.py:CallRunner._refuse_registries @631605a3 -->
- A call that changes a setting the process keeps outside any variable
  (`np.seterr`, a library logger's level, `warnings.simplefilter`,
  `os.environ`, `sys.path`) is not stored, and runs every time: a hit, or a
  new process, would leave the setting unmade
  ([`IMPURE-OBSERVED-EFFECTS`](warnings.md#impure-observed-effects)). Make
  the setting outside the cached function, or, when skipping it is fine,
  waive it where it is made: `# @cash:assume-safe` on its line or on the
  `def` line, or a `with cash.assume_safe():` block around it
  ([which settings, and the waivers' reach](how-it-works/decorator-path.md#side-effects)).

## Generators and async functions

A generator result is stored in chunks as you consume it, and nothing is stored
if you stop early; an infinite generator never finishes, so it never caches. See
[Iterators](tutorials/feature-guides/iterator-caching.md). `async def` functions
cache like sync ones, with one difference: the library settings their body
reads are not checked
([Library settings cash cannot check](#library-settings-cash-cannot-check)).
Async generators are returned undecorated with a warning.
See [Async functions](tutorials/feature-guides/async-caching.md).

## Edits inside a running process

<!-- claim: cash/loaded_code.py:code_objects_of @dd9b7b3c, cash/decorator/function_identity.py:warn_source_changed_since_load @767f9eca -->
cash keys a function by the code that is running. An edit saved while a
long-running process works is seen by the next process, or in this one once
the module is reloaded (`importlib.reload`, IPython's `%autoreload`): the
next call then recomputes. If a source file changes on disk after the process
loaded it (a deploy), cash keys by the running code and warns
[`KEY-SOURCE-CHANGED`](warnings.md#key-source-changed).

## Using a decorated function in a notebook

Decorated functions work in a notebook, but re-running the cell that defines
one creates a new wrapper with new counters, so `cache_info()` may read zero.
Use `explain()` there.

<!-- claim: cash/notebook/statement/store.py:StatementStore._with_decorator_ref @9edb344d -->
A call keeps the key it has in a script, so a notebook and a script that use
one cache folder share its entries. When cash also caches the notebook's
statements, a statement that only binds the call's result (`df = load(path)`)
stores a pointer to the function's entry, not a second copy of the value.
[Moving to a module](tutorials/feature-guides/production-transition.md) covers
the notebook side.

## Weak references to a result you dropped

Within one process, a hit may hand back the same object an earlier hit
returned, when nothing else holds any part of it and nothing in it changed
since; you cannot tell it from a fresh copy by its values. cash keeps that
object alive until the next hit of the same call, or until it evicts the entry.
So a `weakref.finalize` callback, a weak reference's callback, or a
`WeakValueDictionary` entry you attach to such a result fires when cash lets
it go, not at the moment you drop your last reference as in plain Python.
Classes with `__del__` are never kept this way.

A [`frozen=True`](decorator.md#frozen-and-large-arguments) function's hit
hands back the object an earlier hit returned even while you still hold it,
as long as cash can prove it unchanged; cash keeps it until it evicts the
entry or the object changes.

## Related

- [The `@cash.cache` guide](decorator.md): where results go, what invalidates
  an entry, and the parameters.
- [Custom hashers](tutorials/feature-guides/custom-hashers.md): identify your
  own types in the key.
- [File dependencies](tutorials/feature-guides/custom-file-sources.md): which
  file reads are tracked, and how to name the others.
- [Seeing what cash did](decorator.md#seeing-what-cash-did): find out why a
  call missed.
