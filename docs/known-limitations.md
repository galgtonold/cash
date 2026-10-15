# Writing cache-safe cells

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. For limitations of `@cash.cache` functions, see the
    [decorator guide](decorator.md).

cash restores a cell's inputs by working out what each statement reads and
writes. Where a change travels through a channel cash does not see, you can get a
value that looks right and is stale. This page lists those cases: the symptom,
an example and the fix.

Most of them need an **isolated re-run**, meaning you run one cell on its own. A
top-to-bottom **Run All** executes every cell in order, so cash has nothing to
reconstruct and these cases do not arise. Entries that also affect Run All say so.

You do not need to read it all. Follow the checklist below, and come back to a
section when a cell does something you did not expect.

## Checklist

- Seed your random draws, or mark the ones that must change `# @cash:no-cache`.
  See [Randomness](#randomness).
- Rebind instead of changing an earlier cell's object in place:
  `df = df.assign(...)`, not `df["c"] = ...`.
- Pass the time into helpers instead of reading the clock inside them.
- Read files through the usual readers (`pd.read_*`, `np.load`, `open`).
- Define a function above the functions that call it.
- Save the notebook before running a cell, unless your editor has a live reader
  (see [Editing without saving](#editing-without-saving)).

## Randomness

This is the one home for randomness in notebooks; other pages link here.

<!-- claim: cash/notebook/statement/restore.py:StatementRestorer.restore_from_cache @14093389, cash/tracking/randomness/state.py:restore_rng_state @2e1cc6af, cash/tracking/randomness/state.py:capture_rng_state @421bfe05, cash/notebook/statement/carrier_advances.py:advance_carriers @4ebbcaa0 -->
**Symptom:** re-running a cell returns the same random numbers.

An unseeded draw is cached like any other value, so a re-run shows the stored
result. A cheap draw that is never cached is frozen too if it comes from the
`random`, `numpy.random` or `torch` stream: before a cell re-runs, cash rewinds
those streams to where the cell started, so a re-run lands where a top-to-bottom
run would. A generator held in a variable is not rewound but rebuilt: before a
draw from it re-runs, cash runs or restores the cells that created it and drew
from it (see below), so a cheap draw from an unseeded one
(`rng = np.random.default_rng()`) changes on every run. cash warns
`RANDOM-UNSEEDED` when an unseeded draw first runs (not in a `no-cache`
statement), and `RANDOM-REPLAYED` when an unseeded value comes back from the
cache. An estimator fitted with `random_state=None` counts as an unseeded draw,
unless its own settings rule a draw out: `Ridge()` uses the seed only with
`solver='sag'` or `'saga'`, `LogisticRegression()` only with `'sag'`, `'saga'` or
`'liblinear'`, `Lasso()` only with `selection='random'`, `SVC()` only with
`probability=True`.

| You want | Do this |
|---|---|
| A new draw every run | `# @cash:no-cache` on the statement, on the line above it or at the end of its line. The draw then continues the stream from where it stood before cash rewound it, so it changes every run, also below a frozen draw. |
| The same result every run | Seed it: `np.random.seed(0)`, `np.random.default_rng(0)`, `random_state=0`. |
| No warning | `# @cash:allow-random`. It changes the warning only, never what is cached or rewound. |

<!-- claim: cash/tracking/randomness/detect.py:RNG_CARRIER_CONSTRUCTORS @620106b9, cash/tracking/randomness/state.py:capture_object_rng_states @d8dd9223 -->
Seeding counts per source. `np.random.seed(0)` covers `np.random.*` draws for
the rest of the session, but not `random.random()` and not a generator from
`np.random.default_rng()`, which you seed in its constructor. A cache hit puts
the random stream back where the computed run left it, so the next draw matches a
full run.

cash finds a draw by following the generator from where it was created, so these
are **not** flagged and are cached silently:

- a generator cash never saw created: `self.rng.normal()`, a generator returned
  by a helper, `df.sample(100)` without `random_state`;
- a draw inside a helper function (`arr = make_data()`) from a generator the
  helper makes or holds. One from the global `random`, `np.random` or torch
  stream is flagged once the statement has run, by comparing the streams;
- an anonymous generator: `z = np.random.default_rng().normal(size=3)`.

<!-- claim: cash/tracking/randomness/state.py:capture_rng_state @421bfe05, cash/notebook/statement/carrier_advances.py:advance_carriers @4ebbcaa0, cash/tracking/randomness/lineage.py:advanced_carrier_lineage @eaa66fca, cash/notebook/upstream/statement_lineage.py:StatementLineage._recorded_draws @3b7507d9, cash/notebook/upstream/statement_lineage.py:StatementLineage._reachable_generators @bb6221ba -->
A statement that draws from a generator held in a variable, or calls a function
that does (also one the generator is handed to, as in `paths =
simulate(n, rng)`), counts as changing that variable. It is stored like any
other statement, with the generator's state after it, and a cache hit puts the
generator in that state. With `rng = np.random.default_rng(0)` in one cell
and draws from `rng` in the next two, re-running the third cell alone first runs or restores the two above it, so it
draws where a top-to-bottom run does, also after a restart and when a draw
above it was too cheap to store. So running a draw cell twice gives the same
numbers both times, where plain Jupyter would draw new ones. A stored draw from such a generator is
kept apart from one made with it in another position, so a reseed or a restart
does not bring back a value drawn elsewhere in its stream. Four gaps on an
isolated re-run:

- **A generator held by an object.** One reached only through an attribute
  (`self.rng.normal()`, `model.rng`) is not followed: re-running a later draw
  alone draws from wherever it is now. Draw from such a generator in the cell
  that creates it.
- **An edited seed cell you did not re-run.** Edit `np.random.seed(0)` to
  `seed(1)` and run only a later draw: the draw does not see the new seed,
  because a bare `seed()` binds no variable. Re-run the seed cell after editing
  it, or seed in the cell that draws.
- **A cache kept only in memory, after a restart.** Which generator a cheap
  draw moved is noted on disk for the next kernel. Without a disk tier that
  note is gone, and cash recognises the generator only by the call that made
  it (`default_rng`, `RandomState`, `random.Random`, ...). With `rng =
  make_rng(0)`, your own function, re-running a later draw alone after a
  restart skips the cheap draws above it. Run the cells above it first.
- **TensorFlow.** `tf.random.*` draws are flagged, but TensorFlow's stream cannot
  be saved and put back, so a cache hit leaves it where it was. Put
  `# @cash:no-cache` above such draws when later draws must match a clean run.

To silence the warning everywhere, filter it:

```python
import warnings
import cash

warnings.filterwarnings("ignore", category=cash.CashRandomnessWarning)
```

## A helper that reads the clock is frozen

**Symptom:** a timestamp, `uuid4()` or random id from your own function never
changes.

A clock read written in the statement itself (`t = datetime.now()`) runs every
time, and so does a call to a function of yours that only returns one
(`def tick(): return time.time()`). One inside a function that does more is
not seen, and the call is cached with the first value:

<!-- test:skip reason="illustrative: shows a value frozen across two runs of one cell" -->
```python { .nb-cell }
def stamp(x):
    time.sleep(0.2)
    return x, time.time()

# re-run: the same time.time() as the first run, no warning
s = stamp(1)
```

**Fix:** pass the time in (`stamp(1, now=time.time())`), or put
`# @cash:no-cache` on the line above the statement.

## Changes cash cannot see

Each of these misses a change, or applies one a second time.

### Mutating through an alias

<!-- claim: cash/notebook/statement/derivation_edges.py:record_shared_object_edges @1d3cdc8d, cash/notebook/statement/derivation_edges.py:bump_derived_lineages @150acf16, cash/notebook/shared_objects.py:share_group @6b5f6133, cash/notebook/shared_objects.py:view_base_names @aea7d284, cash/notebook/statement/processor.py:StatementProcessor.record_structure_share_edges @4fe4d23f -->
When a statement leaves two names sharing an object, cash links them:

- an alias: `log = history`;
- a container holding another variable's object: `config = {"features":
  features}`, `t = (lst,)`, `b.ref = x`;
- a part taken out of one: `train = data["train"]`;
- a NumPy view and the array it views: `head = big[:5]`.

A change made later through either name moves both, so the cells that read
the other one run again, on Run All and on a cell re-run alike. The links
also hold:

- when something that is not a variable holds the object too (a function
  default `def pick(row, cols=features)`, a class attribute, an
  `argparse.Namespace` or a `deque`). The class or function holding it is
  linked as well, so `slow(Registry.items)` runs again when `rows.append(3)`
  changes a list the registry holds;
- for names a loop, `if` or `try` block makes share an object
  (`for name, rows in pairs: splits[name] = rows`), from the end of the block;
- after a kernel restart: each statement keeps what it linked. A statement
  edited since it last ran links what its text says (`d2 = df`, `y = x[0]`,
  `runs = [cfg]`) until it runs.

A loop that changes the items it walks (`for r in rows: r.append(0)`) changes
`rows`. A pyplot call that draws on the current figure (`plt.plot(...)`,
`plt.title(...)`) changes the names bound to that figure and its axes, or to a
list, tuple or dict holding them (`figs = {"main": plt.figure()}`); such a
call runs every time.

A change made through an object cash cannot link to a name is invisible, and
the cells that read the name keep their cached results:

<!-- test:skip reason="illustrative: alias-mutation shapes, need isolated cell re-runs" -->
```python { .nb-cell }
get = lambda: x
get().append(5)        # changes x through a closure
pick.__defaults__[0].append("vat")  # changes features via the default
```

**Fix:** change the object through its own name (`x.append(5)`), or rebind
(`x = x + [5]`).

<!-- claim: cash/notebook/upstream/stale_values.py:StaleValueGuard._kept_for_partner_below @bebe6e19 -->
A cell that changes a value in place (`history.append(x)`) and is re-run on its
own normally starts from the value as the cells above leave it, so the change
is not applied twice. When a cell below has made another name share that value
(`log = history`), rebuilding it would leave the two names with different
objects, so cash keeps the value as it is, applies the change on top of the
last one, as a plain re-run does, and says so with
[NOTEBOOK-SHARED-KEPT](warnings.md#notebook-shared-kept).

### Mutating an object created in an earlier cell

<!-- test:skip reason="illustrative: contrasts in-place mutation with rebinding across cells" -->
```python { .nb-cell }
# df came from an earlier cell: runs every time
df["score"] = expensive(df)
# cached
df = df.assign(score=expensive(df))
```

A statement that changes an object made in an earlier cell runs every time. The
same statement on an object made in the same cell caches normally. When
another variable holds that object too (`d = dfs[0]`, `b = a`), directly or in
a list, dict, `deque`, `UserDict`, object array or attribute of one (a
`__slots__` attribute too), or the result holds an object that already
existed (`models = {"m": m}`), those variables are stored with the statement
and a hit restores them as one object, as running it would leave them. When
something else holds it -- a closure, a library's module or registry, a pandas
`.attrs` (pandas copies it apart) -- or the statement is in a loop body
(`for d in dfs:`), a restored copy would not be that object, so the statement
runs every time. IPython's output history
(`Out`, `_`) holding a value you displayed is not such a holder. **Fix:**
rebind with `df = df.assign(...)`.

### Mutating global state inside a function

<!-- claim: cash/analysis/callee_effects.py:callee_global_mutations @0563cfbf -->
A function that changes a global (`LOG.append(v)`, `counter["n"] += 1`), itself
or through a helper it calls, is handled: the statement calling it runs every time so the change really happens,
while the call inside it is served from the cache together with its effect on
the global. Nothing to do. If you would rather not rely on this, pass the state
in and return it.

<!-- claim: cash/analysis/handed_callables.py:handed_callables @c55bd405, cash/analysis/handed_callables.py:hands_a_state_changing_callable @4c3c2d42 -->
### A function handed to a call

A function you hand to another call counts as called. One that only reads
keeps the statement cached: a pure lambda in `df.assign(w=lambda d: d.v * 2)`,
`s.apply(f)` or `groupby("g")["v"].transform(lambda s: s - s.mean())` is
fine. One that changes state makes the statement run every time, as calling it
by name would.

cash finds the function however it is handed:

- by name: `s.apply(f)`, `map(f, rows)`;
- picked by an expression: `ops["dbl"]`, `ops.get("dbl")`, `ops[key]`,
  `f if flag else g`, `getattr(tracker, "record")`;
- as a method that changes its object (`s.apply(tracker.record)`), an object
  whose class defines `__call__` and changes it, or a closure that changes
  what it closes over;
- called from a lambda: `s.map(lambda v: tally.record(v))`;
- in a list, dict, deque or object of your class that a function of yours
  calls into: `run_steps(steps, data)` with `steps = [f]`, `run(pipe, data)`
  calling `pipe.steps`.

The same holds for code of yours that the statement runs without naming it:

- a function picked from a list or dict it loops over (`[f() for f in fns]`,
  `{k: f(x) for k, f in table.items()}`);
- a callback an object of yours holds (`bus.emit()`, also from a helper that
  reads `bus` as a global), a `functools.partial`, a lambda default, a
  function passed in `**kwargs`, or a logging handler of your class;
- a method called on an object or a class (`store.process(r)`, a static or
  class method, a hook reached through `super()`), a property, or a context
  manager's `__enter__` and `__exit__` (`with timer("fit"):`);
- the special methods of an object of your class the statement names
  (`bag["k"]`, `len(bag)`, `bag + 1`, `f"{bag}"`, iterating it);
- the function that a wrapper made without `functools.wraps`, or a
  `compose(f, g)` closure, calls; a `functools.partial`, `np.vectorize` or
  `operator.methodcaller` called by its own name; a `functools.singledispatch`
  implementation.

When such a function changes state, the statement runs every time, and the
call of yours that is handed it is not served from the cache; nor is a
`@cash.cache` function handed one
([`IMPURE-HANDED-CALLABLE`](warnings.md#impure-handed-callable)). A method
counts as changing its object when its code changes it, or when it calls a
method on something the object holds that cash does not know to leave it
unchanged (`self.model.fit(x)`).

Two cases run uncached, with the reason on the badge: a function of yours
whose source cannot be read, and one that only running the code can tell
(`s.apply(make_scorer(2))`, the result of a call of yours). A list or dict of
functions handed to a library function is looked into only for `agg`,
`aggregate` and `transform`, or when the statement hands that library
function a function that calls what it is given (`map(lambda f: f(3), fns)`,
`functools.reduce`).

### Setting state on one of your modules

<!-- claim: cash/notebook/statement/mutation_routing.py:MutationRouting.route @b4d452c3, cash/notebook/callee_reach.py:module_state_writes @67827b2e, cash/notebook/call_unit.py:_rebound_unwatched @2d452d22 -->
A statement that sets state on one of your modules runs every time, so that
after a restart, or an edit of the module's file, the module holds what a
top-to-bottom run leaves in it. That covers `mylib.K = slow()`,
`importlib.reload(mylib)`, and a call that changes data the module keeps,
such as `metrics.increment(5)` adding to a counter in `metrics.py`: called
directly, through a helper, or handed to a call (`s.apply(metrics.record)`,
or `s.map(record)` after `from metrics import record`).

A setting the code does not spell out (`globals()[name] = v`, `global K` in a
method of one of your classes) is seen when it runs, in the modules the
statement reaches. It counts the same from then on, in a later kernel too.

Inside an assignment, a slow call is still served from the cache:
`n = metrics.increment(5)` comes back with the globals its code says it
writes put back. A call seen rebinding a global its code does not mention
runs every time, and a cell that is only the call runs it. What Python itself
adds to a module (the registry its first warning creates) is not a setting.

### A setting on your module or the environment, run out of order

<!-- claim: cash/notebook/cache_key.py:_recorded_reads_component @f1b01bbf, cash/notebook/recorded_reads.py:choose @515ac169 -->
A cell that sets data on one of your modules (`mylib.K = 7`,
`mylib.CONFIG["k"] = 7`, `mylib.set_k(7)`) or an environment variable
(`os.environ["MODE"] = "b"`) reaches the statements that read it, in your
functions too, when they run: run the notebook. A statement is keyed on the
values it saw when it last ran, so a setting a cell below it makes does not
reach it. Edit a setting, run it or not, and run only a cell below the
statements that read it, and cash answers as a plain kernel would, with what
they read when they ran, not as a top-to-bottom run would. **Fix:** run the
notebook, or the cells from the edited one down.

<!-- claim: cash/notebook/recorded_reads.py:_HASHED_IN_PLACE_BELOW == 4194304, cash/notebook/recorded_reads.py:outside_changes @c51f572b -->
A change made outside the notebook's cells is seen: a variable the shell or
a launcher sets, a change from a console attached to the kernel, an edit of
the module's file. Running only the last cell rebuilds what was built on the
old value. A change a cell makes is the notebook's own, however the cell makes
it: itself, in a function it calls (`setup()` doing
`os.environ["MODE"] = "b"`), with a magic (`%env`, `%cd`), or by reloading
the module.

To keep this cheap, cash hashes your module's data to look for an outside
change only before a cell that reads it or uses something built from it, and
around a statement that reaches the module or names the value. A cell that
does neither pays nothing. A big value (a few MB or more, such as a table the
module loads) is hashed only before a cell that reads it. So a cell that only
uses what the reader built sees the table rebound from outside
(`mylib.TABLE = load()`), but not changed in place (`mylib.TABLE[0] = 5` in a
console): it answers as a plain kernel would, from what the reader built, and
the next cell that reads the table sees the change. **Fix:** run the
notebook, or the reader and the cells below it.

Two details. A statement that changes the data in place through something
else holding it (`holder["t"][0] = 5`, after `holder["t"] = mylib.TABLE`)
counts as a change from outside, so the cells after it run again rather than
reuse an answer. And when a cell above the reader sets the same module value
(`mylib.K = 3`), that cell's value is the one the reader gets, as in a
top-to-bottom run: the setting runs again over the outside value.

### Re-running a cell above an in-place change

<!-- test:skip reason="illustrative: needs an isolated re-run of a cell above the mutation" -->
```python { .nb-cell }
s = pd.Series(np.arange(200_000, dtype=float))  # cell 1
out = summarize(s)  # cell 2: re-run this alone...
s.iloc[0] = 1e9     # cell 3: ...and it does not see this
```

cash answers cell 2 as a top-to-bottom run would, so it computes with `0.0`
while the live `s` holds `1e9`. **Fix:** move the change above the cells that
must see it, or make it a new value (`s = s.copy(); s.iloc[0] = 1e9`).

### Changing and rebinding the same name in one cell

<!-- test:skip reason="illustrative: mutate-then-reassign raises on isolated re-run" -->
```python { .nb-cell }
df["c"] = df["a"] * 2
df = df.rename(columns={"a": "x"})
```

An isolated re-run raises `KeyError`, because it reads the renamed frame.
**Fix:** split the two statements into two cells.

### Class variables changed from two cells

<!-- test:skip reason="illustrative: cross-cell class-variable accumulator" -->
```python { .nb-cell }
class Widget:          # cell 1
    count = 0
    def __init__(self):
        Widget.count += 1
w0 = Widget()

w = Widget()           # cell 2: each re-run pushes count higher
```

This needs the counter to be changed from both cells; one cell is fine.
**Fix:** keep the counter in a variable, not on the class.

### Writing to a file opened in an earlier cell

<!-- claim: cash/notebook/consumables.py:is_write_stream @90c4d2aa -->
A re-run of a cell that writes through a file opened in an earlier cell writes
again, as in plain Jupyter. cash never opens the file a second time to rebuild
the handle: that would write the earlier lines again too, and a second gzip
writer over the same file corrupts it.

<!-- test:skip reason="illustrative: needs an isolated re-run of a writer cell" -->
```python { .nb-cell }
log = open("run.log", "a")   # cell 1
print("start", file=log)     # cell 2
print("end", file=log)       # cell 3: re-run alone, "end" twice
```

**Fix:** open, write and close the file in one cell
(`with open("run.log", "a") as log: ...`).

### A function that calls one defined in a later cell

<!-- test:skip reason="illustrative: shows staleness across an edit of a later cell" -->
```python { .nb-cell }
def a(n): return b(n) * 2      # cell 1: b does not exist yet
def b(n): return n + 1         # cell 2
r = a(3)                       # cell 3: 8

# edit cell 2 to `return n + 10`, re-run cell 3 alone: still 8, not 26
```

cash follows dependencies upward, so `a` never learns it depends on `b`. cash can
even serve `r` after cell 2 is deleted. **Fix:** define a function above the
functions that call it. A module-level read of a name bound only below raises
[`ForwardReferenceError`](#forwardreferenceerror) instead.

### A name a magic binds

<!-- claim: cash/analysis/code_analyzer.py:_cell_magic_body @3566e904, cash/analysis/code_analyzer.py:_is_magic_line @d120bf9b, cash/notebook/statement/processor.py:StatementProcessor.forget_rebound @70d43250 -->
cash reads a cell's Python, not what IPython makes of its magics. A magic or a
shell command runs every time, uncached, and cash never runs one for you.
<!-- claim: cash/notebook/magic_effects.py:magic_effects @420435ff, cash/notebook/upstream/simulator.py:NotebookSimulator._warn_stale_magic @777a3665, cash/notebook/magic_effects.py:is_rerun_magic @17c29890, cash/notebook/statement/mutations.py:MutationClassifier.magic_snapshots @24994a9f, cash/analysis/code_analyzer.py:magic_python @1a6291b3 -->
The names it binds or changes (`files = !ls`, `t = %time f()`,
`%time x = f()`, the receiver of `%time model.fit()`, an argument
`%time train(model)` was seen changing in place) keep the value it left,
and a statement below that reads them is keyed on that value. They are never
rebuilt from the Python above the magic alone, which would drop what it did.
A `%time`, `%timeit` or `%prun` line runs a Python statement, so a rebuild
runs it again after the Python above it, as a run from the top does: after an
edit above `model = M(k)` and `%time model.fit()`, or a restart, the cell
below gets the model fitted on the new `k`. A `%timeit` rebuilt this way times
its statement again. In a loop or `if` body, such a line reads and changes
what its statement does: `for i in r: %time acc.append(i * k)` changes `acc`. A shell command or any other magic is never run for you:
after an edit to a cell it reads, or a restart, cash keeps the value the name
has (or leaves it undefined) and warns
([NOTEBOOK-MAGIC-STALE](warnings.md#notebook-magic-stale)). A name another
magic binds (`%run`, `%store -r`, `%%capture out`, `%%bash --out o`) has no
producer cash knows of: a statement that reads it runs uncached. The body of a
cell magic other than `%%time`, `%%capture` and `%%prun` is not Python in the
notebook's namespace (`%%writefile`, `%%script`, `%%timeit`, `%%bash`) or runs
under a debugger (`%%debug`), and cash never runs it. **Fix:** re-run the
command's cell after such an edit.

<!-- claim: cash/analysis/code_analyzer.py:exec_literal @1748a9a7 -->
The same holds for `exec(code)` when `code` is not a string literal: cash reads
`exec("w = base * 2")` as the assignment it runs, but not text built at run
time. **Fix:** write the assignment out, or re-run that cell after an edit above
it.

### A function reached by a name spelled at run time

<!-- claim: cash/notebook/string_lookups.py:code_looks_up_by_string @bb240f70, cash/notebook/cache_key.py:unkeyed_reach @f4481b35 -->
A statement's key holds the functions its code names and the globals they
read. Code that finds a function through a string instead
(`globals()["make"](3)`, `eval("make(3)")`, `vars()[name]`,
`getattr(sys.modules[__name__], name)`, a frame's `f_globals`) names no
function, so neither `make` nor what it reads can be in the key. Such a
statement runs every time, uncached, and the badge says why: when the
statement does it itself, and when a function of yours it calls does, at any
depth, in a cell or in a module of your own. `getattr(obj, name)` on an
object, `vars(self)` and a library's own lookups (`namedtuple`) do not count.
**Fix:** call the function by its name, or pass it in as an argument.

### Background threads

A thread that changes data after its cell has finished is outside cash's view.
An earlier cell re-run can see the changed data.

### A loop variable changed before it is read

<!-- claim: cash/notebook/control_structures/for_handler.py:ForLoopHandler._process_one_iteration @5304abe8 -->
This one can give a wrong answer on the first Run All. cash keys each iteration
on the loop variable's value when the `for` binds it. A body that changes that
value, or computes a new body-local variable, before the cached work is not seen:

<!-- test:skip reason="illustrative: pull() stands in for a slow call keyed only by the loop variable" -->
```python { .nb-cell }
for q in [[1], [1]]:           # two iterations, equal when bound
    q.append(len(accm))        # now different: too late for the key
    accm.append(pull(handle))  # 2nd iteration gets the 1st's result
```

**Fix:** make the distinguishing value part of the `for` target:
`for i, base in enumerate(items):`. Only names bound by the `for` statement reach
the per-iteration key. If there is no such value, put `# @cash:no-cache` above
the statement.

## Files

### Reads through a loader cash cannot see

<!-- claim: cash/tracking/reader_patches.py:_install_module_patches @4f31fa44 -->
**Symptom:** you changed a file and the cell still shows the old data, with a
`CACHED` badge and no warning.

cash records a file when it is read through a reader it watches: `pd.read_*`,
`np.load`, `joblib.load`, polars, `sqlite3.connect`, `open()` and
[others](how-it-works/invalidation.md#what-counts-as-a-change). A read through
anything else (a C extension that opens the file itself, a client library, a
subprocess, `os.open`) is invisible, and the statement is cached with no file
recorded. So is a watched reader imported by name in a cell
(`from polars import read_parquet`): call it through its module
(`pl.read_parquet(...)`). A SQLite database in WAL mode is the same: a commit lands in the
`-wal` file.

**Fix:** read the file through a watched reader, or put `# @cash:no-cache` above
the statement if the read is cheap.

<!-- claim: cash/tracking/native_reads.py:close_window @0e2917f4 -->
On Linux (glibc, x86-64 or aarch64) you can also turn on the native file
hook with `CASH_NATIVE_FILE_HOOK=1` (off by default; `%cash_status` shows it as
`native_file_hook`). cash then also sees the files a C, C++ or Rust library
opens itself through the C library's `open`, `fopen` or `opendir`:
`pyarrow.fs`, a database `ATTACH`-ed in SQLite, a compiled extension of your
own. It still misses a library that makes the system call itself, one linked
statically against the C library, and a library first loaded inside the
cached call. A file such a library opens while several cached calls run at
once, in different threads, counts for all of them. On macOS and Windows the
hook is not available yet.

### A file read into a memo before cash was imported

<!-- claim: cash/__init__.py:_watch_reads_from_import @072f3639 -->
**Symptom:** a cached function gets its config from a loader memoised with
`functools.lru_cache`, you edit the config file, and the function returns
the old result with no warning.

cash watches file reads from `import cash` on. A memoised loader that first
ran before that (it was imported, and called, ahead of `cash`) read the file
where cash could not see it, so later calls get the memo and record no file.
A loader first called after `import cash` is recorded, even when that call
came from outside any cached function.

**Fix:** import `cash` before the modules that load config, or pass the path
as an argument (`load_cfg("config.yaml")` inside the cached function).

### An edit that keeps the size and timestamps

<!-- claim: cash/tracking/file_dep_snapshot.py:_unchanged_since_hashed @809a68f2, cash/tracking/file_dep_snapshot.py:_HASH_MEMO_MIN_AGE_SECONDS == 10.0, cash/notebook/upstream/virtual_lineage.py:VirtualLineage._check_cell_file_deps @eeff0da9 -->
cash checks a file's size and timestamps first and reads its content only when
one of them moved. On Linux and macOS every write moves the inode change time,
so this never misses. On **Windows**, two kinds of edit move nothing: a write
whose modification time is put back afterwards (`os.utime`, `shutil.copystat`,
`robocopy /COPY:T`), and a write through `np.memmap(path, mode="r+")`.

<!-- claim: cash/tracking/file_dep_snapshot.py:_remembered_digest @7327a95f, cash/tracking/file_dep_snapshot.py:_HASH_MEMO_TTL_SECONDS == None -->
Such an edit stays unseen for as long as the kernel runs, not only for a
moment: once cash has read a file's content, it reuses that digest while the
size and timestamps match. Restarting does not help either, since the digest
is also kept in the cache folder for the next process.

**Fix:** after such a write, touch the file (`Path(path).touch()`). cash then
reads its content, and if it changed, everything that read it runs again.

## Loops

### A long `for`-append loop can stop caching

<!-- claim: cash/notebook/control_structures/single_unit_policy.py:should_run_as_single_unit @54928083, cash/notebook/control_structures/single_unit_policy.py:MIN_ITERATIONS_FOR_SINGLE_UNIT == 50, cash/notebook/control_structures/single_unit_policy.py:PER_STMT_OVERHEAD_SEC == 0.008, cash/notebook/control_structures/single_unit_policy.py:MIN_OVERHEAD_SEC == 1.0, cash/notebook/control_structures/single_unit_policy.py:ASSUMED_INNER_ITERATIONS == 10 -->
cash caches a `for` loop per iteration. A long loop is run as one unit instead
when all three hold: more than about 50 iterations of known length (a loop
inside the body counts each iteration ten times, as it usually runs about that
often per pass), per-statement bookkeeping estimated above one second (about
8 ms per statement per iteration, a statement in an inner loop ten times over),
and no file write, move or delete written in the loop body. A loop that only
reads files qualifies: the unit depends on every file and folder it read, so
an edited, added or deleted file runs it again. A loop that appends to a list
is an in-place change, so as one unit it is not cached at all. The badge row
then shows `In-place mutation on: out`. A shorter loop runs pass by pass:
each `append` runs, and the expensive call inside it comes from the cache.

<!-- claim: cash/notebook/heavy_loops.py:is_heavy @908c4eba, cash/notebook/heavy_loops.py:HEAVY_FACTOR == 4, cash/notebook/heavy_loops.py:MIN_HEAVY_ITER_SEC == 0.01, cash/notebook/heavy_loops.py:header_identity @ba179718, cash/notebook/control_structures/for_handler.py:ForLoopHandler._prefers_iterations @4510a192 -->
That estimate is a guess made before the loop has run. A loop run as one unit
is measured while it runs, and when one iteration's work takes at least four
times its estimated bookkeeping, and at least 10 ms, the next runs cache it
per iteration again. As one unit, raising its `range` or editing the last
statement of its body runs every iteration again; per iteration, an edit runs
only the iterations it changes or adds. The measurement is kept in the cache
folder under the loop's header with its numbers left out, so it carries over
an edit of the body or of the loop's length (`range(100)` to `range(180)`).
The first edit after the measuring run still runs every iteration, since that
run writes the per-iteration entries; edits after it reuse them. Run again
unchanged, the loop is restored whole from its unit entry. A loop kept per
iteration is measured again, and goes back to one unit once its iterations
have become cheap.

<!-- claim: cash/notebook/control_structures/single_unit_policy.py:iterations_for_one_unit @9bb4425c, cash/notebook/control_structures/single_unit_policy.py:header_names_the_iterator @c82c885c -->
A loop over a named iterator of unknown length (`for row in parsed:` after
`parsed = filter(...)`, or over a generator) runs its first passes one by one
and, once it reaches the length that would have qualified, the rest as one
unit drawing from the same iterator. Items are drawn exactly as plain Python
draws them, so a generator's own side effects interleave with the body's, and
an error leaves the iterator where Python would. Such a unit reads a stream,
so it is never served from the cache. A loop over a call whose value has no
length (`for x in make_rows():` with a generator function) still runs pass by
pass. **Fix** when that is slow: bind it first (`rows = make_rows()`, then
`for x in rows:`). An array built from a list (`enumerate(np.array(lines))`)
counts as long as the list.

<!-- claim: cash/notebook/control_structures/single_unit_policy.py:header_may_be_left_to_the_unit @a78c2c65, cash/notebook/control_structures/processor.py:ControlStructureProcessor.execute_as_single_unit @9f65ba05 -->
A loop's header is evaluated once, as plain Python evaluates it, so the loop
gets the same items as without cash: one random draw
(`np.random.permutation(n)`, `df.sample(n=300).index`), a queue drained once,
a property read once. A header built only by builtins or a progress bar over
plain data (`range(n)`, `tqdm(list(zip(a, b)))`) is evaluated by the unit, so
a cache hit skips it; any other header is evaluated before the loop runs, on a
hit too.

The expensive call inside the loop body (`fetch(e)` in `out.append(fetch(e))`)
is still cached, so usually there is nothing to do. **Fix** when it is not:
build the list with a comprehension, which is cached at any length.

<!-- test:skip reason="illustrative: the comprehension rewrite of an append loop" -->
```python { .nb-cell }
out = [fetch(e) for e in entities]
```

### The first run of a loop hashes what it hands to a library

<!-- claim: cash/notebook/loop_watch.py:LoopWatch.start @31a0eadb, cash/analysis/namespace_effects.py:rewatch_exempt @73357899 -->
cash cannot tell from the text whether a library function changes what it is
handed (`heapq.heappush(h, x)` does, `np.mean(a)` does not), nor whether a
function of yours that hands its argument or a global to one changes it
(`def push(h, x): heapq.heappush(h, x)`). So the first time a version of a
loop, a `with` block or a statement runs, cash hashes each such object before
and after it, about 1 s per GB, and records which ones changed. Later runs,
hits and restarts use that record. Editing the code, or splitting a long loop
into a head and a rest, pays the cost once more. While a version is being
learned, the loop runs rather than being served from the cache.

Not hashed: builtins that only read (`len`, `sum`, `print`), methods known to
only read, and functions of yours that neither change their arguments nor hand
them on.

Two catches:

- A run that changed nothing (a loop over no items yet, a copy of equal
  values) does not settle the record for what the code hands straight to a
  library. Each later run on inputs not seen before in the session hashes
  those objects again, until the record lists them. So a loop whose inputs
  keep changing pays the hashing on each such run.
- What only a function of yours is handed (`bt = run_backtest(series, 3)`),
  what a chart is drawn from (`ax.bar(names, totals)`), and a frame or an
  array a library call reads through an index or an attribute
  (`pd.Categorical(obs[r])`, `df.loc[k].plot()`) keep the record of their
  first run. A helper whose change to its argument depends on the data
  (`def push_all(h, xs): for x in xs: heapq.heappush(h, x)`, first run with
  no `xs`) is then not seen changing it later. Re-run the cell that reads the
  object, or call the library directly in the cell.

### Reordering a loop's items re-runs the tail

A loop that folds into a running total (`s += compute(x)`) keys each iteration
on everything before it, so reordering the items runs the statement again from
the first change. The call inside is cached by its arguments, so a reorder costs
no new `compute()` calls; only a new value does. A function that also writes a
global or a file runs again in full.

### Chained file-writing cells re-run each other

If N cells each read a file, change it and write it back, a Run All costs
N(N+1)/2 executions, not N, because each writer re-runs the writers before it.
**Fix:** write to separate files.

## Editing without saving

<!-- claim: cash/notebook/live_cells.py:handle_message @f101a60b, cash/notebook/server_discovery.py:_try_extension_cells @766cbfd6, cash/notebook/vscode_backup.py:live_cells @b86cd33f -->
**Symptom:** you edited an upstream cell, ran a downstream one, and got the
answer for the old code.

cash reads the cells it did not run from the notebook file. An unsaved edit is
invisible unless your editor has a **live reader**:

- **JupyterLab**, with the `cash-live-cells` extension that `pip install
  cash-lib` installs: it sends the current cells to the kernel before every run.
  It starts working from the run *after* the one that ran `import cash`, so a Run
  All on a fresh kernel still reads the saved file. It must be installed in the
  environment that runs JupyterLab, not only the kernel's.
- **VS Code**: cash reads its hot-exit backup of the notebook.
- **Google Colab**: cash reads the cells from the page.

Elsewhere, or when the reader is unavailable, save (`Ctrl+S`) after editing a
cell you are not about to run, and before a Run All on a fresh kernel.
JupyterLab's autosave runs on a timer, so a quick edit-then-run can miss it. When
the cell you run is itself unsaved, cash can tell and adds a "Notebook file is
stale" warning row.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker._as_run @73be869e, cash/notebook/upstream/checker.py:UpstreamChecker._refuse_to_undo @bc9c4f06 -->
A cell you ran with an unsaved edit is not undone by the cells below it. When
the frontend sends cell ids, cash reads that cell as it ran until the file
changes. Without them it cannot tell which cell the edit belongs to: a later
cell whose check would rebuild a name from the saved code stops with an
`UpstreamStateError` that asks you to save, and runs once you have.

With the same notebook open in two tabs, cash can read the other tab's unsaved
cells. Save before switching tabs.

To turn the extension off, run `jupyter labextension disable cash-live-cells` and
reload the page. To turn it back on:

```bash
jupyter labextension unlock cash-live-cells --level=system
jupyter labextension enable cash-live-cells
```

## Errors you may see

### `AmbiguousCellError`

<!-- claim: cash/exceptions.py:AmbiguousCellError @cc4cd1fd broad="the claim is about when this exception is raised at all" -->
Two cells have byte-identical content and cash cannot get a cell ID to tell
them apart, so it refuses to guess. JupyterLab and VS Code normally supply cell
IDs. **Fix:** add a comment to one of the cells, or save the notebook.

### `ForwardReferenceError`

<!-- claim: cash/exceptions.py:ForwardReferenceError, cash/notebook/upstream/notebook_vetting.py:NotebookVetter._refuse_forward_references -->
A cell reads, at module level, a name that only a later cell binds. It works now
because that cell already ran, but a top-to-bottom run would raise `NameError`.
A name used inside a `def` does not count; a decorator, default argument or base
class does. **Fix:** move the binding above this cell. The error names both
cells.

## Others

- **`from math import pi`-style imports** can block restoring after a restart
  where `import math` restores.
- **An empty cached value** (an empty list or frame) may be computed again
  instead of restored.
- **A pandas table from the cache is read-only past pandas.** Under
  copy-on-write the memory tier shares a table's data between hits, so
  `df["x"].array[0] = v`, or writing to a NumPy view made writable again,
  raises `ValueError: assignment destination is read-only`. `df.loc`,
  `df.iloc` and `df["x"] = ...` work as usual. See
  [the tiers](how-it-works/storage.md#the-tiers).
- **A file you save from a restored object may differ in its bytes.** A
  value restored from the cache equals the one the code computed, but its
  internals can be laid out differently: two attributes that shared one
  string can each hold a copy. So `joblib.dump(model, path)` or `pickle.dump`
  after a restore can write a file a few bytes off from the one plain Python
  writes, which loads to the same model. Compare such files by loading them,
  not by their bytes or a checksum.
- **A restored set may loop in a different order.** A set holds the same
  items after a restore, but the order a `for` loop meets them in can differ
  from the run that made it: Python does not keep a set's order, and a set
  rebuilt from its items lays them out afresh (`copy.deepcopy` and
  multiprocessing do the same). Results are equal; only code that depends on
  that order sees a difference, such as the key order of a dict built by
  looping over the set. Use `sorted(s)` where the order matters.

## Reporting something not on this page

Please [open an issue](https://github.com/galgtonold/cash/issues) with the cells
that trigger it. Say whether it happens on Run All or only when you re-run one
cell, and count calls rather than timing them: a counter your function appends
to shows what ran.

Limitations of `@cash.cache` functions, such as
[code passed as an argument](decorator-limitations.md#code-you-pass-as-an-argument),
are in [Known limitations of `@cash.cache`](decorator-limitations.md).

## Related

- [Debugging](tutorials/feature-guides/debugging-and-monitoring.md): how to
  find the input that moved.
- [Knowing when to recompute](how-it-works/invalidation.md): what cash does
  track.
- [Known limitations of `@cash.cache`](decorator-limitations.md): the
  decorator's own limits.
