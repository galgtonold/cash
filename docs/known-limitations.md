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
each random stream the statement drew from or seeded where the computed run left it, and
leaves every other stream where it is, so the next draw matches a full run.

cash finds a draw by following the generator from where it was created. It also
warns about `df.sample(100)` and other library calls without `random_state`, an
anonymous generator (`z = np.random.default_rng().normal(size=3)`), and a call to
one of your functions that builds its own unseeded generator (`rng = make_rng()`).
<!-- claim: cash/notebook/statement/hidden_generators.py:positions @fc34e5a9 -->
A statement that draws from a generator held where no variable's lineage
follows it is not stored, without a warning: it runs every time, so the draws
below it match plain Python. That is a generator held by an object built in a
module of yours (`tlib.MODEL.rng`), an item of a list, or one a helper keeps in
its closure (`arr = make_data()`). A draw from the global `random`,
`np.random` or torch stream inside a helper is flagged once the statement has
run, by comparing the streams.

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
does not bring back a value drawn elsewhere in its stream. Three gaps on an
isolated re-run:

- **A generator held by an object cash does not rebuild.** One reached
  through an attribute is followed when the object holding it is rebuilt for
  the re-run (`self.rng` set in `__init__`, then `m.rng.normal()`). A class
  attribute read through an instance (`c.rng`, `self.rng` in a method), a
  module's or a function's is not: re-running a later draw alone draws from
  wherever it is now, and cash warns
  [RANDOM-HELD-GENERATOR](warnings.md#random-held-generator). Draw from such a
  generator in the cell that creates it, or hold it in a variable.
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

## A helper that reads the clock runs every time

**Symptom:** a statement that calls one of your functions runs on every run,
with a KEY-AMBIENT-READ warning.

<!-- claim: cash/analysis/cacheability_decision.py:_helper_ambient_reasons @5edc872e, cash/notebook/lineage_formula.py:helper_ambient_reads @609ad208 -->
A clock read written in the statement itself (`t = datetime.now()`) runs every
time, and so does a statement that calls a function of yours which reads the
clock, a fresh UUID or random bytes from the system (`time.time()`,
`uuid.uuid4()`, `os.urandom()`), there or in a helper it calls. A cached result
would return the first value on every run, so cash runs the statement instead
and says so once:

<!-- test:skip reason="illustrative: shows a statement that runs on every run" -->
```python { .nb-cell }
def stamp(x):
    time.sleep(0.2)
    return x, time.time()

# runs every time: KEY-AMBIENT-READ, "stamp(), which reads time.time()"
s = stamp(1)
```

A read whose value only goes into a log line or a `print` does not count, and
neither do the clocks that measure time passing (`time.perf_counter()`,
`time.monotonic()`, `time.process_time()`), as in a timeout loop or a measured
duration: a value built from one differs between two runs without cash too.

**Fix:** to cache it, pass the time in (`stamp(1, now=time.time())`). To cache
the first value on purpose, mark the line `# @cash:assume-safe`.

## Changes cash cannot see

Each of these misses a change, or applies one a second time.

### Mutating through an alias

<!-- claim: cash/notebook/statement/derivation_edges.py:record_shared_object_edges @1d3cdc8d, cash/notebook/statement/derivation_edges.py:bump_derived_lineages @948a7fba, cash/notebook/shared_objects.py:share_group @1523aede, cash/notebook/shared_objects.py:view_base_names @aea7d284, cash/notebook/statement/processor.py:StatementProcessor.record_structure_share_edges @4fe4d23f, cash/notebook/statement/processor.py:StatementProcessor._link_inputs_kept_by @2f874738 -->
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
  `runs = [cfg]`) until it runs;
- when code of yours keeps the object, or a part of it: a function appending
  `rows` to a global, an object, class attribute or module global of yours
  (`keep(rows)`, `Registry.add(rows)`). What keeps it is linked to `rows`, so
  a later call that changes it there (`bump()`) moves `rows` too.

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

<!-- test:skip reason="illustrative: in-place changes to a frame from an earlier cell" -->
```python { .nb-cell }
# df came from an earlier cell
df["score"] = expensive(df)            # cached
# runs every time, but expensive(df) is cached
df.insert(1, "rank", expensive(df))
```

A new column assigned to a frame from an earlier cell (`df["score"] = ...`) is
cached like any other statement. A statement that changes such an object
through a method (`df.insert(...)`, `lst.append(...)`) runs every time, but a
slow call inside it is cached on the values it is handed, `df` included, so
only the cheap change itself repeats. When
another variable holds that object too (`d = dfs[0]`, `b = a`), directly or in
a list, dict, `deque`, `UserDict`, object array or attribute of one (a
`__slots__` attribute too), or the result holds an object that already
existed (`models = {"m": m}`), those variables are stored with the statement
and a hit restores them as one object, as running it would leave them. When
something else holds it -- a closure, a library's module or registry, a pandas
`.attrs` (pandas copies it apart) -- or the statement is in a loop body
(`for d in dfs:`), a restored copy would not be that object, so the statement
runs every time. When that object belongs to a variable the statement leaves
as it was (`grid_search.fit(X, y)` keeps the `full_pipeline` it is handed),
the entry keeps it as that variable instead: a hit binds the live object,
which everything else still holds, after a restart too, and the statement
runs when that variable's content, or which object another variable shares
with it, changed. IPython's output history
(`Out`, `_`) holding a value you displayed is not such a holder, and neither
are an object's own parts: each trace of a plotly figure, or each node of an
XML document, points back at the object it belongs to, and when nothing
outside holds those parts either, the statement is cached. A slow call
inside such a statement (`d["c"] = slow(d)` in the loop) is still cached on
the value it is handed. Nothing to do.

### Mutating global state inside a function

<!-- claim: cash/analysis/callee_effects.py:callee_global_mutations @0563cfbf -->
A function that changes a global of your notebook (`LOG.append(v)`,
`counter["n"] += 1`), itself or through a helper it calls, is handled: the
statement calling it runs every time so the change really happens, while the
call inside it is served from the cache together with its effect on the
global. Nothing to do.

A change to a variable inside one of your modules that the notebook does not
import is not seen: a function that imports the module itself and writes to it
(`import registry; registry.REG["n"] = 1`) is served without the write, so
`registry.REG` keeps what it held, and is empty after a restart. Keep such
state in the notebook, or pass it in and return it.

### A helper memoised with `functools.cache`

<!-- claim: cash/notebook/helper_memos.py:unfilled @36d01050, cash/notebook/helper_memos.py:used @815bf2d3, cash/notebook/helper_memos.py:clears @c8fd014e, cash/notebook/helper_memos.py:MAX_KEYED == 256 -->
A statement served from the cache does not run, so a `@functools.cache` or
`@functools.lru_cache` helper it calls keeps what it held. A run leaves the
answers in it, and a later call of the helper uses them without running its
body. So cash follows these rules:

- A helper whose answer provably comes from its arguments alone: it reads
  no global but plain builtins (`len`, `sum`, `range`, ...) not shadowed
  by a name of yours, and itself (`fib` calling `fib`); no attribute, no
  closure variable; writes nothing; has no default argument but a number,
  string or tuple of them (`out=print` and `acc=[]` do not qualify); and
  takes and returns numbers, strings or tuples of them
  (`return n * n`, `return sum(range(n))`). A call of another function of
  yours does not qualify, since a later cell can redefine it. A statement
  that fills or reads such a memo is cached as usual; served, cash calls
  the helper for each key the run filled that the memo lacks, at the
  statement's place, so the memo holds what a run leaves even if a later
  cell shadows a builtin it reads. It is not served while the memo holds
  a different answer for a key the statement read. One thing still
  differs: `h.cache_info()` counts only those calls, not the hits the
  statement made, so a statement that names `cache_info` (or
  `cache_clear`) reruns every time.
- Any other helper: a statement (or a call cash caches) whose run made the
  memo compute an answer, emptied it (`cache_clear` under any name the
  statement or a function of yours it calls uses, or seen from the keys
  the memo holds afterwards), or changed an answer it holds
  in place (`f(1).append(5)`) changes the memo the way appending to a list
  does: it reruns every time and is never served.
- One that only read answers already there is served only while the memo
  holds every key and answer it read, compared by value. After a kernel
  restart it runs until the memo is filled again, and the first statement
  to read an answer that can change in place after that runs once more,
  so cash has seen the answer as it was. As above, `cache_info()` then
  counts fewer hits than a run would, on one of your classes too
  (`Model.lookup.cache_info()`).

Nothing to do: the results are the ones a run gives, at the cost of those
runs. cash reads a memo's answers only when it holds at most 256 of them;
a statement over a bigger memo reruns every time, and says so. It sees a
memo bound in the notebook or at the top of one of your modules (a
`reload` included), under other decorators that set `__wrapped__`, and on
one of your classes (a `staticmethod`, a `classmethod` or a plain
attribute). One reached only as an argument, an instance attribute or an
item of a dict is not checked, and neither is Python's pure-Python
`lru_cache` (used only when the `_functools` module is missing), nor a
`cache_clear` reached only through a library after which the memo holds
the very same key and answer objects again (cash keeps the ones it saw, so
it compares objects, not addresses). A memo you fill yourself in
a dict (`_cache[k] = v`) is a global the function changes, covered above.

<!-- claim: cash/analysis/handed_callables.py:handed_callables @c55bd405, cash/analysis/handed_callables.py:hands_a_state_changing_callable @4c3c2d42, cash/analysis/handed_callables.py:ITEM_TYPES_AT == 100_000 -->
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
  (`bag["k"]`, `len(bag)`, `bag + 1`, `f"{bag}"`, iterating it), reads
  from a list, tuple or dict (`d["k"] += 1`, `lst[i] + 1`; by an index
  cash cannot read, any of its items), from a class or a library's
  namespace (`C.x + 1`, `ns.x += 1`), or that a function of yours it calls
  reads by name or one step into such a value (`return v + 1`,
  `return d["k"] + 1`, `return C.x + 1`). A container of more than 100,000
  items read by an index cash cannot read, while a class of yours has such
  a method (any class alive, one made inside a function included), runs
  each time with a warning
  ([NOTEBOOK-ITEMS-UNREAD](warnings.md#notebook-items-unread));
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

<!-- claim: cash/notebook/upstream/changed_below.py:warn_changed_below @8bd3d1a3 -->
cash answers cell 2 as a top-to-bottom run would: it rebuilds `s` as cell 1
leaves it, so cell 2 computes with `0.0`, and `s` holds `0.0` until cell 3 runs
again. cash warns [NOTEBOOK-CHANGED-BELOW](warnings.md#notebook-changed-below)
when this happens, naming the variable and the cell that changed it.
**Fix:** move the change above the cells that must see it, or run cell 3 again
after the re-run.

### Changing and rebinding the same name in one cell

<!-- test:skip reason="illustrative: mutate-then-reassign raises on isolated re-run" -->
```python { .nb-cell }
df["c"] = df["a"] * 2
df = df.rename(columns={"a": "x"})
```

An isolated re-run raises `KeyError`, because it reads the renamed frame.
**Fix:** split the two statements into two cells.

### A class edited above a cell that changes it

<!-- claim: cash/notebook/upstream/class_state.py:ClassState.put_back @b3b3a5bc -->
A class whose constructor changes the class itself (`Widget.count += 1`) can
be changed from several cells, and an isolated re-run of one of them counts as
a run from the top does: cash puts the class's attributes back where the cell
started. When they cannot be saved (a lock, over 64 MiB) the re-run keeps the
class as it is and warns ([NOTEBOOK-CLASS-KEPT](warnings.md#notebook-class-kept)).

What is not covered is an edit of the class itself:

<!-- test:skip reason="illustrative: edited class def above a constructing cell" -->
```python { .nb-cell }
class Widget:          # cell 1, edited to count = 10
    count = 0
    def __init__(self):
        Widget.count += 1

w0 = Widget()          # cell 2

w = Widget()           # cell 3, run alone: 11 (from the top: 12)
print(Widget.count)
```

cash runs the edited class def again, but not cell 2, which cell 3 does not
read, so cell 2's change to the class is missing.
**Fix:** run the cells from the edited class down, or keep the counter in a
variable, not on the class.

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

### A name a magic binds

<!-- claim: cash/analysis/code_analyzer.py:_cell_magic_body @3566e904, cash/analysis/code_analyzer.py:_is_magic_line @d120bf9b, cash/notebook/statement/processor.py:StatementProcessor.forget_rebound @70d43250 -->
cash reads a cell's Python, not what IPython makes of its magics. A magic or a
shell command runs every time, uncached, and cash never runs one for you.
<!-- claim: cash/notebook/magic_effects.py:magic_effects @420435ff, cash/notebook/upstream/simulator.py:NotebookSimulator._warn_stale_magic @777a3665, cash/notebook/magic_effects.py:is_rerun_magic @17c29890, cash/notebook/statement/mutations.py:MutationClassifier.magic_snapshots @24994a9f, cash/analysis/code_analyzer.py:magic_python @1a6291b3, cash/notebook/magic_effects.py:magic_base @0d0485ca -->
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
after an edit to a cell it reads, a change to a file or folder its line read
in Python (`!wc -l {max(glob.glob('logs/*.log'))}`), or a restart, cash keeps
the value the name has (or leaves it undefined) and warns
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

<!-- claim: cash/notebook/cache_key.py:lookups_followed @f277d3b1, cash/notebook/string_lookups.py:tree_lookups_traceable @ac8f00df -->
On Python 3.12 and newer, a lookup whose value is called at once
(`globals()[name](3)`, `NS[name](3)` with `NS = globals()`,
`getattr(__main__, name)(3)`) is cached all the same: what it ran is
recorded, the function the string found and the name holding it included,
and a stored result is served only when none of that changed. A value read
by a string and used as data (`globals()[name] + 1`, `f = globals()[name]`),
a name bound through the namespace, `eval`, `exec` of a computed string and
`sys.modules` still run every time.
**Fix:** call the function by its name, or pass it in as an argument.

### A special method that changes state outside its object

<!-- claim: cash/notebook/statement/unseen_writes.py:writers_without_a_trace @fb58b1ed, cash/notebook/statement/unseen_writes.py:own_writes @bb4fc316 -->
`s + 1` on objects of yours runs their `__add__`, which the statement's text
does not name. When such a method changes something outside its object
(appends to a global list), cash runs the statement every time and says so:
[NOTEBOOK-UNSEEN-WRITES](warnings.md#notebook-unseen-writes). On Python 3.12
and newer cash reads what ran from the run itself. Without that record
(Python 3.10 and 3.11, `CASH_TRACE=0`) it reads the special methods' own
text instead, and misses a change made:

- by a function the method calls (`note(1)`, `self.log()`, a lambda held in
  a dict);
- by `__init__`, `__post_init__`, a `default_factory` or an attrs hook;
- by an object held in a slot, a namedtuple, a list or dict subclass, a
  default argument, a `ChainMap`, a property, or a column of a subclass of
  `DataFrame`.

A served result then leaves out that change. Without the record, a writer
the stored run did not call (a `functools.cached_property` already filled,
a tuple's hash Python 3.14 kept) is served after a restart; with it, a
statement refused once for a writer is not stored again in that session
while the writer is still defined. Either way a cell that reads what the
method changed keeps its earlier value in the same session.
**Fix:** pass what the method changes in and return it, or put
`# @cash:no-cache` on the statement.

### Background threads

A thread that changes data after its cell has finished is outside cash's view.
cash does not store the statement that left it running, nor one that reads a
variable the thread's function can reach, so those run every time. An earlier
cell re-run can still see the changed data.

<!-- claim: cash/notebook/background_threads.py:BackgroundThreads.cell_ends @35f8017f -->
cash names a cell that ends while a thread it started, running your code, is
still alive: [NOTEBOOK-THREAD-ALIVE](warnings.md#notebook-thread-alive).
**Fix:** join the thread before the cell ends (`t.join()`).

While another thread is alive during a statement, cash reads each value it
checks afresh, so a change the thread makes in between is seen. A thread a C
library starts that writes memory without running Python code (an audio or
device driver filling a buffer) cannot be seen at all: cash reads that buffer
as it finds it. **Fix:** copy such a buffer (`data = buf.copy()`) in the cell
that stops the stream, and use the copy.

### A class whose pickling changes other values

<!-- claim: cash/notebook/statement/reducer_writes.py:refuses @899c820d, cash/notebook/statement/reducer_writes.py:_handed_to_libraries @b809130a, cash/notebook/statement/reducer_writes.py:_changed_through_copies @a33fedda -->
cash pickles values to hash and store them, and pickling runs a class's own
`__getstate__` or `__reduce__`. One that changes something else (a module's
list, a global array, the object being pickled) would change it where plain
Python never pickles at all, so cash does not run it: the value is not stored,
and the statement runs again next time, with
[NOTEBOOK-REDUCE-WROTE](warnings.md#notebook-reduce-wrote). This also happens
when cash cannot tell: the method's source cannot be read, it changes the
object it pickles, or it hands a global the notebook holds (or a module
holding it) to a library call, such as `np.copyto(ARR, 5)` or `np.mean(ARR)`,
whose effect cash does not read. Builtins that only read (`len`, `map`,
`sorted`, `iter`, ...), the functions of `itertools`, `math` and
`statistics`, `json.dumps` and `json.loads`, `copy.deepcopy` with no memo,
`functools.reduce` of a function that changes nothing, a method of a constant
(`'-'.join`) and a value nothing can change (a number, a string, a class, a
function) do not count; a list, dict or array handed to any other library
function does, even one that only reads it (`copy.copy`, `json.dump`), and
its statement runs every time. A change made through a local bound to a
global or to a shallow copy of one (`c = dict(CFG); c['rows'].append(1)`)
counts as a change of the global, the copy's own top level included.
**Fix:** have `__getstate__` / `__reduce__` only read, and change those values
in the cell that needs them.

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

cash warns [KEY-LOOP-VAR-CHANGED](warnings.md#key-loop-var-changed) when a
loop body changes its loop variable in place before work it caches for each pass.
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
`np.load`, `joblib.load`, polars, `sqlite3.connect`, `h5py.File`,
`xarray.open_dataset`, `rasterio.open`, `open()` and
[others](how-it-works/invalidation.md#what-counts-as-a-change), and on Linux and
Windows the files C libraries open themselves (below). A read through anything
else (a subprocess, a client library that makes the system call itself, and on
macOS a C extension that opens the file itself or `os.open`) is invisible, and
the statement is cached with no file recorded. A watched reader imported by
name counts like its module spelling, also when it was imported before `%cash_on`
or from the submodule that defines it (`from polars.io.parquet import
read_parquet`).

**Fix:** read the file through a watched reader, or put `# @cash:no-cache` above
the statement if the read is cheap.

<!-- claim: cash/tracking/native_reads.py:close_window @0e2917f4, cash/tracking/native_reads.py:_hook_libraries_as_they_load @d228cce2 -->
On Linux (glibc, x86-64 or aarch64) and on Windows (x64) the native file hook
is on by default (`%cash_status` shows it as `native_file_hook`;
`CASH_NATIVE_FILE_HOOK=0` turns it off). cash then also sees the files a
C, C++ or Rust library opens itself through the C library's `open`, `fopen` or
`opendir`, or on Windows through `CreateFileW` or the CRT's `_wopen`,
`_wsopen_s` and `fopen` family: `pyarrow.fs`, a polars `scan_csv`, netCDF4, a
database `ATTACH`-ed in SQLite, `os.open`, a compiled extension of your own.
It still misses a library that makes the system call itself, one linked
statically against the C library, a call through a function pointer the
library looked up itself (`dlsym`, `GetProcAddress`, ctypes), and a plugin a
C library loads by itself and uses within the same cached call. A library your
code imports or loads with `ctypes` inside the call (`import pyarrow.fs` in
the function body) is seen from the moment it is loaded. A file such a
library opens while several cached calls run at once, in different threads, counts for all of them. On
macOS the hook is not available yet; there the readers listed above are still
recorded by their entry points.

### A file read into a memo before cash was imported

<!-- claim: cash/__init__.py:_watch_reads_from_import @0d64857e -->
**Symptom:** a cached function gets its config from a loader memoised with
`functools.lru_cache`, you edit the config file, and the function returns
the old result with no warning.

cash watches file reads from `import cash` on. A memoised loader that first
ran before that (it was imported, and called, ahead of `cash`) read the file
where cash could not see it, so later calls get the memo and record no file.
A loader first called after `import cash` is recorded, even when that call
came from outside any cached function.

<!-- claim: cash/tracking/memos_before_import.py:find_memos_filled_before_import @e09c920f -->
At `import cash`, cash names each memo of yours (`lru_cache`, `cache`)
that already holds results:
[KEY-MEMO-BEFORE-IMPORT](warnings.md#key-memo-before-import).

**Fix:** import `cash` before the modules that load config, or pass the path
as an argument (`load_cfg("config.yaml")` inside the cached function).

### An edit that keeps the size and timestamps

<!-- claim: cash/tracking/file_dep_snapshot.py:_unchanged_since_hashed @59019b2b, cash/tracking/file_dep_snapshot.py:_HASH_MEMO_MIN_AGE_SECONDS == 10.0, cash/notebook/upstream/virtual_lineage.py:VirtualLineage._check_cell_file_deps @eeff0da9 -->
cash checks a file's size and timestamps first and reads its content only when
one of them moved. On Linux and macOS every write moves the inode change time,
so this never misses. On **Windows**, two kinds of edit move nothing: a write
whose modification time is put back afterwards (`os.utime`, `shutil.copystat`,
`robocopy /COPY:T`), and a write through a memory map
(`np.memmap(path, mode="r+")`).

<!-- claim: cash/tracking/written_in_place.py:_on_mmap @fa5b57bf, cash/tracking/written_in_place.py:is_written_in_place @90fefb25, cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @1e9f74d4 -->
A file mapped to write (`np.memmap` in `r+` or `w+` mode,
`np.load(mmap_mode="r+")`, `mmap.mmap` with write access) in a process that
had imported cash is read at every check from then on, in that process and
in later ones that use the same cache folder, so its writes are seen. A
mapping made in a process without cash, or before `import cash`, is not.

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

<!-- claim: cash/notebook/heavy_loops.py:is_heavy @908c4eba, cash/notebook/heavy_loops.py:HEAVY_FACTOR == 4, cash/notebook/heavy_loops.py:MIN_HEAVY_ITER_SEC == 0.01, cash/notebook/heavy_loops.py:header_identity @ba179718, cash/notebook/heavy_loops.py:HeavyLoopStore._find @d8473e07, cash/notebook/control_structures/for_handler.py:ForLoopHandler._prefers_iterations @b5475b31 -->
That estimate is a guess made before the loop has run. A loop run as one unit
is measured while it runs, and when one iteration's work takes at least four
times its estimated bookkeeping, and at least 10 ms, the next runs cache it
per iteration again. As one unit, raising its `range` or editing the last
statement of its body runs every iteration again; per iteration, an edit runs
only the iterations it changes or adds. The measurement is kept in the cache
folder under the loop's header and body with its numbers left out, so it
carries over a change of the loop's length (`range(100)` to `range(180)`) and
an edit to up to half of the body's statements. Another loop with the same
header and a different body keeps a measurement of its own.
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

<!-- claim: cash/notebook/loop_watch.py:LoopWatch.start @31a0eadb, cash/analysis/namespace_effects.py:rewatch_exempt @e0de76f8 -->
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
- What a function of yours hands to a library on every run
  (`bt = run_backtest(series, 3)`), what a chart is drawn from
  (`ax.bar(names, totals)`), and a frame or an array a library call reads
  through an index or an attribute (`pd.Categorical(obs[r])`,
  `df.loc[k].plot()`) keep the record of their first run. What your function
  hands on only on some runs (in a loop over its input, a branch, after an
  early `return`: `def push_all(h, xs): for x in xs: heapq.heappush(h, x)`),
  or what a call of it in a loop hands on, is hashed again on new inputs, as
  a library call's is.

### Reordering a loop's items re-runs the tail

A loop that folds into a running total (`s += compute(x)`) keys each iteration
on everything before it, so reordering the items runs the statement again from
the first change. The call inside is cached by its arguments, so a reorder costs
no new `compute()` calls; only a new value does. A function that also writes a
global or a file runs again in full.

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
cell you are not about to run, and before a Run All on a fresh kernel. cash
says so once per session there
([NOTEBOOK-NO-LIVE-READER](warnings.md#notebook-no-live-reader)), at the first
cell you run after a pause.
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

### A cell reads a name a later cell binds

<!-- claim: cash/exceptions.py:ForwardReferenceError, cash/notebook/upstream/notebook_vetting.py:NotebookVetter._refuse_forward_references -->
A cell reads, at module level, a name that only a later cell binds. It works now
because that cell already ran, but a top-to-bottom run would raise `NameError`.
cash runs the cell uncached, on the value in memory, and warns with
[NOTEBOOK-FORWARD-READ](warnings.md#notebook-forward-read), which
names both cells. A name used inside a `def` does not count; a decorator,
default argument or base class does. **Fix:** move the binding above this cell.

## Others

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
- **A call given a set of strings runs again after a restart.** Inside a
  cell, cash keys a call of your function on its arguments as they are
  stored, and a set or frozenset of strings is stored in an order that
  changes with every Python process. So after a kernel restart that call
  runs once more instead of being served; later runs are served as usual.
  Results are right. A list, a tuple, or a set of numbers is not affected.
- **An object's own pickle recipe runs when cash copies it.** The memory
  tier keeps its own copy of a result, made the way pickle rebuilds it, and
  every hit is such a copy too, as a disk hit is. A class whose
  `__reduce__` rebuilds through a call with an effect (a constructor that
  connects, `logging.getLogger`, which registers a logger) has that effect
  once when the result is stored and once per hit. A rebuild that would
  change a file, by any route (`(open, (path, "w"))`, a `__setstate__`
  reopening its log, a constructor the recipe names, `os.remove`,
  `os.truncate`, `os.rename`, `os.mkdir`, a `sqlite3.connect` that creates
  its file), is stopped there: the result is not cached, in memory or on
  disk, an entry stored before is not served, and the file keeps what it
  held. A scratch file or folder the rebuild makes new in the temp folder
  is allowed, and so is Python's own `.pyc` for a module the rebuild
  imports. A library whose first import writes its own cache (matplotlib's
  font list) is imported as your program would import it, and the rebuild
  tried again. A write through a file already open (one the object held)
  is not seen, and neither are `os.chmod` and `os.mkfifo`.

- **A `__dict__` your class computes may run under `@cash.cache`.** A class
  that defines `__dict__` itself (a `@property` building it) has it read
  when an object of it is part of a cached function's key, its arguments
  or the globals it reads, as Python's own `vars(obj)` would. In a
  notebook, the walks over what a statement reads and what it holds read
  the attributes Python keeps instead, and never run it. Keep such a
  property free of side effects, or give the function what it needs as
  plain arguments. A `dict` subclass set as an object's `__dict__` is read
  the same way, without its own methods, but storing the object still
  pickles it, which calls the subclass's `items` and `__reduce_ex__`: if
  those raise, the statement is not cached and re-runs each time.
- **Python 3.12.0 to 3.12.5 without the C extension: more statements re-run.**
  On those versions, asking Python what the `_asyncio` module holds can loop
  for ever (fixed in 3.12.6), and Jupyter loads that module. cash's C
  extension reads around it. Without the extension, cash cannot tell which
  variables share an object with a statement's outputs, so such a statement
  re-runs with a warning instead of being restored. Upgrade to 3.12.6 or
  later, or install a build with the C extension.

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
