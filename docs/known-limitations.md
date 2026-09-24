# Writing cache-safe cells

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. For limitations of `@cash.cache` functions, see the
    [decorator guide](decorator.md).

Cash restores a cell's inputs by working out what each statement reads and
writes. Where a change travels through a channel cash does not see, you can get a
value that looks right and is stale. This page lists those cases: the symptom,
an example and the fix.

Most of them need an **isolated re-run**, meaning you run one cell on its own. A
top-to-bottom **Run All** executes every cell in order, so cash has nothing to
reconstruct and these cases do not arise. Entries that also affect Run All say so.

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

<!-- claim: cash/notebook/statement/restore.py:StatementRestorer.restore_from_cache @dbaae9db, cash/tracking/randomness/state.py:restore_rng_state @3e10fc77, cash/tracking/randomness/state.py:capture_rng_state @421bfe05 -->
**Symptom:** re-running a cell returns the same random numbers.

An unseeded draw is cached like any other value, so a re-run shows the stored
result. A cheap draw that is never cached is frozen too if it comes from the
`random`, `numpy.random` or `torch` stream: before a cell re-runs, cash rewinds
those streams to where the cell started, so a re-run lands where a top-to-bottom
run would. A generator held in a variable (`rng = np.random.default_rng()`) is
not rewound, so a cheap draw from it changes on every run. Cash warns
`RANDOM-UNSEEDED` when an unseeded draw first runs (not in a `no-cache`
statement), and `RANDOM-REPLAYED` when an unseeded value comes back from the
cache. An estimator fitted with `random_state=None` counts as an unseeded draw.

| You want | Do this |
|---|---|
| A new draw every run | `# @cash:no-cache` on the statement, on the line above it or at the end of its line. It turns off the rewind as well as caching. |
| The same result every run | Seed it: `np.random.seed(0)`, `np.random.default_rng(0)`, `random_state=0`. |
| No warning | `# @cash:allow-random`. It changes the warning only, never what is cached or rewound. |

<!-- claim: cash/tracking/randomness/detect.py:RNG_CARRIER_CONSTRUCTORS @620106b9, cash/tracking/randomness/state.py:capture_object_rng_states @d8dd9223 -->
Seeding counts per source. `np.random.seed(0)` covers `np.random.*` draws for
the rest of the session, but not `random.random()` and not a generator from
`np.random.default_rng()`, which you seed in its constructor. A cache hit puts
the random stream back where the computed run left it, so the next draw matches a
full run.

Cash finds a draw by following the generator from where it was created, so these
are **not** flagged and are cached silently:

- a generator cash never saw created: `self.rng.normal()`, a generator returned
  by a helper, `df.sample(100)` without `random_state`;
- a draw inside a helper function (`arr = make_data()`);
- an anonymous generator: `z = np.random.default_rng().normal(size=3)`.

<!-- claim: cash/tracking/randomness/state.py:capture_rng_state @421bfe05 -->
Three gaps on an isolated re-run:

- **A generator's position across cells.** With `rng = np.random.default_rng(0)`
  in one cell and draws from `rng` in the next two, re-running the third cell
  alone draws from wherever `rng` is now. Draw from a generator in the cell that
  creates it.
- **An edited seed cell you did not re-run.** Edit `np.random.seed(0)` to
  `seed(1)` and run only a later draw: the draw does not see the new seed,
  because a bare `seed()` binds no variable. Re-run the seed cell after editing
  it, or seed in the cell that draws.
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
time. One inside a function you call is not seen, and the call is cached with
the first value:

<!-- test:skip reason="illustrative: shows a value frozen across two runs of one cell" -->
```python { .nb-cell }
def stamp(x):
    time.sleep(0.2)
    return x, time.time()

s = stamp(1)      # re-run: the same time.time() as the first run, no warning
```

**Fix:** pass the time in (`stamp(1, now=time.time())`), or put
`# @cash:no-cache` on the line above the statement.

## Changes cash cannot see

Each of these applies a change a second time, or misses one, on an isolated
re-run. Run All is not affected.

### Mutating through an alias

<!-- claim: cash/analysis/aliases.py:bare_alias_targets @311a21c3, cash/analysis/aliases.py:reference_alias_targets @f052d224 -->
Cash tracks a change through the name the object is bound to. Through another
name it is invisible, so a re-run applies it twice:

<!-- test:skip reason="illustrative: alias-mutation shapes, need isolated cell re-runs" -->
```python { .nb-cell }
b.ref = x
b.ref.append(99)       # changes x, but cash sees only b

t = (lst,)
t[0].append(3)         # changes lst

y = x if flag else z
y.append(3)            # x or z?
```

**Fix:** change the object through its own name (`x.append(99)`), or rebind
(`x = x + [99]`).

### Mutating an object created in an earlier cell

<!-- test:skip reason="illustrative: contrasts in-place mutation with rebinding across cells" -->
```python { .nb-cell }
df["score"] = expensive(df)          # df came from an earlier cell: runs every time
df = df.assign(score=expensive(df))  # cached
```

A statement that changes an object made in an earlier cell runs every time. The
same statement on an object made in the same cell caches normally. **Fix:**
rebind with `df = df.assign(...)`.

### Mutating global state inside a function

<!-- claim: cash/analysis/callee_effects.py:callee_global_mutations @08706d29 -->
A function that changes a global (`LOG.append(v)`, `counter["n"] += 1`) is
handled: the statement calling it runs every time so the change really happens,
while the call inside it is served from the cache together with its effect on
the global. Nothing to do. If you would rather not rely on this, pass the state
in and return it.

### Re-running a cell above an in-place change

<!-- test:skip reason="illustrative: needs an isolated re-run of a cell above the mutation" -->
```python { .nb-cell }
s = pd.Series(np.arange(200_000, dtype=float))   # cell 1
out = summarize(s)                                # cell 2: re-run this alone...
s.iloc[0] = 1e9                                   # cell 3: ...and it does not see this
```

Cash answers cell 2 as a top-to-bottom run would, so it computes with `0.0`
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
class Widget:                    # cell 1
    count = 0
    def __init__(self):
        Widget.count += 1
w0 = Widget()

w = Widget()                     # cell 2: each re-run pushes count higher
```

This needs the counter to be changed from both cells; one cell is fine.
**Fix:** keep the counter in a variable, not on the class.

### A function that calls one defined in a later cell

<!-- test:skip reason="illustrative: shows staleness across an edit of a later cell" -->
```python { .nb-cell }
def a(n): return b(n) * 2      # cell 1: b does not exist yet
def b(n): return n + 1         # cell 2
r = a(3)                       # cell 3: 8

# edit cell 2 to `return n + 10`, re-run cell 3 alone: still 8, not 26
```

Cash follows dependencies upward, so `a` never learns it depends on `b`. Cash can
even serve `r` after cell 2 is deleted. **Fix:** define a function above the
functions that call it. A module-level read of a name bound only below raises
[`ForwardReferenceError`](#forwardreferenceerror) instead.

### Background threads

A thread that changes data after its cell has finished is outside cash's view.
An earlier cell re-run can see the changed data.

### A loop variable changed before it is read

<!-- claim: cash/notebook/control_structures/for_handler.py:ForLoopHandler._process_one_iteration @6a19a3f5 -->
This one can give a wrong answer on the first Run All. Cash keys each iteration
on the loop variable's value when the `for` binds it. A body that changes that
value, or computes a new body-local variable, before the cached work is not seen:

<!-- test:skip reason="illustrative: pull() stands in for a slow call keyed only by the loop variable" -->
```python { .nb-cell }
for q in [[1], [1]]:           # two iterations, equal when bound
    q.append(len(accm))        # now different, but the key was taken already
    accm.append(pull(handle))  # the second iteration gets the first's result
```

**Fix:** make the distinguishing value part of the `for` target:
`for i, base in enumerate(items):`. Only names bound by the `for` statement reach
the per-iteration key. If there is no such value, put `# @cash:no-cache` above
the statement.

## Files

### Reads through a loader cash cannot see

<!-- claim: cash/tracking/reader_patches.py:_install_module_patches @c4527d67 -->
**Symptom:** you changed a file and the cell still shows the old data, with a
`CACHED` badge and no warning.

Cash records a file when it is read through a reader it watches: `pd.read_*`,
`np.load`, `joblib.load`, polars, `sqlite3.connect`, `open()` and
[others](how-it-works/invalidation.md#what-counts-as-a-change). A read through
anything else (a C extension that opens the file itself, a client library, a
subprocess, `os.open`) is invisible, and the statement is cached with no file
recorded. A SQLite database in WAL mode is the same: a commit lands in the
`-wal` file.

**Fix:** read the file through a watched reader, or put `# @cash:no-cache` above
the statement if the read is cheap.

### An edit that keeps the size and timestamps

<!-- claim: cash/tracking/file_dep_snapshot.py:_unchanged_since_hashed @809a68f2, cash/tracking/file_dep_snapshot.py:_HASH_MEMO_MIN_AGE_SECONDS == 10.0 -->
Cash checks a file's size and timestamps first and reads its content only when
one of them moved. On Linux and macOS every write moves the inode change time,
so this never misses. On **Windows**, two kinds of edit move nothing: a write
whose modification time is put back afterwards (`os.utime`, `shutil.copystat`,
`robocopy /COPY:T`), and a write through `np.memmap(path, mode="r+")`.

**Fix:** after such a write, touch the file (`Path(path).touch()`). Cash then
reads its content, and if it changed, everything that read it runs again.

### Large objects are hashed by sampling

<!-- claim: cash/object_hashing.py:compute_hash @a7245478 -->
To keep hashing cheap, cash fingerprints a large DataFrame by its shape, dtypes
and first 5 rows, an array by its first 100 elements, and a long list or dict by
a sample. Normally cash tracks where a value came from and does not rely on this
fingerprint. It matters only after `cash.reset_session()`, when a change deep
inside a large object can go unnoticed. **Fix:** restart the kernel instead of
resetting the session.

## Loops

### A long `for`-append loop can stop caching

<!-- claim: cash/notebook/control_structures/single_unit_policy.py:should_run_as_single_unit @aaa994ff, cash/notebook/control_structures/single_unit_policy.py:MIN_ITERATIONS_FOR_SINGLE_UNIT == 50, cash/notebook/control_structures/single_unit_policy.py:PER_STMT_OVERHEAD_SEC == 0.008, cash/notebook/control_structures/single_unit_policy.py:MIN_OVERHEAD_SEC == 1.0 -->
Cash caches a `for` loop per iteration. A long loop is run as one unit instead
when all three hold: more than about 50 iterations of known length, per-statement
bookkeeping estimated above one second (about 8 ms per statement per iteration),
and no file I/O written in the loop body. A loop that appends to a list is an
in-place change, so as one unit it is not cached at all. The badge row then shows
`In-place mutation on: out`.

The expensive call inside the loop body (`fetch(e)` in `out.append(fetch(e))`)
is still cached, so usually there is nothing to do. **Fix** when it is not:
build the list with a comprehension, which is cached at any length.

<!-- test:skip reason="illustrative: the comprehension rewrite of an append loop" -->
```python { .nb-cell }
out = [fetch(e) for e in entities]
```

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

Cash reads the cells it did not run from the notebook file. An unsaved edit is
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

## Reporting something not on this page

Please [open an issue](https://github.com/galgtonold/cash/issues) with the cells
that trigger it. Say whether it happens on Run All or only when you re-run one
cell, and count calls rather than timing them: a counter your function appends
to shows what ran.

Limitations of `@cash.cache` functions, such as code passed as an argument, are in
the [decorator guide](decorator.md#code-you-pass-as-an-argument).
