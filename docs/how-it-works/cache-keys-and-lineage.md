# Cache keys and lineage

!!! info "Applies to: both paths"
    Anyone who wants to know what a cache key is made of. Most of this page is about the notebook; the decorator's key has [its own page](decorator-path.md#the-key).

A cache key is a fingerprint of a computation: the same code over the same
inputs always gives the same key, so a hit means the stored result can be
reused. An edited line, a changed input or an edited helper gives a different
key, and a miss.

=== "Decorator"

    A call is keyed on `function:state:dynamic:args`: the function's name, its
    code and everything it reads that is not an argument, any
    `dynamic_depends_on=` values, and the arguments hashed by content.
    [How `@cash.cache` decides](decorator-path.md#the-key) has the full list.
    Files are not in the key; they are checked by content before a stored
    value is returned.

=== "Notebook"

    A statement is keyed on its own source and on the *lineage* of each
    variable it reads, plus the functions and modules it calls.
    [In a notebook](#in-a-notebook), below, explains both.

## Hashing values

<!-- claim: cash/content_hashers.py:builtin_hash @54c21e7f, cash/value_hash.py:compute_hash @98f5643e -->
Both paths fingerprint data values with the same built-in hashers. The
decorator uses them for arguments; the notebook uses them wherever it hashes
a value: a loop iteration's values, the arguments of a cached call inside a
statement, and a variable with no lineage.

With the [C fast path](../getting-started/installation.md#the-c-fast-path)
on, cash watches a big container value it has hashed twice (its dicts,
objects, classes and functions through CPython's watchers, its lists and sets
by their item pointers) and hands back the fingerprint it computed last time
while nothing in it has changed. The fingerprint is the same either way; a
value the watchers cannot fully cover is hashed every time.

| Type | Library | What is hashed |
|------|---------|----------------|
| `DataFrame`, `Series` | pandas | column and index labels and axis names, dtypes (with a categorical's categories and order), an index's `freq`, `attrs`, and every value: numbers, booleans and dates from their memory, text columns of the Arrow-backed `str` type from their Arrow memory, other Python objects and strings by their pickled form, so `1` and `'1'` differ (a column holding only floats, only bools or only ints as the typed array it converts to) |
| `ndarray` | numpy | shape, dtype, memory order and every byte; an object array by its items, column by column |
| `DataFrame`, `Series` | polars | schema and every value, each column from its Arrow memory; an `Object` column by its values' content |
| `LazyFrame` | polars | the serialised plan, including in-memory data; a plan reading a file holds the path, and the decorator records the files it scans as the call's reads |
| `Table`, `RecordBatch`, `Array`, `ChunkedArray` | pyarrow | schema or type and every row, each column from its offset; a dictionary column by its values |
| sparse matrices and arrays | scipy.sparse | type, shape, dtype, format and every array the format stores; a DOK or LIL matrix as the CSR matrix it converts to |
| `DataFrame`, `Series` | modin | converted to pandas, then hashed as pandas |
| any collection | dask | the task-graph keys and the schema |

<!-- claim: cash/content_hashers.py:ArrowDigests.digest @55bde937, cash/backends/frame_sharing.py:arrow_data_owned @d173c9fa -->
A pyarrow value whose memory Arrow allocated itself (read from a file,
computed by Arrow, built from Python values) cannot change, so it is read
once and then recognised for as long as it exists. One over memory something
else can write, such as `pa.array(a)` or `pa.table({"x": a})` over a numpy
array `a`, is read every time. A polars value is read every time: one built
over a numpy array shares that array's memory, and polars cannot say which
of its memory is its own.

<!-- claim: cash/bulk_digest.py:_leaf_digests @60885a16, cash/bulk_digest.py:BulkHasher.digest @ce05cc07, cash/bulk_digest.py:LEAF_BYTES == 1048576, cash/bulk_digest.py:MAX_THREADS == 8, cash/bulk_digest.py:BLAKE3_MAX_THREADS == 4, cash/bulk_digest.py:_scheme @93bd31ed -->
The bytes are cut into 1 MiB pieces, hashed with BLAKE3, whose digests are
digested together: a big array or column is hashed on several threads at
once (up to four), every buffer past 16 KiB goes in as its pieces' digest,
and the result is the same however many threads there are. Where the
`blake3` package cannot load, the pieces go through SHA-256 instead, on up
to eight threads. The digest's tag names the algorithm, so an install
hashing with BLAKE3 and one with SHA-256 key the same array apart: a cache
written by one misses under the other, never serves it.

Arrays are hashed in full, never sampled. Two arrays with equal values but
different memory order (C and Fortran) are different keys, because
order-sensitive code can treat them differently; a strided view and its
contiguous copy share a key. The dtypes are in a pandas key, so `int64` and
`Int64` columns, or a tz-naive and a tz-aware series, never share an entry.

<!-- claim: cash/value_hash.py:_pickle_digest @5b9dad60, cash/content_hashers.py:hash_sparse @b0bdb6a6, cash/stored_pickle.py:pickle_value @16e67246 -->
Any other value, such as a networkx graph or a fitted model, is pickled and
the pickle is hashed, so it is compared by content too, at the cost of
pickling it. That pickle is the one cash stores the value as, its large
arrays hashed as arrays are above, so in a notebook a value a statement makes
is pickled once to hash it and store it. Where the stored bytes would miss
something a key must see (an attribute a class's own pickling leaves out),
the key is taken over a pickle that keeps it, in the same format. To key a
type faster, or one that cannot be pickled, register a hasher; see [custom hashers](../tutorials/feature-guides/custom-hashers.md).

<!-- claim: cash/core.py:Cash.register_hasher @f8a61573, cash/value_hash.py:compute_hash @98f5643e -->
=== "Decorator"

    Registered hashers apply to call arguments and to the values inside a
    list, tuple, set or dict argument. The order in which an argument
    is tried against them is under
    [how arguments are hashed](decorator-path.md#how-arguments-are-hashed).

=== "Notebook"

    A registered hasher does not change a statement's key. A variable made by
    a statement is keyed on its lineage, not on its content, so this rarely
    matters. The exception is the output of a statement that runs every time
    (`no-cache`, the clock, a fresh id), whose lineage includes a hash of its
    value.

## In a notebook

### The statement key

<!-- claim: cash/notebook/cache_key.py:compute_cache_key @f900638a -->
Every statement key is built by one function. It joins these parts with
colons, in this order, and hashes the result:

| Part | What it holds |
|---|---|
| Source | A hash of the statement's own text, normalised. |
| Inputs | One lineage per input variable, ordered by variable name. |
| Called functions | `name:hash` per function it calls, sorted. |
| Local modules | `name:hash` per local module it reads, sorted. |
| Occurrence | `occ` and a 0-based index, which tells a statement repeated in the cell apart. |
| Callee globals | `callees:`, then `name:lineage` per global a called function reads. That includes a function the statement reaches through what holds it: `fns[0](3)` with `fns = [make]`, a dict of builders, a `functools.partial`, a function kept as another's default or in its closure, a class attribute, an object of a class of yours, or a library object such as `FunctionTransformer(make)`. Then `heldmod:module:digest` per module of yours that defines the class of an object the inputs hold (`orders` read back with `pickle.load`), by what the class reaches in the module's file and the code the module has loaded, and `helddata:digest` of the modules' data those classes' methods read, as a cell may set it (`mylib.K = 7`). |
| Installed code | `installed:name:...` per function or class of yours installed on a pandas or polars frame, series, index or expression class under an attribute name the statement or a function it calls reads (an accessor, `pd.DataFrame.report = report`), by its code, and `installed-reads:name:lineage` per global that code reads. |
| Environment | `env:`, then a digest per `os.getenv("NAME")`, `os.environ["NAME"]` or `os.getcwd()` it, or a function of yours it calls, reads. |
| Module data | `moddata:`, then `module.name=digest` per piece of a local module's data it, or a function of yours it calls, reads, by its value. A function, or an object of a class of yours, kept in that data (a registry dict, a handler set with a setter) counts by its code too, so editing its body moves the key. |

The key is `namespace:` followed by the SHA-256 of the joined parts, for
example `stmt:ffd3d255…`.

- Called functions, local modules, callee globals, installed code,
  environment reads and module data are left out when there are none.
- A value too large to search to the end for the functions it holds (more
  than two million objects that are not plain values) adds `unfollowed:` and
  a random token, so the key never matches, and the statement runs uncached
  with a reason. So does code that finds a function through a string
  (`globals()["make"]`, `eval("make(3)")`), in the statement or in a function
  of yours it reaches, with `by-string:`: the function it finds, and what that
  reads, is known only when it runs. So does an object of a class whose module
  file cannot be read, with `unkeyed-modules:`. A call inside such a statement
  stores no entry either.
- The namespace is `stmt` for a statement and `call` for a
  [call inside a statement](../annotations.md#call-level-caching-default-and-cashno-cache-calls),
  so the two never collide.
- A function your notebook defines counts as an input by its code and what
  its `def` evaluates, such as default values. The globals its body reads
  count as callee globals, as they are when the statement runs. A restart
  then keys a call the same even when a global changed since the `def` cell
  last ran.
- A module is not an input lineage. A statement that reads `helpers.load` is
  keyed on `load` and what it reaches inside `helpers`, so editing another
  function in that file leaves it cached.
- An `import` statement also includes the source of the module it binds, so
  re-running it after a reload does not return the old module.
- An environment read is keyed on a digest of the value, never the value
  itself, and the digest also goes into the lineage of what the statement
  assigns.

<!-- claim: cash/notebook/upstream/simulated_callables.py:SimulatedCallables.register_def @448d76eb -->
After a restart, cash computes these keys from the notebook's code before
your `def` cells have run again, so a statement that calls a notebook
function still hits.

**Files are not in the key.** A file you read enters the *lineage* of the
variable the read produced, and it is checked again before each reuse; see
[what counts as a change](invalidation.md#what-counts-as-a-change).

??? warning "Keys survive a restart, not a move to another machine"
    <!-- claim: cash/notebook/statement/file_deps.py:compute_file_hash_component @903b4459 -->
    A statement that reads a file folds the file's modification time, size
    and inode change time into its lineage, and one that only checks whether
    a path exists folds whether it is there. A file rewritten with the same
    bytes keeps the times cash saw before once a content check has found it
    unchanged, so what is built on it keeps its lineage after a restart. Copy the `.cash` folder to another machine, or point two
    machines at one shared backend, and those statements miss there, because
    the two machines compute different keys. The decorator checks files by
    content instead, so its entries can survive a move while the file paths
    still resolve. See
    [sharing a cache](../tutorials/feature-guides/sharing-caches.md).

### The lineage chain

Each variable carries a **lineage hash** that encodes the code that produced it
and the lineage of everything that fed into it. Edit `a`, and `lineage(a)`
changes; `c` was built from `a`, so `lineage(c)` changes too, and so on down
the chain. You declare no dependencies.

`lineage(x)` is the SHA-256 of these parts:

| Part | What it holds |
|---|---|
| Source | A hash of the statement that produced `x`. |
| Inputs | The lineage of each input, sorted by hash (the key orders them by name). |
| Files | `path:mtime:size` per file read. |
| Called functions | Their source hashes, sorted. |
| Local modules | Their source hashes, sorted. |
| Callee globals | A digest of the key's callee globals part, less the globals the statement itself changes, when a function the statement calls reads globals. |
| Value | A digest of `x`'s content, only when its statement is `no-cache`, or runs every time because what it binds changes from run to run: the clock, a fresh id, a `@stateful` function, pyplot's current figure read through a function of yours. |

Files, called functions, local modules, callee globals and the value are left
out when there are none.

```python { .nb-cell }
a = 1
b = 2
c = a + b      # lineage(c) folds in lineage(a) and lineage(b)
d = c * 2      # lineage(d) folds in lineage(c)
```

<!-- claim: cash/notebook/value_cutoff.py:ValueCutoff.lineage_for @7f9a2f17, cash/notebook/value_cutoff.py:_COMPARED @18236e1f, cash/notebook/statement/lineage.py:StatementLineageBuilder._hash_once @5f30bf0b -->
**A value rebuilt to the same content keeps its lineage.** When a statement
runs again because an input moved on, and the pandas frame or series, numpy
array or networkx graph (`Graph`, `DiGraph`, `MultiGraph`, `MultiDiGraph`)
it binds has the same content as one an earlier statement bound, it takes
that one's lineage, and what was built on it is served. A table filtered
back to the rows it held, or a giant component that new data outside it left
unchanged, does not run the analyses below it again. The comparison reads
the whole value: for a graph its class, its nodes and each node's edges in
order, and every node, edge and graph attribute by content. Any difference
keeps the new lineage. The graph's digest is the one cash already takes of a
statement's output, so this costs no extra pass. After a restart a value
restored from the cache counts as seen, and the equalities found before are
kept in the cache folder. A statement below that also depends on a file that
changed since (new rows read upstream) still runs, as any reader of that file
does; the library calls inside it that are keyed on the value are served.

### Inputs a statement never names

A statement's inputs come from the names it mentions. Two real dependencies
are not among them, and each has its own part of the key.

<!-- claim: cash/tracking/randomness/lineage.py:hidden_lineage_reads @e9ddd20b, cash/tracking/randomness/lineage.py:hidden_lineage_writes @1369d609, cash/tracking/randomness/lineage.py:advanced_rng_lineage @d2d070a9, cash/tracking/randomness/lineage.py:drawn_rng_vars @e538facc, cash/notebook/callee_reach.py:helper_seeded_modules @8917bd4e -->
**The random seed.** `x = np.random.rand(3)` mentions no variable, so editing
`np.random.seed(0)` above it would change nothing. Each global random module
therefore gets a hidden lineage variable: a `seed()` call writes it, and a draw
reads it and moves it on, to a lineage made from the draw's own key. A re-seed
changes the draw's key and the lineage of everything built from it, and so
does a change in the draws above it: a cell that draws 5 numbers instead of 3,
or a draw cell inserted, removed or moved. A draw from a stream no seed set
leaves the variable unset, so an unseeded draw stays frozen. A seed set
through a function defined in the notebook (`set_seed(42)`, whose body calls
`random.seed` and `np.random.seed`) writes the variables as the calls written
out would.

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._key_a_newly_seen_draw @0ff7f4ba -->
A draw inside a function the statement calls (`X = make_data(1000)`) names no
module. The first run sees the seeded stream move, and from then on the
statement's key reads that module's variable too. The value is stored under
that key on the run that saw the draw, so the next Run All restores it. When
that key cannot be built yet (an input changed while the statement ran, or the
statement also seeds), the value is not stored on that run.

**Globals that called functions read.** `r = a(3)` names `a`, not the globals
`a` reads when it runs. cash follows the called functions' global names and
adds `name:lineage` for each. A name that no longer exists counts as
`ABSENT`, so deleting a function makes the call re-run and raise `NameError`,
as a plain kernel would, instead of replaying a stored value.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker._callee_globals @c63352d8 -->
Those names count as the cell's inputs for the upstream check too, so a
function defined in a later cell is followed like one above it:

<!-- test:skip reason="illustrative: shows an edit of a later cell reaching an earlier caller" -->
```python { .nb-cell }
def a(n): return b(n) * 2      # cell 1: b does not exist yet
def b(n): return n + 1         # cell 2
r = a(3)                       # cell 3: 8

# edit cell 2 to `return n + 10`, run cell 3 alone:
# cell 2 runs again, r is 26
```

### Resolving an input

<!-- claim: cash/notebook/lineage_store.py:resolve_lineage @778e6d73, cash/value_hash.py:compute_hash @98f5643e -->
For each input variable, the statement key uses the first of these that
exists:

1. The lineage the upstream check simulated for it, while a check is running.
2. The lineage recorded when the variable was last assigned.
3. The lineage tag cash keeps for that object (beside it, never in its
   attributes).
4. A content hash of the value, over all of it: every row of a DataFrame,
   every element of an array or a list.
5. A hash of `str(value)`.

A value that cannot be pickled at all ends up keyed on its memory address.
The statement still runs and is stored, but that entry cannot be found again
after a restart.

## Related

- [The decorator path](decorator-path.md#the-key): the decorator's key, part
  by part.
- [Knowing when to recompute](invalidation.md): which changes move a key or a
  lineage.
- [Custom hashers](../tutorials/feature-guides/custom-hashers.md): fingerprint
  a type cash does not know.
