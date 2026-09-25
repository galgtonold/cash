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

<!-- claim: cash/object_hashing.py:builtin_hash @bd4210c7, cash/object_hashing.py:compute_hash_full @244ad49e -->
Both paths fingerprint data values with the same built-in hashers. The
decorator uses them for arguments; the notebook uses them for a loop
iteration's values and for the arguments of a cached call inside a statement.

| Type | Library | What is hashed |
|------|---------|----------------|
| `DataFrame`, `Series` | pandas | column and index labels and axis names, dtypes (with a categorical's categories and order), an index's `freq`, `attrs`, and every value: Python objects and strings by their pickled form, so `1` and `'1'` differ |
| `ndarray` | numpy | shape, dtype, memory order and every byte |
| `DataFrame`, `Series` | polars | schema and every row; an `Object` column by its values' content |
| `LazyFrame` | polars | the serialised plan, including in-memory data; a plan reading a file holds the path, not the contents |
| `Table`, `RecordBatch` | pyarrow | schema and every row, as Arrow's IPC format writes it: a slice from its offset, a dictionary column with its dictionary |
| `DataFrame`, `Series` | modin | converted to pandas, then hashed as pandas |
| any collection | dask | the task-graph keys and the schema |

Arrays are hashed in full, never sampled. Two arrays with equal values but
different memory order (C and Fortran) are different keys, because
order-sensitive code can treat them differently; a strided view and its
contiguous copy share a key. The dtypes are in a pandas key, so `int64` and
`Int64` columns, or a tz-naive and a tz-aware series, never share an entry.

For other types, register a hasher; see
[custom hashers](../tutorials/feature-guides/custom-hashers.md).

<!-- claim: cash/core.py:Cash.register_hasher @f48a324b, cash/object_hashing.py:compute_hash @7aa554ba -->
=== "Decorator"

    Registered hashers apply to call arguments and to the values inside a
    list, tuple, set or dict argument. The order in which an argument
    is tried against them is under
    [how arguments are hashed](decorator-path.md#how-arguments-are-hashed).

=== "Notebook"

    A registered hasher does not change a statement's key. A variable made by
    a statement is keyed on its lineage, not on its content, so this rarely
    matters. The exception is the output of a `no-cache` statement, whose
    lineage includes a hash of its value.

## In a notebook

### The statement key

<!-- claim: cash/notebook/cache_key.py:compute_cache_key @c2321118 -->
Every statement key is built by one function, from these parts:

```
combined  = source_hash                      # the statement's own text, normalised
          + ":" + input_lineages             # one lineage per input, ordered by variable name
          + [":" + func_source_hashes]       # "name:hash" per called function, sorted
          + [":" + module_source_hashes]     # "name:hash" per local module read, sorted
          + ":occ" + occurrence_index        # 0-based; tells a repeated statement apart
          + [":callees:" + callee_globals]   # "name:lineage" per global a called function reads
          + [":env:" + environment_reads]    # a digest per os.getenv("NAME"), os.environ["NAME"], os.getcwd()

cache_key = namespace + ":" + SHA256(combined)
```

- Bracketed parts are left out when empty.
- The namespace is `stmt` for a statement and `call` for a
  [call inside a statement](../annotations.md#call-level-caching-default-and-cashno-cache-calls),
  so the two never collide.
- A module is not an input lineage. A statement that reads `helpers.load` is
  keyed on `load` and what it reaches inside `helpers`, so editing another
  function in that file leaves it cached.
- An `import` statement also includes the source of the module it binds, so
  re-running it after a reload does not return the old module.
- An environment read is keyed on a digest of the value, never the value
  itself, and the digest also goes into the lineage of what the statement
  assigns.

<!-- claim: cash/notebook/upstream/virtual_lineage.py:VirtualLineage._register_virtual_callable @57dbc4f5 -->
After a restart, Cash computes these keys from the notebook's code before
your `def` cells have run again, so a statement that calls a notebook
function still hits.

**Files are not in the key.** A file you read enters the *lineage* of the
variable the read produced, and it is checked again before each reuse; see
[what counts as a change](invalidation.md#what-counts-as-a-change).

??? warning "Keys survive a restart, not a move to another machine"
    <!-- claim: cash/notebook/statement/file_deps.py:compute_file_hash_component @ce37ff53 -->
    A statement that reads a file folds the file's modification time and size
    into its lineage. Copy the `.cash` folder to another machine, or point two
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

```
lineage(x) = SHA256(
    source_hash                                  # of the statement that produced x
    + ":" + sorted(lineage(i) for i in inputs)   # sorted by hash here, unlike the key
    + [file_component]                           # "path:mtime:size" per file read
    + [func_source_hashes]                       # called functions, sorted
    + [module_source_hashes]                     # local modules, sorted
    + [value_digest]                             # x's content, if its statement is no-cache
)
```

```python { .nb-cell }
a = 1
b = 2
c = a + b      # lineage(c) folds in lineage(a) and lineage(b)
d = c * 2      # lineage(d) folds in lineage(c)
```

### Inputs a statement never names

A statement's inputs come from the names it mentions. Two real dependencies
are not among them, and each has its own part of the key.

<!-- claim: cash/tracking/randomness/lineage.py:hidden_lineage_reads @e9ddd20b, cash/tracking/randomness/lineage.py:hidden_lineage_writes @1369d609 -->
**The random seed.** `x = np.random.rand(3)` mentions no variable, so editing
`np.random.seed(0)` above it would change nothing. Each global random module
therefore gets a hidden lineage variable: a `seed()` call writes it and a draw
reads it. A re-seed changes the draw's key and the lineage of everything built
from it.

**Globals that called functions read.** `r = a(3)` names `a`, not the globals
`a` reads when it runs. Cash follows the called functions' global names and
adds `name:lineage` for each. A name that no longer exists counts as
`ABSENT`, so deleting a function makes the call re-run and raise `NameError`,
as a plain kernel would, instead of replaying a stored value. For the case of
calling a function defined in a later cell, see
[known limitations](../known-limitations.md#a-function-that-calls-one-defined-in-a-later-cell).

### Resolving an input

<!-- claim: cash/notebook/lineage_store.py:resolve_lineage @e6dc6918, cash/object_hashing.py:compute_hash @7aa554ba -->
For each input variable, the statement key uses the first of these that
exists:

1. The lineage the upstream check simulated for it, while a check is running.
2. The lineage recorded when the variable was last assigned.
3. A lineage tag on the value itself.
4. A content hash of the value, *sampled* for large objects (the first rows
   of a DataFrame, the first elements of an array).
5. A hash of `str(value)`.

A value that cannot be pickled at all ends up keyed on its memory address.
The statement still runs and is stored, but that entry cannot be found again
after a restart.
