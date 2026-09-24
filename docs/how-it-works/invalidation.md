# Knowing when to recompute

!!! info "Applies to: both paths"
    Anyone who wants to know which changes make Cash recompute, and how it notices them.

Cash recomputes when something it tracks has changed since the result was
stored. This page lists what it tracks and how it checks.
[Known limitations](../known-limitations.md) lists what it does not see.

## What counts as a change

=== "Decorator"

    A call recomputes when any of these changed:

    - **Its code:** the function's source, the helpers and classes it reaches,
      the module globals it or its helpers read, captured values and defaults.
      [What goes into the state](decorator-path.md#what-goes-into-the-state)
      has the full list.
    - **Its arguments**, compared by content.
    - **A file it read**, or a file named in `file_depends_on=`
      ([files](#files), below).
    - **An environment variable it reads** by name, or the working directory
      it reads with `os.getcwd()`.
    - **The random seed in force**, for a function that draws from a global
      random stream.
    - **Its age**, when `ttl=` is set and the entry is older.

=== "Notebook"

    A statement recomputes when any of these changed:

    - **Its source.**
    - **An input's lineage**, which changes whenever anything upstream of that
      input changed ([below](#lineage-propagation)).
    - **A function or local module it calls.** An imported module of yours
      (outside the standard library and `site-packages`) is tracked as soon
      as a cell imports it, and Cash reloads it in the kernel when you edit
      it. Editing one function in a module re-runs only what uses that
      function.
    - **A global that a called function reads**, even one bound below the
      statement.
    - **A file it or an input read** ([files](#files), below).
    - **An environment variable it reads** by name.
    - **Its age**, when a `ttl` applies. A statement that calls a
      `@cash.cache` function with a shorter `ttl` takes that shorter one.
    - **An in-place change** to a variable it reads
      ([below](#mutation-bumps-the-receivers-lineage)).

### Files

<!-- claim: cash/tracking/file_tracker.py:FileDependencyRegistry._initialize_defaults @b63601b2, cash/tracking/file_tracker.py:_is_read_mode @238e2cb8 -->
Cash records a file when your code reads it through one of these:

| Library | Readers |
|---|---|
| built-in | `open()` in a read mode (`'r'`, `'rb'`, `'r+'`, `'a+'`), and `pathlib.Path.read_text()`, `read_bytes()`, `open()` |
| pandas | every `read_*` function |
| polars | `read_csv`, `read_parquet`, `read_json`, `read_ndjson`, `read_ipc`, `read_avro`, `read_excel`, and `scan_csv`, `scan_parquet`, `scan_ipc`, `scan_ndjson` |
| pyarrow | `csv.read_csv`, `csv.open_csv`, `parquet.read_table`, `parquet.read_pandas`, `feather.read_table`, `feather.read_feather`, `json.read_json` |
| numpy | `load`, `loadtxt`, `genfromtxt`, `fromfile`, `memmap` |
| others | `joblib.load`, `pickle.load` and `json.load` of an opened file, `sqlite3.connect` |
| directories | `glob.glob`, `glob.iglob`, `os.listdir`, `os.scandir`: the directory, so a new matching file counts |
| missing files | `os.path.exists` and `os.path.isfile` when they return `False`, so a file that appears counts |

A file opened for writing only (`'w'`, `'x'`) is not a dependency. For a file
read another way, name it with `file_depends_on=` on the decorator, or see
[custom file sources](../tutorials/feature-guides/custom-file-sources.md).

<!-- claim: cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @3dd62608, cash/tracking/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES_DEFAULT == 268435456 -->
When the result is stored, each file is recorded with its size, modification
time and a content hash. Before the result is reused:

1. A different size means changed.
2. If the size, the modification time (to the nanosecond), the file's identity
   and, on Linux and macOS, the inode change time all match, and the file had
   been left alone for ten seconds before it was hashed, it is unchanged and
   is not read.
3. Otherwise the content hash decides. A bare `touch` does not count as a
   change; a same-size edit within the same second does.

Files over 256 MiB (`file_hash_full_max_bytes`) are hashed from three sampled
regions, so for them the timestamps must also match exactly. The one edit this
misses is on Windows: a same-size write that puts the old modification time
back, or a write through `np.memmap`, which moves no timestamp. See
[an edit that keeps the size and timestamps](../known-limitations.md#an-edit-that-keeps-the-size-and-timestamps).

In a notebook, a statement is checked against the files it read itself and
the files its inputs were built from, so a changed CSV invalidates the whole
chain that read it.

## In a notebook

### Lineage propagation

Each variable's lineage hash encodes everything that fed into it
([the lineage chain](cache-keys-and-lineage.md#the-lineage-chain)). So if
`raw` changes, `clean`, `features` and `model`, built from it, all get new
lineages and miss, however many cells separate them.

<div class="cash-invalidation-playground" markdown="1">

| Cell | If you change `raw` | If you change `features` |
|---|---|---|
| `raw = pd.read_csv('data.csv')` | recompute | cache hit |
| `clean = raw.dropna()` | recompute | cache hit |
| `features = engineer(clean)` | recompute | recompute |
| `model = train(features)` | recompute | recompute |

</div>

### Upstream simulation

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker.check_and_reexecute @057d4768, cash/notebook/upstream/simulator.py:NotebookSimulator.simulate_upstream @877db1a9 -->
You edit cell 1, then run cell 3 directly. Before cell 3 runs, Cash reads the
notebook's current cells and *simulates* the cells above: it computes, from
their code alone and without running them, the lineage each statement would
produce, and compares it with the lineage from the last real run. Statements
whose lineage differs run again; a value that matches is used as it is, and one
missing from memory is restored from the cache.

Only what the cell you run depends on is considered. A stale chart or export
above it that it does not read stays as it is. A statement that writes a file
counts as needed when something the cell depends on reads that file, even
through a helper or a loop over a list of paths.

### Finding your notebook

<!-- claim: cash/notebook/server_discovery.py:NotebookCellReaders.read @d0b4a970 -->
To simulate, Cash needs the notebook's current cells. It tries, in order:
cells pushed by Cash's JupyterLab extension before each run, Colab's
frontend, and VS Code's unsaved-changes backup. All three see edits you have
not saved. Otherwise it reads the saved `.ipynb`, found through VS Code's
notebook variable, the `ipynbname` package or the Jupyter server API. See
[editing without saving](../known-limitations.md#editing-without-saving).

If none of these works (a plain IPython shell, for example), upstream checking
is off for the session and Cash says so once
([`NOTEBOOK-NOT-FOUND`](../warnings.md#notebook-not-found)). Each cell still
uses its own code and inputs. Cash never guesses the notebook from the files
on disk.

### Mutation bumps the receiver's lineage

<!-- claim: cash/analysis/mutation_effects.py:classify_receivers @704f9e6f -->
`items.append(x)` assigns nothing, but it changes `items`. When Cash decides a
method call changed its receiver, the receiver gets a new lineage from that
statement, so everything built from it downstream misses. How Cash decides
which calls change their receiver is under
[the mutation problem](safety.md#the-mutation-problem).

### Randomness: re-seeding invalidates the draws below it

<!-- claim: cash/tracking/randomness/lineage.py:hidden_lineage_writes @1369d609 -->
Three rules keep random draws right when you edit a seed:

- **A seed is an input.** A `seed()` call sets a hidden lineage variable that
  every later draw from that module reads, so a re-seed changes the draw's key
  and everything built from it.
- **A restored draw restores the generator.** A cached draw also stores the
  generator state it left behind, and restores it only while the same seed is
  in force.
- **A re-run draw starts from the right place.** If a draw re-runs because an
  ordinary input changed, Cash first puts the generator where it would be in a
  top-to-bottom run.

!!! warning "An unseeded draw is usually frozen, not redrawn"
    A re-run of an unseeded draw gives the same number: a cached draw comes
    back from the cache, and a cheap one from the `random`, `numpy.random` or
    `torch` stream repeats because the stream is rewound. A cheap draw from a
    generator held in a variable (`rng = np.random.default_rng()`) is not
    rewound and changes on every run. `# @cash:allow-random` only silences the
    warning; `# @cash:no-cache` on a line of its own makes the statement draw
    fresh every run. See [Randomness](../known-limitations.md#randomness).
