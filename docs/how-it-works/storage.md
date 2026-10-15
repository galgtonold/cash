# Where your cache lives

!!! info "Applies to: both paths"
    Anyone who wants to know where cash writes, how much it keeps, and what it deletes first.

By default cash keeps results in two tiers: memory, for the running process,
and a cache folder on disk, which survives a restart.

## Where the cache folder is

<!-- claim: cash/_location.py:_own_project_anchor @38fe0ec3, cash/_location.py:_own_installed_entry_point_cache_dir @7a17793c, cash/config/schema.py:CashConfig.cache_dir == ".cash" -->
Unless you set it, the folder is called `.cash`, and where it goes depends on
what is running:

=== "Decorator"

    - **A script** (`python train.py`): in the project root above the script,
      which is the first folder up that holds a `pyproject.toml`, `setup.py`,
      `setup.cfg` or `.git`, or a `.cash` cache folder cash already made there
      (not in your home folder or above it). Without one, next to the script.
      The same script uses the same cache whichever directory you run it
      from.
    - **Installed code run inside a project** (`pytest`, a console script,
      `python -m` of an installed package): the root of the project you are
      in.
    - **An installed tool run outside any project**: a per-user folder named
      after the tool: `~/.cache/cash/<tool>` on Linux (or under
      `$XDG_CACHE_HOME`), `~/Library/Caches/cash/<tool>` on macOS,
      `%LOCALAPPDATA%\cash\<tool>` on Windows.
    - **A REPL or `python -c`**: the current directory.
    - **A pool worker** (`multiprocessing`, `ProcessPoolExecutor`, joblib):
      the folder of the process that started it.

=== "Notebook"

    The kernel's working directory, which Jupyter and VS Code set to the
    notebook's folder. Every notebook in a folder shares one `.cash`.

<!-- claim: cash/_location.py:announce_new_cache_dir @5260de8d -->
When a script creates the default folder, cash says where, once
([`CACHE-DIR-NEW`](../warnings.md#cache-dir-new)), and names an existing cache
nearby if there is one.

To choose the folder yourself, set `CASH_CACHE_DIR`, `cache_dir` under
`[tool.cash]` in `pyproject.toml`, or `Cash(cache_dir=...)`. A relative path
in `pyproject.toml` is relative to that file; one in the environment variable
or in code (including `cash.configure(cache_dir=...)`) is relative to the
current directory. The path is made absolute when the setting is read, so a
later `os.chdir()` does not move it. A tier's own `cache_dir` follows the same
rule.
[Configuration](../getting-started/configuration.md) lists every setting and
which one wins.

<!-- claim: cash/backends/adaptive_caps.py:DISK_FRACTION == 0.25, cash/backends/adaptive_caps.py:DISK_FLOOR @a6159b9c, cash/backends/adaptive_caps.py:DISK_CEILING @774c769b, cash/backends/adaptive_caps.py:DISK_SAFETY == 0.8, cash/backends/adaptive_caps.py:RAM_FRACTION == 0.20, cash/backends/adaptive_caps.py:RAM_FLOOR @bb77c2b3, cash/backends/adaptive_caps.py:RAM_CEILING @a21285e0, cash/backends/adaptive_caps.py:adaptive_disk_cap_for @eb1486f7 -->
Each tier has a size cap, sized to the machine unless you set one:

| Tier | Default cap |
|---|---|
| Disk | A quarter of the room on the cache's volume (free space plus what the cache already holds), at least 8 GiB and at most 100 GiB, and never more than 80% of that room. Set `max_cache_size` to fix it. |
| Memory | A fifth of the memory the process may use (the machine's RAM, or a container's limit if lower), at least 512 MiB and at most 4 GiB. |

<!-- claim: cash/backends/budget_notices.py:describe_budget @663afadf, cash/backends/adaptive_caps.py:disk_cap_reason @8e67909a, cash/backends/file_eviction.py:FileEvictor.budget @9ff2d2c0 -->
cash says how big the disk cache may grow when caching starts, once per
process and cache folder. The line names the folder, the cap and the rule that
set it, so a cache of many GiB is never a surprise:

```text title="Output"
caching in /work/.cash, up to 26.0 GiB (a quarter of the free disk space; set max_cache_size to change it)
```

When the 8 GiB floor or the 100 GiB ceiling decided, it says that instead. On
a volume too full for the 8 GiB floor, it reads `(80% of the free disk space,
the most cash takes; the 8 GiB minimum does not fit)`. A cap you set reads
`(set by max_cache_size)`.

=== "Decorator"

    <!-- claim: cash/backends/file_eviction.py:FileEvictor.ensure_size_scanned @adb3043c, cash/decorator/run_summary.py:RunSummary._budget @b7ff011b -->
    The first result a process writes to disk logs the line on the
    `cash.storage` logger when you ask for it: `verbose=True`, `debug=True` (or
    `CASH_VERBOSE=1`, `CASH_DEBUG=1`), or a level of INFO you set on the `cash`
    or `cash.storage` logger yourself. An application that logs everything at
    INFO from the root logger does not get it on every run. `summary=True` names the cap on the summary's `cache:` line.

=== "Notebook"

    <!-- claim: cash/notebook/ipython/magics.py:CashMagics._show_disk_budget @e5d4f9bb -->
    `%cash_on` prints it, capitalised, under `Cash enabled.`.

<!-- claim: cash/__main__.py:cmd_info @148bc6bb, cash/__main__.py:cmd_clear @fd92c8a3 -->
`cash info` prints the folder in use, where that setting came from, and each
tier's cap. `cash clear` deletes a cache folder: `cash clear analysis.ipynb` clears
that notebook's whole cache folder, shared with its neighbours, and
`--all` clears the folder in use. `--function NAME` and `--entry ID` delete
less; [the CLI page](../cli.md) has every option.

## The tiers

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.get @549a750b, cash/backends/memory_backend.py:_cheap_to_keep @c2798702 -->
| Tier | Where | Survives a restart? |
|------|-------|---------------------|
| Memory | RAM | No |
| Disk | the cache folder | Yes |

A read tries memory first. A pandas table, or a polars `DataFrame`,
`Series` or `LazyFrame`, found on disk is kept in memory for the next read without a copy (see below). Any other value found on disk is
copied into memory on its second read from disk in the same process, so a
value read once after a restart costs no extra copy. Neither is kept when it
would take more than 90% of the memory cap.

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend._copy_frame @aba2fd9a, cash/backends/frame_sharing.py:_freeze @d0721597, cash/backends/frame_sharing.py:own_parts @5b3bb403, cash/backends/tiered_backend.py:TieredBackend.get @549a750b, cash/backends/memory_backend.py:_copy_polars @d34b28a6 -->
The memory tier keeps a copy of each value that only it holds, so nothing you
do to a value you were handed reaches the stored one, or the other way round.
Under pandas copy-on-write (pandas 3, or `pd.options.mode.copy_on_write = True`
on pandas 2), a table of numbers, dates and text is copied once, when it is
stored, and that copy is shared by every hit: cash marks it read-only and
shared, and each hit is a shallow copy that costs microseconds whatever the
table's size. The table you stored stays as writable as it was. pandas
copies a column of a hit before any write you make through it
(`df.loc[...] = ...`, `df["x"] = ...`, `inplace=True` methods), so the write
stays in your table. A write to a hit that goes past pandas to the shared
memory -- `df["x"].array[0] = ...`, or a NumPy view made writable again --
raises `ValueError: assignment destination is read-only`; write through
`.loc` or `.iloc` instead. The copy keeps the table's columns laid out as
yours were, so `df[cols].to_numpy()` gives a new array you can write
wherever it does on your own table (`X -= X.mean(axis=0)`). A table read from
disk is shared the same way. A table with nullable,
categorical or period columns, a `MultiIndex`, or a subclass of `DataFrame` is
copied on the store and on every hit. A subclass's own attributes (those it
names in `_metadata`, such as a GeoDataFrame's `crs` or an `info` dict) are
copied with it, so editing `df.info` on a result changes neither the stored
table nor a later hit; when such an attribute can be neither deep-copied nor
pickled, the decorator does not cache the table
([`STORE-FAILED`](../warnings.md#store-failed)). A column of lists or dicts is kept as
compact bytes and read back into new lists on every hit. Each copy holds
labels of its own (row and column labels, a categorical column's
categories), so writing one in place (`df.index.array[0] = ...`) changes
neither the stored table nor a later hit. An Arrow-backed column's data
can be your NumPy array itself (`pa.array(a)` and
`to_pandas(types_mapper=pd.ArrowDtype)` copy nothing): each chunk of it
that is not memory Arrow allocated itself (read from a file, computed by
Arrow or pandas) is copied, so writing into `a` after the call changes
nothing stored, and the rest is shared, as nothing but Arrow holds it.
A polars `DataFrame`, `Series` or `LazyFrame` is copied on the store, for
the same reason: `pl.Series(a)` and `pl.DataFrame({"x": a})` copy nothing
from a NumPy array `a`, and polars cannot say which of its memory it
allocated itself. A hit is a clone of the stored copy: it shares that
memory, which polars never writes in place (an edit such as
`df[0, "x"] = ...` or `df.insert_column(...)` changes only the frame it is
made on), so a hit costs well under a millisecond whatever the frame's size.
The Python objects in an `Object` column are copied.
`SQLiteBackend`, `RedisBackend` and `S3Backend` can replace or join these
tiers; see
[choosing a backend](../tutorials/feature-guides/choosing-a-backend.md).

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.get @549a750b, cash/backends/_base.py:effective_ttl @c0f1dab8 -->
An entry's ttl is checked on every tier's copy as it is read, so the memory
copy expires with the disk copy, and a backend used on its own applies the
same rule. The ttl is the decorator's `ttl=`, or else the shorter of the ttl
the entry was written with and the tier's current `default_ttl`.
`cash.cleanup()` and `cash clear --expired` use the same rule.

## What's worth persisting

=== "Decorator"

    Every result is written to memory and to disk, however quick it was to
    compute. The only thing that stops the disk write is a size cap (below).

=== "Notebook"

    <!-- claim: cash/backends/persistence_policy.py:COMPUTE_FLOOR_S == 0.1, cash/config/schema.py:CashConfig.min_cache_savings_pct == 0.2, cash/config/schema.py:CashConfig.min_execution_time_to_cache_seconds == 0.01, cash/backends/value_policy.py:WORTH_CEILING_BYTES_PER_SECOND == 134217728 -->
    Each statement is judged on its own. A result that took less than 10 ms
    is not stored at all. One that took less than 0.1 s stays in memory. Above
    that, it goes to disk only when reading it back is predicted to be at
    least 20% cheaper than recomputing it, and when it costs no more than
    128 MiB of disk per second of compute saved. `# @cash:persist` or
    `%cash_persist on` writes it anyway. The [cost model](../cost-model.md)
    explains the prediction and every setting.

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.set @8aecb084 -->
Each tier turns down a single value too large for its cap, so a 20 MB frame
can be stored in memory and on disk while skipping a Redis tier limited to
10 MB. When a value was meant for disk but every disk tier refused it, cash
keeps it in memory if it fits and warns once
([`CACHE-VALUE-TOO-BIG`](../warnings.md#cache-value-too-big)).

<span id="when-the-disk-fills-up"></span>

## Cache size limit and eviction

<!-- claim: cash/backends/file_backend.py:FileBackend._do_set_sync @d7ba8c7f, cash/backends/file_eviction.py:FileEvictor.evict @697d1c43 -->
Only a write can start eviction. After each write to disk, the background
writer adds the entry's size to a running total. If the total is over the cap,
it deletes entries until the cache is under 90% of the cap, so the next few
writes fit without another round. A read never deletes anything to make room;
it deletes only the entry it read, and only if that entry's `ttl` has run out.

The entries deleted first are those worth least per byte: how long the value
took to compute, times how often it has been read, divided by its size. A
result that took 30 seconds outlives a newer one that took 50 ms, and one huge
cheap value goes before many small expensive ones. Entries nobody reads lose
their standing over time. An entry read since the ranking was made, or one
about to be rewritten, is skipped in that round.

<!-- claim: cash/backends/file_eviction.py:FileEvictor.report_eviction @0a7cf82f, cash/backends/budget_notices.py:eviction_text @172c867a -->
The first time the cap makes cash remove entries, it says how much it removed
and that those were the entries worth least per byte. It says so once per
process and cache folder; later removals only reach the debug log. It is a
notice, not a warning: a cache at its cap is doing its job.

=== "Decorator"

    A `cash.storage` log record, shown the same way as the cap above.

=== "Notebook"

    <!-- claim: cash/notebook/ipython/magics.py:CashMagics._show_storage_notices @91b990ad, cash/notebook/ipython/magics.py:CashMagics._after_cell @ac507171 -->
    A line at the end of the first cell to finish after the write that
    filled the cache:

    ```text title="Output"
    [cash] The cache in /work/.cash reached its 26.0 GiB cap, so cash removed 41 entries (2.6 GiB), ...
    ```

<!-- claim: cash/backends/eviction_log.py:EvictionLog.record @cc6659e6, cash/backends/file_eviction.py:FileEvictor.clear @03ded076, cash/backends/eviction_log.py:EvictionLog.MAX_NOTES == 10000, cash/backends/file_backend.py:FileBackend.eviction_note @7f6d5fd0 -->
cash also notes each entry the cap removes: a hash of its key, when, how long
it took to compute, and its size, never the arguments or the value. It keeps
the newest 10,000 such notes, in `_evicted.log` inside the cache folder, and
`cash clear` removes them with the rest. When a later run misses on a noted
entry, the miss says "evicted to make room", and a recompute of two seconds or
more warns once
([`CACHE-EVICTED-RECOMPUTE`](../warnings.md#cache-evicted-recompute)).

<!-- claim: cash/backends/chunk_store.py:ChunkStore.share @b16f5d81, cash/backends/file_eviction.py:FileEvictor.remove_path @76e08300, cash/backends/chunk_store.py:ORPHAN_CHUNK_AGE == 3600.0 -->
The size counted is what the folder holds: a buffer several entries share
([below](#shared-buffers)) counts once. For the ranking, its bytes are shared
out evenly among the entries holding it, and it is deleted with the last of
them. A chunk no entry holds any more, left by a process killed between
writing it and its entry, goes once it is an hour old.

<!-- claim: cash/backends/file_eviction.py:FileEvictor.touched_since @1e44fa35 -->
The value just written is not protected. If it is the least valuable entry, it
goes first; when that keeps happening, the cap is too small, and
[`CACHE-THRASH`](../warnings.md#cache-thrash) warns once per session.

Each process enforces the cap on its own writes, so several processes sharing
one folder can together go over it before one of them evicts. A long-running
process re-measures the disk about once a minute while it writes. A re-run
notebook statement also drops its older versions when the new one is written.

## When a result reaches the disk

<!-- claim: cash/backends/_writes.py:PendingWrites.wait_for_backlog @1faf5bd4, cash/backends/_writes.py:MAX_BACKLOG_S == 2.0, cash/backends/_writes.py:_finish_before_child @342230fe, cash/backends/file_backend.py:_finish_before_listing @891535e1, cash/backends/file_backend.py:FileBackend.get @5568ac7d -->
The disk write runs on a background thread, and your code carries on while it
does. cash waits for a write only when something needs it on disk:

- a read of that result in the same process,
- code in the same process listing the cache folder,
- another process this one starts (`subprocess`, `os.system`, a `!` line in
  a notebook, a forked worker), for up to 30 seconds,
- the end of the process, for up to `shutdown_write_timeout`
  ([`CACHE-WRITE-ABANDONED`](../warnings.md#cache-write-abandoned)).

A process this one did not start, reading the same folder at the same time,
finds a result once its write has finished, and computes it until then.

<!-- claim: cash/backends/memory_backend.py:_snapshot @579b150a, cash/backends/tiered_backend.py:TieredBackend._stored_payload @92ed63c1, cash/backends/memory_backend.py:InMemoryBackend.set @3eb6a25c, cash/notebook/call_refs.py:digest_and_size @0dc444f6, cash/kept_state.py:stored_views @8fe0d16f, cash/stored_pickle.py:keeping @7af52dad, cash/notebook/statement/store.py:_nestable @73df97fe -->
An object of a class of its own (a fitted model, a graph) is kept in memory
as the bytes its disk entry holds, made when it is stored: its arrays are
copied out of it then, so a later change to your object, by Python or by
NumPy writing into an array, never reaches them. Each hit reads a new copy
out of the bytes, as a disk hit does, and the disk write is those same bytes,
so the object is pickled once. It counts against the memory cap by those
bytes. In a notebook, a cached call's result also gets a digest, so later
statements can refer to the stored entry instead of storing the object again.
That digest is taken from the same pickle, not from a second one. The same
holds for a statement: the content hash cash takes of a value the statement
made is taken over the pickle its entry stores, and the entry holds those
bytes as they are, unless another of the statement's values could share an
object with it. A networkx
graph's views (`G.adj`, `G.nodes`) are left out of the bytes, and networkx
makes them again on first use.

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend.holds_own_copy @d82553cc, cash/backends/memory_backend.py:InMemoryBackend.private_copy @ef57a587, cash/backends/memory_backend.py:InMemoryBackend.write_copy @987f7a47, cash/backends/memory_backend.py:_frames_own @d957b9f5 -->
The disk entry is the result as it was computed: a change you make to it
after the call never reaches the disk. The background write takes the memory
tier's copy when that copy keeps nothing of yours (numbers, plain data, a
table shared as above, a table of nullable or Python-string columns copied
in full, anything copied through pickle), even when the memory tier lets
that copy go as soon as it stored it, to stay under its cap. When the memory
tier has no such copy (the value is over its cap, or slower to read back
than to compute), one is made for the write before the call returns, at about
the cost of copying its memory, if the value is tables of numbers, dates,
text and nullable columns, arrays and plain values: the pickling, checksum
and write still happen in the background. Anything else (a categorical
column, a subclass, an object column holding lists) is serialized before
the call returns instead, which costs its pickle time on a miss.

=== "Decorator"

    A call returns as soon as its result is computed; the write follows.

=== "Notebook"

    A cell does not wait for its writes either: they finish while the next
    cell runs. It waits only for what would take more than about two
    seconds more to write. That bounds what a restart can lose, since
    Jupyter gives the kernel a few seconds to exit before it stops it, and
    what a crash or a killed kernel can lose.

<!-- claim: cash/backends/file_backend.py:FileBackend._write_new_in_place @88db24c4, cash/backends/file_backend.py:FileBackend._atomic_write @1d55d3e9 -->
A write cut short never leaves a readable entry: a new entry's header is
written last, and a replaced one goes to a temporary file first. What a
killed process was still writing is missing, and is computed again.

## Turning objects into bytes

<!-- claim: cash/decorator/store.py:ResultStore.store @e289b59d, cash/backends/serialization.py:PickleSerializer.serialize @0f446a8f -->
Every result is stored with `pickle` (protocol 5), pandas DataFrames
included, so a hit hands back the same frame: its class, index, dtypes, cell
types, `attrs` and `flags`. Pickle is several times faster than Parquet to
write and read. Parquet files are smaller for columns with few distinct values
(small integers, repeated strings, mostly missing values); if disk space
matters more than speed, the [`compress`](../getting-started/configuration.md#all-settings)
setting gets most of that back.

<!-- claim: cash/backends/serialization.py:_reduce_arrow_array @388e99d3, cash/backends/serialization.py:_compact @79a515ab -->
A pandas column backed by Arrow (pandas' `str` columns, `ArrowDtype`) is
stored as its Arrow chunks, written as they are: pandas' own pickle first
joins a column's chunks into one copy. A short slice of a long column is
copied down to its rows, so `df.head()` does not write the whole column.

<!-- claim: cash/backends/serialization.py:_reduce_polars @f7539f57, cash/backends/serialization.py:_polars_from_arrow @654cf355 -->
A polars `DataFrame` or `Series` is stored as its Arrow data, so its
buffers are written to disk and read back as they are, as a pandas frame's
are: it comes back with its dtypes (an `Enum` stays an `Enum`, a time zone
stays), its column names and sorted flags. Without `pyarrow`, or for a dtype
the installed `pyarrow` lacks, it is stored with polars' own pickle, which
is several times slower for a large frame. polars does not pickle an
`Object` column, so a frame with one stays in memory and is computed again
in a new process.

<span id="shared-buffers"></span>

<!-- claim: cash/backends/chunk_store.py:ChunkStore.put @1a6e2443, cash/backends/chunk_store.py:CHUNK_MIN_BYTES @fdf218ec, cash/backends/file_backend.py:FileBackend._write_cache_files @b971c62d -->
Each buffer of a megabyte or more that a value is written with (a column of
a table, an array's data) is stored once per cache folder, in its `chunks`
folder, named by a digest of its bytes. An entry names the chunks it holds.
When another entry already holds the same bytes -- a table with one column
added, the same data stored by two statements, a loop whose every result
carries its input's columns -- the write costs a digest of the buffer instead
of writing it again. The digest is taken with the same algorithm as cash's
other digests of big data (BLAKE3 with `cash-lib[fast]`, SHA-256 otherwise),
and the name says which, so installs hashing differently never share a
chunk. With `compress` on, values are not split this way.

<!-- claim: cash/backends/chunk_store.py:read_chunk @aa8ba979, cash/backends/entry_format.py:_read_apart @97c522a8 -->
A chunk is checked against the checksum its entry recorded each time it is
read. A chunk that is missing or damaged makes the entry a miss, computed
again, and a damaged chunk is removed so the next write stores it afresh.

<span id="compression"></span>

<!-- claim: cash/backends/compression.py:ZSTD_LEVEL == 3, cash/backends/compression.py:ZLIB_LEVEL == 1, cash/backends/compression.py:MIN_SAVING == 0.1, cash/backends/compression.py:compress @9df58020 -->
With `compress` on, each entry is compressed on the background writer: with
zstd on Python 3.14 and later, with zlib before, both at fast settings. cash
first tries the first megabyte; an entry that does not shrink by a tenth
(random floats, images already compressed) is stored as it is. An entry
compressed with zstd on Python 3.14 reads as missing on Python 3.13 or
earlier, so it is computed again there, once: the new entry is written with
zlib, which every Python reads. A zstd entry is never read wrongly.

!!! warning "Only use caches you trust"
    Loading a pickle can run arbitrary code. A cache folder, Redis database or
    S3 bucket is as trustworthy as whoever can write to it. Never point cash at
    one filled by someone you do not trust.

## Damaged entries and format changes

<!-- claim: cash/backends/entry_format.py:pack_entry @cd68bcc1, cash/backends/entry_format.py:_verify @8a91dffb -->
Every entry carries a checksum of its payload, checked on every read. An entry
that does not match (a half-written file, a bad sector, a sync client that
merged two versions) counts as missing, and the value is recomputed. So does
an entry with no checksum. The check finds damage, not tampering.

<!-- claim: cash/backends/cache_dir.py:remove_orphan_temp_files @f7c8f067, cash/backends/file_backend.py:FileBackend._do_set_sync @d7ba8c7f -->
A write that replaces an entry goes to a hidden `.tmp-*.part` file first,
renamed into place once complete. A process killed in between (a kernel
restart, an out-of-memory kill) leaves that file behind. The first write of a
later process on the same machine removes it; one left by another machine
goes once nobody has written to it for a day.

<!-- claim: cash/backends/entry_format.py:update_metadata_in_place @0db18fc6, cash/backends/file_backend.py:FileBackend.get @5568ac7d -->
Several processes can share one folder. Each one counts the reads of the
entries it used and writes those counts back later. When another process has
stored a new result under the same entry meanwhile, the next read takes the
new entry as it is on disk, and the counts are dropped rather than written over
it.

<!-- claim: cash/backends/cache_dir.py:CACHE_FORMAT_VERSION == 7, cash/backends/cache_dir.py:CacheDirStamp.check @7bdce11c -->
The cache folder records the storage format it was written in. When cash opens
a folder written in a different format, by an older or newer cash, it logs a
warning and deletes the old entries, so the first run afterwards recomputes.
Within one format, entry metadata tolerates fields it does not know.

## Related

- [Configuration](../getting-started/configuration.md): every setting,
  including `cache_dir` and `max_cache_size`.
- [Choosing a backend](../tutorials/feature-guides/choosing-a-backend.md):
  SQLite, Redis and S3 tiers.
- [Command-line interface](../cli.md): `cash info`, `cash inspect` and
  `cash clear`.
- [Cost model](../cost-model.md): which notebook results are worth writing to
  disk.
