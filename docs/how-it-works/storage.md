# Where your cache lives

!!! info "Applies to: both paths"
    Anyone who wants to know where Cash writes, how much it keeps, and what it deletes first.

By default Cash keeps results in two tiers: memory, for the running process,
and a cache folder on disk, which survives a restart.

## Where the cache folder is

<!-- claim: cash/_location.py:project_anchor @46e903a7, cash/_location.py:installed_entry_point_cache_dir @76af1800, cash/config.py:CashConfig.cache_dir == ".cash" -->
Unless you set it, the folder is called `.cash`, and where it goes depends on
what is running:

=== "Decorator"

    - **A script** (`python train.py`): in the project root above the script,
      which is the first folder up that holds a `pyproject.toml`, `setup.py`,
      `setup.cfg` or `.git`. Without one, next to the script. The same script
      uses the same cache whichever directory you run it from.
    - **Installed code run inside a project** (`pytest`, a console script,
      `python -m` of an installed package): the root of the project you are
      in.
    - **An installed tool run outside any project**: a per-user folder named
      after the tool: `~/.cache/cash/<tool>` on Linux (or under
      `$XDG_CACHE_HOME`), `~/Library/Caches/cash/<tool>` on macOS,
      `%LOCALAPPDATA%\cash\<tool>` on Windows.
    - **A REPL or `python -c`**: the current directory.

=== "Notebook"

    The kernel's working directory, which Jupyter and VS Code set to the
    notebook's folder. Every notebook in a folder shares one `.cash`.

To choose the folder yourself, set `CASH_CACHE_DIR`, `cache_dir` under
`[tool.cash]` in `pyproject.toml`, or `Cash(cache_dir=...)`. A relative path
in `pyproject.toml` is relative to that file; one in the environment variable
or in code is relative to the current directory. The path is made absolute at
startup, so a later `os.chdir()` does not move it.
[Configuration](../getting-started/configuration.md) lists every setting and
which one wins.

<!-- claim: cash/backends/adaptive_caps.py:DISK_FRACTION == 0.25, cash/backends/adaptive_caps.py:DISK_FLOOR @a6159b9c, cash/backends/adaptive_caps.py:DISK_CEILING @774c769b, cash/backends/adaptive_caps.py:DISK_SAFETY == 0.8, cash/backends/adaptive_caps.py:RAM_FRACTION == 0.20, cash/backends/adaptive_caps.py:RAM_FLOOR @bb77c2b3, cash/backends/adaptive_caps.py:RAM_CEILING @a21285e0, cash/backends/adaptive_caps.py:adaptive_disk_cap_for @5e8f2ff8 -->
Each tier has a size cap, sized to the machine unless you set one:

| Tier | Default cap |
|---|---|
| Disk | A quarter of the room on the cache's volume (free space plus what the cache already holds), at least 8 GiB and at most 100 GiB, and never more than 80% of that room. Set `max_cache_size` to fix it. |
| Memory | A fifth of the memory the process may use (the machine's RAM, or a container's limit if lower), at least 512 MiB and at most 4 GiB. |

<!-- claim: cash/backends/budget_notices.py:describe_budget @663afadf, cash/backends/adaptive_caps.py:disk_cap_reason @988059b0, cash/backends/file_eviction.py:FileEvictor.budget @9ff2d2c0 -->
Cash says how big the disk cache may grow when caching starts, once per
process and cache folder. The line names the folder, the cap and the rule that
set it, so a cache of many GiB is never a surprise:

    caching in /work/.cash, up to 26.0 GiB (a quarter of the free disk space; set max_cache_size to change it)

When the 8 GiB floor or the 100 GiB ceiling decided, it says that instead, and
a cap you set reads `(set by max_cache_size)`.

=== "Decorator"

    <!-- claim: cash/backends/file_eviction.py:FileEvictor.ensure_size_scanned @adb3043c, cash/core.py:Cash._summary_budget @187bcd8a -->
    The first result a process writes to disk logs the line on the
    `cash.storage` logger. `verbose=True`, `debug=True` (or `CASH_VERBOSE=1`,
    `CASH_DEBUG=1`) or your own `logging` at INFO shows it; otherwise it stays
    quiet. `summary=True` names the cap on the summary's `cache:` line.

=== "Notebook"

    <!-- claim: cash/notebook/ipython/magics.py:CashMagics._show_disk_budget @e5d4f9bb -->
    `%cash_on` prints it, capitalised, under `Cash enabled.`.

<!-- claim: cash/__main__.py:cmd_info @f8794ec8, cash/__main__.py:cmd_clear @a08b9044 -->
`cash info` prints the folder in use, where that setting came from, and both
caps. `cash clear` deletes a cache folder: `cash clear analysis.ipynb` clears
that notebook's whole cache folder, shared with its neighbours, and
`--all` clears the folder in use. `--function NAME` and `--entry ID` delete
less; [the CLI page](../cli.md) has every option.

## The tiers

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.get @231ca6c0 -->
| Tier | Where | Survives a restart? |
|------|-------|---------------------|
| Memory | RAM | No |
| Disk | the cache folder | Yes |

A read tries memory first. A value found on disk is copied back into memory
for the next read, unless it would take more than 90% of the memory cap.
`SQLiteBackend`, `RedisBackend` and `S3Backend` can replace or join these
tiers; see
[choosing a backend](../tutorials/feature-guides/choosing-a-backend.md).

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.get @231ca6c0, cash/backends/_base.py:effective_ttl @7c55c336 -->
An entry's ttl is checked on every tier's copy as it is read, so the memory
copy expires with the disk copy. The ttl is the decorator's `ttl=`, or else
the shorter of the ttl the entry was written with and the tier's current
`default_ttl`. `cash.cleanup()` and `cash clear --expired` use the same rule.

## What's worth persisting

=== "Decorator"

    Every result is written to memory and to disk, however quick it was to
    compute. The only thing that stops the disk write is a size cap (below).

=== "Notebook"

    <!-- claim: cash/backends/persistence_policy.py:COMPUTE_FLOOR_S == 0.1, cash/config.py:CashConfig.min_cache_savings_pct == 0.2, cash/config.py:CashConfig.min_execution_time_to_cache_seconds == 0.01, cash/backends/value_policy.py:WORTH_CEILING_BYTES_PER_SECOND == 134217728 -->
    Each statement is judged on its own. A result that took less than 10 ms
    is not stored at all. One that took less than 0.1 s stays in memory. Above
    that, it goes to disk only when reading it back is predicted to be at
    least 20% cheaper than recomputing it, and when it costs no more than
    128 MiB of disk per second of compute saved. `# @cash:persist` or
    `%cash_persist on` writes it anyway. The [cost model](../cost-model.md)
    explains the prediction and every setting.

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.set @8f018587 -->
Each tier turns down a single value too large for its cap, so a 20 MB frame
can be stored in memory and on disk while skipping a Redis tier limited to
10 MB. When a value was meant for disk but every disk tier refused it, Cash
keeps it in memory if it fits and warns once
([`CACHE-VALUE-TOO-BIG`](../warnings.md#cache-value-too-big)).

## When the disk fills up

<!-- claim: cash/backends/file_backend.py:FileBackend._do_set_sync @39e0412a, cash/backends/file_eviction.py:FileEvictor.evict @290394ae -->
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

<!-- claim: cash/backends/file_eviction.py:FileEvictor.report_eviction @aa4b93be, cash/backends/budget_notices.py:eviction_text @172c867a -->
The first time the cap makes Cash remove entries, it says how much it removed
and that those were the entries worth least per byte. It says so once per
process and cache folder; later removals only reach the debug log. It is a
notice, not a warning: a cache at its cap is doing its job.

=== "Decorator"

    A `cash.storage` log record, shown the same way as the cap above.

=== "Notebook"

    <!-- claim: cash/notebook/ipython/magics.py:CashMagics._show_storage_notices @91b990ad, cash/notebook/ipython/magics.py:CashMagics._flush_pending_writes @52688dae -->
    A line at the end of the cell whose results filled the cache:

        [cash] The cache in /work/.cash reached its 26.0 GiB cap, so cash removed 41 entries (2.6 GiB), ...

<!-- claim: cash/backends/eviction_log.py:EvictionLog.record @cc6659e6, cash/backends/file_eviction.py:FileEvictor.clear @03ded076, cash/backends/eviction_log.py:EvictionLog.MAX_NOTES == 10000, cash/backends/file_backend.py:FileBackend.eviction_note @7f6d5fd0 -->
Cash also notes each entry the cap removes: a hash of its key, when, how long
it took to compute, and its size, never the arguments or the value. It keeps
the newest 10,000 such notes, in `_evicted.log` inside the cache folder, and
`cash clear` removes them with the rest. When a later run misses on a noted
entry, the miss says "evicted to make room", and a recompute of two seconds or
more warns once
([`CACHE-EVICTED-RECOMPUTE`](../warnings.md#cache-evicted-recompute)).

<!-- claim: cash/backends/file_eviction.py:FileEvictor.touched_since @1e44fa35 -->
The value just written is not protected. If it is the least valuable entry, it
goes first; when that keeps happening, the cap is too small, and
[`CACHE-THRASH`](../warnings.md#cache-thrash) warns once per session.

Each process enforces the cap on its own writes, so several processes sharing
one folder can together go over it before one of them evicts. A long-running
process re-measures the disk about once a minute while it writes. A re-run
notebook statement also drops its older versions when the new one is written.

## Turning objects into bytes

<!-- claim: cash/backends/serialization.py:get_serializer @76cf2c1b, cash/backends/serialization.py:ParquetSerializer.serialize @97962311, cash/backends/serialization.py:_parquet_keeps @7d0d2a54 -->
A pandas `DataFrame` is stored as Parquet when pyarrow or fastparquet is
installed, and comes back as it was stored, `RangeIndex` included. A frame
Parquet cannot give back unchanged (non-string, duplicate or multi-level
column labels, an index with a frequency, no columns, or a column Parquet
cannot convert) is pickled instead. Everything else, and a DataFrame without a
Parquet engine, is pickled.

!!! warning "Only use caches you trust"
    Loading a pickle can run arbitrary code. A cache folder, Redis database or
    S3 bucket is as trustworthy as whoever can write to it. Never point Cash at
    one filled by someone you do not trust.

## Damaged entries and format changes

<!-- claim: cash/backends/entry_format.py:pack_entry @f6d2f3e2, cash/backends/entry_format.py:_verify @8a91dffb -->
Every entry carries a checksum of its payload, checked on every read. An entry
that does not match (a half-written file, a bad sector, a sync client that
merged two versions) counts as missing, and the value is recomputed. So does
an entry with no checksum. The check finds damage, not tampering.

<!-- claim: cash/backends/cache_dir.py:CACHE_FORMAT_VERSION == 2, cash/backends/cache_dir.py:CacheDirStamp.check @c89cf812 -->
The cache folder records the storage format it was written in. When Cash opens
a folder written in a different format, by an older or newer Cash, it logs a
warning and deletes the old entries, so the first run afterwards recomputes.
Within one format, entry metadata tolerates fields it does not know.
