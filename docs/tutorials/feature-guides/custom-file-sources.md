# Custom file sources — automatic file tracking and the escape hatch

Cash automatically tracks file dependencies. When you call `pd.read_csv('data.csv')` inside a cached function, the file's mtime and size get recorded; the next time you call the function, the cache invalidates if the file changed. This guide covers what's tracked, what isn't, and how to add tracking for non-standard access patterns.

## Why this exists

Without file tracking, every CSV/parquet load you make from a cached function would either *always* hit the cache (silently stale when the file changes on disk) or *always* miss (slow). Neither is acceptable. The conservative middle is to record a cheap fingerprint of every file the function reads — mtime and size — and invalidate when that fingerprint moves. You get hits on identical inputs and re-runs on changed inputs without thinking about it.

The mechanism is a one-time monkey-patch of the popular reader functions: when a `@cash.cache` function executes, Cash installs `FileAccessTracker` around the call, intercepts reads from `builtins.open`, pandas, polars, numpy, joblib, json, and pickle, and stores the resulting file dictionary in the cache metadata. On the next lookup, Cash re-stats every recorded file and re-runs the function if anything moved.

## Quick start

<!-- test:skip reason="demonstrates file mutation (edit file between calls) which cannot be reproduced in a single script" -->
```python
import cash
import pandas as pd

@cash.cache
def load_features():
    return pd.read_csv("data/features.csv")     # auto-tracked

load_features()                                  # compute + record file
load_features()                                  # hit
# (edit data/features.csv on disk)
load_features()                                  # MISS — file_changed
```

No decorator argument, no manual registration. Cash sees the `read_csv` call, records the path, and notices the next time the mtime or size differs.

## What's automatically tracked

The default handler set is registered in `FileDependencyRegistry._initialize_defaults` (`src/cash/notebook/file_tracker.py:148-185`):

| Module | Functions |
|---|---|
| `builtins` | `open()` (any read mode — `'r'`, `'r+'`, `'rb'`, `'r+b'`, …) |
| `io` | `open()` (the pathlib-backed variant) |
| `pandas` | `read_*` — every reader: `read_csv`, `read_parquet`, `read_excel`, `read_json`, `read_pickle`, `read_feather`, `read_hdf`, `read_orc`, `read_sas`, `read_spss`, `read_stata`, `read_table`, `read_xml`, `read_html`, `read_fwf`, `read_clipboard`, `read_sql*` |
| `polars` | `read_csv`, `read_parquet`, `read_json`, `read_ndjson`, `read_ipc`, `read_avro`, `read_excel`, plus the lazy variants `scan_csv`, `scan_parquet`, `scan_ipc`, `scan_ndjson` |
| `numpy` | `load`, `loadtxt`, `genfromtxt`, `fromfile` |
| `joblib` | `load` |
| `pickle` | `load` |
| `json` | `load` |

The pandas entry is the glob `read_*`, expanded by `_find_patch_targets` (`src/cash/notebook/file_tracker.py:63-70`) against the live `pandas` module — so any reader pandas adds in a future release is picked up too. Both top-level reads (`pd.read_csv`) and submodule reads (`pd.read_csv` via the `pandas.io.parsers` shim) flow through the patched attribute.

For `open()`, the wrapper records the path only when the mode contains `'r'` or `'+'` (read or read/write), not pure writes — see `_create_open_handler` at `src/cash/notebook/file_tracker.py:216-222`. So an `open(path, 'w')` for output does *not* get tracked, which is what you want: writes are accounted for by hashing the function's return value, not its outputs.

## How to verify what's tracked

`cache_info()` and `f.explain()` surface the recorded dictionary. After a compute, the entry's metadata holds `auto_file_deps`:

<!-- test:skip reason="demonstrates file mutation between calls which cannot be reproduced in a single script" -->
```python
import cash
import pandas as pd

@cash.cache
def load_features():
    return pd.read_csv("data/features.csv")

load_features()
load_features.explain()
# CacheExplanation(would_hit=True, reason='hit', ...)

# Edit data/features.csv, then:
load_features.explain()
# [MISS] __main__.load_features — file_changed
#   changed_files: {'data/features.csv': 'mtime changed'}
```

The `file_changed` reason and the `changed_files` dict are emitted by `_explain_call` at `src/cash/core.py:878-900`. The dict's values are short human-readable strings: `'mtime changed'`, `'size changed'`, or `'file missing'`. See [Debugging and Monitoring](debugging-and-monitoring.md) for the full `explain()` story.

## What's NOT tracked

The patch set is a curated list. Reads that go through anything else slip past the tracker:

- **Direct `pyarrow` / `fastparquet` calls** — `pyarrow.parquet.read_table('data.parquet')` is not patched. `pd.read_parquet(...)` (which calls pyarrow internally) *is* — the patch is at the pandas entry point.
- **Specialized format libraries** — `feather.read_dataframe`, `h5py.File`, `netCDF4.Dataset`, custom binary readers in vendored utilities.
- **C extensions and subprocesses** — anything that opens a file descriptor outside the Python-level `open()` (e.g. a C library called via `ctypes`, a `subprocess.run` that reads the file) is invisible. The monkey-patch only intercepts Python-side dispatch.
- **Database files** — `sqlite3.connect('db.sqlite')` or a SQLAlchemy engine pointed at a file URL doesn't open the file via the patched readers. The query itself goes through the driver and Cash sees nothing.
- **Lazy scans you don't materialize** — `polars.scan_csv(...)` *is* tracked at scan time, but a remote URL passed to any reader is not (the path-arg handler at `src/cash/notebook/file_tracker.py:233-238` only records `str | bytes | os.PathLike` values that point at real filesystem entries).

For each of these gaps, use the `file_depends_on=` escape hatch.

## Escape hatch 1: `file_depends_on=` on `@cash.cache`

When Cash can't see the read, tell it explicitly:

```python
import cash
import pyarrow.parquet as pq

@cash.cache(file_depends_on="data/events.parquet")
def load_events():
    return pq.read_table("data/events.parquet").to_pandas()
```

`file_depends_on` accepts a single path or a list:

```python
@cash.cache(file_depends_on=["models/embeddings.bin", "models/vocab.json"])
def load_model():
    return MyModel.from_disk("models/embeddings.bin", "models/vocab.json")
```

Under the hood, `_register_func` wraps each path in a `FileDataSource` and folds it into the function's static dependency list (`src/cash/core.py:669-673`). `FileDataSource.has_changed()` re-reads the file's mtime on every lookup (`src/cash/data_source.py:41-43`); a change propagates into the cache key and forces a miss.

A subtle behavior worth knowing: `FileDataSource.__init__` snapshots the mtime *at decoration time* (`src/cash/data_source.py:30`). If the file doesn't exist yet when the decorator runs, the snapshot is `0.0` (the OSError fallback at `src/cash/data_source.py:34-36`). That's fine — the next stat sees the real mtime and triggers a miss for the first real run. But it means `file_depends_on` on a not-yet-created file does *not* fail loudly; you have to remember it's there.

## Escape hatch 2: registering a custom file source for auto-tracking

For libraries you use across many cached functions, manually adding `file_depends_on=` to each decorator is repetitive. `Cash.register_file_handler` (`src/cash/core.py:2380-2458`) lets you teach the auto-tracker about a new reader once and have every subsequent call site picked up automatically:

<!-- test:skip reason="fictional my_lib module cannot be registered without existing in sys.modules" -->
```python
import cash

c = cash.Cash()

# my_lib.read_data(path) reads a custom binary format.
def custom_reader_handler(original_func, track_callback):
    def wrapper(path, *args, **kwargs):
        track_callback(path)              # record the dep
        return original_func(path, *args, **kwargs)
    return wrapper

c.register_file_handler("my_lib", "read_data", custom_reader_handler)

@c.cache
def load_features():
    import my_lib
    return my_lib.read_data("/data/features.bin")
    # ^ now auto-tracked, no file_depends_on= needed
```

The handler is a factory: Cash calls it with the original function and a `track_callback(path)` shim; your wrapper records and forwards. `func_name` supports glob patterns (`"read_*"` catches every reader in one call), and `module_name` may be dotted (`"my_lib.io"`). The wrapper is installed on the live module via the same `_install_module_patches` path used for the built-ins (`src/cash/notebook/file_tracker.py:73-106`).

Two caveats from the docstring:

- The wrapper replaces the attribute on the module object, so existing imports (`from my_lib import read_data`) still see the original unwrapped version. Track via the module namespace (`my_lib.read_data(...)`) or import after registering.
- Pass an absolute or resolvable path to `track_callback`. Relative paths are resolved against `os.getcwd()` at tracking time by `_track_path` (`src/cash/notebook/file_tracker.py:377-386`).

## Staleness detection

Cash checks freshness on every lookup, not at write time. `_auto_file_deps_fresh` (`src/cash/core.py:1026-1049`) walks the recorded dictionary, stats each path, and short-circuits to a miss as soon as any of three things is true:

- The file is unreadable (`os.stat` raises) — recorded as `file missing` in `explain()`.
- The mtime moved — `st.st_mtime != recorded['mtime']`.
- The size moved — `st.st_size != recorded['size']`.

**Why mtime+size and not a content hash?** Hashing a 2 GB parquet file on every cache lookup would defeat the point of caching. mtime+size is one `stat()` syscall per file — sub-millisecond, no I/O on the data itself. The trade-off: if something rewrites a file without changing the mtime *and* the size matches byte-for-byte (rare, but possible with coarse-resolution filesystems and content-preserving touches), Cash will miss the change. In practice the false-negative rate is low enough that nobody cares.

The notebook-statement layer uses a slightly different policy in `_invalidate_if_direct_file_changed` (`src/cash/notebook/cache_freshness.py:149-185`): it tolerates mtime deltas under 10 ms and falls back to a size check, which handles filesystems with coarse mtime granularity (HFS+, some ext4 configs). The decorator path is the strict "any drift counts" version above.

Race condition to be aware of: if a file is rewritten *while* a cached function is running, the snapshot captures the post-write mtime. On the next call Cash sees the same mtime and returns the cached value — which now reflects half-old, half-new data. The window is small and rarely matters, but for high-churn pipelines wrap the write in a tempfile-then-rename so each run sees a consistent snapshot.

## Caveats

### Symlinks are followed

`_track_path` resolves the path through `os.path.realpath` before storing it (`src/cash/notebook/file_tracker.py:382`). If you read a symlink, Cash records and stats the *target*. Editing the symlink target invalidates the cache; replacing the symlink to point at a different file with the same target mtime+size does not. This matches what most users expect ("the data file changed"), but if you genuinely care about the symlink identity, you'll need to use `file_depends_on=` with the link path explicitly.

### Paths are absolute and platform-normalized

Stored paths are absolute and use forward slashes regardless of OS (`normalize_path` in `src/cash/utils.py:24-36`). Moving the cache directory to a different machine where the same files live at different paths invalidates everything — paths are part of the dependency key. For portable cache archives, expect a full re-compute after relocation.

### Network-mounted filesystems

NFS, SMB, and similar network mounts often have coarse mtime resolution (1-second granularity) and the timestamp source is the *server*, not the client. Two writes within the same second can produce identical mtimes — the size check is your safety net but won't catch in-place edits that preserve size. This isn't a Cash-specific limitation; it's general advice for any tooling that relies on mtime. If you're hitting it, switch to `file_depends_on=` and write a `FileDataSource` subclass that uses a content hash on critical files.

### Files outside the working directory

The tracker records full absolute paths and stats them on every lookup. There's no working-directory boundary — a cached function that reads `/etc/hosts` or `~/.config/foo.yml` will invalidate when those files change, which is usually right but occasionally surprising in containerized or CI environments where unrelated host files churn.

### Writes are intentionally not tracked

`FileAccessTracker` only intercepts read paths (`src/cash/notebook/file_tracker.py:336-339`). A cached function that writes a file does not record the write as a dependency — and shouldn't, because the function's return value is what Cash hashes for downstream cache keys. If you need a downstream function to invalidate when an upstream wrote a file, depend on the upstream function's return value, not its on-disk side effect.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `file_depends_on=path` | `@cash.cache` kwarg | Wraps *path* in `FileDataSource` and adds it to the function's static dependencies. Accepts `str` or `list[str]`. |
| `c.register_file_handler(module, func, factory)` | `Cash` method | Register a wrapper factory for an additional reader. Catches every subsequent call to `module.func` from cached code. Glob wildcard supported in *func*. |
| `cash.FileDataSource(path)` | Public class | mtime-based change detection for a single file. Use in `depends_on=[...]` for advanced cases or subclass for content-hashing. |
| `f.explain(*args).reason == 'file_changed'` | Diagnostic | Explanation reason emitted when one or more recorded files changed. `details['changed_files']` maps each path to `'mtime changed'`, `'size changed'`, or `'file missing'`. |
| `FileAccessTracker` | Internal | Context manager that drives the monkey-patch. Auto-installed by `_compute_and_store`; not intended for direct use. |
| `FileDependencyRegistry` | Internal | Singleton holding the registered handler factories. Accessed through `register_file_handler`; direct use is unsupported. |

## Related

- [Decorator (`@cash.cache`)](../../decorator.md) — full reference for the decorator including `file_depends_on=`.
- [Data Engineering](../use-cases/data-engineering.md) — file-based ETL is the canonical use case for automatic tracking.
- [Debugging and Monitoring](debugging-and-monitoring.md) — `f.explain()` shows `file_changed` and the changed-files dict.
- [Dynamic Dependencies](dynamic-dependencies.md) — `dynamic_depends_on=` for cases where the file set depends on the call's arguments.
- [Choosing a Backend](choosing-a-backend.md) — backend choice affects how invalidation propagates across processes.
