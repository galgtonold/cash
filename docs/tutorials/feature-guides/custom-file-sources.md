# Custom file sources — automatic file tracking and the escape hatch

Cash automatically tracks file dependencies. When you call `pd.read_csv('data.csv')` inside a cached function, the file's size and a content hash get recorded; the next time you call the function, the cache invalidates if the file's *contents* changed. This guide covers what's tracked, what isn't, and how to add tracking for non-standard access patterns.

> **Two mechanisms, two different signals.** This page covers both, and they do not work the same way. **Auto-tracking** (`pd.read_csv`, `open`, … — everything Cash intercepts for you) is **content-authoritative**: it records a content hash and ignores the mtime. The **`file_depends_on=` / `FileDataSource` escape hatch** (below) is **mtime-based**: it folds the file's modification time into the cache key. So a touch that doesn't change any bytes leaves an auto-tracked dependency valid but *does* invalidate a `file_depends_on=` one. Keep the distinction in mind as you read.

## Why this exists

Without file tracking, every CSV/parquet load you make from a cached function would either *always* hit the cache (silently stale when the file changes on disk) or *always* miss (slow). Neither is acceptable. The middle ground is to record a fingerprint of every file the function reads and invalidate when that fingerprint moves. You get hits on identical inputs and re-runs on changed inputs without thinking about it.

The fingerprint is `(mtime, size, hash)`, and **content is authoritative whenever the size matches**. The cheap size check runs first — a differing size proves staleness without reading a byte of data — and only when the size is equal does Cash hash the file to decide. The mtime is recorded but no longer arbitrates: a touch that leaves the bytes alone is a **hit**, and a same-size edit under an indistinguishable mtime is still a **miss**.

The mechanism is a one-time monkey-patch of the popular reader functions: when a `@cash.cache` function executes, Cash installs `FileAccessTracker` around the call, intercepts reads from `builtins.open`, pandas, polars, numpy, joblib, json, and pickle, and stores the resulting file dictionary in the cache metadata. On the next lookup, Cash re-checks every recorded file and re-runs the function if the contents moved.

## Quick start

<!-- test:expect-warning reason="load_features reads a file; the impurity advisory is expected and cash still caches" -->
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

No decorator argument, no manual registration. Cash sees the `read_csv` call, records the path, and notices the next time the file's contents differ.

## What's automatically tracked

<!-- claim: cash/notebook/file_tracker.py:FileDependencyRegistry._initialize_defaults @9540ade4, cash/notebook/file_tracker.py:_find_patch_targets @720455ed -->
The default handler set is registered in `FileDependencyRegistry._initialize_defaults`:

| Module | Functions |
|---|---|
| `builtins` | `open()` (any read mode — `'r'`, `'r+'`, `'rb'`, `'r+b'`, …) |
| `io` | `open()` (alias of the built-in `open`) |
| `pandas` | `read_*` — every reader: `read_csv`, `read_parquet`, `read_excel`, `read_json`, `read_pickle`, `read_feather`, `read_hdf`, `read_orc`, `read_sas`, `read_spss`, `read_stata`, `read_table`, `read_xml`, `read_html`, `read_fwf`, `read_clipboard`, `read_sql*` |
| `polars` | `read_csv`, `read_parquet`, `read_json`, `read_ndjson`, `read_ipc`, `read_avro`, `read_excel`, plus the lazy variants `scan_csv`, `scan_parquet`, `scan_ipc`, `scan_ndjson` |
| `numpy` | `load`, `loadtxt`, `genfromtxt`, `fromfile` |
| `joblib` | `load` |
| `pickle` | `load` |
| `json` | `load` |
| `glob` | `glob`, `iglob` — tracks the *directory* enumerated (see below) |
| `os` | `listdir`, `scandir` — tracks the *directory* enumerated (see below) |

The pandas entry is the glob `read_*`, expanded by `_find_patch_targets` against the live `pandas` module — so any reader pandas adds in a future release is picked up too. Both top-level reads (`pd.read_csv`) and submodule reads (`pd.read_csv` via the `pandas.io.parsers` shim) flow through the patched attribute.

<!-- claim: cash/notebook/file_tracker.py:FileDependencyRegistry._create_open_handler @2c6b1e1b -->
For `open()`, the wrapper records the path only when the mode contains `'r'` or `'+'` (read or read/write), not pure writes — see `_create_open_handler`. So an `open(path, 'w')` for output does *not* get tracked, which is what you want: writes are accounted for by hashing the function's return value, not its outputs.

### Directory enumeration tracks the directory

Reading the files a `glob` matched only ever records the files that *existed* on the first run — so a **new** matching file would be invisible. Cash therefore tracks the enumerated directory itself as a dependency:

```python
import glob

@cash.cache
def load_all():
    return [open(p).read() for p in glob.glob("data/*.csv")]
```

Drop a new `data/extra.csv` in and `load_all` recomputes. Adding or removing a directory entry bumps the directory's own mtime on local filesystems, which the existing freshness check already notices. For a `glob` pattern, the tracked directory is the longest leading magic-free part of the pattern (`data/` for `data/*.csv`); `os.listdir` / `os.scandir` track the path passed to them.

Two limits worth knowing: a *modification* to an existing file is caught by that file's own dependency, not the directory's; and this leans on the filesystem bumping the directory mtime, which is not guaranteed on some network mounts (see [Network-mounted filesystems](#network-mounted-filesystems)).

## How to verify what's tracked

`cache_info()` and `f.explain()` surface the recorded dictionary. After a compute, the entry's metadata holds `auto_file_deps`:

```python
import cash
import pandas as pd

@cash.cache
def load_features():
    return pd.read_csv("data/features.csv")

load_features()                                  # cache hit — same body as above
load_features.explain()
# CacheExplanation(would_hit=True, reason='hit', ...)

# Edit data/features.csv, then:
# test:inject: import pathlib, time as _t; pathlib.Path("data/features.csv").write_text("col1,col2\nnew1,new2\n"); _t.sleep(0.01)
load_features.explain()
# [MISS] __main__.load_features — file_changed
#   changed_files: {'data/features.csv': 'content changed'}
```

The `file_changed` reason and the `changed_files` dict are emitted by `Cash._explain_call`. The dict's values are short human-readable strings: `'content changed'`, `'size changed'`, or `'file missing'`.

`explain()` decides freshness through the same content-authoritative `file_dep_is_fresh` helper a real lookup uses, so it cannot disagree with the call: a **touch** (identical bytes, bumped mtime) explains as `hit`, exactly as it behaves. See [Debugging and Monitoring](debugging-and-monitoring.md) for the full `explain()` story.

## What's NOT tracked

The patch set is a curated list. Reads that go through anything else slip past the tracker:

- **Direct `pyarrow` / `fastparquet` calls** — `pyarrow.parquet.read_table('data.parquet')` is not patched. `pd.read_parquet(...)` (which calls pyarrow internally) *is* — the patch is at the pandas entry point.
- **Specialized format libraries** — `feather.read_dataframe`, `h5py.File`, `netCDF4.Dataset`, custom binary readers in vendored utilities.
- **C extensions and subprocesses** — anything that opens a file descriptor outside the Python-level `open()` (e.g. a C library called via `ctypes`, a `subprocess.run` that reads the file) is invisible. The monkey-patch only intercepts Python-side dispatch.
- **Database files** — `sqlite3.connect('db.sqlite')` or a SQLAlchemy engine pointed at a file URL doesn't open the file via the patched readers. The query itself goes through the driver and Cash sees nothing.
- **Lazy scans you don't materialize** — `polars.scan_csv(...)` *is* tracked at scan time.

Remote URLs are the exception to the "not tracked" list: `pd.read_parquet("s3://bucket/key")` **is** tracked, just not by fingerprinting bytes — see [Remote objects](#remote-objects-tracked-by-the-stores-own-validator) below.

For the local-file gaps, use the `file_depends_on=` escape hatch below.

!!! warning "`file_depends_on=` does not work for a remote URL"
    It builds a `FileDataSource`, whose token is the file's mtime — and
    `os.path.getmtime("s3://…")` fails, so the token is a constant `0.0` and the
    entry **never invalidates**. Remote objects are tracked automatically
    (below); to declare one explicitly, use
    `depends_on=[RemoteFileDataSource(url)]`, not `file_depends_on=`.

## Remote objects: tracked by the store's own validator

A remote object can't be checked the way a local file is — downloading it to
see whether it moved defeats the point of caching. Object stores answer the
question directly instead: every object carries a validator the store maintains
itself — an **ETag**, a **version id**, a GCS **generation** — that changes when
the bytes change. Cash reads that with a single metadata request.

This happens **automatically**. A read of `s3://`, `gs://`, `az://` or
`https://` inside a cached function is recorded as a dependency and rechecked on
every hit:

<!-- test:skip reason="needs a reachable bucket; the behaviour below is verified against an S3 emulator in tests/test_core/test_remote_s3_contract.py" -->
```python
import cash
import pandas as pd

@cash.cache
def load_events(url):
    return pd.read_csv(url)

load_events("s3://bucket/events.parquet")   # first call — downloads, records the ETag
load_events("s3://bucket/events.parquet")   # cache hit — ETag check only, no download
```

Overwrite the object and the next call recomputes, because the store hands back
a different ETag:

<!-- test:skip reason="needs a reachable bucket; see test_remote_s3_contract.py" -->
```python
# Someone overwrites s3://bucket/events.parquet, then:
load_events.explain("s3://bucket/events.parquet")
# [MISS] __main__.load_events — file_changed
#   changed_files: {'s3://bucket/events.parquet': 'remote object changed'}
```

Re-uploading *identical* bytes does **not** invalidate: an ETag is derived from
content, so a no-op rewrite doesn't throw away everyone's cache.

!!! note "One exception, for large multipart uploads"
    S3 computes a **multipart** ETag from the checksums of the parts plus a
    `-N` part-count suffix, so it depends on how the object was uploaded as
    well as on its bytes. Re-uploading identical data with a *different part
    size* — switching between the AWS CLI and a boto3 default, say — produces a
    different ETag, and cash recomputes once even though nothing changed.

    Correctness is never at risk (changed content always moves the ETag), and
    the next run caches under the new token. Keeping your upload tooling
    consistent avoids it entirely.

Two things follow, and both are the point:

- **A hit costs no download.** The metadata request is tens of milliseconds
  against a `GET` that may be hundreds of megabytes, so tracking a remote read
  usually *reduces* network traffic rather than adding to it.
- **The dependency travels.** A local path is a fact about one filesystem, so a
  cache keyed on one can't be shared; an ETag is a fact about the object, so
  your teammate's key matches yours. See
  [Sharing a cache](sharing-caches.md).

To declare a remote dependency the tracker can't see — a read through `boto3`, a
subprocess, a format library — name it explicitly:

<!-- test:skip reason="illustrative — requires a reachable bucket" -->
```python
from cash import RemoteFileDataSource

@cash.cache(depends_on=[RemoteFileDataSource("s3://bucket/events.parquet")])
def load_via_boto3():
    return read_via_boto3("bucket", "events.parquet")
```

!!! tip "Data that can never change? Say so, and pay nothing"
    `RemoteFileDataSource(url, immutable=True)` resolves once per session and
    skips every later request. Cash infers it for free when the URL **pins a
    version** (`?versionId=…`, `#generation=…`), because the storage contract
    guarantees those bytes can't change — the pin *is* the token, so no request
    is made at all. It is never inferred from a path that merely looks
    write-once: a wrong guess there never invalidates, silently, forever.

**When the store can't be reached**, cash recomputes rather than serving a
result whose freshness nobody could verify, and warns once. A bad day costs you
the speedup, never correctness. If freshness checking starts costing real
time — many sources, a slow link — cash says so on the badge's `remote` overhead
line and warns with the source count.

## Escape hatch 1: `file_depends_on=` on `@cash.cache`

When Cash can't see the read, tell it explicitly:

```python
import cash
import pyarrow.parquet as pq

@cash.cache(file_depends_on="data/events.parquet")
def load_local_events():
    return pq.read_table("data/events.parquet").to_pandas()
```

`file_depends_on` accepts a single path or a list:

```python
@cash.cache(file_depends_on=["models/embeddings.bin", "models/vocab.json"])
def load_model():
    return MyModel.from_disk("models/embeddings.bin", "models/vocab.json")
```

<!-- claim: cash/data_source.py:FileDataSource @4099fc64 broad="the mtime-at-init behaviour is a property of the whole class" -->
Under the hood, `_register_func` wraps each path in a `FileDataSource` and folds it into the function's static dependency list. `FileDataSource.state_token()` re-reads the file's mtime on every lookup; a change propagates into the cache key and forces a miss.

A subtle behavior worth knowing: `FileDataSource.__init__` snapshots the mtime *at decoration time*. If the file doesn't exist yet when the decorator runs, the snapshot is `0.0` (the `OSError` fallback in `_get_mtime`). That's fine — the next stat sees the real mtime and triggers a miss for the first real run. But it means `file_depends_on` on a not-yet-created file does *not* fail loudly; you have to remember it's there.

## Escape hatch 2: registering a custom file source for auto-tracking

<!-- claim: cash/core.py:Cash.register_file_handler @5731a107, cash/notebook/file_tracker.py:_install_module_patches @4cabaa21 -->
For libraries you use across many cached functions, manually adding `file_depends_on=` to each decorator is repetitive. `Cash.register_file_handler` lets you teach the auto-tracker about a new reader once and have every subsequent call site picked up automatically:

<!-- test:skip reason="illustrative — the handler wraps `my_lib`, which does not exist; executing it only proves a def parses, while shadowing the real load_features above" -->
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

The handler is a factory: Cash calls it with the original function and a `track_callback(path)` shim; your wrapper records and forwards. `func_name` supports glob patterns (`"read_*"` catches every reader in one call), and `module_name` may be dotted (`"my_lib.io"`). The wrapper is installed on the live module via the same `_install_module_patches` path used for the built-ins.

Two caveats from the docstring:

- The wrapper replaces the attribute on the module object, so existing imports (`from my_lib import read_data`) still see the original unwrapped version. Track via the module namespace (`my_lib.read_data(...)`) or import after registering.
- Pass an absolute or resolvable path to `track_callback`. Relative paths are resolved against `os.getcwd()` at tracking time by `_track_path`.

## Staleness detection

Cash checks freshness on every lookup, not at write time. `_auto_file_deps_fresh` walks the recorded dictionary and delegates each entry to `file_dep_is_fresh` — the **same** helper the notebook-statement layer uses, so the two subsystems cannot drift. An entry is stale as soon as any of these is true:

- The file is unreadable (`os.stat` raises) — reason `unreadable`.
- The size moved — reason `size`. Checked first, and Cash never hashes on this path.
- The size matches but the content hash differs — reason `content`. **This is the authoritative check.**

A matching size *and* a matching content hash is fresh, **regardless of the mtime**. Touching a file does not invalidate an auto-tracked dependency.

**Why a content hash and not just mtime+size?** Because `(mtime, size)` was ambiguous in both directions, and both failure modes were real bugs. A touch-only change (identical bytes, bumped mtime) recomputed needlessly; a same-size edit written under an mtime the check couldn't distinguish was missed and served stale. Content is the signal that actually answers the question. The cost is bounded by checking size first and by sampling large files (below), so the common case is still one `stat()` and — only when the size matches — a bounded read.

### Large files are sampled, not fully hashed

<!-- claim: cash/notebook/file_dep_snapshot.py:file_dep_is_fresh @5f35e472, cash/notebook/file_dep_snapshot.py:file_content_hash @6bdf50df, cash/notebook/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES == 8388608, cash/notebook/file_dep_snapshot.py:_HASH_SAMPLE_REGION_BYTES == 262144 -->
Hashing a multi-GB parquet on every lookup would defeat the point of caching, so the hash is size-bounded (`file_content_hash`):

- Files **≤ 8 MiB** (`_HASH_FULL_MAX_BYTES`) are hashed **in full**.
- Files **> 8 MiB** are **sampled** at three deterministic, size-derived offsets — head, middle, and tail, **256 KiB each** (`_HASH_SAMPLE_REGION_BYTES`) — with the byte length folded into the digest.

A sampled hash on its own would miss an edit that changes only unsampled interior bytes while preserving the exact size. **It doesn't, because sampled files carry an mtime backstop**: above the 8 MiB cap a matching hash is trusted only when the mtime *also* matches, so any real in-place write is caught (`stale_reason` reads `'mtime-sampled'`). Below the cap the hash is authoritative and mtime is ignored, which is what makes a content-preserving `touch` free.

The tradeoff therefore inverted rather than disappearing. What you pay for a large file is the opposite error: **touching** it — `touch`, a re-checkout that rewrites identical bytes, an rsync that resets timestamps — forces one spurious recompute. That is the safe direction to be wrong in, and it is why the two regimes differ:

| File size | Hash covers | mtime | You can be surprised by |
|---|---|---|---|
| ≤ 8 MiB | every byte | ignored | nothing — content decides |
| > 8 MiB | head/middle/tail | must also match | a needless recompute after a touch |

If a spurious recompute on a multi-GB input is itself too expensive, write a `DataSource` subclass whose `state_token()` returns whatever cheap, authoritative version marker your data already has (a manifest hash, an ETag, a build id) and pass it via `depends_on=`.

Race condition to be aware of: if a file is rewritten *while* a cached function is running, the snapshot captures the post-write content. On the next call Cash sees a matching hash and returns the cached value — which now reflects half-old, half-new data. The window is small and rarely matters, but for high-churn pipelines wrap the write in a tempfile-then-rename so each run sees a consistent snapshot.

## Caveats

### Symlinks are followed

`_track_path` resolves the path through `os.path.realpath` before storing it. If you read a symlink, Cash records and checks the *target*, and the resolution is frozen at track time. Editing the symlink target's contents invalidates the cache. Repointing the symlink at a different file does **not** — Cash goes on checking the original target, which hasn't changed. This matches what most users expect ("the data file changed"), but if you genuinely care about the symlink identity itself, use `file_depends_on=` with the link path explicitly.

### Relative paths re-resolve against the live cwd

A relative-path read records **two** dependencies: the absolute path resolved at execution time, *and* the un-resolved relative path. The freshness check re-resolves the relative one against the *current* working directory on every lookup, so changing `os.chdir` to a directory holding a different file of the same name is detected:

<!-- test:skip reason="illustrative — spans a chdir cell edit across runs; run_a/run_b don't exist in the harness" -->
```python { .nb-cell }
os.chdir("run_a")
df = pd.read_csv("results.csv")   # run_a/results.csv

# Edit the chdir cell to point at run_b and re-run:
os.chdir("run_b")
df = pd.read_csv("results.csv")   # run_b/results.csv — recomputed, not run_a's data
```

Without the relative dependency, the frozen `run_a/results.csv` realpath still existed and was unmodified, so even `run_all` served the old directory's data. Re-runs from the same cwd resolve to the same file and stay cached.

### Paths are absolute and platform-normalized

Stored paths are absolute and use forward slashes regardless of OS (`cash.utils.normalize_path`). Moving the cache directory to a different machine where the same files live at different paths invalidates everything — paths are part of the dependency key. For portable cache archives, expect a full re-compute after relocation.

### Network-mounted filesystems

NFS, SMB, and similar network mounts often have coarse mtime resolution (1-second granularity) and the timestamp source is the *server*, not the client, so two writes within the same second can produce identical mtimes. **Auto-tracking is immune to this** — it reads content, not timestamps, so a same-second in-place edit that preserves size is still caught.

Two things on network mounts do still deserve care:

- **`file_depends_on=` remains mtime-based**, so the coarse-resolution problem applies to it in full. On a network mount, prefer auto-tracking for critical files, or write a `DataSource` subclass whose `state_token()` returns a content hash.
- **Directory dependencies are mtime-based too.** A directory has no content to hash, so the [directory tracking](#directory-enumeration-tracks-the-directory) added for `glob` / `listdir` / `scandir` falls back to the mtime path. It relies on the filesystem bumping a directory's mtime when an entry is added or removed — true on local filesystems, not guaranteed on every network mount. If a new file appearing in a globbed directory must invalidate on such a mount, list the files explicitly via `file_depends_on=`.
- **Content hashing costs a network read.** On a slow mount the hash is I/O over the wire whenever the size matches. The size check short-circuits the common "file was replaced wholesale" case first, and files over 8 MiB only pull 768 KiB of samples, but a large directory of same-size files re-hashed on every lookup is worth measuring.

### Files outside the working directory

The tracker records full absolute paths and stats them on every lookup. There's no working-directory boundary — a cached function that reads `/etc/hosts` or `~/.config/foo.yml` will invalidate when those files change, which is usually right but occasionally surprising in containerized or CI environments where unrelated host files churn.

### Writes are intentionally not tracked

`FileAccessTracker` only intercepts read paths. A cached function that writes a file does not record the write as a dependency — and shouldn't, because the function's return value is what Cash hashes for downstream cache keys. If you need a downstream function to invalidate when an upstream wrote a file, depend on the upstream function's return value, not its on-disk side effect.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `file_depends_on=path` | `@cash.cache` kwarg | Wraps *path* in `FileDataSource` and adds it to the function's static dependencies. Accepts `str` or `list[str]`. |
| `c.register_file_handler(module, func, factory)` | `Cash` method | Register a wrapper factory for an additional reader. Catches every subsequent call to `module.func` from cached code. Glob wildcard supported in *func*. |
| `cash.FileDataSource(path)` | Public class | mtime-based change detection for a single file. Use in `depends_on=[...]` for advanced cases or subclass for content-hashing. |
| `f.explain(*args).reason == 'file_changed'` | Diagnostic | Explanation reason emitted when one or more recorded files changed. `details['changed_files']` maps each path to `'content changed'`, `'size changed'`, or `'file missing'`. |
| `FileAccessTracker` | Internal | Context manager that drives the monkey-patch. Auto-installed by `_compute_and_store`; not intended for direct use. |
| `FileDependencyRegistry` | Internal | Singleton holding the registered handler factories. Accessed through `register_file_handler`; direct use is unsupported. |

## Related

- [Decorator (`@cash.cache`)](../../decorator.md) — full reference for the decorator including `file_depends_on=`.
- [Data Engineering](../use-cases/data-engineering.md) — file-based ETL is the canonical use case for automatic tracking.
- [Debugging and Monitoring](debugging-and-monitoring.md) — `f.explain()` shows `file_changed` and the changed-files dict.
- [Dynamic Dependencies](dynamic-dependencies.md) — `dynamic_depends_on=` for cases where the file set depends on the call's arguments.
- [Choosing a Backend](choosing-a-backend.md) — backend choice affects how invalidation propagates across processes.
