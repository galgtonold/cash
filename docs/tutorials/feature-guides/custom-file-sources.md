# Custom file sources — automatic file tracking and the escape hatch

Cash automatically tracks file dependencies. When you call `pd.read_csv('data.csv')` inside a cached function, the file's size and a content hash get recorded; the next time you call the function, the cache invalidates if the file's *contents* changed. This guide covers what's tracked, what isn't, and how to add tracking for non-standard access patterns.

> **Two ways in, one signal.** **Auto-tracking** (`pd.read_csv`, `open`, … — everything Cash intercepts for you) and the **`file_depends_on=` escape hatch** (below) are both **content-authoritative**: each records a content hash and ignores the mtime, so a touch that doesn't change any bytes leaves either dependency valid. The exception is **`FileDataSource`**, the bundled `DataSource` for `depends_on=` / `dynamic_depends_on=`: it folds the file's modification time into the cache key.

## Why this exists

Without file tracking, every CSV/parquet load you make from a cached function would either *always* hit the cache (silently stale when the file changes on disk) or *always* miss (slow). Neither is acceptable. The middle ground is to record a fingerprint of every file the function reads and invalidate when that fingerprint moves. You get hits on identical inputs and re-runs on changed inputs without thinking about it.

The fingerprint is `(mtime, size, hash)`, and **content is authoritative whenever the size matches**. The cheap size check runs first — a differing size proves staleness without reading a byte of data — and only when the size is equal does Cash hash the file to decide. The mtime is recorded but no longer arbitrates: a touch that leaves the bytes alone is a **hit**, and a same-size edit under an indistinguishable mtime is still a **miss**.

<!-- claim: cash/tracking/io_watch.py:hold @fcfb42b4, cash/tracking/file_tracker.py:_on_open @0ac87d87 -->
The mechanism has two parts. Every Python-level `open()` — `builtins.open`, `io.open`, `pathlib`, and every library that opens its file through them (`json`, `pickle`, `joblib`, `numpy`) — and every directory listing reaches cash as a Python audit event (`sys.addaudithook`), whoever calls it and however it was imported. Readers that open files in C, C++ or Rust (pyarrow, polars, sqlite3, some pandas readers), and the calls that raise no event (`os.path.exists`, `Path.stat`), are wrapped. When a `@cash.cache` function executes, Cash opens a `FileAccessTracker` around the call, which records both kinds of read; the first tracker to open installs the wrappers and the last one to close restores the originals, so outside a cached call (and outside `%cash_on`) `pd.read_csv` is pandas' own function. The resulting file dictionary goes into the cache metadata. On the next lookup, Cash re-checks every recorded file and re-runs the function if the contents moved.

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

<!-- claim: cash/tracking/file_tracker.py:FileDependencyRegistry._initialize_defaults @b63601b2, cash/tracking/file_tracker.py:_find_patch_targets @30ce0d87, cash/tracking/file_tracker.py:_on_listing @c2c76d1d -->
The `open` and listing audit events are handled by `_on_open` and `_on_listing`; the wrapped readers are registered in `FileDependencyRegistry._initialize_defaults`:

| Module | Functions |
|---|---|
| `builtins`, `io` | `open()` (any read mode — `'r'`, `'r+'`, `'rb'`, `'r+b'`, …), seen as the `open` audit event, including through a reference taken before cash was imported. `os.open` is not tracked. |
| `pandas` | `read_*` — every reader: `read_csv`, `read_parquet`, `read_excel`, `read_json`, `read_pickle`, `read_feather`, `read_hdf`, `read_orc`, `read_sas`, `read_spss`, `read_stata`, `read_table`, `read_xml`, `read_html`, `read_fwf`, `read_clipboard`, `read_sql*` |
| `polars` | `read_csv`, `read_parquet`, `read_json`, `read_ndjson`, `read_ipc`, `read_avro`, `read_excel`, plus the lazy variants `scan_csv`, `scan_parquet`, `scan_ipc`, `scan_ndjson` |
| `pyarrow` | `csv.read_csv`, `csv.open_csv`, `parquet.read_table`, `parquet.read_pandas`, `feather.read_table`, `feather.read_feather`, `json.read_json` |
| `sqlite3` | `connect` — the database file |
| `numpy` | `load`, `loadtxt`, `genfromtxt`, `fromfile`, `memmap`, which open their file through `open()` |
| `joblib`, `pickle`, `json` | `joblib.load(path)`, and `pickle.load(f)` / `json.load(f)` of a file opened with `open()` |
| `glob` | `glob`, `iglob` — tracks the *directory* enumerated (see below); seen as the `glob.glob` audit event |
| `os` | `listdir`, `scandir` — tracks the *directory* enumerated (see below); seen as audit events, as is every `pathlib` `glob` / `iterdir` that lists through them |
| `os.path` | `exists`, `isfile` (and their `genericpath` originals) — records a path that was looked for and was **not** there (see below) |
| `pathlib` | `Path.read_text()`, `read_bytes()` and `open()`, which read through `open()`; and `Path.stat()` on a regular file (wrapped) — so `p.stat().st_size` or `.st_mtime` shown after the file changed is the new value, not the cached one |

The pandas entry is the glob `read_*`, expanded by `_find_patch_targets` against the live `pandas` module — so any reader pandas adds in a future release is picked up too. Both top-level reads (`pd.read_csv`) and submodule reads (`pd.read_csv` via the `pandas.io.parsers` shim) flow through the patched attribute. A reader that opens its file through `open()` is tracked through the audit event even where no wrapper reaches it.

A reader may be given its path positionally or by keyword — `pd.read_csv(filepath_or_buffer=p)`, `np.load(file=p)`, `pq.read_table(source=p)` — and both are tracked. pyarrow reads files in C++, so none of its reads pass through `open()`; before its readers were registered, a function that switched to `pyarrow.csv` for speed recorded no dependency at all and kept returning the old file's answer. `pyarrow.parquet.ParquetFile` is not wrapped (it is a class, and replacing it with a function would break `isinstance` checks), and `pyarrow.dataset.dataset` records only the path it was given; read through them and name the files with `file_depends_on=`.

<!-- claim: cash/tracking/file_tracker.py:_is_read_mode @238e2cb8, cash/tracking/file_tracker.py:_on_open @0ac87d87 -->
For `open()`, cash records the path as a *dependency* only when the call can read what was there before: a mode containing `'r'`, or `'+'` without `'w'` or `'x'` (`'r+'`, `'a+'`) — see `_is_read_mode`. An `open(path, 'w')` for output does **not** become a dependency, which is what you want: folding a file the function writes into its own cache key would invalidate the entry on its own output. Nor does `'w+'` / `'x+'`, which start from an empty file — Pillow saves every image with `'w+b'`, so a `savefig` used to depend on the PNG it had just written.

A write is not ignored, though — it is an *effect*, and it is reported as one. The same `open` event handler hands a write-mode open to the [effect observer](purity-decorators.md#observed-effects-what-the-first-call-actually-did), which warns once if the first call wrote a file the static analyzer never saw. That matters because every cache hit from then on skips the write.

### A file that was not there is a dependency too

`os.path.exists` and `os.path.isfile` are tracked, but only when the answer is **False**:

```python
@cash.cache
def load_config():
    if os.path.exists("local_overrides.toml"):
        return read_overrides()
    return DEFAULTS
```

The absence of `local_overrides.toml` is what selected the defaults branch — it is an input, and it was the one input Cash could not see, because a file that is never opened produces no read to track. The entry recorded no dependencies at all, so it looked valid everywhere: one directory's answer was served in another, silently. A negative probe is now recorded as `{'absent': True}`, and the entry stops being valid once the file appears.

A probe that finds the file records nothing here; the read that follows it tracks the file properly. Absent paths are kept **as written** rather than resolved: a relative probe asks "is there a file with this name, *here*", and freezing the directory it happened to run in would reintroduce the same bug in mirror image.

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

The `file_changed` reason and the `changed_files` dict are emitted by `Cash._explain_call`. The `changed_files` values are short human-readable strings: `'content changed'`, `'size changed'`, `'file missing'`, `'mtime changed'` and `'mtime changed (sampled file)'`, `'the file was written (sampled file)'`, `'a file the call looked for and did not find now exists'`, or — for a remote source — `'remote object changed'` / `'remote object could not be checked'`. After `file_hash_full_max_bytes` moves across a file's size, its entry was fingerprinted one way and is now checked the other, and the two cannot be compared: that reads `'fingerprinted under a different file_hash_full_max_bytes, so it could not be compared -- the file itself may be unchanged'`, and the entry recomputes once. `file_deps` labels a fingerprint of a file above that size `sampled hash`, since only its head, middle and tail were hashed.

`explain()` decides freshness through the same content-authoritative `file_dep_is_fresh` helper a real lookup uses, so it cannot disagree with the call: a **touch** (identical bytes, bumped mtime) explains as `hit`, exactly as it behaves. See [Debugging and Monitoring](debugging-and-monitoring.md) for the full `explain()` story.

## What's NOT tracked

Anything that opens a file through Python's `open()` is seen. Reads that go through anything else slip past the tracker unless their reader is in the wrapped list above:

- **`fastparquet` and `pyarrow.parquet.ParquetFile`** (see above).
- **Specialized format libraries that open files in C** — `h5py.File`, `netCDF4.Dataset`, custom binary readers in vendored utilities.
- **C extensions, `os.open` and subprocesses** — anything that opens a file descriptor without Python's `open()` (e.g. a C library called via `ctypes`, `os.open`/`os.read`, a `subprocess.run` that reads the file) is invisible.
- **Database engines other than `sqlite3.connect`** — a SQLAlchemy engine pointed at a file URL goes through the driver and Cash sees nothing.
- **C-level readers outside every cached call** — a memo filled through `pl.read_csv` before the first cached call is not seen, because the wrappers are only installed while a cached call runs; one filled through `open()` is (see [the decorator guide](../../decorator.md#file-reads-are-tracked-automatically)).
- **Lazy scans you don't materialize** — `polars.scan_csv(...)` *is* tracked at scan time.

### Reads that are ignored on purpose

<!-- claim: cash/tracking/file_tracker.py:incidental_read @6f65994d -->
Some reads happen while your code runs but are not your data, and cash leaves them out: files of the **Python installation itself** (the standard library), **package metadata** lookups (`importlib.metadata`, `importlib.resources`, `pkg_resources` — the import system listing every `sys.path` folder, your working directory included, and reading `entry_points.txt` files), anything a library reads **while it is being imported** (matplotlib's style sheets and font cache), and files that belong to an **installed package other than your own** — matplotlib's fonts on first draw, scikit-learn's HTML template, a zone `zoneinfo` loads from `tzdata` (or from the system time zone database) the first time you use it. They only happen the first time, so recording them gave the same statement a different key on its second run, and a new file anywhere next to a notebook invalidated everything after an `import`. The zone is the case that shows why the reader does not decide: `zoneinfo` is the standard library, so "a library reading its own package" did not cover it, and a load's lineage carried that file after a restart but not on a re-run in the same session — everything below it missed, once, on the first restart.

A library reading a file **for you** is still tracked — `PIL.Image.open(p)`, `torch.load(p)` and `pd.read_csv(p)` read a path outside that library. So is your own module reading its configuration at import, and so is an installed tool reading data from its own package folder when the cached function belongs to that tool.

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

<!-- claim: cash/decorator/file_deps.py:FileDepsMixin._track_declared_files @1a1a4d66 -->
Under the hood, `_register_func` records each path (made absolute at decoration time), and every miss adds them to the call's file tracker as if the body had read them. The entry therefore stores their content fingerprint beside anything the body read itself, every lookup checks it the way it checks an auto-tracked read, and a cached function that calls this one inherits the files on a hit too.

A declared file that does not exist yet is recorded as *absent*, like a lookup for a missing file: creating it later forces a miss. It does *not* fail loudly; you have to remember it's there.

## Escape hatch 2: registering a custom file source for auto-tracking

<!-- claim: cash/core.py:Cash.register_file_handler @e2ebcc87, cash/tracking/file_tracker.py:_install_module_patches @03e888c6 -->
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

The handler is a factory: Cash calls it with the original function and a `track_callback(path)` shim; your wrapper records and forwards. `func_name` supports glob patterns (`"read_*"` catches every reader in one call), and `module_name` may be dotted (`"my_lib.io"`). The wrapper is installed on the live module via the same `_install_module_patches` path used for the built-ins, while a cached call runs (or `%cash_on` is on), and the original is put back afterwards.

Two caveats from the docstring:

- The wrapper replaces the attribute on the module object, so imports that bound the function itself (`from my_lib import read_data`) see whatever was there when they ran — usually the original. Track via the module namespace (`my_lib.read_data(...)`).
- Pass an absolute or resolvable path to `track_callback`. Relative paths are resolved against `os.getcwd()` at tracking time by `_track_path`.

## Staleness detection

Cash checks freshness on every lookup, not at write time. `_auto_file_deps_fresh` walks the recorded dictionary and delegates each entry to `file_dep_is_fresh` — the **same** helper the notebook-statement layer uses, so the two subsystems cannot drift. An entry is stale as soon as any of these is true:

- The file is unreadable (`os.stat` raises) — reason `unreadable`.
- The size moved — reason `size`. Checked first, and Cash never hashes on this path.
- The size matches but the content hash differs — reason `content`. **This is the authoritative check.**

A matching size *and* a matching content hash is fresh, **regardless of the mtime**. Touching a file does not invalidate an auto-tracked dependency.

**Why a content hash and not just mtime+size?** Because `(mtime, size)` was ambiguous in both directions, and both failure modes were real bugs. A touch-only change (identical bytes, bumped mtime) recomputed needlessly; a same-size edit written under an mtime the check couldn't distinguish was missed and served stale. So the content decides whenever the metadata moved. When nothing moved — the size, the modification time to the nanosecond, which file it is, and on Linux and macOS the inode change time are as recorded, and the file had been left alone for ten seconds before it was hashed — the file is not read at all: the common case is one `stat()`. That gives up an edit that keeps the size and puts the timestamp back, on Windows (see [known limitations](../../known-limitations.md#an-edit-that-keeps-the-size-and-timestamps)).

### Large files are sampled, not fully hashed

<!-- claim: cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @3dd62608, cash/tracking/file_dep_snapshot.py:file_content_hash @8404c5db, cash/tracking/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES_DEFAULT == 268435456, cash/tracking/file_dep_snapshot.py:_HASH_SAMPLE_REGION_BYTES == 262144 -->
Hashing a multi-GB parquet on every lookup would defeat the point of caching, so the hash is size-bounded (`file_content_hash`), at a threshold you can move (`file_hash_full_max_bytes`):

- Files **≤ 256 MiB** (`_HASH_FULL_MAX_BYTES`) are hashed **in full**.
- Files **> 256 MiB** are **sampled** at three deterministic, size-derived offsets — head, middle, and tail, **256 KiB each** (`_HASH_SAMPLE_REGION_BYTES`) — with the byte length folded into the digest.

<!-- claim: cash/tracking/file_dep_snapshot.py:_HASH_MEMO_TTL_SECONDS == 5.0 -->
A full hash costs about 0.72 ms per MiB, and it is only taken when the file's metadata moved (above). Within one process a digest is reused for up to five seconds against the file's stat fields (in a notebook, also for the rest of the cell run), so an aggregate whose ten cached helpers read the same fifty freshly written inputs hashes each input once.

A sampled hash on its own would miss an edit that changes only unsampled interior bytes while preserving the exact size. **It doesn't, because sampled files carry a timestamp backstop**: above the cap a matching hash is trusted only when the mtime *also* matches, so any real in-place write is caught (`stale_reason` reads `'mtime-sampled'`). Below the cap, once the metadata has moved, the hash is authoritative and mtime is ignored, which is what makes a content-preserving `touch` free.

That comparison is **exact on the integer nanoseconds**, not a tolerance — here the timestamp stands in for bytes the hash never read, and a tolerance is a window an edit can sit inside. What it catches depends on the resolution the restoring tool stores: whole seconds (a plain `tar` ustar header, rsync's protocol) cannot reproduce the original nanoseconds, so the edit shows up; the exact nanoseconds (`cp -p`, `shutil.copystat`, GNU tar's pax headers) reproduce them, and it does not. See [known limitations](../../known-limitations.md#an-edit-that-keeps-the-size-and-timestamps).

The tradeoff therefore inverted rather than disappearing. What you pay for a large file is the opposite error: **touching** it — `touch`, a re-checkout that rewrites identical bytes, an rsync that resets timestamps — forces one spurious recompute. That is the safe direction to be wrong in, and it is why the two regimes differ:

| File size | Hash covers | mtime | You can be surprised by |
|---|---|---|---|
| ≤ 256 MiB | every byte | ignored once it moved | an edit that puts the timestamp back (Windows) |
| > 256 MiB | head/middle/tail | must also match | the same, and a needless recompute after a touch |

If a spurious recompute on a multi-GB input is itself too expensive, write a `DataSource` subclass whose `state_token()` returns whatever cheap, authoritative version marker your data already has (a manifest hash, an ETag, a build id) and pass it via `depends_on=`.

**A file that changes while the call is running.** Cash records the size and timestamps of each file when the function first reads it, and checks them again before storing the result. If a file moved in between — a sync job landing mid-run, or an input rewritten after an inner cached call had already read it — the result is still returned, but **not cached**, with a [`STORE-INPUT-CHANGED`](../../warnings.md#store-input-changed) warning: there is no way to tell which version of the file it was computed from. The next call reads the settled file and caches normally.

It used to be cached. The fingerprint was taken at store time, so it described the *new* file while the result came from the old one, and every later call was a hit with the old answer — for as long as the entry lived. Writing the file via a temp file and a rename did not help, because the rename lands before the store.

One residual: a same-size edit that also puts the file's mtime back leaves the size and timestamps identical on Windows, during the call or after it. See [known limitations](../../known-limitations.md#an-edit-that-keeps-the-size-and-timestamps).

## Caveats

### Symlinks and junctions are checked as they point now

A path read through a symlink or a directory junction records **two** dependencies: the resolved target (`os.path.realpath`), and the path as you wrote it. Both must be fresh for a hit. Editing the target's contents invalidates, through the first. **Re-pointing the link** at a different file invalidates too, through the second — it is checked through the link *as it points now*, so a `current` release pointer switched to a new release, or back to an old one, gets that release's answer.

It used to record the resolved target only, frozen at write time: re-pointing a link was invisible, and a rollback through a `current` junction returned the *newer* release's report.

### Files beside your code belong to that copy of the code

A file your function reads from its **own** code's directory — package data through `importlib.resources`, `Path(__file__).parent / "ref.csv"`, a release's bundled config — is also recorded relative to that code, and every process checks **its own copy**. Two installs of one tool (a checkout and a wheel), or two releases of one job laid out side by side, run byte-identical code and so share cache keys; with the same data they share entries, and with different data each gets its own answer. A copy that is *missing* the file is a miss, and the call raises as it should rather than being served a value it could never have computed.

"Your own code's directory" means the top-level package that defines the cached function, or the script's directory for a function defined in the script you run.

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

Stored paths are absolute and use forward slashes regardless of OS (`cash._paths.normalize_path`). Moving the cache directory to a different machine where the same files live at different paths invalidates everything — paths are part of the dependency key. For portable cache archives, expect a full re-compute after relocation.

### Network-mounted filesystems

NFS, SMB, and similar network mounts often have coarse mtime resolution (1-second granularity) and the timestamp source is the *server*, not the client, so two writes within the same second can produce identical mtimes. **Auto-tracking is immune to this** — it reads content, not timestamps, so a same-second in-place edit that preserves size is still caught.

Three things on network mounts do still deserve care:

- **`FileDataSource` remains mtime-based**, so the coarse-resolution problem applies to it in full. On a network mount, prefer auto-tracking or `file_depends_on=` for critical files, or write a `DataSource` subclass whose `state_token()` returns a content hash.
- **Directory dependencies are mtime-based too.** A directory has no content to hash, so the [directory tracking](#directory-enumeration-tracks-the-directory) added for `glob` / `listdir` / `scandir` falls back to the mtime path. It relies on the filesystem bumping a directory's mtime when an entry is added or removed — true on local filesystems, not guaranteed on every network mount. If a new file appearing in a globbed directory must invalidate on such a mount, list the files explicitly via `file_depends_on=`.
- **Content hashing costs a network read.** On a slow mount the hash is I/O over the wire whenever the size matches. The size check short-circuits the common "file was replaced wholesale" case first, and files over 256 MiB only pull 768 KiB of samples, but a large directory of same-size files re-hashed on every lookup is worth measuring.

### Files outside the working directory

The tracker records full absolute paths and stats them on every lookup. There's no working-directory boundary — a cached function that reads `/etc/hosts` or `~/.config/foo.yml` will invalidate when those files change, which is usually right but occasionally surprising in containerized or CI environments where unrelated host files churn.

### Writes are intentionally not tracked

`FileAccessTracker` only intercepts read paths. A cached function that writes a file does not record the write as a dependency — and shouldn't, because the function's return value is what Cash hashes for downstream cache keys. If you need a downstream function to invalidate when an upstream wrote a file, depend on the upstream function's return value, not its on-disk side effect.

## API reference

| Symbol | Surface | Effect |
|---|---|---|
| `file_depends_on=path` | `@cash.cache` kwarg | Records *path* on every miss as if the function read it, so its content is checked on each lookup. Accepts `str` or `list[str]`. |
| `c.register_file_handler(module, func, factory)` | `Cash` method | Register a wrapper factory for an additional reader. Catches every subsequent call to `module.func` from cached code. Glob wildcard supported in *func*. |
| `cash.FileDataSource(path)` | Public class | mtime-based change detection for a single file. Use in `depends_on=[...]` for advanced cases or subclass for content-hashing. |
| `f.explain(*args).reason == 'file_changed'` | Diagnostic | Explanation reason emitted when one or more recorded files changed. `details['changed_files']` maps each path to `'content changed'`, `'size changed'`, or `'file missing'`. |
| `FileAccessTracker` | Internal | Context manager that records a block's reads, installing the reader wrappers while it is open. Auto-installed around the body by `Cash._body_scope`; not intended for direct use. |
| `FileDependencyRegistry` | Internal | The class of the process's one registry of handler factories (`file_registry()`). Accessed through `register_file_handler`; direct use is unsupported. |

## Related

- [Decorator (`@cash.cache`)](../../decorator.md) — full reference for the decorator including `file_depends_on=`.
- [Data Engineering](../use-cases/data-engineering.md) — file-based ETL is the canonical use case for automatic tracking.
- [Debugging and Monitoring](debugging-and-monitoring.md) — `f.explain()` shows `file_changed` and the changed-files dict.
- [Dynamic Dependencies](dynamic-dependencies.md) — `dynamic_depends_on=` for cases where the file set depends on the call's arguments.
- [Choosing a Backend](choosing-a-backend.md) — backend choice affects how invalidation propagates across processes.
