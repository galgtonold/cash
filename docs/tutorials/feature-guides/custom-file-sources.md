# File dependencies

!!! info "Applies to: decorator"
    Code whose cached functions read files, on local disk or in object storage.

When a cached function reads a file, cash records it with the entry and checks
it on every later lookup. A changed file makes the call run again. You usually
declare nothing:

```python
# test:inject: import pathlib
import cash
import pandas as pd

@cash.cache
def load_features():
    return pd.read_csv("data/features.csv")   # recorded automatically

load_features()   # first call: reads the file
load_features()   # cache hit
# test:inject: pathlib.Path("data/features.csv").write_text("col1,col2\nnew1,new2\n")
print(load_features.explain())
# [MISS] __main__.load_features - file_changed
#   ...
#   changed_files:
#     /home/you/project/data/features.csv: content changed
```

<!-- claim: cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @b9d64ecd, cash/tracking/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES_DEFAULT == 268435456 -->
The check is by **content**. A `touch`, or a re-save of identical bytes, still
hits. A same-size edit within the same second still recomputes. When a file's
size and timestamps have not moved, the check is one `stat` call, so a hit stays
cheap. Files over 256 MiB are hashed in three sampled regions, and their
modification time must match too. The full rule is in
[What counts as a change](../../how-it-works/invalidation.md#what-counts-as-a-change).

## What's automatically tracked

<!-- claim: cash/tracking/reader_patches.py:FileDependencyRegistry._initialize_defaults @56d3c684, cash/tracking/read_events.py:_on_open @5461415d, cash/tracking/read_events.py:_on_listing @82829f4a -->
Cash tracks `open()` in a read mode and what reads through it, the pandas, polars,
pyarrow and numpy readers, `sqlite3.connect`, directory listings (a new
matching file recomputes the call) and existence checks (the call recomputes
once a missing file appears, or once a file it found is gone). The full reader list is under
[Files](../../how-it-works/invalidation.md#files).

A path passed by keyword counts the same as one passed by position.

<!-- claim: cash/tracking/read_events.py:_is_read_mode @238e2cb8 -->
A file opened for writing is not a dependency: the entry would then depend on
its own output. Cash reports the write as a side effect instead
([Side effects](../../decorator.md#side-effects)).

<!-- claim: cash/tracking/read_classification.py:incidental_read @75e075c3 -->
Some reads are left out on purpose because they are not your data: files of the
Python installation, package metadata, reads a library makes while it is
imported, and an installed package reading its own files (fonts, templates,
time zones). A library reading a file you named, such as `pd.read_csv(p)` or
`torch.load(p)`, is tracked.

## What is not tracked

- Readers that open files in C without going through `open()`: `h5py`,
  `netCDF4`, `fastparquet`, `pyarrow.fs` file systems
  (`LocalFileSystem().open_input_file(...)`), a library called through
  `ctypes`.
- A tracked reader imported by name in a notebook cell
  (`from polars import read_parquet`) or wrapped in `functools.partial`: call
  it through its module (`pl.read_parquet(...)`) instead.
- `os.open`, and files a subprocess reads.
- The size or time a directory listing reports (`entry.stat()` on an
  `os.scandir` entry): use `os.stat(entry.path)` or `Path.stat()`.
- Database engines other than `sqlite3.connect`, such as SQLAlchemy.

Name such files with `file_depends_on=`, or teach cash the reader with
`register_file_handler` (both below).

## `file_depends_on=`: name a file yourself

```python
import cash
import pyarrow.parquet as pq
from pyarrow import fs

@cash.cache(file_depends_on="data/events.parquet")
def load_events():
    with fs.LocalFileSystem().open_input_file("data/events.parquet") as f:
        return pq.read_table(f).to_pandas()
```

<!-- claim: cash/decorator/file_deps.py:FileDeps.track_declared_files @4027a947 -->
Pass one path or a list. Each file is recorded as if the body had read it, and
is checked by content like an automatic read. A path that does not exist yet is
recorded as absent, so creating the file later recomputes the call. Nothing
warns you that the file is missing.

A URL (`s3://…`, `https://…`) is **not** a file here. It is treated as a local
path that does not exist, so the entry never notices the object changing. Use
`RemoteFileDataSource` for remote data (below).

<!-- claim: cash/file_source.py:FileDataSource @69335436 broad="the content-digest contract is a property of the whole class" -->
!!! note "`file_depends_on=` or `FileDataSource`?"
    `FileDataSource(path)` is the `DataSource` for `depends_on=` and
    `dynamic_depends_on=`. It puts the file's **content digest** in the key,
    the same check `file_depends_on=` gets, so a `touch` alone does not
    recompute. Use `file_depends_on=` for a fixed path, and `FileDataSource`
    only when the path depends on the arguments
    ([Dynamic dependencies](dynamic-dependencies.md)).

## Remote objects: tracked by the store's own validator

<!-- claim: cash/remote_source.py:RemoteFileDataSource @754fe5e0 broad="the scheme list and validator contract are properties of the whole class" -->
Object stores keep a validator for every object, such as an ETag, a version id
or a GCS generation, and it changes when the bytes change. A read of `s3://`,
`gs://`, `az://` or `https://` through a tracked reader records that validator.
Each hit asks the store for it again: one small metadata request instead of a
download.

<!-- test:skip reason="needs a reachable bucket" -->
```python
@cash.cache
def load_events(url):
    return pd.read_parquet(url)

load_events("s3://bucket/events.parquet")   # first call: downloads, records the ETag
load_events("s3://bucket/events.parquet")   # cache hit: one metadata request
```

<!-- claim: cash/remote_source.py:_fsspec_token @e1bf519f, cash/remote_source.py:_listing_token @fe6b3d29 -->
A prefix or a glob (`pd.read_parquet("s3://bucket/events/")`) is tracked by
its listing: every object's name and validator, read with one LIST request
per check. A new, removed or rewritten partition recomputes the next call.

Overwriting the object recomputes the next call; re-uploading identical bytes
does not. One exception: S3's ETag for a multipart upload also depends on the
part size, so re-uploading the same data with a different tool can cost one
recompute. The validator belongs to the object, not to a machine, so remote
reads share well ([Sharing a cache](sharing-caches.md)).

For a read cash cannot see, such as one through `boto3`, declare it:

<!-- test:skip reason="needs a reachable bucket" -->
```python
from cash import RemoteFileDataSource

@cash.cache(depends_on=[RemoteFileDataSource("s3://bucket/events.parquet")])
def load_via_boto3():
    return read_via_boto3("bucket", "events.parquet")
```

<!-- claim: cash/remote_source.py:RemoteFileDataSource._warn_failure @bc945f69 -->
- `RemoteFileDataSource(url, immutable=True)` checks once per process. A URL
  that pins a version (`?versionId=…`, `#generation=…`) counts as immutable
  without being told.
- If the store can't be reached, the call recomputes rather than serve an
  unchecked result, and warns once.

## Teaching cash a new reader

<!-- claim: cash/core.py:Cash.register_file_handler @15d195a5 -->
If many functions read through a reader cash doesn't know, register it once
instead of adding `file_depends_on=` everywhere:

<!-- test:skip reason="illustrative: my_lib is not a real library" -->
```python
from cash import Cash

app = Cash()

def track_read_data(original, track_callback):
    def wrapper(path, *args, **kwargs):
        track_callback(path)                  # record the file
        return original(path, *args, **kwargs)
    return wrapper

app.register_file_handler("my_lib", "read_data", track_read_data)
```

Cash calls your factory with the original function and a `track_callback`, and
installs the wrapper while a cached call runs. `func_name` may be a glob
(`"read_*"`), and `module_name` may be dotted (`"my_lib.io"`). The wrapper
replaces the attribute on the module, so call `my_lib.read_data(...)`: a name
bound earlier by `from my_lib import read_data` keeps the original. Pass
`track_callback` an absolute path, or one relative to the current directory.

## Edge cases

- **A file that changes during the call.** If a file moved between the read and
  the end of the call, the result is returned but not stored
  ([`STORE-INPUT-CHANGED`](../../warnings.md#store-input-changed)), since cash
  can't tell which version it came from. The next call caches normally.
- **Symlinks** record both the link and its target, so editing the target and
  re-pointing the link both recompute.
- **Relative paths** are also recorded as written and re-resolved against the
  current directory on each lookup. The same name read from another directory is
  another file, so switching directories recomputes.
- **Files next to your code** (package data, `Path(__file__).parent / "x.csv"`)
  are checked in each install's own copy. Two installs of the same code share
  entries when their data matches.
- **Network file systems.** Content checks work there, but folder dependencies
  (`glob`, `listdir`) rely on the folder's modification time, which some network
  mounts don't update. List such files with `file_depends_on=`.
- **Recorded paths are absolute.** On a machine where the files live at other
  paths, calls that read them recompute.
- **Timestamps put back by hand.** On Windows, a same-size edit that restores
  the original modification time is not seen; see
  [Known limitations](../../known-limitations.md#an-edit-that-keeps-the-size-and-timestamps).

## Related

- [Dynamic dependencies](dynamic-dependencies.md): a dependency chosen by the arguments.
- [Sharing a cache](sharing-caches.md)
- [The `@cash.cache` guide](../../decorator.md)
