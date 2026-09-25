# Sharing a cache

!!! info "Applies to: both paths"
    Teams and jobs on several machines that want one run's results to be
    another run's hits.

When one person has computed a four-minute feature build, a shared cache lets
everyone else get it as a hit. Cash has no export format: you share a cache by
pointing everyone at the same backend.

## Keys have to match

A shared backend only means two runs look in the same place. A hit still needs
both runs to compute the **same key**, and keys are built from content: the
function's code and its helpers, the arguments, and the data it reads. So:

- **Same code in the same module.** A decorated function is keyed by its module
  and name, so `features.build` imported from a shared package matches across
  machines. The same function pasted into a notebook cell is a different
  function and does not share entries with it.
- **Same library and Python versions.** Cash keys your code, not your
  dependencies, and entries are pickled. Pin versions, and let everyone who
  reads a shared store run the same Python.

## Point everyone at a shared backend

Redis or S3, set per environment so no code changes. Install the extra you use
(`pip install "cash-lib[redis]"` or `"cash-lib[s3]"`):

```bash
export CASH_BACKEND=redis
export CASH_REDIS_HOST=redis.internal
export CASH_REDIS_PREFIX=team-analytics:
```

Use `CASH_REDIS_PREFIX` (or `CASH_S3_PREFIX`) to keep projects apart in one
store. [Choosing a backend](choosing-a-backend.md) compares the options.

<!-- claim: cash/backends/_base.py:CacheBackend.lock @2c1d7483, cash/backends/redis_backend.py:RedisBackend.lock @cfdf2e01 -->
A RAM tier in front of the shared store serves repeat hits without a network
round trip. The trade-off: `use_locking=True` then locks within each process
only. Cross-process locking needs Redis as the backend itself; see
[Deploying](deploying.md#sharing-a-cache-between-processes-and-hosts).

## What travels between machines

<!-- claim: cash/notebook/statement/file_deps.py:compute_file_hash_component @ce37ff53, cash/tracking/file_dep_snapshot.py:file_dep_is_fresh @b9d64ecd -->
<!-- claim: cash/remote_source.py:RemoteFileDataSource @754fe5e0 broad="the portability claim is a property of the whole source type" -->
=== "Decorator"

    | What the cached function does | Hits on another machine? |
    |---|---|
    | Pure computation | **Yes**: code and arguments are hashed by content |
    | Reads local files | **If the paths match.** Each entry records the paths it read and checks their content on a hit. A container image, a shared mount or a CI checkout at a fixed path hits; a checkout in another home folder recomputes |
    | Reads a file next to its own code (package data) | **If that copy matches.** Each install checks its own copy |
    | Reads object storage (`s3://`, `gs://`, `https://`) | **Yes**: the recorded dependency is the object's ETag or version, the same on every machine |

    If two machines' paths differ, each one's miss stores an entry with its own
    paths, so they keep trading misses. Pin the data location: a mount point, a
    container path, an absolute path from configuration.

=== "Notebook"

    | What the statement does | Hits on another machine? |
    |---|---|
    | Pure computation | **Yes**, as long as none of its inputs came from a statement that read a file |
    | Reads a local file | **No**: the file's modification time is part of the key, and a fresh checkout has new timestamps. Every statement downstream inherits that |
    | Reads object storage | **Yes**: keyed by the object's ETag or version |

    To share expensive file-reading work, move it into a decorated function in
    a module and call that from the notebook; see
    [Moving to a module](production-transition.md).

<!-- claim: cash/tracking/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES_DEFAULT == 268435456 -->
Files over 256 MiB are hashed by sampling, so their check also compares the
modification time. A fresh clone of a large file therefore recomputes once,
even when the paths match.

## Data in object storage

If the shared data lives in S3, GCS or Azure, the path problem goes away. Cash
tracks a remote read by the object's ETag, version id or generation, which is
the same on every machine. That is automatic for reads cash sees, such as
`pd.read_parquet("s3://bucket/key")`. For a read it can't see (through `boto3`,
say), declare it with `depends_on=[RemoteFileDataSource(url)]`. Don't use
`file_depends_on=` for a URL: it is treated as a missing local file. See
[Remote objects](custom-file-sources.md#remote-objects-tracked-by-the-stores-own-validator).

## The trust boundary

!!! danger "A cache runs code: share it only with people you'd share code with"
    Cash stores results with `pickle`, and loading a pickle can run any code.
    Reading a shared cache is running code written by whoever filled it.

    - Fine: a team Redis inside your own infrastructure, a bucket only your
      organisation can write to.
    - Not fine: a cache folder from outside that circle, or a store anyone can
      write to.

    Treat write access to a shared cache like commit access to the repository.
    See [Backends](../../api/backends.md#security).

## Related

- [Choosing a backend](choosing-a-backend.md)
- [Deploying](deploying.md): locking, workers and CI for decorated code.
- [Cache keys and lineage](../../how-it-works/cache-keys-and-lineage.md): what goes into a key.
