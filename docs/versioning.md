# Versioning & compatibility

!!! info "Applies to: both paths"
    Anyone pinning or upgrading cash.

cash uses [semantic versioning](https://semver.org/) with the usual `0.x`
rule: **a minor release (a change in the second number) can remove or change
public APIs outright**, with no deprecation period. Each removal is listed in the
[CHANGELOG](https://github.com/galgtonold/cash/blob/main/CHANGELOG.md). Patch
releases (a change in the third number) only fix bugs.

## The public API

The public API is what the [API reference](api/index.md) documents, except
the [Internals](api/backend_internals.md) page: the names in `cash.__all__`,
plus `cash.cache`, `cash.show_stats`, `cash.register_hasher` and `cash.help`
(left out of `__all__` so that `from cash import *` leaves your `help()` alone
and builds nothing), the magics, the `# @cash:` annotations and the `cash`
command. Anything else, including any name that starts with an underscore, can
change in any release.

## The cache format

<!-- claim: cash/backends/cache_dir.py:CacheDirStamp.check @5ae84fb2 -->
The on-disk cache format is not part of the public API and can change in any
minor release. For a cache folder (the default) you do not need to do
anything when it changes: cash clears a folder written in an incompatible
format when it opens it, logs how many entries it removed, and treats any
entry it cannot read as a miss. The first run afterwards computes everything
again.

An older cash never clears a folder written in a newer format. It leaves the
folder as it is, warns once
([CACHE-FORMAT-NEWER](warnings.md#cache-format-newer)) and writes nothing to
disk for that run, so two environments on different versions do not delete
each other's results. Give each its own `cache_dir`, or upgrade every
environment that shares a folder together.

<!-- claim: cash/backends/sqlite_backend.py:SQLiteBackend._create_tables @d3a9a0c1, cash/backends/redis_backend.py:RedisBackend._entry_prefix @a96d9ffd, cash/backends/s3_backend.py:S3Backend._entry_prefix @6fbcba0f -->
The other backends follow the same rule. A SQLite cache is stamped like a
folder: a database in an older format is emptied when cash opens it, and one
written by a newer cash is left alone with the same warning. On Redis and S3,
which several machines share, every entry carries the format in its name, so
each cash version reads and writes only entries of its own format and versions
with different formats share a server or bucket without touching each other's
entries. Entries of a format nobody uses any more stay there until you remove
them: clear the cache from the old version before upgrading, or delete them
with the store's own tools. The release's CHANGELOG entry says when the format
changed.

There is no tool to migrate a cache between versions. The cache is a
performance layer, not a store of record: write anything you cannot afford to
recompute to a file or a database yourself.

## Pinning

Pin cash like any dependency you rely on, for example
`cash-lib<!-- docnum:version_pin -->~=0.15.0<!-- /docnum -->` to take patch
releases but not <!-- docnum:version_next_minor -->0.16<!-- /docnum -->, and
read the CHANGELOG before you move to a new minor release.

## Related

- [CHANGELOG](https://github.com/galgtonold/cash/blob/main/CHANGELOG.md):
  what changed in each release, removals first.
- [Where your cache lives](how-it-works/storage.md): the cache directory an
  upgrade may clear.
- [API reference](api/index.md): the names the public API covers.
