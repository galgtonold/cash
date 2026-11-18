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

The on-disk cache format is not part of the public API and can change in any
minor release.

<!-- claim: cash/backends/cache_dir.py:CacheDirStamp.check @7bdce11c, cash/backends/file_backend.py:FileBackend -->
A cache folder (the default file cache) carries a stamp of its format. When
cash opens a folder stamped with another format, it logs a warning with the
number of files, deletes the entries, and the next run computes them again.
You do not need to do anything, and any entry cash cannot read is a miss, so
an upgrade costs a recompute, never a wrong value. The check runs both ways:
an older cash opening a folder a newer one wrote clears it too, so machines
that share a folder should run the same version.

A SQLite, Redis or S3 tier carries no stamp and is not cleared. An entry an
older version stored there can be served for a call the new version runs
every time, so clear such a tier after a minor upgrade: a SQLite cache with
`cash clear <dir>`, a Redis database or an S3 prefix with its own tools, or
`f.cache_clear()` for each cached function.

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
