# Versioning & compatibility

!!! info "Applies to: both paths"
    Anyone pinning or upgrading cash.

cash uses [semantic versioning](https://semver.org/) with the usual `0.x`
rule: **a minor release (a change in the second number) can remove or change
public APIs outright**, with no deprecation period. Each removal is listed in the
[CHANGELOG](https://github.com/galgtonold/cash/blob/main/CHANGELOG.md). Patch
releases (a change in the third number) only fix bugs.

## The public API

The public API is what the [API reference](api/index.md) documents: the names
in `cash.__all__` (including `@cash.cache`), the magics, the `# @cash:`
annotations and the `cash` command. Anything else, including any name that
starts with an underscore, can change in any release.

## The cache format

The on-disk cache format is not part of the public API and can change in any
minor release. You do not need to do anything when it changes: cash clears a
local cache written in an incompatible format when it opens it, and treats any
entry it cannot read as a miss. An upgrade can cost a recompute, never a wrong
value. Run `cash clear --all` only if you want the space back at once.

There is no tool to migrate a cache between versions. The cache is a
performance layer, not a store of record: write anything you cannot afford to
recompute to a file or a database yourself.

## Pinning

Pin cash like any dependency you rely on, for example
`cash-lib<!-- docnum:version_pin -->~=0.11.0<!-- /docnum -->` to take patch
releases but not <!-- docnum:version_next_minor -->0.12<!-- /docnum -->, and
read the CHANGELOG before you move to a new minor release.
