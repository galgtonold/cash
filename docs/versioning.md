# Versioning & compatibility

Cash follows [semantic versioning](https://semver.org/), with the usual `0.x`
caveat: **while cash is in `0.x`, a minor version bump (`0.1` → `0.2`) may make
breaking changes.** Patch releases (`0.1.1` → `0.1.2`) are for fixes and are
safe to take.

## What "the public API" means

The supported surface is what's documented in the [API reference](api/index.md):
the `cash` top-level names, the `@cash.cache` decorator, the magics (`%cash_on`,
`%cash_stats`, …), the annotations (`# @cash:...`), and the public backend
classes. Anything under a leading underscore, or not shown in the API reference,
is internal and may change at any time.

## The cache format is *not* covered by the API guarantee

This is the one to remember. The on-disk cache format — how entries are keyed,
serialized, and laid out under `.cash/` — is an **implementation detail** and
**may change between minor versions**. Cache entries written by one minor version
are not guaranteed to be readable by another.

**After upgrading cash, run `%cash_repair --full`** (or delete `.cash/`) to clear
any entries written by the previous version. A stale entry from an older format
is discarded rather than misread — you never get a *wrong* value from a format
change, only a recompute — but clearing avoids the wasted space and the
first-run misses.

There is deliberately no cross-version cache-migration tool: the cache is a
disposable performance layer, not a source of truth. Anything you can't afford
to recompute should be written out explicitly (to a file, a database, a
model artifact), not left to live only in the cache.

## Deprecations

When a public API needs to change, the previous form is kept working for at
least one minor version and emits a `DeprecationWarning` pointing at the
replacement, wherever that's feasible. Removals are called out in the
[CHANGELOG](https://github.com/galgtonold/cash/blob/main/CHANGELOG.md).

## Pinning

Pin cash the way you'd pin any dependency you rely on in production — e.g.
`cash-lib<!-- docnum:version_pin -->~=0.9.0<!-- /docnum -->` to take patch fixes but not an
automatic `<!-- docnum:version_next_minor -->0.10<!-- /docnum -->` — and upgrade
deliberately, reading the CHANGELOG and running `%cash_repair --full` as part of
the bump.
