# Glossary

Cash uses a handful of specific terms throughout the docs and in the per-cell
badge. This page defines them in one place; each entry links to the page that
covers it in depth.

<div class="cash-glossary" markdown="1">

## Annotation

A `# @cash:...` comment that overrides cash's default decision for one
statement or cell — e.g. `@cash:no-cache`, `@cash:persist`, `@cash:cache-fit`,
`@cash:allow-random`. Annotations are how you tell cash you know better than the
heuristic. See [Annotations](annotations.md).

## Badge

The compact summary cash renders above each executed cell, reporting what it did
for every statement — **restored**, **computed**, or **skipped** — plus the
storage tier, timing, and any warnings (randomness, mutation, side effects). The
badge is the primary way to *see* caching happening. See
[Reading the Cash badge](badges.md).

## Cache key

The hash that decides whether a statement **hits** or **misses**. It is computed
from the statement's code, the [lineages](#lineage) of the variables it reads,
and its [file dependencies](#file-dependency). Same key → restore the cached
result; different key → recompute. See
[Cache keys, lineage & hashing](how-it-works/cache-keys-and-lineage.md).

## Cost model

The fitted model cash uses to estimate how long restoring a result would take
versus recomputing it. It drives [promotion](#promotion) and the decision to
[skip](#skip-uncacheable) caching a result that would be slower to reload than
to recompute. See [Cost model](cost-model.md).

## File dependency

A data file a statement read (via `pd.read_csv`, `np.load`, `open`, …) that cash
intercepted and folded into the [cache key](#cache-key). When the file changes,
dependent statements recompute automatically. See
[Knowing when to recompute](how-it-works/invalidation.md).

## Freshness

Whether a cached statement's tracked inputs — especially [file
dependencies](#file-dependency) — are unchanged since the entry was written.
A stale input makes the entry unusable and forces a recompute. See
[Knowing when to recompute](how-it-works/invalidation.md).

## Hit / Miss

A **hit** means the [cache key](#cache-key) matched an existing entry and the
result was [restored](#restore) without executing. A **miss** means it was
recomputed (new code, changed input, or first run). Session totals are shown by
`%cash_stats`. See [Seeing what Cash did](how-it-works/inspecting.md).

## Lineage

A hash attached to a produced variable that identifies *how it was derived*. It
equals the producing statement's [cache key](#cache-key), and is propagated to
every downstream statement that reads the variable — so a change anywhere flows
through the dependency graph:

```
lineage(result) = hash(code + sorted(input_lineages) + file_deps)
```

Lineage (propagated downstream) is distinct from the cache key (which decides
one statement's hit/miss) even though they share a value. See
[Cache keys, lineage & hashing](how-it-works/cache-keys-and-lineage.md).

## Mutation detection

AST-based detection of in-place changes — `df['x'] = 0`, `lst.append(...)`,
`+=` — so that mutating a cached object correctly invalidates it and its
readers, rather than serving a stale value. See
[Knowing when to recompute](how-it-works/invalidation.md).

## Promotion

The [cost model](#cost-model)'s decision to move a result from the RAM tier (L1)
to the disk tier (L2) so it survives a kernel restart. A result is promoted when
recomputing it would cost more than restoring it from disk, above a ~100 ms
compute floor. Force it with `@cash:persist`. See
[Smart persistence](tutorials/feature-guides/smart-persistence.md).

## Provenance

The recorded computational history of a variable — which statements and inputs
produced it. Surfaced by `%cash_provenance`. See
[Seeing what Cash did](how-it-works/inspecting.md).

## Purity

Whether a function is free of side effects and therefore safe to cache. Cash
warns when it caches an impure function (LLM calls, HTTP, file writes) because
the side effect only runs on the first call. Declare intent with `@pure` /
`@stateful`. See [Purity decorators](tutorials/feature-guides/purity-decorators.md)
and [Knowing when not to cache](how-it-works/safety.md).

## Restore

Rebuilding a variable's value from the cache instead of executing the statement
that produces it — what happens on a [hit](#hit-miss). See
[The notebook path](how-it-works/notebook-path.md).

## Side-effect detection

Static flagging of statements that write files, hit the network, or touch a
database. These are not safe to silently cache, so cash warns or declines. See
[Knowing when not to cache](how-it-works/safety.md).

## Skip / Uncacheable

A statement cash decides *not* to cache — because it is too cheap to be worth
caching (under the ~10 ms floor), too large to store economically (the [cost
model](#cost-model) declines), or has side effects. The [badge](#badge) reports
it as skipped. See [Knowing when not to cache](how-it-works/safety.md).

## Statement-level caching

Caching each individual statement (line) in a cell independently, rather than
the whole cell. Edit one line in a 20-line cell and only that line and its
dependents recompute. This is cash's defining characteristic. See
[Overview](how-it-works/overview.md).

## Tier / TieredBackend

The default backend, layering a fast **RAM** tier (L1, `InMemoryBackend`) over a
persistent **disk** tier (L2, `FileBackend`). Results live in RAM and are
[promoted](#promotion) to disk when worth persisting. Other backends — SQLite,
Redis, S3 — are available for custom stacks. See
[Choosing a backend](tutorials/feature-guides/choosing-a-backend.md).

## Unseeded randomness

A random draw (`np.random.rand()`, `random.random()`, …) with no fixed seed.
Because the value can't be reproduced, caching it would freeze one draw forever;
cash warns and, by design, freezes the drawn value. Opt in explicitly with
`@cash:allow-random`. See [Known limitations](known-limitations.md).

## Upstream simulation

Cash's dry-run analysis of the notebook that determines, before executing, which
statements can be restored and which must recompute — including recomputing
skipped intermediate cells to satisfy a downstream read. See
[The notebook path](how-it-works/notebook-path.md).

</div>
