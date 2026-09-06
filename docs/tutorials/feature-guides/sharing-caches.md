# Sharing a cache across machines and teammates

One person waits four minutes for a feature build; everyone else on the team
waits four minutes for the same feature build. A shared cache fixes that — the
second person's run hits an entry the first person's run stored.

This guide is about making that work, and being clear-eyed about when it can't.
The short version: **point everyone at a shared backend, and put the expensive
work behind [`@cash.cache`](../../decorator.md).** Exporting cache files by hand
is a different tool for a different job — see [the end of this page](#when-a-file-handoff-is-the-right-tool).

---

## The precondition: the key has to come out the same

A cache hit happens when two runs compute the **same cache key**. Nothing about
sharing changes that — a shared backend just means both runs look in the same
place. So sharing works exactly as well as your keys match, and keys are built
from content: the function's source, the source of helpers it calls, the
arguments, and the state it reads.

Two consequences worth knowing before you wire anything up:

- **Same code, same import path.** A decorated function is keyed as
  `module.qualname`, so `features.build(df)` imported from a shared package
  matches across machines — but the *same* function pasted into a notebook cell
  is `__main__.build` and will not share entries with it. Sharing rewards code
  that lives in a real module.
- **Same library and Python versions, in practice.** Cash hashes your source,
  not your dependencies. If a pandas upgrade changes results, the key won't
  notice — pin what matters, or the shared cache will faithfully serve you
  agreement rather than correctness. Entries are pickled, so a shared store is
  also happiest when everyone reading it is on the same Python version.

---

## Use a shared backend

The recommended setup: a Redis or S3 backend that everyone's cash instance
points at. Configure it once, per environment, and sharing is automatic — no one
has to remember to export anything.

Both are optional extras, so install the one you're using
(`pip install "cash-lib[redis]"` or `"cash-lib[s3]"`); without it the backend
raises `DependencyNotFoundError` on first use.

<!-- test:skip reason="illustrative — requires a reachable Redis server" -->
```python
import cash

cash.configure(backend="redis", redis_host="redis.internal", redis_port=6379)
```

In practice you don't want that host hardcoded in a notebook. Every setting has
a `CASH_*` environment variable, so the usual pattern is to set it per
environment and leave the code alone:

```bash
export CASH_BACKEND=redis
export CASH_REDIS_HOST=redis.internal
export CASH_REDIS_PREFIX=team-analytics:
```

Use `CASH_REDIS_PREFIX` (or `CASH_S3_PREFIX`) to keep separate projects from
colliding in one store. For choosing between Redis, S3, and a tiered stack — and
the latency trade-offs of each — see
[Choosing a backend](choosing-a-backend.md).

!!! tip "Keep a local tier in front"
    A shared remote backend on its own means every hit pays a network round
    trip. A tiered stack — RAM in front of the shared store — serves repeat hits
    in-process and only reaches out when it must. That is also why the cost
    model's predictions are calibrated for LAN Redis and same-region S3; over a
    WAN they under-charge the real cost. See [Cost model](../../cost-model.md#remote-backends-the-predictions-are-estimates).

---

## What travels between machines, and what doesn't

This is the part that decides whether a shared backend actually pays off, and
it differs between the two caching paths:

<!-- claim: cash/notebook/statement/file_deps.py:compute_file_hash_component @3dc65a7c, cash/notebook/file_dep_snapshot.py:file_dep_is_fresh @5f35e472 -->
<!-- claim: cash/remote_source.py:RemoteFileDataSource @16c56807 broad="the portability claim is a property of the whole source type" -->

| What you cached | Hits on another machine? |
|---|---|
| `@cash.cache` function, pure compute | **Yes** — source and arguments are hashed by content |
| `@cash.cache` function that reads files | **If the paths line up** — freshness is re-checked by content rather than timestamp, but against the file paths recorded when the entry was written (the absolute path among them). Identical layouts — a container image, a shared mount, CI — hit; a checkout under a different home directory can't resolve the recorded path, revalidates as stale, and recomputes |
| `@cash.cache` function that reads **object storage** | **Yes** — the recorded dependency is the object's ETag/version, a fact about the object rather than about one filesystem, so there are no paths to line up |
| Notebook statement, pure compute | **Yes** — the statement key carries no notebook path or cell id, *provided* none of its inputs descend from a file-reading statement (the file component rides down the lineage chain) |
| Notebook statement that **reads a file** | **No** — the file's path, mtime and size are folded into the statement's lineage, and a fresh checkout has different timestamps. Every statement downstream inherits that through its inputs |

The last row is the one to plan around, and no backend can fix it: the two runs
compute *different keys*, so they never look at the same entry.

<!-- claim: cash/notebook/file_dep_snapshot.py:file_content_hash @6bdf50df, cash/notebook/file_dep_snapshot.py:_HASH_FULL_MAX_BYTES == 8388608 -->
!!! note "Large files carry an extra condition"
    Files over 8 MiB are hash-**sampled** rather than read end to end, so their
    freshness check also consults the mtime. A fresh clone of a big parquet
    therefore recomputes once even when the paths do line up.

**The practical rule:** move expensive file-reading work into a decorated
function and share that.

<!-- test:skip reason="illustrative — references a project-specific data path" -->
```python
# features.py — shared module, shared cache
@cash.cache
def load_features(path):
    return pd.read_parquet(path).pipe(build_features)
```

<!-- test:skip reason="illustrative — imports the project-specific features module from the fence above" -->
```python { .nb-cell }
# notebook — the expensive read happens inside the decorated function
from features import load_features
feats = load_features("data/events.parquet")
```

Now the heavy step's key no longer bakes in a timestamp — a `touch`, a re-save,
or an identical re-download stops invalidating it — and where the data file sits
at the same path on both sides (a container image, a shared mount, CI), your
teammate hits it.

If the two layouts *don't* match, be aware of the failure mode: the machine that
misses recomputes and stores the entry with **its** paths, so the next run on the
other machine misses in turn. Two differing layouts trade misses rather than
settling. Pinning the data location — a mount point, a container path, an
absolute path from config — is what makes sharing pay off.

The notebook keeps its statement-level caching for everything else; that part
stays local, which is usually what you want anyway.

!!! tip "Data in object storage? The problem disappears"
    If the shared data already lives in S3/GCS/Azure, none of the above applies.
    Cash tracks a remote read by the object's **ETag / version-id / generation**
    — the validator the store maintains itself — and that token is a fact about
    the *object*, not about one machine's filesystem. It is therefore identical
    on every machine, so your teammate's key matches yours. The portability
    problem doesn't get worked around; it stops existing.

    This is automatic for reads cash can see
    (`pd.read_parquet("s3://bucket/key")`); declare the ones it can't with
    `depends_on=[RemoteFileDataSource(url)]`. `file_depends_on=` is the wrong
    tool here — its token is a local mtime, which for a URL is a constant, so
    the entry would never invalidate. See
    [Remote objects](custom-file-sources.md#remote-objects-tracked-by-the-stores-own-validator).

---

## The trust boundary

!!! danger "A cache is executable — only share within a circle you'd share code with"
    Cash stores results by **pickling** them, and unpickling runs arbitrary
    code. Reading from a shared cache is therefore equivalent to running code
    written by whoever populated it.

    - Fine: a team Redis inside your own infrastructure, an S3 bucket only your
      org can write to.
    - Not fine: a cache directory from someone outside that circle, or a store
      any untrusted party can write to.

    Treat write access to a shared cache exactly like commit access to the
    repository. See [Backends](../../api/backends.md#security) for the full
    trust model.

---

## When a file handoff is the right tool

[`%cash_export`](../../magics.md#cash_export) and
[`%cash_import`](../../magics.md#cash_import) move cache entries as a file.
They are a **point-in-time snapshot**, not a sharing mechanism: the moment
either side edits code, the exported entries stop matching. For a team, prefer a
shared backend.

They earn their place in narrower situations:

- **Attaching a cache to a bug report** — so someone else can reproduce the
  exact state you saw.
- **Archiving a run** for reproducibility, alongside the code that produced it.
- **No shared infrastructure**, and a one-off handoff is genuinely simpler than
  standing up Redis.

<!-- test:skip reason="IPython magic commands — require kernel context" -->
```python
%cash_export handoff.cache          # write entries + lineage to a file
%cash_import handoff.cache --merge  # on the other side; --merge keeps existing entries
```

The same portability rules apply — an exported cache is subject to every row of
the table above, so a file-reading notebook statement won't hit after the move
either.

---

## Related

- [Choosing a backend](choosing-a-backend.md) — Redis vs S3 vs tiered, and the latency trade-offs.
- [Where your cache lives](../../how-it-works/storage.md) — tiers, promotion, and the pickle trust model.
- [Cache keys, lineage & hashing](../../how-it-works/cache-keys-and-lineage.md) — what actually goes into a key.
- [Production transition](production-transition.md) — moving notebook work into modules, which is also what makes it shareable.
- [Troubleshooting & debugging](debugging-and-monitoring.md) — including `%cash_diff` for "why does my session differ from yours?".
