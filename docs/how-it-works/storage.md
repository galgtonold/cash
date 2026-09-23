# Where your cache lives

A cache is only useful if reading it back is cheaper than recomputing. Cash's
default backend is **tiered**: a fast in-memory layer in front of a persistent
on-disk layer, with a promotion policy that decides what's worth writing down.

## The tiers

<!-- claim: cash/backends/tiered_backend.py:TieredBackend @29f9b0de, cash/backends/memory_backend.py:InMemoryBackend, cash/backends/file_backend.py:FileBackend, cash/backends/sqlite_backend.py:SQLiteBackend, cash/backends/redis_backend.py:RedisBackend, cash/backends/s3_backend.py:S3Backend broad="tier ordering and read-repair are properties of the class as a whole" -->
The default `TieredBackend` stacks two layers, fastest first:

| Tier | Backend | Speed | Survives restart? |
|------|---------|-------|-------------------|
| **L1** | `InMemoryBackend` | Fastest (RAM) | No — cleared on kernel restart |
| **L2** | `FileBackend` | Fast (local disk) | Yes |

On a read, a hit in L2 is **promoted back up to L1** (read-repair) with no policy
check at all — if it was worth persisting, it's worth holding hot, and the memory
tier does its own eviction. The one value it turns away is one it could never
keep: over 90% of its own cap, where holding it would first evict everything
else. Writes are the opposite: they're gated, and the next
section is about that gate. Cash also ships backends you can swap in or stack —
`SQLiteBackend`, `RedisBackend` and `S3Backend`; `Cash(backends=[...])` stacks
the ones you pass in a `TieredBackend` of their own (see
[Choosing a Backend](../tutorials/feature-guides/choosing-a-backend.md)).

## What's worth persisting

Writing every result to disk would be wasteful — a value that recomputes in
20 ms isn't worth a disk write. But "wasteful" has to be measured, not guessed,
so the promotion decision runs against a **fitted cost model**
(`cash/notebook/cost_model.py`): a table of `(family, backend, operation)`
coefficients measured offline, which predicts how long it will actually take to
read this value back and deserialize it. [Cost model](../cost-model.md) covers
the fit itself and every knob; what follows is only the promotion decision.

**This is the notebook's decision, not the decorator's.** Cash caches every
statement of a notebook by itself, so it has to judge which are worth keeping. A
`@cash.cache` function was singled out by hand: its result is written whatever it
cost to produce and whatever it costs to read back, and neither gate below is
consulted. The only thing that can still stop that write is one of the per-tier size caps
described below.

For a notebook statement, three gates, in order:

<!-- claim: cash/backends/factory.py:_SMART_PERSIST_COMPUTE_FLOOR_S == 0.1 -->
1. **A compute floor.** Anything under **0.1 s** never leaves RAM — disk I/O
   alone would cost more than rerunning it.
<!-- claim: cash/config.py:CashConfig.min_cache_savings_pct == 0.2 -->
2. **A savings test.** Promote only when the restore is meaningfully cheaper
   than the recompute:
   `execution_time - est_restore > min_cache_savings_pct × execution_time`,
   with `min_cache_savings_pct` defaulting to 0.20. Twenty percent isn't a
   rounding error's worth of headroom; it's the margin below which the write
   isn't paying for itself.
<!-- claim: cash/backends/value_policy.py:worth_its_bytes, cash/backends/value_policy.py:WORTH_CEILING_BYTES_PER_SECOND == 134217728, cash/backends/value_policy.py:WORTH_FLOOR_BYTES == 8388608 -->
3. **A rate ceiling.** Both gates above ask whether restoring beats recomputing;
   neither asks what the answer costs in disk. So cash spends at most **128 MiB
   of cache per second of compute saved**, and refuses anything over 8 MiB that
   exceeds that rate
   ([`CACHE-NOT-WORTH-BYTES`](../warnings.md#cache-not-worth-bytes) says so once
   per cell, naming each statement it refused).

The cost model uses the value's *real type* when the entry carries one
(`DataFrame`, `ndarray`, `dict`, …); when it doesn't — an injected policy, an
untyped entry — it falls back to a `_GENERIC` family, which
is the slowest family measured for that backend. Being conservative in the
fallback means an unrecognised type is under-promoted rather than promoted onto
a treadmill.

The important consequence of a fitted model is that a **big** result is *more*
likely to persist when it's expensive, not less. The old raw-bandwidth
arithmetic had the sign backwards on large objects. Drag the sliders to see
where the line falls:

<div class="cash-promotion-explorer" markdown="0">
  <table>
    <thead><tr><th>Compute time</th><th>Result size</th><th>Decision</th></tr></thead>
    <tbody>
      <tr><td>0.05 s</td><td>5 MB</td><td>RAM only — under the 0.1 s compute floor</td></tr>
      <tr><td>0.3 s</td><td>5 MB</td><td>Persisted — predicted restore ~0.02 s</td></tr>
      <tr><td>3 s</td><td>100 MB</td><td>Persisted — predicted restore ~0.22 s</td></tr>
      <tr><td>1.5 s</td><td>1 GB</td><td>RAM only — predicted restore ~2.1 s beats recomputing</td></tr>
    </tbody>
  </table>
</div>

!!! warning "The floor is per entry — it does not compose"
    The 0.1 s floor is applied to each cached value on its own. A cell of 120
    statements at 0.05 s each takes six seconds of wall clock and persists
    **nothing**: no individual statement cleared the floor, and Cash never sums
    them. If you have a genuinely slow cell made of individually cheap steps,
    the floor is why nothing survives your kernel restart. Force the write with
    a `# @cash:persist` comment on the statement, flip `%cash_persist on`
    (`persist_all=True`) for the session, or restructure so the expensive work
    lands in one statement.

    **A long loop is the exception, and it cuts the other way.** Once
    per-iteration bookkeeping stops paying for itself — over ~50 iterations, and
    roughly `iterations × body-statements × 8 ms` above one second — Cash caches
    the loop as a *single unit*. That one entry's compute is the whole loop, so
    it clears the floor and does persist. A shorter loop stays per-iteration and
    does not: 60 iterations of two cheap statements produce 120 sub-0.1 s
    entries and nothing on disk, while the same loop with a third statement in
    the body crosses the threshold, becomes one entry, and survives a restart.

<!-- claim: cash/config.py:CashConfig.min_execution_time_to_cache_seconds == 0.01 -->
Before promotion is even considered, a separate and much lower floor decides
whether a notebook statement is written at all: compute under
`min_execution_time_to_cache_seconds` (default **10 ms**) writes no entry, not
even a metadata-only one, so the next lookup is a fast clean miss instead of a
file read that only discovers the entry is a skip marker.

`force_persist` — set by a `# @cash:persist` annotation or by
`persist_all=True` — bypasses the compute floor entirely. It does not escape the
per-tier size caps below.

<!-- claim: cash/backends/tiered_backend.py:TieredBackend.__init__ @694d21ad, cash/backends/tiered_backend.py:TieredBackend._default_promotion_policy @c919ec7c, cash/config.py:CashConfig.smart_persistence == True -->
Two places the 0.1 s number quietly becomes 1.0 s. The 0.1 s floor is installed
by the backend *factory* when `smart_persistence` is on (the default); setting
`smart_persistence=False`, or constructing a `TieredBackend([...])` by hand,
falls back to `_default_promotion_policy` — same cost-model rule, same 20%
savings test, but a 1.0 s floor (and the conservative `_GENERIC` family for
entries that carry no recorded type). "Off" means "less eager", not
"unconditional". The floor itself is not configurable in
either case; changing it means supplying your own `promotion_policy`.

Size caps are the last word. Each persistent tier declines objects larger than a
safe fraction of its own cap, per tier — so a 20 MB frame can land in RAM and on
disk while skipping a Redis tier with a 10 MB limit. If a value cleared the
compute floor but *every* persistent tier refused it, Cash keeps it in RAM if it
fits there and warns once per session with `CashCacheIneffectiveWarning`, because
otherwise you would be left wondering why a value you were told was worth
caching evaporated on restart. A value that big is often over the RAM tier's own
limit too (90% of its cap), and then nothing holds it; see
[`CACHE-VALUE-TOO-BIG`](../warnings.md#cache-value-too-big).

## When the disk fills up

The disk tier has a size cap (`max_cache_size`; by default a quarter of the
room on the disk). Going over it is what makes Cash delete entries, and this
section is about when and how that happens.

<!-- claim: cash/backends/file_backend.py:FileBackend._do_set_sync @0e010aaf, cash/backends/file_backend.py:FileBackend._check_and_evict @92057a04 -->
**Only a write can trigger eviction.** Each time an entry lands on disk, the
background write thread adds its size to a running total and compares that
total to the cap. If the cache is over, it deletes entries until the cache is
back under **90%** of the cap. The extra 10% of room means the next few writes
fit without each one starting another round.

<!-- claim: cash/backends/file_backend.py:FileBackend._ensure_size_scanned @1ec3940e, cash/backends/file_backend.py:FileBackend.get @a300927d -->
**Reading never evicts.** A cache hit deletes nothing from disk to make room, however
full the cache is. A process that only reads, such as a kernel restart that
replays everything from cache, never even adds up the directory's size. That
one-time walk over every file happens on the process's first write, and on the
write thread, so no cell waits for it. The one thing a read can delete is the
entry it just read, when that entry is older than its `ttl`. That is expiry, not
making room, and it happens whether or not a cap is set.

**What goes first is whatever is worth least per byte:** how long the value took
to compute, multiplied by how often it has been read, divided by its size.
A result that took 30 seconds outlives a newer one that took 50 ms, and one huge
cheap value goes before many small expensive ones. Entries nobody reads slowly
lose their standing and age out, however valuable they once were, and entries
of about equal worth go least recently used first.
[Choosing a Backend](../tutorials/feature-guides/choosing-a-backend.md#filebackend)
covers how that ranking is kept cheap on a directory of 100k files.

<!-- claim: cash/backends/file_backend.py:FileBackend._touched_since @7f4b852e -->
A few entries are passed over in a round:

- **One that was read since the ranking was made.** The ranking is reused
  across many rounds, so a read after it was taken would not otherwise count.
- **One that has another write on its way.** That write will replace it anyway.

The value that was just written is **not** protected. If it is the least
valuable thing in the cache, it goes first. When that keeps happening, the cap
is too small for what is being cached, and
[`CACHE-THRASH`](../warnings.md#cache-thrash) warns once per session. If a round
runs out of entries it is allowed to delete, it stops and the cache stays a
little over its cap until the next write tries again.

**Each process enforces the cap on its own writes.** Several processes sharing
one cache directory each count what they wrote, so together they can go over
the cap before one of them evicts. With the default automatic cap, a
long-running process also re-measures the disk about once a minute while it
writes, so a disk that other programs are filling can shrink the cap and start
a round on the next write.

Separately from the cap, a re-run notebook statement drops its old versions
when the new one is written, so a notebook you keep editing doesn't fill the
cache with results nobody will ask for again (see
[old versions](../tutorials/feature-guides/choosing-a-backend.md#filebackend)).

## Turning objects into bytes

<!-- claim: cash/backends/serialization.py:get_serializer @76cf2c1b, cash/backends/serialization.py:ParquetSerializer, cash/backends/serialization.py:PickleSerializer, cash/backends/serialization.py:CloudPickleSerializer -->
To persist a value, Cash serializes it. A small factory — `get_serializer` —
picks the strategy from the data's type:

| Data | Serializer | Why |
|------|-----------|-----|
| pandas `DataFrame` (with a parquet engine — pyarrow or fastparquet) | `ParquetSerializer` | Columnar, compact, fast |
| Everything else | `PickleSerializer` | The general-purpose default |

If a `DataFrame` has no parquet engine installed, the factory falls back to
`PickleSerializer`, so persistence always works — just less optimally. Cash also
ships a `CloudPickleSerializer` for objects plain pickle can't handle (lambdas,
closures, dynamically-defined classes); the factory doesn't select it
automatically, but it's available when you need it and itself falls back to
standard `pickle` if `cloudpickle` isn't installed.

!!! warning "Pickle executes code on load — only trust caches you created"
    The default serializers are built on Python's `pickle`, and
    **`pickle.loads()` can execute arbitrary code while unpickling.** A cache
    file is therefore as trusted as the code that wrote it. Never point Cash at
    a cache directory (or a Redis/S3 bucket) populated by an untrusted party,
    and don't ship prebuilt caches between mutually-distrusting environments.
    Within your own project — your machine, your CI — this is a non-issue;
    across a trust boundary, treat a cache like any other executable artifact.

## A damaged entry is a miss

<!-- claim: cash/backends/entry_format.py:pack_entry @f6d2f3e2, cash/backends/entry_format.py:_verify @8a91dffb -->
Every persisted entry carries a crc32 of its payload, and every read that
touches the payload checks it. An entry that does not match -- a half-written
file, a bad sector, a sync client that merged two versions -- raises
`CorruptEntry`, which every caller already treats the way it treats a missing
file: the value is recomputed. Cash would rather spend the compute again than
hand you bytes that are not the ones it stored.

The check costs about 4.6 ms per 64 MB, a fraction of what reading those same
bytes from disk costs, which is why it is unconditional rather than an option.
It detects damage, not tampering: a cache is as trusted as whoever can write to
it (see the pickle warning above). An entry with no checksum at all is treated
the same way: it is not a readable entry, and the value is recomputed.

## Deciding before you deserialize

<!-- claim: cash/backends/lazy.py:LazyProxy @33f4359b, cash/backends/lazy.py:make_lazy_loader @11b3e48d broad="the defer-until-touched contract is the class as a whole" -->
Deserializing a multi-gigabyte object only to discover you wanted its size is
wasted work. `LazyProxy` exists for that: a handle that carries the entry's
metadata but defers `backend.get` until you reach for the value.

Be clear about what this is and isn't. It is **not** on the automatic read
path — `backend.get(key)` always returns a real value, never a proxy, at any
size. `make_lazy_loader` is a tool you call yourself when you are writing code
*over* a cache (an inspector, a migration script, a size audit) and want to look
before you load:

<!-- test:skip reason="requires a live backend instance — illustrative" -->
```python
from cash.backends.lazy import make_lazy_loader

proxy = make_lazy_loader(backend, cache_key)   # None if the key is absent

print(proxy.metadata)   # read from backend.get_metadata — no deserialization
print(proxy.is_resolved)  # False
df = proxy.value        # backend.get happens here, once, on first access
```

The metadata read goes through `backend.get_metadata`, which is cheap on every
backend precisely so existence and size can be established without touching the
payload. `resolve()` is idempotent: the loader runs at most once.

??? question "How does cache metadata stay typed without locking the backends in?"
    <!-- claim: cash/backends/_base.py:CacheMetadata @381f348f, cash/notebook/statement/_metadata.py:StatementCacheMetadata @cd79d868 broad="the frozen-dataclass-in, dict-on-the-wire contract is a property of both classes" -->
    Each entry carries metadata — execution time, size, ttl, type. Inside the
    cash layer that metadata is a **frozen dataclass** (`CacheMetadata` for the
    decorator layer, `StatementCacheMetadata` for the notebook layer), so call
    sites get typed attribute access (`meta.execution_time`) instead of
    stringly-typed `dict.get('execution_time')`. But on the wire — into and out
    of every backend — it's a **plain dict**. Producers call `.to_dict()` just
    before `backend.set(...)`; consumers call `from_dict(...)` just after
    `backend.get(...)`. Backends never see the dataclass; they round-trip an
    opaque `dict[str, Any]` and may inject their own private keys, which
    `from_dict` simply ignores. The payoff: the persisted format stays a
    version-tolerant dict — caches written by an older or newer Cash stay
    readable — while in-memory code gets real types. `to_dict()` omits `None`
    fields so the historical "only-set keys" shape is preserved, and
    `from_dict()` is lenient about unknown keys.
