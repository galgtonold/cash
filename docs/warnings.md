# Warnings

!!! info "Applies to: both paths"
    Every warning code cash emits, for `@cash.cache` users and notebook users. The index says which path each code comes from.

<!-- claim: cash/diagnostics.py:DIAGNOSTIC_CODES @9c9e1bfe -->
Every cash warning starts with a code in square brackets, such as
`[CACHE-THRASH]`, and ends with a link to that code's section below.

### Silencing one code

Filter on the code. The message always starts with it, so this silences one
code and nothing else:

    import warnings
    warnings.filterwarnings("ignore", message=r"\[CACHE-THRASH\]")

<!-- claim: cash/diagnostics.py:warn_diagnostic @af27ed90 -->
Every warning also carries its code on `.code`, so a handler can test
`w.message.code == "CACHE-THRASH"` instead of matching the wording. Filtering
by class catches many codes at once (the index shows which); the class
recipes are in [Exceptions and warnings](api/exceptions.md#filtering-warnings).
For a decorated function, `f.cache_info()["warnings"]` keeps the last twenty
warnings it raised, in case one scrolled past.

Where a section says to put `# @cash:assume-safe` on a line, that comment
waives that one finding and keeps the rest of the function checked.
`@cash.cache(assume_safe=True)` waives every finding in the function,
including code added to it later.

### Index

| Code | Applies to | Class | Meaning |
|---|---|---|---|
| [ANNOT-TTL-INVALID](#annot-ttl-invalid) | notebook | `CashCacheIneffectiveWarning` | `# @cash:ttl=` is not whole seconds; ignored |
| [ANNOT-UNKNOWN-DIRECTIVE](#annot-unknown-directive) | both | `CashCacheIneffectiveWarning` | `# @cash:<name>` is not a directive; ignored |
| [CACHE-ASYNC-GENERATOR](#cache-async-generator) | decorator | `CashCacheIneffectiveWarning` | async generators are not cached |
| [CACHE-DIR-UNWRITABLE](#cache-dir-unwritable) | both | `CashCacheStoreFailedWarning` | the cache directory cannot be written |
| [CACHE-EVICTED-RECOMPUTE](#cache-evicted-recompute) | both | `CashCacheIneffectiveWarning` | the size cap had evicted a costly result, so it was computed again |
| [CACHE-FRESHNESS-COST](#cache-freshness-cost) | decorator | `CashCacheIneffectiveWarning` | checking tracked files costs much of what a hit saves |
| [CACHE-IDENTITY-COUPLED](#cache-identity-coupled) | decorator | `CashCacheIneffectiveWarning` | a live matplotlib figure is never stored |
| [CACHE-IF-BYPASSED](#cache-if-bypassed) | decorator | `CashCacheIneffectiveWarning` | a chunked result was stored without running `cache_if` |
| [CACHE-IF-RAISED](#cache-if-raised) | decorator | `CashCacheIneffectiveWarning` | the `cache_if` predicate raised |
| [CACHE-LOOP-GROWTH](#cache-loop-growth) | notebook | `CashCacheIneffectiveWarning` | a loop stores every state of a growing value |
| [CACHE-NET-LOSS](#cache-net-loss) | decorator | `CashCacheIneffectiveWarning` | caching this function costs more time than it saves |
| [CACHE-NOT-WORTH-BYTES](#cache-not-worth-bytes) | notebook | `CashCacheIneffectiveWarning` | a big, cheap value was not written to disk |
| [CACHE-RESULT-SHARED](#cache-result-shared) | decorator | `CashImpurityWarning` | the result shares state with the caller's object |
| [CACHE-THRASH](#cache-thrash) | both | `CashCacheIneffectiveWarning` | the cache is full and evicts what it just stored |
| [CACHE-VALUE-TOO-BIG](#cache-value-too-big) | both | `CashCacheIneffectiveWarning` | a value is bigger than the disk cap |
| [CACHE-WRITE-ABANDONED](#cache-write-abandoned) | both | `CashCacheStoreFailedWarning` | the process exited before its writes finished |
| [CONFIG-FILE-MISSING](#config-file-missing) | both | `CashCacheIneffectiveWarning` | `Cash(config_path=...)` names a missing file |
| [CONFIG-INVALID](#config-invalid) | both | `CashCacheIneffectiveWarning` | a setting or config file cash cannot use |
| [CONFIG-TOML-UNREADABLE](#config-toml-unreadable) | both | `CashCacheIneffectiveWarning` | no TOML parser on Python 3.10 |
| [CONFIG-UNKNOWN-KEY](#config-unknown-key) | both | `CashCacheIneffectiveWarning` | a config key that is not a setting |
| [IMPURE-OBSERVED-EFFECTS](#impure-observed-effects) | decorator | `CashImpurityWarning` | the first call wrote, sent or changed something |
| [IMPURE-SCOPE-MUTATION](#impure-scope-mutation) | decorator | `CashImpurityWarning` | the function rewrites a global it reads |
| [IMPURE-SIDE-EFFECTS](#impure-side-effects) | decorator | `CashImpurityWarning` | the source has likely side effects |
| [KEY-AMBIENT-READ](#key-ambient-read) | decorator | `CashImpurityWarning` | the body reads the clock or a fresh UUID |
| [KEY-BOOL-STATE-TOKEN](#key-bool-state-token) | decorator | `CashCacheIneffectiveWarning` | `state_token()` returned a bool |
| [KEY-BUILD-FAILED](#key-build-failed) | decorator | `CashCacheIneffectiveWarning` | building the key raised; the call ran uncached |
| [KEY-CALLABLE-HASHER](#key-callable-hasher) | decorator | `CashCacheIneffectiveWarning` | a hasher registered for every function |
| [KEY-DEPENDS-ON-OPAQUE](#key-depends-on-opaque) | decorator | `CashCacheIneffectiveWarning` | a `depends_on=` target has no source or bytecode to fingerprint |
| [KEY-DYNAMIC-DEP-FAILED](#key-dynamic-dep-failed) | decorator | `CashCacheIneffectiveWarning` | a `dynamic_depends_on` resolver failed |
| [KEY-DYNAMIC-DEPENDENCY](#key-dynamic-dependency) | decorator | `CashImpurityWarning` | code in an argument picks what it calls at run time |
| [KEY-FROZEN-MUTATED](#key-frozen-mutated) | decorator | `CashImpurityWarning` | a `frozen=True` result was modified |
| [KEY-FROZEN-NO-EFFECT](#key-frozen-no-effect) | decorator | `CashCacheIneffectiveWarning` | `frozen=True` cannot mark this result |
| [KEY-INSTANCE-STATE](#key-instance-state) | decorator | `CashCacheIneffectiveWarning` | a bound method's instance cannot be hashed |
| [KEY-NETWORK-READ](#key-network-read) | decorator | `CashImpurityWarning` | the body reads from a server or database |
| [KEY-OPAQUE-CALLABLE](#key-opaque-callable) | decorator | `CashImpurityWarning` | a callable's code cannot be hashed |
| [KEY-SOURCE-CHANGED](#key-source-changed) | decorator | `CashCacheIneffectiveWarning` | a code file changed after import |
| [KEY-UNHASHABLE-ARG](#key-unhashable-arg) | decorator | `CashCacheIneffectiveWarning` | an argument cannot be hashed; not cached |
| [KEY-UNHASHABLE-DEFAULT](#key-unhashable-default) | decorator | `CashCacheIneffectiveWarning` | a parameter default cannot be hashed; not cached |
| [KEY-UNHASHABLE-GLOBAL](#key-unhashable-global) | decorator | `CashImpurityWarning` | a global the function reads cannot be hashed |
| [NOTEBOOK-BAILOUT](#notebook-bailout) | notebook | `CashCacheIneffectiveWarning` | an internal error; the cell ran uncached |
| [NOTEBOOK-CELL-SYNTAX](#notebook-cell-syntax) | notebook | `CashUpstreamSyntaxWarning` | an earlier cell does not parse |
| [NOTEBOOK-NOT-FOUND](#notebook-not-found) | notebook | `CashWarning` | the notebook file is unknown; cross-cell tracking is off |
| [NOTEBOOK-SAVEFIG-SKIP](#notebook-savefig-skip) | notebook | `CashWarning` | a `plt.savefig` was not re-run |
| [RANDOM-REPLAYED](#random-replayed) | notebook | `CashRandomnessWarning` | a restored value is an earlier random draw |
| [RANDOM-SEED-NONE](#random-seed-none) | notebook | `CashRandomnessWarning` | `seed(None)` cannot refresh cached values |
| [RANDOM-UNSEEDED](#random-unseeded) | both | `CashRandomnessWarning` | a draw from an unseeded source is not reproducible |
| [REMOTE-FRESHNESS-COST](#remote-freshness-cost) | both | `CashCacheIneffectiveWarning` | checking remote files costs more than it protects |
| [REMOTE-SIZE-ONLY](#remote-size-only) | both | `CashCacheIneffectiveWarning` | a remote file is tracked by size alone |
| [REMOTE-STATE-UNREADABLE](#remote-state-unreadable) | both | `CashCacheIneffectiveWarning` | a remote file's state could not be read |
| [STORE-CHUNK-FAILED](#store-chunk-failed) | decorator | `CashCacheStoreFailedWarning` | one chunk of an iterator failed to write |
| [STORE-CODE-CHANGED](#store-code-changed) | decorator | `CashCacheStoreFailedWarning` | code changed on disk during the call; not stored |
| [STORE-FAILED](#store-failed) | both | `CashCacheStoreFailedWarning` | the backend refused the write |
| [STORE-INPUT-CHANGED](#store-input-changed) | decorator | `CashCacheStoreFailedWarning` | an input file changed during the call; not stored |
| [STORE-LOCK-FAILED](#store-lock-failed) | decorator | `CashCacheIneffectiveWarning` | the per-key lock failed; ran without it |
| [STORE-METADATA-INVALID](#store-metadata-invalid) | decorator | `CashCacheIneffectiveWarning` | a stored entry's metadata is unreadable |

## ANNOT-TTL-INVALID {#annot-ttl-invalid}

*Notebook.*

<!-- claim: cash/analysis/annotations.py:parse_annotation_line @5d8ea461 -->
**What happened.** The value after `# @cash:ttl=` is not a whole number of
seconds (`ttl=5m`, `ttl=300.0`, `ttl=-5`, or no `=`), so cash ignored the
annotation.

**Why it matters.** The statement is still cached, with no expiry of its own.
If the TTL was the only thing meant to refresh it, the stored value is served
until something else invalidates it.

<!-- claim: cash/analysis/annotations.py:ANNOTATION_PATTERN @412c3ce1, cash/analysis/annotations.py:parse_annotation_line @5d8ea461 -->
**What to do.** Write plain seconds: `# @cash:ttl=300` for five minutes. Put
one directive per line: a second `# @cash:` on the same line is not read. See
[Annotations](annotations.md).

**When it is safe to ignore.** When cash already tracks what the value depends
on (a file, a `DataSource`) and the TTL was only a backup.

## ANNOT-UNKNOWN-DIRECTIVE {#annot-unknown-directive}

*Both paths: a notebook cell or a cached function's source.*

<!-- claim: cash/analysis/annotations.py:KNOWN_DIRECTIVES @91e03db7, cash/analysis/annotations.py:_warn_unknown_directive @24d9ef6e -->
**What happened.** A comment starts `# @cash:` but the word after it is not a
directive, so cash ignored it. The known directives are `no-cache`, `persist`,
`ttl=N`, `allow-random`, `cache-fit`, `no-cache-calls` and `assume-safe`. When
your spelling is close to one, the message suggests it.

**Why it matters.** The code behaves as if the comment were not there. A
misspelled `no-cache` means the statement is cached anyway.

**What to do.** Fix the spelling and run the cell again. See
[Annotations](annotations.md).

**When it is safe to ignore.** When the comment was never meant as a
directive. Reword it so it does not start `# @cash:`.

## CACHE-ASYNC-GENERATOR {#cache-async-generator}

*Decorator.*

<!-- claim: cash/core.py:Cash.cache @df737927 -->
**What happened.** You put `@cash.cache` on an async generator (an
`async def` that `yield`s). Cash does not cache those, so it returned your
function unwrapped.

**Why it matters.** Every call runs the whole body.

**What to do.** If the results fit in memory, move the work into a plain
`async def` that returns a list and cache that. Otherwise cache the expensive
step inside the generator.

**When it is safe to ignore.** When the generator is cheap and you do not
need it cached.

## CACHE-CLEAR-INCOMPLETE {#cache-clear-incomplete}

*Decorator.*

<!-- claim: cash/core.py:Cash._delete_backend_entries @b7c16174, cash/backends/file_eviction.py:FileEvictor.remove_path @ec89275a -->
**What happened.** `f.cache_clear()` could not remove some of the function's
entries. The message says how many.

**Why it matters.** Those entries are still stored, so the next call with
their arguments is served the result you meant to clear.

**What to do.** On Windows, a file cannot be removed while another process has
it open: a reader, a virus scanner, a search indexer. Cash retries briefly
first. Close whatever holds the cache folder and clear again, or run
`cash clear --function NAME` once it has let go.

**When it is safe to ignore.** Never when you cleared to get rid of a wrong
result. For freeing space only, the entries go at the next clear.

## CACHE-DIR-UNWRITABLE {#cache-dir-unwritable}

*Both paths.*

<!-- claim: cash/backends/cache_dir.py:warn_if_unwritable @ca7bc994 -->
**What happened.** Cash could not create a file in its cache directory: a
read-only mount, missing permissions, or a path that no longer exists. The
message names the directory and the OS error.

**Why it matters.** Nothing is written to disk for the rest of the run, so
every new process recomputes. Results are still correct, and in-process repeats
still hit the RAM tier.

**What to do.** Point cash at a writable directory, or give this account write
permission on the one named:

<!-- test:skip reason="repoints the cache at a fixed absolute path; running it would send every later fence on this page to a directory that outlives the test" -->
```python
cash.configure(cache_dir="/var/tmp/cash")   # or CASH_CACHE_DIR=... in the env
```

In a container, check the cache path is on a writable volume.

**When it is safe to ignore.** When the directory is read-only on purpose: a
shared cache you only read from.

## CACHE-EVICTED-RECOMPUTE {#cache-evicted-recompute}

*Both paths.*

<!-- claim: cash/backends/budget_notices.py:evicted_recompute_warning @0dd945bf, cash/effectiveness.py:CUMULATIVE_WASTE_SECONDS == 2.0, cash/decorator/reporting.py:Notices.evicted_recompute @94976b5e, cash/notebook/statement/evictions.py:EvictedRecomputes.attribute @c24d956b -->
**What happened.** A result was stored on disk, then removed when the cache
reached its size cap, and now it was needed again and had to be computed
again. Recomputing it took at least two seconds. The message names the
function or statement, the time it took and the cap.

**Why it matters.** The cap removes the entries worth least per byte, but
something was still lost: you waited for work the cache had once held. If it
keeps happening, the cap is too small for what you use.

**What to do.** Raise `max_cache_size` above the cap named, if the disk has
room (the message says how much is free). If it does not, cache smaller
results, or move `cache_dir` to a bigger volume.

<!-- claim: cash/decorator/explain.py:MissHistory.absent_entry_reason @de955740 -->
**When it is safe to ignore.** When the result is rarely needed and a
recompute now and then is cheaper than the disk it would take. It is said once
per function or statement per process. Cheaper recomputes are not warned
about, but `f.explain()`, the per-call log and a notebook badge still say
"evicted to make room" for them.

## CACHE-FRESHNESS-COST {#cache-freshness-cost}

*Decorator.*

<!-- claim: cash/decorator/file_deps.py:FileDeps._warn_if_local_validation_is_expensive @ba7ad550, cash/remote_source.py:validation_is_expensive @18292cc6 -->
**What happened.** Before serving a hit, cash checks every file the call read.
Here that check cost more than half of the compute it saved, or more than two
seconds. The result was correct.

**Why it matters.** A hit should be nearly free. The usual causes are many
files, very large files, or a slow filesystem such as a network share.
File dependencies also pass up to cached callers, so a wide set is checked on
every caller's hit too.

**What to do.** Make the entry depend on less. Split the function so each
input is read by its own cached loader and the aggregate depends on their
results. For a few very large files, lowering `file_hash_full_max_bytes`
makes each check cheaper: files above it are sampled instead of hashed whole
([Configuration](getting-started/configuration.md)).

**When it is safe to ignore.** When the numbers in the message are still a good
trade, such as half a second of checking against a five-minute job. It fires
once per function per process.

## CACHE-IDENTITY-COUPLED {#cache-identity-coupled}

*Decorator.*

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.refuses_identity_coupled @9a87665b -->
**What happened.** The function returned a live matplotlib `Figure` or `Axes`,
or a container holding one, and cash did not store it.

**Why it matters.** A restored copy would not be the figure pyplot tracks, so
`plt.savefig()` could write a blank image. Refusing keeps your plot correct;
the function simply runs every time.

**What to do.** If the function does real work before it plots, split it:
cache the part that computes the numbers, and draw in an uncached function.

**When it is safe to ignore.** Almost always, when the function only draws.

## CACHE-IF-BYPASSED {#cache-if-bypassed}

*Decorator.*

<!-- claim: cash/decorator/store.py:ResultStore._warn_cache_if_bypassed @e205ecf4 -->
**What happened.** The function returned an iterator big enough to be stored
in chunks. Chunks are written as they arrive, so the result was stored without
calling your `cache_if=` predicate.

**Why it matters.** Whatever the predicate was meant to keep out of the cache
is now in it.

**What to do.** Raise `chunk_max_items` / `chunk_max_bytes` above this result's
size, or return a list instead of an iterator. Both hold the whole result in
memory. If that is too big, gate the call before it runs instead.

**When it is safe to ignore.** When the predicate only saved space.

## CACHE-IF-RAISED {#cache-if-raised}

*Decorator.*

<!-- claim: cash/decorator/reporting.py:Notices.cache_if_raised @6298df6f -->
**What happened.** Your `cache_if=` predicate raised when called with the
function's result. The call returned normally; only storing was skipped.

**Why it matters.** Every result that makes the predicate raise is recomputed
on every call. The warning appears once per function, not once per call.

**What to do.** Read the exception and make the predicate handle every result
shape, for example `lambda r: r is not None and len(r) > 0`.

**When it is safe to ignore.** When those results are cheap to recompute.

## CACHE-LOOP-GROWTH {#cache-loop-growth}

*Notebook.*

<!-- claim: cash/notebook/statement/amplification.py:AmplificationGuard._warn @5193540b -->
**What happened.** A statement in a loop stores a value that grows on each
pass (a list being appended to, a frame being concatenated). The copies now
add up to many times the value's size, so cash stopped storing further passes.

**Why it matters.** Caching a growing value on every pass costs the sum of
every intermediate size.

**What to do.** Build the finished value in one statement, such as a
comprehension or a function call, so it is stored once. Calls inside the loop
are still cached per call.

**When it is safe to ignore.** It is not urgent: the extra writing has already
stopped. A new kernel re-runs the loop from where storing stopped.

## CACHE-NET-LOSS {#cache-net-loss}

*Decorator.*

<!-- claim: cash/effectiveness.py:CUMULATIVE_WASTE_SECONDS == 2.0 -->
**What happened.** Cash times what it spends on each call (building the key,
the lookup, storing the result) and compares it with the function's own run
time. For this function, caching has lost more than two seconds in total so
far. The message gives the numbers and names the costliest argument.

<!-- claim: cash/effectiveness.py:EffectivenessLedger.final_verdicts @0e2aaab4 -->
**Why it matters.** The decorator makes your program slower. The usual cause
is a large argument, such as a big DataFrame, being hashed on every call.
The check also runs at exit, so a script that calls each function once is
covered, and several small losers are named together.

<!-- claim: cash/core.py:Cash.register_hasher @f48a324b -->
**What to do.** If a cached function produced the argument and nothing
changes it afterwards, mark the producer `@cash.cache(frozen=True)`: the
argument is then keyed by the call that made it. Otherwise register a cheap
hasher for its type that returns a version or content id. For numpy and
dataframe types this needs `override=True`, and what it returns becomes the
value's whole identity. If neither fits, remove the decorator. See
[`frozen=` and large arguments](decorator.md#frozen-and-large-arguments).

    cash.register_hasher(pd.DataFrame, lambda df: df.attrs["version"], override=True)

**When it is safe to ignore.** When speed is not why you cache, for example to
avoid a paid API call or to hold a result steady.

## CACHE-NOT-WORTH-BYTES {#cache-not-worth-bytes}

*Notebook.*

<!-- claim: cash/backends/value_policy.py:worth_its_bytes, cash/backends/store_notices.py:StoreNotices.not_worth_bytes @bac7fc82 -->
**What happened.** A value over 8 MiB was cheap enough to rebuild that storing
it would cost more than 128 MiB of disk per second of compute saved, so it was
not written to disk. The message names the size and compute time. A cached
call's result faces the same rule unless a statement stores a reference to it
(`b = load(p)`); the message then names the function, as in `load(...)`.

**Why it matters.** A new kernel recomputes it, which is usually faster than
you would notice. The rule keeps big, cheap values from filling the disk.

**What to do.** Usually nothing. To store it anyway, put `# @cash:persist` on
the statement ([Cost model](cost-model.md)). Better still, store something
smaller: the aggregate or the columns you use.

**When it is safe to ignore.** Usually. Act on it when the message names a
recompute time you can feel on every restart.

## CACHE-RESULT-SHARED {#cache-result-shared}

*Decorator.*

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.warn_shared_result @8c38fe43, cash/decorator/purity_checks.py:PurityChecks._shared_with @aabac7c2 -->
**What happened.** The result shares state with an object the caller still
holds: it is an argument, holds one, is a view of an array argument, or is a
module global. On the first run, a write through one shows in the other. A
cache hit returns a separate copy, so from then on it does not.

**Why it matters.** A caller that writes through the result, such as filling
a preallocated array, works on the first run and silently stops working on
later runs.

**What to do.** Return a value of its own: `.copy()`, `list(...)`,
`dict(...)`. If the sharing is the point, do not cache the function.

**When it is safe to ignore.** When callers only read the result. Pass
`assume_safe=True` to record that.

## CACHE-THRASH {#cache-thrash}

*Both paths.*

<!-- claim: cash/backends/file_eviction.py:FileEvictor.warn_thrash @9c41f109 -->
**What happened.** The cache is at its size cap and evicts entries within a
few writes of storing them. The message names the cap and how much room the
volume has.

**Why it matters.** You pay for computing and for writing, and almost nothing
survives to be reused.

**What to do.** Raise `max_cache_size` if the volume has room. If not, cache
smaller values, or move `cache_dir` to a bigger volume.

**When it is safe to ignore.** Never: it always costs time.

## CACHE-VALUE-TOO-BIG {#cache-value-too-big}

*Both paths.*

<!-- claim: cash/backends/store_notices.py:StoreNotices.too_big @0b79929c, cash/backends/file_backend.py:FileBackend.promotion_size_cap @ef38a34e -->
**What happened.** One value, serialized, is bigger than every disk tier's
whole cap, so it was not written to disk. The message names its size and the
cap.

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend._evict_to_byte_cap @2e17ed9e, cash/backends/memory_backend.py:InMemoryBackend.set @980d4716 -->
**Why it matters.** It is offered to the RAM tier instead, but the RAM cap is
usually smaller, so usually nothing is cached at all.

**What to do.** Raise `max_cache_size` well above the size named, or cache
something smaller.

**When it is safe to ignore.** When you did not need that value cached. For a
decorated function, `f.cache_info()` shows whether you get any hits.

## CACHE-WRITE-ABANDONED {#cache-write-abandoned}

*Both paths.*

<!-- claim: cash/config.py:CashConfig.shutdown_write_timeout @5ac9f606, cash/backends/_writes.py:PendingWrites.shutdown @298bbecd, cash/backends/_writes.py:_DaemonWriterPool.shutdown @814ad0be -->
**What happened.** At exit, cash waited for its background writes (60 s by
default) and some were still running, so the process exited without them. The
message says how many.

**Why it matters.** Those results were not stored; the next run recomputes
them. The results you already got are unaffected. The wait is bounded so a
stuck write never keeps a finished process alive.

**What to do.** Check the cache directory first: this usually means storage
that cannot accept writes. If it is only slow, raise the deadline:

```bash
CASH_SHUTDOWN_WRITE_TIMEOUT=300 python nightly_job.py
```

**When it is safe to ignore.** A one-off after writing an unusually large
result. If it fires every run, those entries are never stored.

## CONFIG-FILE-MISSING {#config-file-missing}

*Both paths.*

<!-- claim: cash/config.py:_resolve_config @18f97088 -->
**What happened.** Your code passed `Cash(config_path=...)` naming a file that
does not exist. Cash used the other configuration layers.

**Why it matters.** None of the file's settings apply. Usual causes: a wheel
that did not include the file, or a path relative to the working directory.

**What to do.** Build the path from the module, and ship the file as package
data:

<!-- test:skip reason="illustrative: a packaged tool's layout" -->
```python
from pathlib import Path
app = cash.Cash(config_path=Path(__file__).parent / "cash.toml")
```

`cash info --config PATH` shows what a file resolves to.

**When it is safe to ignore.** When the file is optional. Then check for it
before passing it.

## CONFIG-INVALID {#config-invalid}

*Both paths.*

<!-- claim: cash/config.py:_validated_layer @84048bbe, cash/config.py:_warn_toml_malformed @ca4597b4, cash/config.py:_load_toml_layer @045509b5, cash/config.py:_build_tiers @d9b42b7d, cash/config.py:TierConfig.__post_init__ @afa4a855 -->
**What happened.** Cash could not use part of its configuration:

* a value of the wrong type in a config file or `CASH_*` variable (that
  setting keeps its default);
* a config file that is not valid TOML, including one saved with a UTF-8 BOM
  (every setting in it is ignored);
* a config file with its settings outside a `[cash]` or `[tool.cash]` table;
* a tier with no `type` (left out of the stack);
* a tier key its `type` does not use, such as `default_ttl` on a `memory`
  tier or `wal_mode` on a `file` tier (the tier is built without it).

A bad value passed in code, to `Cash(...)` or `cash.configure()`, raises
`ValueError` instead.

**Why it matters.** The setting you wrote is not in effect.

**What to do.** Fix the value or the file (save it as UTF-8 without a BOM),
then check with `cash info`.

**When it is safe to ignore.** When the default is what you want. Then delete
the line.

## CONFIG-TOML-UNREADABLE {#config-toml-unreadable}

*Both paths.*

<!-- claim: cash/config.py:_load_toml_config @9d135d5c, cash/config.py:_warn_toml_unreadable @b0598e4e -->
**What happened.** Cash found a config file with a `[tool.cash]` or `[cash]`
table but has no TOML parser. Python 3.10 has none built in.

**Why it matters.** Every setting in that file is ignored, `cache_dir`
included, so cash runs on defaults.

**What to do.** Install `cash-lib[toml]` (it adds `tomli`), set the values
through `CASH_*` environment variables, or use Python 3.11 or newer.

**When it is safe to ignore.** When that file is not meant for this
environment.

## CONFIG-UNKNOWN-KEY {#config-unknown-key}

*Both paths.*

<!-- claim: cash/config.py:_validated_layer @84048bbe, cash/config.py:_unknown_key @87165926 -->
**What happened.** A `[tool.cash]` table, a `[cash]` table or a
`CASH_TIER_<N>_*` variable sets a key that is not a cash setting. The message
names the closest real setting:

```text
[CONFIG-UNKNOWN-KEY] …/pyproject.toml sets `max_cache_siz`, which is not a cash
setting, so it does nothing. Did you mean `max_cache_size`?
```

Plain `CASH_*` variables that are not settings are not reported, because
other tools use that prefix.

**Why it matters.** The setting you meant is not in effect.

**What to do.** Rename or remove the key. `cash info` lists every setting in
effect and where it came from.

**When it is safe to ignore.** Never for long: the key does nothing.

## IMPURE-OBSERVED-EFFECTS {#impure-observed-effects}

*Decorator.*

**What happened.** Cash watched the first (missing) call and saw it reach
outside its return value: a `file write`, a `network` connection, a
`subprocess`, or an `argument mutation` (an object you passed in changed).
Each line names the path or address and the line of your code that led to it.

**Why it matters.** A hit runs none of the body, so these effects happened
once and will not happen again. If the next step relies on them, later runs
behave differently from the first.

<!-- claim: cash/decorator/store.py:ResultStore.refusal @50f5a969, cash/decorator/purity_checks.py:PurityChecks.argument_snapshot @334133e6 -->
<!-- claim: cash/decorator/purity_checks.py:PurityChecks.argument_identities @7507ae49, cash/_plain_data.py:identity_changed @a853a1cf -->
A call that changes an argument is **not stored**, so it runs every time. For
arguments that take more than about 50 ms to hash, this check is skipped.

**What to do.** If the effect is part of the job, split the function: cache the
computation and do the writing in an uncached caller. For an argument
mutation, return a modified copy instead. If the effect is incidental (a log
file, a temp file), put `# @cash:assume-safe` on the line the message names.

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.report_observed_effects @5af70afb -->
**When it is safe to ignore.** When everything listed is bookkeeping nobody
reads back. Only the path this call took was watched, so an empty report does
not prove the function is pure.

## IMPURE-SCOPE-MUTATION {#impure-scope-mutation}

*Decorator.*

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.learn_mutating_captures @6045b36f -->
**What happened.** The function reads a module global or captured variable,
and calling the function changed it. The message names the variable and the
line that changes it, which may be in a helper. A callable object that changes
what it holds when called (an instance memoising into `self`, a library
wrapper filling its cache) is not reported: cash just stops folding what it
holds, after one extra miss.

**Why it matters.** A hit skips the change, so a counter stops counting. Cash
also stops folding that variable into the key, so a change you make to it
elsewhere no longer invalidates the entry.

**What to do.** Pass the value in as an argument and return the new value. If
updating shared state is the function's job, cache only the expensive part.

**When it is safe to ignore.** When the variable is the function's own memo,
or a log you are happy for a hit to skip. Put `# @cash:assume-safe` on the
line that changes it. `assume_safe=True` on the decorator does not silence
this code; filter it by code if you must.

## IMPURE-SIDE-EFFECTS {#impure-side-effects}

*Decorator.*

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @21132aa4 -->
**What happened.** Before the first call, cash read the source of the
function and its helpers and found shapes that make a cached result doubtful.
Each finding has a line number and a label:

<!-- claim: cash/decorator/purity_checks.py:PurityChecks._mutable_global_is_keyed @df6f788b -->
<!-- claim: cash/analysis/purity_flow.py:_FreshFlow._check_insertion @70874c70, cash/analysis/purity_flow.py:_FreshFlow._loop_targets @bd0ebd1c, cash/analysis/purity_flow.py:is_log_line @90d04d4b, cash/effects.py:is_read_only_sql @94e903d1 -->
| Label | What it flags | Reported as |
|---|---|---|
| `impure_call` | A call made for its effect: `print` to stdout, `open(..., "w")`, `os.remove`, `subprocess.run`, `requests.post`, a write method such as `df.to_csv` on an object the function did not create, or a `@stateful` function | this code |
| `scope_mutation` | `global` / `nonlocal`, or assigning to another object's attribute or item | this code |
| `discarded_call` | A call whose return value is thrown away | this code |
| `mutable_global` | A module global that other code in the module reassigns, and that the key does not fold | this code |
| `dynamic_pattern` | A callable picked at run time from a table built in the body (`t = {...}; t[kind]()`), from a parameter (`router.table[key]()`), or from `globals()[name]` | this code |
| `ambient_read` | The clock, a fresh UUID, an environment variable named at run time | [KEY-AMBIENT-READ](#key-ambient-read) |
| `network_read` | A GET request or a read-only SQL query | [KEY-NETWORK-READ](#key-network-read) |
| `untrackable_dep` | `eval` / `exec` / `compile`, `getattr(obj, name)()` with a run-time name, `importlib.import_module` | raises `CashImpureFunctionError` |

Log lines (`logging`, `print(..., file=sys.stderr)`) are not reported. A
module-level table such as `HANDLERS[kind]()` is hashed as a global and is not
reported either.

**Why it matters.** `impure_call`, `scope_mutation` and `discarded_call` mean
a hit will not repeat the effect. `mutable_global` and `dynamic_pattern` are
worse: something the result depends on is not in the key, so editing it does
not invalidate the entry.

**What to do.** Fix the findings that are real. For each line you have checked
and accept, add `# @cash:assume-safe`:

    def build_report(rows):
        print(f"building {len(rows)} rows")  # @cash:assume-safe
        return summarise(rows)

On the `def` line, the comment waives findings about the whole body. A line
that changes an argument in place is better fixed than waived: return a
modified copy.

<!-- claim: cash/decorator/reporting.py:Notices._first_showing @8007ca34 -->
**When it is safe to ignore.** When every line is a `print` to stdout or a
progress bar: you only lose the printout on hits. Never ignore
`mutable_global` or `dynamic_pattern`. This warning is shown once per cache,
not once per process.

## KEY-AMBIENT-READ {#key-ambient-read}

*Decorator.*

<!-- claim: cash/effects.py:MODULE_CALLS @c6f9471b, cash/analysis/purity_analyzer.py:_PurityVisitor.visit_Subscript @f19bc8dc -->
<!-- claim: cash/analysis/purity_analyzer.py:_ambient_call @81835f7e, cash/effects.py:_canonical_names @e0692d46 -->
<!-- claim: cash/effects.py:CLOCK_WHEN_ARGS_OMITTED @3c78d511, cash/effects.py:_reads_clock_when_omitted @b543a896 -->
**What happened.** The function reads the clock or a fresh UUID
(`datetime.now()`, `date.today()`, `time.time()`, `uuid.uuid4()`,
`pd.Timestamp.now()`), or an environment variable whose name is only known at
run time (`os.getenv(name)`).

<!-- claim: cash/effects.py:environment_input @bed3e42a, cash/decorator/globals_fold.py:GlobalsFold.fold_environment @0398e851 -->
<!-- claim: cash/analysis/purity_flow.py:is_log_helper @6bf250bd, cash/analysis/purity_analyzer.py:_log_helper_names @c43afd2c -->
<!-- claim: cash/analysis/purity_analyzer.py:_clock_helper_read @c1abcb81 -->
An environment read with the name written out (`os.getenv("TENANT")`) and
`os.getcwd()` are not reported: their values are folded into the key. A
reading that only goes into a log line is not reported either.

**Why it matters.** The value is an input the key cannot see. The first call's
value is stored and returned to every later call, in every later process.

**What to do.** Pass the value in, so it reaches the key:

```python
from datetime import date

import cash

@cash.cache
def report(rows, as_of):        # as_of is an argument, so it is in the key
    return sum(rows), as_of

report([1, 2, 3], as_of=date.today())
```

For an environment variable, write its name out.

**When it is safe to ignore.** When freezing the value is the point, such as
a timestamp of when the result was computed. Put `# @cash:assume-safe` on the
line.

## KEY-BOOL-STATE-TOKEN {#key-bool-state-token}

*Decorator.*

<!-- claim: cash/data_source.py:state_token_of @914de552 -->
**What happened.** Your `DataSource.state_token()` returned `True` or
`False`.

**Why it matters.** A bool cannot say "the data is different now", so entries
never invalidate when the source changes.

**What to do.** Return something that changes with the data: a version, a
digest, an mtime, an ETag. See [Data sources](api/data_sources.md).

**When it is safe to ignore.** Never, if the source can change while your
program runs.

## KEY-BUILD-FAILED {#key-build-failed}

*Decorator.*

<!-- claim: cash/decorator/runtime.py:KeyBuilder.resolve @b144476a -->
**What happened.** Something raised while cash built the cache key. The
message names the exception and, when it can, the argument type. The call ran
and returned its real result, uncached.

**Why it matters.** That call is not cached. Nothing stale can be served.

**What to do.** If the exception points at a type of yours, register a hasher
for it with `cash.register_hasher`. If it does not, please report it as a bug
with the traceback.

**When it is safe to ignore.** When the function is cheap enough to run every
time.

## KEY-CALLABLE-HASHER {#key-callable-hasher}

*Decorator.*

<!-- claim: cash/core.py:Cash.register_hasher @f48a324b -->
**What happened.** You registered a hasher for `types.FunctionType`,
`types.MethodType` or `functools.partial`. The registration took effect.

**Why it matters.** That hasher decides the identity of every function passed
to a cached function. Closures from one factory share a name and a body, so a
hasher keyed on the name gives them one entry, and the second call gets the
first one's result.

**What to do.** Pass a module-level function and give the captured value to
the cached function as an argument. If you keep the hasher, make it return the
captured values too.

**When it is safe to ignore.** When your hasher already returns everything that
tells two such functions apart.

## KEY-DEPENDS-ON-OPAQUE {#key-depends-on-opaque}

*Decorator.*

<!-- claim: cash/decorator/registry.py:FunctionRegistry._register_declared_callable_dep @cdef7cb8, cash/decorator/registry.py:warn_inert_dependency @6242cef6 -->
**What happened.** A callable in `depends_on=` has no source and no Python
bytecode (a builtin, a NumPy ufunc, or an installed compiled extension). Cash
can key it only by its name, so the declaration does next to nothing. An
extension built inside your project (`build_ext --inplace`, an editable
install) does not warn: it is keyed by the content of its built file.

**Why it matters.** Changing that callable will not invalidate the entry.

**What to do.** For an extension you build and install yourself, pass its
version as an argument, or use a `DataSource` whose token is the build id.

**When it is safe to ignore.** Usually: a stdlib or pinned third-party builtin,
such as `depends_on=[math.sqrt]`, will not change between runs.

## KEY-DYNAMIC-DEP-FAILED {#key-dynamic-dep-failed}

*Decorator.*

<!-- claim: cash/decorator/registry.py:resolve_dynamic_dependencies @44d428bd -->
**What happened.** A `dynamic_depends_on=` resolver raised, or returned
something that is not a `DataSource`, a list of them, or `None`. The call ran
uncached.

**Why it matters.** Every such call recomputes. Nothing stale is served.

**What to do.** Fix the resolver. It receives the same arguments as your
function. Wrap a raw value in a `DataSource`, and return `None` for calls with
no dependency.

**When it is safe to ignore.** When the function is cheap. If the dependency
no longer changes, remove `dynamic_depends_on=`.

## KEY-DYNAMIC-DEPENDENCY {#key-dynamic-dependency}

*Decorator.*

<!-- claim: cash/decorator/code_args.py:CodeArgs._warn_untrackable_in_carrier_once @477865a2 -->
**What happened.** An object you passed to a cached function carries code, and
that code picks what it calls at run time: `getattr(module, name)()` with
`name` in a variable, `eval`, a dynamic import. The message names the method,
line and argument.

**Why it matters.** Cash cannot follow the call, so editing the function it
lands on will not invalidate the entry. (The same line in the cached
function's own body raises `CashImpureFunctionError` instead.)

**What to do.** Call the function by name where you can. Otherwise list the
candidates with `depends_on=[...]`.

**When it is safe to ignore.** When the functions it can pick never change.
Put `# @cash:assume-safe` on the line.

## KEY-FROZEN-MUTATED {#key-frozen-mutated}

*Decorator.*

<!-- claim: cash/decorator/frozen.py:FrozenResults.audit @0786c6c6 -->
**What happened.** A function marked `@cash.cache(frozen=True)` promised its
result is not modified, and a later check found one of its results modified.

**Why it matters.** Until the check, cached functions receiving that object
could be served results computed for the unmodified object. From the check on,
it is keyed by content, so later calls are correct. Checks run at the 8th use
and every 64th after that, or every use under `CASH_DEBUG=1`.

**What to do.** Find what modifies it (often `model.fit(...)`), then either
remove `frozen=True` or modify a copy.

**When it is safe to ignore.** Never.

## KEY-FROZEN-NO-EFFECT {#key-frozen-no-effect}

*Decorator.*

**What happened.** A `frozen=True` function returned a value cash cannot mark:
a `set`, or an object that takes no new attributes (`__slots__`, many C
types).

**Why it matters.** `frozen=True` does nothing for this result; cached
functions receiving it still hash it in full.

**What to do.** Return a numpy array, a dataframe, a list, tuple or dict, or an
object that takes attributes. Or remove `frozen=True`.

**When it is safe to ignore.** When the result is small.

## KEY-INSTANCE-STATE {#key-instance-state}

*Decorator.*

<!-- claim: cash/decorator/closure_fold.py:ClosureFold.fold_bound_self @29550361 -->
**What happened.** You cached a bound method (`c.cache(obj.method)`) and the
instance could not be hashed, so cash keyed on the object's in-memory identity.

**Why it matters.** Identity means nothing in a new process, so every new run
recomputes.

**What to do.** Register a hasher for the class that returns what the result
depends on:

    cash.register_hasher(Config, lambda c: c.fingerprint)

**When it is safe to ignore.** In a long-running process with one instance.

## KEY-NETWORK-READ {#key-network-read}

*Decorator.*

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @21132aa4, cash/analysis/purity_analyzer.py:DECORATOR_POLICY @44b8bc03, cash/effects.py:MODULE_CALLS @c6f9471b -->
<!-- claim: cash/analysis/purity_analyzer.py:_opens_tracked_database @35da8b91 -->
**What happened.** The function fetches from a server (`requests.get`,
`httpx.get`, `urlopen(url)`) or queries a database (`cur.execute("SELECT
...")`, `pd.read_sql`). A query over a SQLite file the function opens itself
is not reported: that file is tracked. Writes (`requests.post`, `INSERT`) are
[IMPURE-SIDE-EFFECTS](#impure-side-effects) instead.

**Why it matters.** The answer is an input the key cannot see. The first
answer is stored and served to every later call.

**What to do.** Say how old an answer may be:

```python
import cash
import requests

@cash.cache(ttl=3600)            # an hour at most
def rates():
    return requests.get("https://api.example.com/rates").json()
```

A `ttl=` silences this warning. Under `strict=True` the call raises unless a
`ttl=` is set.

**When it is safe to ignore.** When the answer never changes for the arguments
you pass, such as a fetch by pinned version. Put `# @cash:assume-safe` on the
line.

## KEY-OPAQUE-CALLABLE {#key-opaque-callable}

*Decorator.*

<!-- claim: cash/decorator/code_args.py:is_user_code_carrier @c7843ce8, cash/decorator/code_identity.py:is_user_module @8bcf4264 -->
**What happened.** A function, class or object from your own code reached a
cached call (as an argument or a default), and cash could not hash its code.
Typical cases: `numpy.frompyfunc(my_fn, 1, 1)`, or a bound method of a
compiled object such as `re.compile(p).match`. Library callables do not
trigger it.

**Why it matters.** Editing that code will not invalidate the entry.

**What to do.** If the result depends on it, name the wrapped function with
`depends_on=[my_fn]`. If it does not, mark the type with
`cash.opaque(TheType)` (or `@cash.opaque` on a class you own), which silences
this code for every object of that type.

**When it is safe to ignore.** For a bound method of a compiled library object
like `re.compile(p).match`: it cannot change under you.

## KEY-SOURCE-CHANGED {#key-source-changed}

*Decorator.*

<!-- claim: cash/source_norm.py:loaded_code_matches_disk @f140e8b2, cash/decorator/code_identity.py:warn_source_changed_since_load @4f566032 -->
<!-- claim: cash/source_norm.py:_pyc_proves_unchanged @5d0686e2 -->
**What happened.** A file holding a cached function or a helper was edited
after this process imported it, or the import loaded bytecode compiled from an
earlier save of it. The process runs the old code.

<!-- claim: cash/decorator/code_identity.py:CodeIdentity.pin_own_source @ede4d2d0 -->
**Why it matters.** Cash keys that code by what is actually running, so
results in this process are correct, and they are not reused after a restart
on the new code.

**What to do.** Restart the process to run the new code. A deploy that copies
files before restarting a service triggers this.

If it appears again right after a restart, Python is loading bytecode from an
earlier save: its `.pyc` records the file's modification time in whole
seconds and its size, so an edit that keeps the size within a second of the
last import looks current. Save the file again, or delete its `__pycache__`
entry.

**When it is safe to ignore.** Always, for correctness. Look into it if you
did not expect the file to change.

## KEY-UNHASHABLE-ARG {#key-unhashable-arg}

*Decorator.*

<!-- claim: cash/decorator/runtime.py:KeyBuilder.resolve @b144476a -->
**What happened.** An argument could not be hashed, so no key could be built.
The message names the type, or says the value is nested in a container. The
call ran uncached.

**Why it matters.** Every call with that argument recomputes. Nothing stale
is served.

**What to do.** Register a hasher for the type, or pass something hashable in
its place (a connection string, not a connection):

    cash.register_hasher(DatabaseSession, lambda s: s.database_url)

For a closure, `lambda` or `functools.partial`, pass a module-level function
and give the captured values as arguments
([Code you pass as an argument](decorator.md#code-you-pass-as-an-argument)).

**When it is safe to ignore.** When you do not need that call cached.

## KEY-UNHASHABLE-DEFAULT {#key-unhashable-default}

*Decorator.*

<!-- claim: cash/decorator/closure_fold.py:ClosureFold._defaults_unhashable @26da6217, cash/decorator/closure_fold.py:HelperIdentity.identity @ea302891 -->
**What happened.** A parameter default of the function, or of a helper it
calls, could not be hashed, so the call was not cached. The message names the
type.

**Why it matters.** The function is not cached for any caller, including
those that pass the argument.

**What to do.** Move the value out of the signature (build it in the body, or
require it), or register a hasher for its type. `def load(session=Session())`
is the classic case.

**When it is safe to ignore.** When you do not need the function cached.

## KEY-UNHASHABLE-GLOBAL {#key-unhashable-global}

*Decorator.*

<!-- claim: cash/decorator/globals_fold.py:GlobalsFold.fold_read_globals @34ac7e63 -->
**What happened.** The function (or a helper) reads a module global that
could not be hashed, so it was left out of the key.

**Why it matters.** Changing that global will not invalidate the entry.

**What to do.** Have the function read what the result depends on (a
connection string, not a connection), pass the value as an argument, or
register a hasher for its type.

**When it is safe to ignore.** When the global is set once at import and never
changed: a client, a logger, a compiled pattern.

## NOTEBOOK-BAILOUT {#notebook-bailout}

*Notebook.*

**What happened.** Cash hit an internal error while processing the cell,
stepped aside, and let IPython run it normally. The message names the
exception.

**Why it matters.** The result is correct, but nothing from this cell was
stored, and cells below lose the lineage they would have inherited.

**What to do.** Re-running is safe. Please report the exception as a bug.
Restarting the kernel often clears it.

**When it is safe to ignore.** When it happens once. If it repeats on the same
cell, that cell is never cached.

## NOTEBOOK-CELL-SYNTAX {#notebook-cell-syntax}

*Notebook.*

<!-- claim: cash/notebook/upstream/notebook_vetting.py:NotebookVetter._warn_broken_upstream_cells @6ebee167 -->
**What happened.** An earlier cell in the notebook has a syntax error. The
message gives its number (counting code cells from the top, not the `[7]`
execution count) and quotes its first line.

**Why it matters.** Cash skips that cell, so cells that use its output are no
longer invalidated when it changes.

**What to do.** Fix the cell, then re-run it and the cells below that use it.
If it is not code, delete it or make it a markdown cell.

**When it is safe to ignore.** When nothing below uses that cell.

## NOTEBOOK-NOT-FOUND {#notebook-not-found}

*Notebook.*

<!-- claim: cash/notebook/server_discovery.py:warn_notebook_not_found_once @f1dad161 -->
**What happened.** Cash could not find which notebook file this kernel runs,
so cross-cell tracking is off for the session. Cells are still cached and
restored.

**Why it matters.** Editing a cell no longer invalidates the cells below that
used its output.

**What to do.** Under papermill, nbconvert or CI this is expected. In
JupyterLab or VS Code, restart the kernel; if it persists, close and reopen the
notebook.

**When it is safe to ignore.** In a run that executes the notebook top to
bottom once. It is shown once per session.

## NOTEBOOK-SAVEFIG-SKIP {#notebook-savefig-skip}

*Notebook.*

<!-- claim: cash/notebook/upstream/reexecution_planner.py:ReexecutionPlanner._warn_orphaned_figure_write @9973d398 -->
**What happened.** While re-running earlier statements, cash skipped a
`plt.savefig(path)` whose figure was not being redrawn. The file on disk is
unchanged.

**Why it matters.** Running it without the drawing would save a blank figure
over your chart.

**What to do.** To rewrite the image, re-run the cell that draws the plot. To
avoid this, save through the figure object:

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    fig.savefig("chart.png")

**When it is safe to ignore.** Almost always: the figure did not change, so the
file on disk is the one you want.

## RANDOM-REPLAYED {#random-replayed}

*Notebook.*

<!-- claim: cash/notebook/upstream/rng_rewind.py:RngRewind._opts_out_of_rng_rewind @494a3663 -->
**What happened.** A restored value came from an unseeded random source (a
named call such as `np.random.normal()`, or an estimator fitted with
`random_state=None`). You are seeing an earlier draw.

**Why it matters.** Re-running the cell will not change the value, which is
easy to mistake for stability.

**What to do.** For a fresh draw every run, put `# @cash:no-cache` on a line of
its own above the statement. For a reproducible value, seed the source. To
keep it frozen on purpose, add `# @cash:allow-random`.

**When it is safe to ignore.** When holding the value steady is why you cached
it. Not when you are measuring how much a result varies.

## RANDOM-SEED-NONE {#random-seed-none}

*Notebook.*

<!-- claim: cash/notebook/statement/randomness.py:StatementRandomness.warn_entropy_reseed @23925b73 -->
**What happened.** A statement called `seed(None)` (`np.random.seed(None)`,
`random.seed()`), asking for a new random stream every run. Cached values
below it cannot follow.

**Why it matters.** New draws and restored values from the old stream appear
side by side with nothing to tell them apart.

<!-- claim: cash/notebook/upstream/rng_rewind.py:RngRewind._opts_out_of_rng_rewind @494a3663 -->
**What to do.** If the values below must follow the new stream, put
`# @cash:no-cache` on a line of its own above them. If you want them
reproducible, seed with a fixed number instead.

**When it is safe to ignore.** When nothing cached depends on that stream.

## RANDOM-UNSEEDED {#random-unseeded}

*Both paths.*

**What happened.** A statement or a cached function draws from an unseeded
random source.

**Why it matters.** The value is not reproducible. A cached draw is frozen: the
first result is returned from then on, and clearing the cache or running on
another machine gives a different frozen value.

<!-- claim: cash/tracking/randomness/state.py:capture_rng_state @421bfe05, cash/tracking/randomness/detect.py:RandomnessDetector.analyze_code @2471d11d -->
In a notebook, a draw too cheap to cache is frozen too when it comes from the
`random`, `numpy.random` or `torch` stream, because Cash rewinds those streams
before a re-run. A cheap draw from a generator held in a variable
(`rng = np.random.default_rng()`) is not rewound, so it changes on every run.
The message for a generator draw says both.

**What to do.** Seed the source (`random_state=42`, `np.random.default_rng(0)`)
for a stable, reproducible value. Or accept the frozen value and silence the
warning, as below.

<!-- claim: cash/notebook/upstream/rng_rewind.py:RngRewind._opts_out_of_rng_rewind @494a3663, cash/notebook/statement/randomness.py:StatementRandomness.warn_unseeded @79868eb7 -->
In a notebook: for a fresh draw every run, put `# @cash:no-cache` on the
statement, on the line above it or at the end of its line. It turns off the
rewind as well as caching, and the statement no longer raises this warning.
`# @cash:allow-random` only silences the warning.

<!-- claim: cash/decorator/rng.py:RngWatch.warn_unseeded_randomness @52f9e356 -->
With `@cash.cache`: the check runs when the decorator is applied, once per
function, and reads only that function's source, so a `random.seed(0)`
elsewhere does not silence it. A `seed=None` parameter passed on to the
generator warns for calls that leave it out; pass `seed=i` per replicate. To
keep the frozen value, use `@cash.cache(allow_random=True)`. For a fresh draw,
do not cache the function.

**When it is safe to ignore.** When any fixed value will do: an exploratory
split, a demo, a smoke test. Not when the number goes into a report or a test
assertion.

## REMOTE-FRESHNESS-COST {#remote-freshness-cost}

*Both paths.*

<!-- claim: cash/remote_source.py:VALIDATION_WARN_RATIO == 0.5, cash/remote_source.py:VALIDATION_WARN_FLOOR_SECONDS == 0.25, cash/remote_source.py:VALIDATION_WARN_ABSOLUTE_SECONDS == 2.0 -->
**What happened.** Checking whether remote files (`s3://`, `gs://`,
`https://`) changed took more than two seconds, or more than half of the
compute it protects (ignoring checks under a quarter of a second).

**Why it matters.** Each check is a network round trip on the hit path, so hits
are slower than they look.

**What to do.** If the object cannot change, use
`RemoteFileDataSource(url, immutable=True)` or a version-pinned URL
(`?versionId=`, `#generation=`), which needs no request. If it changes
rarely, raise `remote_revalidate_max_age_seconds`; a change then goes unseen
for that long.

**When it is safe to ignore.** When the object can change and a stale answer
would hurt: then the check is what you pay for.

## REMOTE-SIZE-ONLY {#remote-size-only}

*Both paths.*

<!-- claim: cash/remote_source.py:_warn_weak_token @d860c31b -->
**What happened.** The store gave no ETag, version id or last-modified time
for a remote file, so cash tracks it by size alone.

**Why it matters.** A same-size edit does not change the key, so the old
result is served.

**What to do.** Pin a version in the URL (`?versionId=...`,
`#generation=...`), or write a `DataSource` whose token you control (see
[Data sources](api/data_sources.md)).

**When it is safe to ignore.** When the object is append-only or written once,
such as dated partitions.

## REMOTE-STATE-UNREADABLE {#remote-state-unreadable}

*Both paths.*

<!-- claim: cash/remote_source.py:RemoteFileDataSource._warn_failure @bc945f69 -->
**What happened.** Reading a remote file's state failed. The message names
the URL and the exception. Cash recomputed rather than serve an unchecked
result.

**Why it matters.** While it lasts, every call recomputes and leaves an entry
no later call can reach. It warns once per URL and error type.

**What to do.** Read the exception: usually expired credentials, a permissions
change, or network trouble. A 404 means the URL is wrong or the object was
removed. Caching resumes by itself once access works.

**When it is safe to ignore.** For a short blip on a cheap function.

## STORE-CHUNK-FAILED {#store-chunk-failed}

*Decorator.*

<!-- claim: cash/decorator/runtime.py:CallRunner._chunks_are_intact @898490ec -->
<!-- claim: cash/decorator/runtime.py:CallRunner.compute_with_lock @b4c5c8c2 -->
**What happened.** A chunk of a large iterator result failed to write. The
message names the chunk, the backend and the exception.

**Why it matters.** Cash treats an entry with a missing chunk as absent, so
every call recomputes until a write succeeds.

**What to do.** Call `f.cache_clear()`, then fix the cause: a full disk,
permissions, or an item that cannot be serialized.

**When it is safe to ignore.** Not for long: the function recomputes on every
call.

## STORE-CODE-CHANGED {#store-code-changed}

*Decorator.*

<!-- claim: cash/decorator/file_deps.py:FileDeps.code_moved_since_keyed @34c666d8, cash/decorator/registry.py:FunctionRegistry.code_functions @031ca888 -->
**What happened.** A file holding the function, a helper, or a cached function
it depends on changed on disk during the call, in code this call runs. The
result was returned but not stored.

**Why it matters.** Worker processes started now import the new code, so the
result may mix old and new code. Storing it could serve a wrong result later.

**What to do.** Restart the process. If you are not editing the code,
something is replacing files under a running job, such as a deploy.

**When it is safe to ignore.** Once, while you edit a helper during a run.

## STORE-FAILED {#store-failed}

*Both paths.*

<!-- claim: cash/decorator/store.py:ResultStore.store @cc2d1ab2 -->
**What happened.** The result was computed, but writing it to the cache
failed. The message names the backend and the exception. Whatever the
exception, the call returns its result; a failed write never fails the call.

**Why it matters.** Your result is correct. If it happens every call, the cache
does nothing.

**What to do.** Read the exception. Common causes: a full disk, no write
permission on `cache_dir`, a value that cannot be pickled (a socket, a file
handle), or on Windows a file held open by another process.

**When it is safe to ignore.** A one-off on a cheap function.

## STORE-INPUT-CHANGED {#store-input-changed}

*Decorator.*

<!-- claim: cash/decorator/file_deps.py:FileDeps.inputs_moved_during_call @5b620397, cash/tracking/file_tracker.py:FileAccessTracker.inputs_changed_since_read @a2ece8d1 -->
<!-- claim: cash/tracking/file_tracker.py:FileAccessTracker._digest_now @270aaafd, cash/tracking/file_dep_snapshot.py:snapshot_file_deps @fac08483 -->
**What happened.** A file the function read changed before it returned. The
result was returned but not stored.

**Why it matters.** Nobody can say which version of the file the result came
from, so storing it could serve a stale result later.

**What to do.** Usually nothing: the next call caches normally. If it fires on
every run, the function probably writes a file it also reads. Split the read
from the write:

<!-- test:skip reason="illustrative: the point is the split, not a value" -->
```python
@cash.cache
def summarise(path):
    return build_summary(pd.read_csv(path))      # reads only

summary = summarise("state.csv")
summary.to_csv("state.csv")                      # the write happens outside
```

**When it is safe to ignore.** When another process was writing the file and
has finished.

## STORE-LOCK-FAILED {#store-lock-failed}

*Decorator.*

<!-- claim: cash/backends/_base.py:CacheBackend.lock @2c1d7483 -->
**What happened.** Cash could not take the per-key lock that stops two
callers computing the same thing at once, and went ahead without it.

**Why it matters.** Concurrent calls with the same arguments may each compute
the result. Nothing becomes wrong.

**What to do.** Read the exception: a Redis timeout, a stale lock file, a full
disk, or a filesystem where locking does not work.

**When it is safe to ignore.** When nothing runs concurrently.

## STORE-METADATA-INVALID {#store-metadata-invalid}

*Decorator.*

<!-- claim: cash/decorator/reporting.py:Notices.metadata_invalid @4b14c4a0 -->
**What happened.** Cash found an entry but could not read its metadata, so it
treated the entry as missing and recomputed.

**Why it matters.** Little: one recompute per entry.

**What to do.** Nothing for a one-off. If it keeps appearing, run
`f.cache_clear()`. Usual causes: an entry from another cash version, or an
interrupted write.

**When it is safe to ignore.** Usually. Look into it if it persists after
`cache_clear()`.
