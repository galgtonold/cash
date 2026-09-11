# Warnings

<!-- claim: cash/diagnostics.py:DIAGNOSTIC_CODES @17f7cb28, cash/experimental/__init__.py:_warn_experimental @5dcce1c0 -->
Every warning in the `CashWarning` hierarchy carries a code in square brackets
and a link to its section here. To look one up, search this page for the code.
The one exception is the import-time notice from `cash.experimental`: it is a
plain `FutureWarning` outside that hierarchy — it flags an unstable API rather
than a diagnosable condition — and it carries neither a code nor a link.

Warnings are informational by default. To turn them into errors:

    import warnings, cash
    warnings.filterwarnings("error", category=cash.CashWarning)

To silence one kind:

    warnings.filterwarnings("ignore", category=cash.CashCacheIneffectiveWarning)

Handlers can branch on the code rather than the wording, which is free to change:

    with warnings.catch_warnings(record=True) as caught:
        ...
    if any(getattr(w.message, "code", None) == "CACHE-THRASH" for w in caught):
        ...

<!-- claim: cash/diagnostics.py:warn_diagnostic_explicit @8c8ea21a, cash/diagnostics.py:warn_diagnostic @4b561034 -->
**That recipe does not reach every warning.** `.code` is an attribute set on a
warning *object*, and the three notebook-side diagnostics are raised through
`warnings.warn_explicit`, which takes a message *string* and offers no way to
pass an object — so there is nothing for the attribute to ride on.
[`RANDOM-REPLAYED`](#random-replayed) and
[`NOTEBOOK-CELL-SYNTAX`](#notebook-cell-syntax) never carry it, and
[`RANDOM-UNSEEDED`](#random-unseeded) carries it when the decorator raises it
and not when a notebook statement does. Every other code always carries it.

The rendered text always starts `[CODE] `, whichever way the warning was
raised, so read the attribute when it is there and fall back to the text when
it is not:

```python
import warnings
import cash

def code_of(w):
    """The diagnostic code of a caught warning, however it was raised."""
    attached = getattr(w.message, "code", None)
    if attached is not None:
        return attached
    text = str(w.message)
    return text[1:text.index("]")] if text.startswith("[") and "]" in text else None

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    # the shape almost every Cash warning arrives in: an object, with .code set
    instance = cash.CashCacheIneffectiveWarning("[CACHE-THRASH] the cache is full.")
    instance.code = "CACHE-THRASH"
    warnings.warn(instance)
    # the shape warn_explicit produces: a bare string, no attribute to read
    warnings.warn_explicit(
        "[RANDOM-REPLAYED] the value you are seeing is a replay.",
        cash.CashRandomnessWarning, filename="<cash>", lineno=1, registry=None,
    )

assert [getattr(w.message, "code", None) for w in caught] == ["CACHE-THRASH", None]
assert [code_of(w) for w in caught] == ["CACHE-THRASH", "RANDOM-REPLAYED"]
```

If a warning scrolled past, most of what the decorator raises about a function
is also kept in a rolling log on the function itself — `f.cache_info()`
returns a dict whose `"warnings"` entry holds the last twenty, each with the
code, the message and a timestamp. See
[Debugging and monitoring](tutorials/feature-guides/debugging-and-monitoring.md).

For the class hierarchy, see [Exceptions & warnings](api/exceptions.md).

## ANNOT-TTL-INVALID {#annot-ttl-invalid}

<!-- claim: cash/notebook/annotations.py:parse_annotation_line @f843a7c6 -->
**What happened.** You put `# @cash:ttl=` on a statement in a notebook and the
value after the `=` is not a whole number of seconds, so Cash ignored that
annotation entirely and the statement keeps whatever caching it would have had
without it. There is no unit suffix, no decimal point and no negative: five
minutes is `ttl=300`, not `ttl=5m`, not `ttl=300.0` and not `ttl=-5`. Only ASCII
digits count, so a superscript or a full-width digit pasted in from elsewhere
looks right in the cell and is still rejected. `# @cash:ttl` with no `=` at all
is a different case: it is not a TTL Cash rejected, it is a directive it does
not recognise, and those drop silently with no warning of any kind.

**Why it matters.** The statement is still cached, and unless something else is
setting a TTL it now has no expiry. If you set one because the value goes out of
date on a clock rather than because of anything Cash can watch change — a query
against a table that reloads overnight, a rate-limited API response — then
nothing is going to expire it, and the stored value will be served for as long
as the entry survives. Meanwhile the line sits in your notebook looking like it
works. Two things can still expire it, and neither is what you wrote: a
session-wide TTL from `%cash_on ttl=N`, and a `default_ttl` set on a backend
tier, which stamps every entry that arrives without one of its own.

<!-- claim: cash/notebook/annotations.py:ANNOTATION_PATTERN @95980cce, cash/notebook/annotations.py:parse_annotation_line @f843a7c6 -->
**What to do.** Rewrite the value as a bare count of seconds: `# @cash:ttl=300`
for five minutes, `3600` for an hour, `86400` for a day. Annotations on *other*
lines were parsed normally and still apply — a `# @cash:persist` above the
statement survives. One on the **same line** does not: the pattern is matched
with `search`, which takes the first `@cash:` on a line and stops, so
`x = f()  # @cash:ttl=5m  # @cash:persist` loses the `persist` as well. Put one
directive per line. See [Annotations](annotations.md).

**When it is safe to ignore.** When something else already invalidates that
statement. A TTL added out of habit on top of a value derived from a file or a
`DataSource` Cash tracks is belt-and-braces: the entry invalidates when the data
changes, which is what you wanted, and losing the timer costs nothing. It is not
safe to ignore when the TTL was the *only* thing that would ever have
invalidated the statement — there the clock is the whole mechanism, and it is
switched off.

## CACHE-ASYNC-GENERATOR {#cache-async-generator}

<!-- claim: cash/core.py:Cash.cache @2431dc44 -->
**What happened.** You put `@cash.cache` on an async generator — an `async def`
function that `yield`s. Cash does not cache those in this release, so the
decorator handed your function straight back, unwrapped.

**Why it matters.** The function still behaves exactly as written, so nothing
is broken. But nothing about it is cached either: every call runs the whole
body, and no other signal will tell you that. If you added the decorator
because the body is expensive, you did not get what you asked for.

**What to do.** If the results fit in memory, move the work into a plain
`async def` that returns a list, cache that, and iterate the list at the call
site — ordinary coroutines are cached normally. If they do not fit in memory,
leave the generator undecorated and cache the expensive step *inside* it
instead.

**When it is safe to ignore.** When the generator is cheap and you were
decorating a batch of functions in one sweep — nothing is slower than it would
have been without Cash, you simply have one function that does not cache. Do
not ignore it if this is the function you were trying to speed up.

## CACHE-IDENTITY-COUPLED {#cache-identity-coupled}

<!-- claim: cash/core.py:Cash._refuses_identity_coupled @27e72e7a -->
**What happened.** Your cached function returned a live matplotlib `Figure` or
`Axes` — or a list, tuple, dict or array holding one — and Cash refused to
store it. "Identity-coupled" is Cash's term for an object that a library keeps
its own reference to: pyplot tracks the current figure, and the object you got
back has to *be* that same figure, not a copy of it.

**Why it matters.** Storing such an object means copying it, and a copied
`Figure` re-registers itself with pyplot as it is restored. You would then draw
on your figure while `plt.savefig()` wrote the cache's private copy — a blank
image, with no error anywhere. The refusal is the protection, not the failure.

**What to do.** Nothing, if the plot was the point. If the function is
expensive, split it in two: cache the part that computes the numbers, and draw
the figure from those numbers in an uncached function. Drawing is almost always
the cheap half.

**When it is safe to ignore.** Almost always. It means one plotting function
runs every time instead of being cached, which is exactly what you want.
It is worth acting on only when the function does real work before it plots —
that work is being repeated on every call, and splitting the function recovers
the caching.

## CACHE-DIR-MOVED {#cache-dir-moved}

<!-- claim: cash/config.py:project_anchor @c7c2a516, cash/config.py:_anchor_cache_dir @7f3408d4 -->
**What happened.** Cash resolves its default cache directory next to the code
being run -- the project the running script belongs to -- rather than next to
wherever the process was launched from. This run found a cache in the current
directory that the new location does not have, so it is telling you that cache
will not be used. Two things leave one there: a cache from before this
anchoring, when it followed the directory a process ran from, and a project
marker (`pyproject.toml`, `setup.py`, `setup.cfg`, `.git`) added or moved since
the cache was written.

**Why it matters.** For this run only: it is a cold one, and everything
recomputes. From then on the project's cache is in one place no matter where the
job runs from, which is the point. Before this anchoring, running the same
script from a different directory -- a cron job, a CI step, a colleague's
terminal -- silently built a second cache and looked exactly like a cold start
forever, with the disk filling up with duplicates.

**What to do.** Usually nothing: let the new location fill up and delete the old
one when you are happy. To keep using the existing cache instead, point cash at
it explicitly, which always wins:

```bash
CASH_CACHE_DIR=/path/to/the/old/.cash python job.py
```

Note that `CASH_CACHE_DIR` and `Cash(cache_dir=...)` are taken exactly as
written -- relative to your current directory, like any other path you type. A
`cache_dir` in `pyproject.toml` is relative to that file. Only the default is
anchored to the project.

**When it is safe to ignore.** Always, once. Seeing it every run means each run
starts in a different directory that has its own old cache -- the situation the
anchoring exists to end; set `CASH_CACHE_DIR` or delete the strays.

## CACHE-DIR-UNWRITABLE {#cache-dir-unwritable}

<!-- claim: cash/backends/file_backend.py:FileBackend._warn_if_unwritable @8edc0db6 -->
**What happened.** On its first cache operation, cash tried to create a file in
its cache directory and could not: a read-only mount, a directory this account
has no write permission on, a container volume, a path owned by another user
after a deployment. The warning names the exact directory and the OS error.

**Why it matters.** Nothing will be cached to disk for the rest of the run, so
every call recomputes. Your results are correct -- they are computed the normal
way -- but the cache is doing nothing at all, and without this warning that has
no symptom you could see. The job is simply always slow, and nothing names the
directory responsible.

The same code fires for the harder case: a directory cash cannot even open --
a stale mount, a path that no longer exists, permissions changed under a
running deployment. There the file tier turns itself **off** for the rest of
the process rather than raising into your code, which is what it used to do:
"cash cannot cache" became "your job does not run", exit 1, before the caller's
own work had started. In a tiered stack the RAM tier is untouched, so repeats
within the process still hit.

**What to do.** Point cash somewhere writable, or grant this account write
permission on the path it named:

```python
cash.configure(cache_dir="/var/tmp/cash")   # or CASH_CACHE_DIR=... in the env
```

In a container, check that the cache path is on a writable volume rather than
the image's read-only layer. Under a service account, check that the account --
not your user -- owns or can write the directory.

**When it is safe to ignore.** When the directory is read-only deliberately (a
shared, pre-populated cache you only ever read from). Cash still reads it; only
new entries are lost. If that is the intent, the warning is telling you the
truth and there is nothing to fix.

## CACHE-FRESHNESS-COST {#cache-freshness-cost}

<!-- claim: cash/core.py:Cash._warn_if_local_validation_is_expensive @b7e7f0ec, cash/remote_source.py:validation_is_expensive @18292cc6 -->
**What happened.** Before serving a cached result, cash re-checks every file the
call read, to be sure none of them changed. On this call that check cost a
serious share of the compute it saved — more than half of it, or more than two
seconds outright. Your result was correct and came from the cache; the warning
is about what proving it cost.

**Why it matters.** A cache hit is supposed to be nearly free. Checking is
normally microseconds per file after the first look, so reaching this threshold
means one of three things: the call depends on **many** files, on very **large**
ones, or the filesystem is slow (a network share, a fuse mount, a spinning disk
under load). File dependencies also *propagate* — an aggregate that calls ten
cached functions inherits their inputs — so a wide dependency set is paid for on
every one of those hits, not once.

**What to do.** Reduce what the entry depends on, or make the dependency
cheaper:

* **Depend on a summary, not on every input.** A function that reads fifty
  shards and returns an aggregate can be split: one cached loader per shard, and
  an aggregate that depends on their *results* rather than inheriting all fifty
  files.
* **Split so the inputs are read once.** If the expensive reads live in a callee
  that the aggregates do not inherit from, only that callee pays the check.
* **Check the storage.** The same fifty files on a local disk and on an SMB
  share are not the same cost; the message gives you the seconds to compare.
* `file_hash_full_max_bytes` decides where cash stops hashing a file in full and
  starts sampling three regions of it. Sampling is cheaper **per file** and no
  cheaper per file *count*, so it helps with a few huge inputs and not with
  hundreds of small ones.

**When it is safe to ignore.** When the numbers say the trade is still worth it
— half a second of checking against a five-minute pipeline is a good deal, and
the message prints both. Ignore it once, though, not every run: it fires once per
function per session precisely so it stays worth reading.

## CACHE-IF-BYPASSED {#cache-if-bypassed}

**What happened.** You passed `cache_if=` to decide whether a result is worth
storing, and the function returned an iterator big enough that Cash split it
into chunks. Chunks are written as they are produced, so by the time the last
item arrives the earlier ones are already stored and there is nothing left to
gate. The result was cached without your predicate ever running. The message
names the two thresholds that decide the split, `chunk_max_items` and
`chunk_max_bytes`.

**Why it matters.** `cache_if` did not run. Whatever you were using it to keep
out of the cache is in the cache — which matters a great deal if the predicate
was there to stop an incomplete or unwanted result being stored, and not at all
if it was there to save space.

<!-- claim: cash/core.py:Cash._warn_cache_if_bypassed @f39e544a -->
**What to do.** To get the predicate back, the result has to arrive in one
piece. Either **raise** `chunk_max_items` / `chunk_max_bytes` above the size
this result actually reaches, or return a list instead of an iterator — a
non-iterator result always consults `cache_if`. Both mean holding the whole
result in memory at once, which is what chunking exists to avoid; if you cannot
afford that, `cache_if` cannot gate this function, and the gate belongs
somewhere else (decide before the call, or delete the entry afterwards).

**When it is safe to ignore.** When the predicate was an optimisation — "don't
bother caching empty results", "skip the cheap cases". You lose a little disk
and nothing else. Do not ignore it when the predicate was a correctness or
policy gate: the thing you were keeping out is now stored, and it will be
served back on the next call.

## CACHE-IF-RAISED {#cache-if-raised}

**What happened.** The `cache_if=` predicate you gave `@cash.cache` raised an
exception when Cash called it with the function's return value. The message
names the exception and its text. Your call itself returned normally — only the
storing was abandoned.

<!-- claim: cash/core.py:Cash._warn_cache_if_raised @ed2eb8f1 -->
**Why it matters.** Cash treats a predicate that raises as "do not cache", so
every call whose *result* makes it raise goes uncached and recomputes. This is
scoped to the result, not to the function: calls returning a shape the predicate
handles keep caching normally, which is why the symptom is usually a function
that caches for most inputs and never for a few. It is a bug in your predicate
rather than a limitation of Cash.

**What to do.** Read the exception. The usual cause is a predicate that assumes
a shape the result does not always have — `lambda r: len(r) > 0` meeting a
`None`, or `lambda r: r["rows"]` meeting an error payload. Make the predicate
total, or drop `cache_if=` if you no longer need the gate.

**When it is safe to ignore.** Only if you were about to remove `cache_if=`
anyway, or if the results that trip it are cheap to recompute. Note that this
fires once per function, not once per call: seeing it a single time does not
mean it happened a single time. Until the predicate is fixed, every call landing
on one of those results pays full compute, every time.

## CACHE-LOOP-GROWTH {#cache-loop-growth}

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._warn_persist_amplification @bf8e0721 -->
**What happened.** A statement marked `# @cash:persist` sits inside a loop, and
the value it stores grows on each pass — a list being appended to, a frame
being concatenated. Cash wrote a fresh copy of the whole thing every iteration,
noticed the copies now add up to many times the value's real size, and stopped
storing further iterations. The message names both numbers: what has been
written so far, and how big the value currently is.

**Why it matters.** Caching a growing object every iteration costs the *sum* of
every intermediate size, not the final one. A 100 MB result assembled over a
hundred passes writes gigabytes to get one useful entry.

**What to do.** Move `# @cash:persist` off the loop body and onto a statement
that produces the finished object, so it is stored once. The loop itself does
not need the annotation.

**When it is safe to ignore.** It is not urgent — Cash has already stopped
writing, so the disk churn is over for this run — but leaving it is not free
either. The annotation is now inert, so nothing in that loop is being cached at
all, and running from a clean cache will pay the same write storm again from
the start. Treat it as something to fix before the next run rather than
something to fix right now.

## CACHE-NET-LOSS {#cache-net-loss}

**What happened.** This is a measurement, not a guess and not an error. Cash
times what it spends on each call building the cache key and looking the entry
up, and compares that against how long your function's own body takes. For this
function the account has been running for several calls and caching has come
out behind. The message gives the numbers it used: how many calls, how much
total time went on keys and lookups, the largest body time it has ever
observed, and the net loss so far.

<!-- claim: cash/effectiveness.py:CUMULATIVE_WASTE_SECONDS == 2.0 -->
**Why it matters.** `@cash.cache` on this function is making your program
slower, and not on some pathological call — on every call. The usual cause is a
large argument whose fingerprint is computed from its actual bytes. A 153 MiB
DataFrame passed to a function that sums one column measured 390 ms to
fingerprint against 11 ms of work: 34 times slower, every time. Cash only
speaks up once the loss has accumulated past a couple of real seconds *and* its
per-call overhead exceeds even the largest body time it has seen, so a function
that is usually fast but occasionally very slow will not be flagged.

**What to do.** The message names the costliest argument. If a cached function
produced it and nothing modifies it afterwards — a trained model, a lookup
table — declare that on the producer, and the argument is keyed by the call
that made it instead of being hashed
([passing large objects](decorator.md#passing-large-objects-between-cached-functions)):

    @cash.cache(frozen=True)
    def train(data): ...

Otherwise keep the caching and make the key cheap, by registering a hasher for
the expensive argument's type:

    cash.register_hasher(pd.DataFrame, lambda df: df.attrs["version"], override=True)

<!-- claim: cash/core.py:Cash.register_hasher @eed1ca57 -->
`override=True` is not decoration. For the types Cash fingerprints itself —
numpy arrays, pandas / polars / PyArrow / modin frames, dask collections — its
own content hasher runs first, and a plain registration for one of those types
is rejected outright rather than silently ignored. Whatever your hasher returns
becomes the entire identity of that value, so return something that genuinely
changes when the data changes: a version, a content id, an immutable
fingerprint. For a pandas frame, pandas 3 (copy-on-write) lets cash check the
frame for changes instead of re-hashing it, so this warning usually goes away
with the upgrade. If there is no such handle to be had, remove the decorator
from this function. It is not the right tool for this shape of work.

**When it is safe to ignore.** When speed is not why the decorator is there.
Caching to avoid a metered API call, to hold a result steady across a session,
or to stop a nondeterministic step from re-running are all real reasons, and
Cash measures seconds and nothing else — so its verdict is accurate and beside
the point. If you added the decorator to make something faster, this is the
warning on this page most worth reading, and the last one to filter.

## CACHE-THRASH {#cache-thrash}

<!-- claim: cash/backends/file_backend.py:FileBackend._warn_evict_after_write @d3225bf8 -->
**What happened.** The cache reached its size cap and is evicting entries
within a couple of writes of storing them, so it re-writes and re-evicts
instead of caching durably. The message names the cap, how much room is left on
that volume, and — when only a handful of entries fit at once — roughly how big
the typical entry is.

**Why it matters.** This is slower than no cache at all: you pay the compute
*and* the storage churn, and almost nothing survives to be reused.

**What to do.** Raise `max_cache_size` if the volume has the room — the message
says whether it does, which is why it measures. If it does not, the useful
lever is a smaller value rather than a bigger cache: cache the summary you
actually use downstream instead of the full result, or point `cache_dir` at a
roomier volume.

**When it is safe to ignore.** Never — this one always costs you time.

## CACHE-VALUE-TOO-BIG {#cache-value-too-big}

<!-- claim: cash/backends/tiered_backend.py:TieredBackend._warn_oversize_not_persisted @80a5a388, cash/backends/file_backend.py:FileBackend._promotion_size_cap @cc93a731 -->
**What happened.** A single value is larger than every persistent tier's whole
cap, so there is nowhere durable to put it and Cash offered it to the RAM tier
instead. The message names the value's size and the cap it was measured
against.

The size named is the **serialized** one — the same number `cash inspect`
reports in its SIZE column, so the two agree. It did not always: the gate
compared the value's in-memory footprint, which for a frame of strings is two
or more times larger, so a cap that looked generous next to `inspect`'s numbers
refused entries anyway. And the threshold used to be *half* the cap, which
meant a 500 MB cap cached nothing at all for a 263 MB working set. It is the
whole cap now, measured on the serialized bytes, so a value that fits is stored
and LRU eviction does the rest.

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend._evict_to_byte_cap @9dbcf959 -->
**Why it matters.** Usually you get no caching at all, not RAM-only caching.
The RAM tier has its own byte cap, scaled to the machine and independent of
`max_cache_size`, which caps only disk — and it is the *smaller* of the two.
Anything big enough to be refused by every persistent tier is normally also over
the RAM cap, so `InMemoryBackend.set` evicts down to 90% of that cap in the same
call that wrote the entry, taking the entry with it. Measured, on a 4 MiB value
that trips the warning in both arms: with a RAM cap of 100 MiB, two calls run
the body **once**; with a RAM cap of 1 MiB, two calls run it **twice** and the
RAM store is empty immediately. The second arm is the shape of the defaults —
measured on one machine, a 4.0 GiB RAM cap against an 18.7 GiB per-object
refusal threshold on disk, so every value that can reach this warning is over
four times the RAM cap.

**What to do.** Raise `max_cache_size` above the size the message names — a
comfortable multiple of it, so the cache can hold more than that one entry —
and the value is written durably rather than juggled in RAM. If that
room is not available, cache something smaller — the aggregate, the sample, or
the columns you actually use rather than the whole object. Smaller is the fix
that works on both tiers at once.

**When it is safe to ignore.** When you did not need that value cached. Do not
read this one as "restarts are the only thing I lose": check first whether you
are getting in-process hits at all, because on default caps you usually are not.
`f.cache_info()` on the decorated function is the quickest answer — if the hits
are not climbing, the entry is being evicted as fast as it is written and the
decorator is buying you nothing.

## CONFIG-TOML-UNREADABLE {#config-toml-unreadable}

<!-- claim: cash/config.py:_load_toml_config @d2daeca0, cash/config.py:_warn_toml_unreadable @d0af8e40 -->
**What happened.** Cash found a config file — `pyproject.toml` with a
`[tool.cash]` section, or the XDG user config — and has nothing that can parse
it. A TOML parser entered the standard library in **Python 3.11** (`tomllib`);
on 3.10 the job falls to `tomli`, and cash cannot depend on it, because cash
has no required dependencies at all.

Without a parser, cash decides whether the file holds settings by looking for
the section header itself: a `pyproject.toml` with no `[tool.cash]` table says
nothing, because there is nothing of cash's in it to ignore.

**Why it matters.** Every setting in that file is ignored — `cache_dir` among
them — so cash runs on the defaults the file was written to change. Nothing
fails and nothing looks wrong: you get a working cache in the wrong place, with
the wrong caps, and a `[tool.cash] cache_dir` that appears to do nothing. This
was a debug-level log line until 0.10.1, which is to say it had no symptom at
all.

**What to do.** Any one of:

```bash
pip install tomli          # the parser 3.10 is missing
```

or set the same values through the environment, which needs no parser:

```bash
export CASH_CACHE_DIR=/srv/cache
```

or run on Python 3.11 or newer, where the parser ships with the interpreter.

**When it is safe to ignore.** When the file is not meant for this environment
— a `pyproject.toml` that carries `[tool.cash]` for a different deployment, say.
The warning fires once per process, and only when a config file is actually
there.

## CACHE-WRITE-ABANDONED {#cache-write-abandoned}

<!-- claim: cash/config.py:CashConfig.shutdown_write_timeout @5ac9f606, cash/backends/_base.py:PendingWrites.shutdown @a2c42b01, cash/backends/_base.py:_DaemonWriterPool.shutdown @a6534a5d -->
**What happened.** Your program finished, and cash was still writing cache
entries in the background. It waits for them at exit -- for 60 seconds by
default -- and this time the deadline expired with writes still running. The
process exited anyway; the warning says how many entries were dropped.

**Why it matters.** Only for speed: those keys were not stored, so the next run
recomputes that work. Nothing about the results you already got is affected --
they were computed, returned and used normally.

What the deadline buys is the opposite failure. Cash's writer threads are
daemons and this wait is bounded on purpose, because a cache write that cannot
finish must never keep a finished process alive. A job that prints its answer
and then hangs forever is an outage, and a hard one to diagnose: the logs show
a healthy run, and the scheduler's next tick piles up behind a process that
looks busy.

**What to do.** Look at the cache directory first -- this almost always means
storage that cannot accept the write rather than storage that is merely slow: a
read-only mount, a service account without write permission, a container volume,
a directory owned by another user after a deployment, a network mount that has
stalled. `cash.configure(cache_dir=...)` and `CASH_CACHE_DIR` name the path
cash is using.

If the storage really is that slow and the entries are worth waiting for, raise
the deadline:

```bash
CASH_SHUTDOWN_WRITE_TIMEOUT=300 python nightly_job.py
```

**When it is safe to ignore.** A one-off at the end of a run that wrote an
unusually large result to a slow disk. If it fires every run, the entries it
names are never being stored, and the cache is doing less than it appears to.

## IMPURE-OBSERVED-EFFECTS {#impure-observed-effects}

**What happened.** Cash watched the first call to this function — the one that
missed and had to run — and saw it reach outside its own return value. The
message lists what it saw, one line per effect: a `file write` and the path, a
`network` connection and the address, a `subprocess` and the command, or an
`argument mutation`, meaning an object you passed in was different after the
call than before it. These happen inside library code, which the source scan
does not walk into, so watching the call was the only way to find them.

**Why it matters.** A cache hit returns the stored value and runs none of the
body, so every effect on that list happened exactly once and will not happen
again. If the effect was part of the point — the file the next step reads, the
row posted to a service, the dict the caller inspects afterwards — the program
is correct on the run that filled the cache and quietly different on every run
after it. `argument mutation` is the one that catches people out: an object the
caller still holds stopped being changed, and nothing at the call site says so.

**What to do.** Decide whether the effect is part of the result. If it is, split
the function: cache the computation that produces the data, and do the writing,
posting or mutating in an uncached caller. If it is incidental — a log file, a
progress marker, a temp file the function cleans up itself — record that
decision with `@cash.cache(assume_safe=True)`, which is what the message
suggests.

<!-- claim: cash/core.py:Cash._report_observed_effects @960249ed -->
One caveat worth knowing: only the path this particular call took was watched.
An effect behind a branch that did not run was not seen, so silence here is not
a proof of purity — this supplements the source scan behind
[IMPURE-SIDE-EFFECTS](#impure-side-effects) rather than replacing it. The two
never appear together: if the source scan already flagged this function, this
warning stays quiet.

**When it is safe to ignore.** When everything on the list is bookkeeping nobody
reads back — a log line, a metrics counter, a `.tmp` file, a progress bar
writing to disk. You lose it on cache hits and nothing downstream notices. Do
not ignore an `argument mutation` line without first checking what the caller
does with that object next: that one is a change in your program's behaviour
rather than in Cash's, and it only shows up once the cache is warm, which is
usually not the run you were watching.

## IMPURE-SCOPE-MUTATION {#impure-scope-mutation}

**What happened.** Cash folds the module globals and captured variables a
function *reads* into its cache key, so that changing one invalidates the entry.
It hashed those values again when the first call returned and found one had
moved — which means calling the function is what moves it. The message names the
variable and says whether it is a module global or a variable captured from an
enclosing scope.

<!-- claim: cash/core.py:Cash._learn_mutating_captures @367b8947 -->
**Why it matters.** Two things follow, and neither is visible at the call site.
A cache hit runs no body, so the write stops happening: a counter stops
counting, an accumulator stops accumulating, and code that reads the variable
later sees whatever it held at the last miss. And Cash stops folding that one
name into the key from here on — it has to, because keying an entry on a value
the function's own body produces would mean never getting a hit. The function
keeps caching on everything else, but a change *you* make to that variable from
somewhere else no longer invalidates it.

**What to do.** Pass the value in as an argument and return the new one, instead
of reaching out and rewriting it. That single change fixes both halves: the
value becomes an input the key can see, and the update becomes something the
caller receives rather than something a cache hit skips. If updating shared
state really is the function's job, it is not a caching candidate — move the
expensive part into its own function, cache that, and do the update at the call
site.

**When it is safe to ignore.** When the variable is the function's own private
memo: a `_SEEN = {}` it fills in to avoid repeating work, a handle it builds
lazily on first use, a compiled pattern. Nothing outside reads it for its own
sake, and dropping it from the key costs nothing because it is derived from
inputs the key already carries. Do not ignore it when the variable is program
state something else reads — configuration, a counter, a registry, a list of
results. Note too that `@cash.cache(assume_safe=True)` does not silence this
one, because it is decided by watching the call rather than by reading the
source; a filter on `CashImpurityWarning` is the only way to mute it, and muting
it is rarely what you want.

## IMPURE-SIDE-EFFECTS {#impure-side-effects}

<!-- claim: cash/core.py:Cash._surface_purity @86786923 -->
**What happened.** Before the first call, Cash reads the source of your function
and of the helpers it calls, looking for shapes that make a cached result
questionable. It found some. The message lists each one with its line number and
a short label in square brackets, and the label is the part that tells you how
much to care:

- `impure_call` — a call whose job is a side effect: `print`, `input`,
  `open(..., "w")`, `os.remove`, `subprocess.run`, `requests.post`,
  `logging.info`, `json.dump`, or a write-shaped method on a receiver the
  function did not create itself — `df.to_csv(...)`, `fig.savefig(...)`,
  `session.post(...)`, `cursor.execute(...)`, `RESULTS.append(...)`. The
  "did not create itself" part matters: `rows.append(x)` on a list the function
  built a line earlier is not flagged, and neither is `a, b = [], []` followed
  by `a.append(...)`, a view of a local array (`inner = u[1:-1]; inner[m] = 0`),
  or a frame rebound to a copy before it is changed
  (`df = load(p); df = df[mask]; df["x"] = ...`). The same frame changed
  *before* the copy is still flagged: it may be the helper's own object.
- `scope_mutation` — a `global` or `nonlocal` statement, or an assignment to
  someone else's attribute or subscript: `obj.attr = ...`, `d[k] = ...`.
- `discarded_call` — a method call whose return value is thrown away, which
  usually means it was made for its effect.
- `mutable_global` — the function reads a module global that other code in the
  same module reassigns.
- `dynamic_pattern` — a callable chosen at run time, `HANDLERS[kind]()` or a
  name bound from a lookup, so editing whichever callable it lands on will not
  invalidate the entry.

**Why it matters.** It depends on the label, which is why the message prints
them. `impure_call`, `scope_mutation` and `discarded_call` are one story: a
cache hit will not repeat that effect. `mutable_global` and `dynamic_pattern`
are the other story, and the more serious one — something the result depends on
is missing from the cache key, so an edit to it will not invalidate and you get
the old answer back. None of it is proof of anything: this is a reading of the
source, and it recognises shapes rather than observing behaviour.

Each line number is the line in the file that defines the function, and each
group names that file — so a finding in a helper points at the helper's
module, not at the call that surfaced it.

**What to do.** Go down the list one line at a time and fix the ones that are
real. For each one you have read and decided is fine, put `# @cash:assume-safe`
on that line:

    def build_report(rows):
        print(f"building {len(rows)} rows")  # @cash:assume-safe
        return summarise(rows)

On a line of its own the comment waives the statement below it as well as
itself; on the `def` line it waives the findings that belong to the whole body
rather than to any single line, which is where a `mutable_global` lands.

Adding the first annotation recomputes the function once. `# @cash:` comments
are directives — several of them change how a result is cached — so unlike an
ordinary comment they are part of the function's source identity.

Reach for `@cash.cache(assume_safe=True)` only when you mean the whole function
for good. It silences the check for everything in the body *including code added
to it later* — audit the function today, add a `session.post(...)` next month,
and nothing says a word. A comment written next to the statement cannot do that:
new code arrives unannotated and is reported, and the scope of the exemption is
visible in the diff that granted it.

A line that **changes an argument in place** is the one to fix rather than
annotate: the message says "changes the argument '…' in place". On a miss the
caller's object is changed; on a hit the stored result comes back and the
object is not, so whatever the caller does next sees two different objects
depending on whether the call hit. Return a modified copy instead
(`feats = feats.copy(); feats["x"] = ...; return feats`).

**When it is safe to ignore.** When every line it names is a `print`, a
`logging` call or a progress bar. That is far and away the commonest reason this
fires, and it is as harmless as it looks: you lose the printout on cache hits
and nothing else. Even then, the per-line comment is a better response than a
warning filter, because it leaves the rest of the function watched. Do not
ignore a `mutable_global` or a `dynamic_pattern` line — those two are the
stale-result kinds, and nothing else will tell you when they bite.

## KEY-AMBIENT-READ {#key-ambient-read}

<!-- claim: cash/notebook/purity.py:_AMBIENT_READ_CALLS @23eb97e5, cash/purity_analyzer.py:_PurityVisitor.visit_Subscript @c9ab46b9 -->
**What happened.** Reading the source of the function you decorated found a
call that asks the world what time it is, what the environment says, where the
process is running, or for a fresh UUID: `datetime.now()`, `date.today()`,
`time.time()`, `os.getenv(...)`, `os.environ["..."]`, `os.getcwd()`,
`uuid.uuid4()`. The named line ran, and the result was cached as normal.

A read whose value goes only into a `print`, a `logging` call or
`warnings.warn` — `t = time.perf_counter()` feeding an elapsed-time line — is
not reported: it cannot reach the result. Once the value is returned, stored,
tested in a condition or passed to any other call, it is.

**Why it matters.** That value is an *input* to your result, and it is not one
Cash can see: it does not arrive as an argument, so it is not in the cache key.
The first call's value is therefore baked into the stored result, and every
later call gets it back — in this process and in every process afterwards,
because the cache is on disk. A nightly report stamped with `date.today()`
keeps the date of the night it first ran. A job that reads
`os.environ["TENANT"]` serves the first tenant's answer to every other tenant.
Nothing raises; the run exits 0 with the wrong number in it.

This is not the same as [IMPURE-SIDE-EFFECTS](#impure-side-effects), which is
about work a cache hit *skips*. Here nothing is skipped — a hidden input is
frozen — so the fix is different too.

**What to do.** Pass the value in, so it reaches the key and a new value means
a new entry:

```python
from datetime import date

@cash.cache
def report(rows, as_of):        # as_of is an argument, so it is in the key
    return sum(rows), as_of

report([1, 2, 3], as_of=date.today())
```

Read the clock at the call site, where it is obvious, rather than inside the
body, where it is invisible. If the value genuinely never changes for the
program's lifetime (`os.getcwd()` in a job that never chdirs, a build ID read
once from the environment), hoist it to a module-level constant computed at
import.

**When it is safe to ignore.** When the frozen value is the point — a timestamp
recording when the *computation* happened rather than when you asked for it, or
a UUID used only as a stable identifier for the cached result. Say so on the
line with `# @cash:assume-safe`, which keeps the rest of the function watched;
`@cash.cache(assume_safe=True)` waives the whole function, including code added
to it later.

## KEY-BOOL-STATE-TOKEN {#key-bool-state-token}

<!-- claim: cash/data_source.py:DataSource.state_token @fb386b76 -->
**What happened.** You wrote a `DataSource` subclass, and Cash asked it for the
value to fold into the cache key. That value comes from `has_changed()` unless
you override `state_token()` — and yours returned `True` or `False`. Despite
the method's name, what Cash needs there is a *token*, not a yes/no.

**Why it matters.** A boolean has two values, so it cannot represent "the data
is different now". Entries keyed on one do not invalidate when the source
changes: you get stale results, silently, which is the one failure a cache must
not have.

**What to do.** Return something that moves with the data — a version string, a
content digest, an mtime, an ETag. Either return it from `has_changed()`
directly, or leave `has_changed()` as a real boolean and override
`state_token()` to return the token. See [Data sources](api/data_sources.md).

**When it is safe to ignore.** Effectively never, if the source can change
while your program runs. The only exception is a source that is genuinely fixed
for the process's lifetime — and in that case the `DataSource` is not earning
its place and can be removed instead.

## KEY-BUILD-FAILED {#key-build-failed}

**What happened.** Something raised while Cash was assembling the cache key,
somewhere it did not anticipate. The message names the exception and, where it
can identify one, the argument type most likely responsible. Your call ran and
returned its real result; only the caching was skipped.

<!-- claim: cash/core.py:Cash._resolve_cache_key @4adf0746 -->
**Why it matters.** That call did not cache. Correctness is not at risk — with
no key, nothing is written and nothing is read, so this cannot produce a stale
answer — but you are paying full compute every time it happens.

**What to do.** Read the exception. If it points at a type of your own,
register a hasher for it: `cash.register_hasher(YourType, ...)`. Unlike
[KEY-UNHASHABLE-ARG](#key-unhashable-arg), which is the ordinary,
user-fixable case, this site is Cash's catch-all for key failures it did not
expect — so if the exception does not obviously belong to your code, it is
worth reporting as a bug with the traceback attached.

**When it is safe to ignore.** When the function is cheap enough that running
it every time is fine. Nothing on this path can hand you a wrong result.

## KEY-CALLABLE-HASHER {#key-callable-hasher}

**What happened.** You registered a hasher for `types.FunctionType`,
`types.MethodType` or `functools.partial` — a type every function, bound method
or partial passed to any cached function in the process belongs to. Cash warns
when you register it; the registration still takes effect.

<!-- claim: cash/core.py:Cash.register_hasher @eed1ca57 -->
**Why it matters.** What the hasher returns becomes that argument's identity in
the key. Cash still folds in each function's code, so two functions with
different bodies stay apart — but closures one factory makes have the same
body and the same name, and differ only in what they capture:

<!-- test:skip reason="illustrative: shows the collision the warning is about" -->
```python
def make_model(omega):
    def model(t):
        return np.cos(omega * t)
    return model

cash.register_hasher(types.FunctionType, lambda f: f.__qualname__)
fit(make_model(3.0), data)
fit(make_model(5.0), data)   # make_model(3.0)'s result
```

A hasher keyed on the name gives them one cache entry, and the second call gets
the first one's result — a wrong answer, not a slow one.

**What to do.** Don't pass the closure. Pass a module-level function and give
the captured value to the cached function as a plain argument, where the key
sees it: `fit(model, data, omega=5.0)`. If you keep the hasher, make it return
what the function captures as well as its name.

**When it is safe to ignore.** When your hasher already returns everything that
tells two of these values apart — the captured values, or a version you
control.

## KEY-DEPENDS-ON-OPAQUE {#key-depends-on-opaque}

<!-- claim: cash/core.py:Cash._register_declared_callable_dep @ab2edea1 -->
**What happened.** You named a callable in `depends_on=`, and Cash could not
read its source to fingerprint it — it is a builtin, or it lives in a compiled
extension. The declaration was accepted and does nothing.

**Why it matters.** `depends_on=` is a promise that changing the named thing
invalidates the entry. For this entry the promise is not being kept, and
nothing at the call site shows it. If the target changes, your cached results
will not.

**What to do.** Depend on something Cash can actually see. If the opaque target
is code you build yourself — a Cython or Rust extension — pass its version into
the cached function as an argument, or declare a `DataSource` whose token is
that version or build id. Wrapping the extension call in a small Python
function and depending on the wrapper does not work: the wrapper's source is
what gets hashed, and it does not change when the extension does.

**When it is safe to ignore.** Usually, and this is the common case. If you
wrote `depends_on=[json.loads]`, or named any other stdlib or pinned
third-party builtin, that target is not going to change under you between runs,
so an invalidation you were never going to need costs nothing. Take it
seriously only when the opaque target is code you compile yourself.

## KEY-DYNAMIC-DEP-FAILED {#key-dynamic-dep-failed}

<!-- claim: cash/core.py:Cash._resolve_dynamic_dependencies @8ba1651b -->
**What happened.** A resolver you passed to `dynamic_depends_on=` raised when
Cash called it to find out which data sources this particular call depends on.
Cash carried on and built the key without that dependency. The message names
the exception.

**Why it matters.** Whatever the resolver was tracking is missing from the
cache key. Entries written now keep being served after the underlying data
changes, because nothing in the key moves when it does. This is a stale-result
risk, not just a lost speedup.

**What to do.** Fix the resolver. It is called with exactly the same arguments
as your function, which is the usual source of the failure — a resolver written
against one signature and attached to another, or one that assumes an argument
is always present. Until it is fixed, treat results from this function as
possibly stale, and clear its entries after changing the source data.

**When it is safe to ignore.** Never, while the data behind that resolver can
change. If the thing it tracked has since become static, the right response is
to delete the `dynamic_depends_on=` argument, not to filter the warning — a
silent declaration that does nothing is worse than no declaration.

## KEY-FROZEN-MUTATED {#key-frozen-mutated}

<!-- claim: cash/core.py:Cash._audit_frozen @0f673f82 -->
**What happened.** A function is decorated `@cash.cache(frozen=True)` — a
promise that its result is not modified after it is returned — and one of its
results was modified anyway: a later audit found it had changed since cash
first saw it. The message names the function that produced it.

**Why it matters.** A frozen result is keyed downstream by the call that
produced it, not by its contents; that is what makes passing it on cheap. While
it was modified and not yet audited, a cached function receiving it could be
served the result it computed for the unmodified object. From the audit on,
that object is keyed by its contents, so later calls are correct. The audit
runs at the object's 8th use as an argument and every 64th after that, and at
every use under `CASH_DEBUG=1` — so the window can be wide.

**What to do.** Find what modifies it — `model.fit(...)` on a model a frozen
step trained is the usual one — and either take `frozen=True` off the producer
or modify a copy (`copy.deepcopy(obj)`). While looking, `CASH_DEBUG=1` audits
every use, so the warning points at the first call after the change.

**When it is safe to ignore.** Never: the promise the key relies on was broken.

## KEY-INSTANCE-STATE {#key-instance-state}

<!-- claim: cash/core.py:Cash._fold_bound_self @22b72518 -->
**What happened.** You cached an already-bound method — `c.cache(obj.method)`.
Cash folds the instance into the key so that two objects in different states do
not share results, but this instance could not be hashed, so it fell back to
the object's in-memory identity. The message names the class.

**Why it matters.** Keying on identity is correct but narrow. Two equal
instances get separate entries instead of sharing one, and an in-memory
identity means nothing in a new process — so a fresh run finds no entry and
recomputes everything, however full the cache on disk is. If you decorated this
method to get reuse across runs, you are not getting any.

**What to do.** Register a hasher for the owning class, returning something
derived from the state that actually affects the result:

    cash.register_hasher(Config, lambda c: c.fingerprint)

If the instance holds something unhashable but irrelevant — an open connection,
a logger, a thread pool — a hasher that simply skips it is exactly the right
answer.

**When it is safe to ignore.** When the process is long-lived and there is one
instance. A service that builds its object at startup and calls the method for
hours gets the full benefit of the cache, and has nothing to share it with. Do
not ignore it in a script that runs repeatedly: that is the case where the
cache looks healthy and is silently doing nothing.

## KEY-OPAQUE-CALLABLE {#key-opaque-callable}

<!-- claim: cash/core.py:Cash._is_user_code_carrier @a334d114, cash/core.py:Cash._is_user_module @1b6836eb -->
**What happened.** A function, a class, or an object whose class carries code
reached a cached call — as an argument you passed, or
as a parameter default you never typed — and Cash could not fingerprint its
body. Cash normally folds the code of such things into the key, so that editing
them invalidates. This one it could not.

Cash only says so when it judges the carrier to be **your** code: the check
excludes anything defined under `site-packages`, `dist-packages` or the stdlib
paths. Passing a library callable straight in does not reach this warning at
all — measured, none of `json.dumps`, `operator.itemgetter(0)`, `len`,
`numpy.sum`, `functools.reduce` or `functools.partial(json.dumps)` produce it.
The shapes that do are narrow, and both are things built out of your own code:
a compiled wrapper around one of your functions (`numpy.frompyfunc(my_fn, 1, 1)`),
and a bound method of a C-implemented object whose class Cash cannot re-resolve
from the module it claims — `re.compile(p).match` is the one you are most likely
to meet. A `functools.partial` is not one of them: it is keyed by the function it
wraps and by its arguments, so editing either invalidates.

**Why it matters.** The result depends on what that callable does, and the key
does not. Edit the callable, run again, and you get the old answer back with
nothing to indicate anything is wrong. This is the residue where Cash genuinely
cannot work the answer out, which is why it says so rather than staying quiet.

**What to do.** If the result really does depend on that implementation, name
the *wrapped* function explicitly with `@cash.cache(depends_on=[the_function])`
— for a `frompyfunc`, that is the function it wraps. That closes
the invalidation gap; it does not silence the warning, because the carrier
itself is still unhashable. If the result does not depend on it — the identity
is already covered by another argument, or the carrier is library code Cash
misjudged — say so deliberately: `cash.mark_opaque(TheType)`, which does silence
it, or `@cash.opaque` on a class you own. (`cash.mark_opaque(functools.partial)`,
once the advice for partials, no longer applies to them: it silenced every partial
in the process, including ones over code you were still editing.) Both record the decision in the code,
which is what makes them better than a warning filter.

**When it is safe to ignore.** In the one case where the carrier only *looks*
like yours: a bound method of a compiled library object, such as
`re.compile(p).match`, which reaches this warning because Cash cannot walk back
to it from `re` — not because you can edit it. Nothing will change under you
there, so the missing fingerprint costs nothing.

It is *not* safe in the cases the gate is actually designed to catch, which are
the common one: a `frompyfunc` over a function of your own that you are still
editing. Editing the wrapped body will not invalidate anything,
and you will be served the old answer with nothing to indicate it — the "why is
my cache serving me the old version" problem, with this warning as the only
notice you get. The instinct to read "it's a library callable, it can't change"
is exactly backwards here: a library callable would not have warned.

## KEY-SOURCE-CHANGED {#key-source-changed}

<!-- claim: cash/source_norm.py:loaded_code_matches_disk @cb3b7996, cash/core.py:_warn_source_changed_since_load @02d3e452 -->
**What happened.** A file holding your cached function, or a helper it calls,
was edited after this process imported it. The process is still running the
*old* code; the file now holds the *new* code. cash noticed the difference the
first time the function was called.

**Why it matters.** cash identifies code by its source. Read after the edit,
that source describes code the process is not running — and a result computed
by the old code would have been stored under the new code's identity, then
served to the restarted process as a hit. Measured in round 17 on a helper: a
service answered 0.500504 where the new code computes 0.530876, and kept
answering it after the restart. Round 18 found the same for the cached function
itself, in a worker that imported the old code and made its first call after
the deploy landed.

<!-- claim: cash/core.py:Cash._pin_own_source @54026f75 -->
So cash keys that code by what is **actually running** instead: a cached
function by the source it was imported with (its identity is taken when the
decorator runs, not at its first call), and a helper by its loaded bytecode.
Results in this process are correct for the code it is running, and they are
not reused once the process restarts on the new code.

**What to do.** Restart the process to run the new code. This fires in the
window a deploy opens when it puts new files on disk before restarting the
service — which is how most deploys work, and why this is a notice rather than
something to fix.

**When it is safe to ignore.** Always, as far as correctness goes. It is worth
reading when you did *not* expect the file to change: then something rewrote
your code under a running process.

## KEY-UNHASHABLE-ARG {#key-unhashable-arg}

**What happened.** One of the arguments could not be turned into a stable
fingerprint, so Cash could not build a cache key at all. The message names the
type when it can identify one; when the offending value is nested inside a
container it says so instead, because it cannot see which element is to blame.
The call ran and returned normally.

<!-- claim: cash/core.py:Cash._resolve_cache_key @4adf0746 -->
**Why it matters.** That call did not cache, and calls like it will not cache
either — this is not first-call warm-up. Every call passing that argument pays
full compute. Nothing can go stale, because nothing is being stored.

**What to do.** Register a hasher for the type:

    cash.register_hasher(DatabaseSession, lambda s: s.database_url)

Or pass something Cash can fingerprint in its place: the connection string
rather than the connection, the path rather than the open file handle. When
Cash could not name the type, the culprit is nested — a list of custom objects,
a dict holding a live handle — and the same two fixes apply once you find it.

The exception is a closure, a `lambda` or a `functools.partial`, and the
message says so instead: a hasher for `function` would cover every function in
the process, and the obvious one — by name — gives every closure a factory
makes the same identity ([KEY-CALLABLE-HASHER](#key-callable-hasher)). Pass a
module-level function and hand what it captures to the cached function as a
plain argument; see [a closure or `lambda` passed as an
argument](known-limitations.md#a-closure-or-lambda-passed-as-an-argument-stops-the-call-caching-entirely).

**When it is safe to ignore.** When you do not need that call path to be fast.
There is no correctness risk here whatsoever: an unbuildable key means no entry
is written and none is read. What you lose is the caching, completely, for
every call that passes that argument — so ignore it only if you have decided
that is fine.

## KEY-UNHASHABLE-DEFAULT {#key-unhashable-default}

**What happened.** One of the function's *parameter defaults* — a value in the
`def` line, not something a caller passed — could not be fingerprinted, so Cash
declined to cache the call. The message names the type.

<!-- claim: cash/core.py:Cash._defaults_unhashable @ce13fcc1 -->
**Why it matters.** Cash folds defaults into the key so that `build()` and
`build(Schema)` are recognised as the same call, and so that changing a default
invalidates. It cannot tell whether an unhashable default has changed, and it
chooses to skip caching rather than serve a result that might be stale. That is
the safe choice, but the consequence is broad: the function does not cache for
*any* caller, including ones that pass the argument explicitly.

**What to do.** Get the value out of the signature — build it inside the
function body, or require it at the call site — or register a hasher for its
type with `cash.register_hasher`. The classic case is a live default such as
`def load(session=Session())`, which is worth moving for reasons that have
nothing to do with caching.

**When it is safe to ignore.** When you do not need that function cached. As
with an unhashable argument, there is no staleness risk: Cash refused to store
anything precisely so that there could not be one. What it costs you is the
whole function's caching, not just the calls that rely on the default.

## KEY-UNHASHABLE-GLOBAL {#key-unhashable-global}

<!-- claim: cash/core.py:Cash._fold_read_globals @2f7c4ca9 -->
**What happened.** The function reads a module-level variable — its own
module's, or a helper's, in which case the message shows a dotted name — and
Cash could not fingerprint that variable's value. Cash normally folds the
globals a function reads into its key, so that changing one invalidates. This
one it had to leave out.

**Why it matters.** Change that global and the cached results will not change
with it. This is the staleness-shaped member of the `KEY-` family: the function
keeps returning what it computed under the old value, with nothing to say so.

**What to do.** Look at what the global actually holds. If it is a live handle
— a database connection, an HTTP session, a thread pool, an open file — replace
the global the function reads with the part the result really depends on: the
connection string rather than the connection, the base URL rather than the
session. If the object itself is what matters, register a hasher for its type
with `cash.register_hasher` and return something that changes with its state.
Passing the value in as an argument works too, and makes the dependency
visible at every call site.

**When it is safe to ignore.** When the global is set once at import and never
touched again — a client handle, a logger, a compiled pattern, a thread pool.
Those are the common case here, and an invalidation you will never need is not
worth anything. Do not ignore it when the global is configuration or data that
your program rewrites while it runs: that is a stale-result bug waiting for the
first person who changes the value and does not see the output change.

## NOTEBOOK-BAILOUT {#notebook-bailout}

**What happened.** Cash hit an internal error while processing the cell, stepped
aside, and handed the cell straight to IPython. Your code then ran normally,
uncached. The warning names the exception that caused it.

**Why it matters.** The result you got is correct — the cell really ran, and it
ran without Cash in the way. What you did not get is caching: nothing from this
cell was stored, so it will run in full again next time, and cells downstream of
it lose the lineage they would have inherited. Nothing is silently wrong, but a
cell you expect to be instant on the next run will not be.

Before this warning existed the failure was logged to the kernel log and nowhere
else, so in a notebook the only trace was a cell that appeared to do nothing.
The badge said `BYPASSED` at best, and before that it wrongly said `EXECUTED`.

**What to do.** Nothing in your code caused this, and re-running is safe. It is
a bug in Cash, and the exception name in the message is the part worth reporting
— please open an issue with it. If you need the cell cached in the meantime,
restarting the kernel clears any accumulated state that may have triggered it.

**When it is safe to ignore.** If it fires once and the cell is cached normally
on a later run, you lost one cache write and nothing else. It is not safe to
ignore if it repeats on the same cell: that cell is permanently uncached, so
every run pays its full cost, and any timing you read from the badge for cells
below it is measuring a pipeline Cash is no longer helping with.

## NOTEBOOK-CELL-SYNTAX {#notebook-cell-syntax}

**What happened.** To work out what the cell you just ran depends on, Cash
re-reads the notebook's earlier cells. One of them does not parse — it has a
syntax error. The message gives that cell's number and quotes its first
non-blank line so you can find it. The number counts code cells from the top of
the file and ignores markdown ones, so it is not the `[7]` execution count in
the margin; the quoted line is the reliable way to identify it.

**Why it matters.** A cell that will not parse cannot be analysed, so Cash
cannot see what it defines or what it reads, and it is skipped. Caching carries
on as normal for every cell that does not touch it. What stops is dependency
tracking for anything downstream: a cell taking its input from the broken one no
longer gets invalidated when things change, so it can be served a value computed
from an older version of the notebook.

**What to do.** Fix the syntax error, then re-run that cell and the cells below
it that use its output, so their dependencies are rebuilt from a cell Cash can
actually read. If the cell is not really code — pasted output, a traceback,
notes you were half-way through typing — delete it or turn it into a markdown
cell. Markdown cells are not parsed and never trip this.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker._warn_broken_upstream_cells @f225e598 -->
The warning repeats when the break changes and stays quiet while it does not, so
re-running cells *below* the broken one will not spam you; fixing it and later
breaking it again will warn again. One gap in that promise: the scan only looks
at cells above the one you are running, and the ledger of what it has already
warned about is replaced by whatever that scan found. Run a cell above the break
and the break is not in view, so the ledger empties — and the next run of a cell
below it warns again with nothing having changed. Measured: warn, quiet, run a
cell above, warn again.

**When it is safe to ignore.** When nothing you are running depends on that
cell — a scratch cell of half-typed notes near the top that defines no name
anything below it uses. Nothing you have cached is at risk in that case. Do not
ignore it while you are relying on edit-a-cell-and-the-rest-catches-up: that is
exactly the feature switched off for everything downstream of the break, so if
you are not going to fix the cell now, re-run the affected cells top to bottom
rather than trusting the invalidation.

## NOTEBOOK-NOT-FOUND {#notebook-not-found}

**What happened.** Cash could not work out which notebook file this kernel is
running, so upstream dependency tracking is off for the session. It asks the
running Jupyter Server which document the kernel belongs to, and got no answer.

**Why it matters.** Statement-level caching still works — cells are cached and
restored exactly as usual. What you lose is the cross-cell half: editing one
cell no longer invalidates the cells below that used its output. On a run that
goes top to bottom once, that costs nothing, because everything runs in order
anyway. It matters in the edit-and-re-run-one-cell loop, where a downstream cell
can be handed a value computed from the previous version of the cell above.

<!-- claim: cash/notebook/server_discovery.py:warn_notebook_not_found_once @7cabb93c -->
**What to do.** Under papermill, nbconvert or a CI job there is no live Jupyter
Server to ask, so this is expected and there is nothing to fix. In JupyterLab or
VS Code it usually means a stale runtime: restart the kernel, and if it persists
close and reopen the notebook so the frontend reconnects. Colab has never had a
discoverable path, and neither has a remote or containerised kernel whose
JupyterLab extension is pushing cell contents — Cash stays quiet in both,
because there the check is about to run on the live cells and the message would
be false.

**When it is safe to ignore.** In any run that executes the notebook start to
finish exactly once: papermill, nbconvert, `jupyter execute`, CI. There is no
partial re-run for the tracking to protect, so all you lose is the message. In
interactive work it is worth fixing, because the feature that is off is
precisely the one that stops you reading a stale number. It is emitted at most
once per session, so do not read the silence afterwards as tracking having come
back — once you have changed something, restart the kernel and watch whether a
fresh session says it again.

## NOTEBOOK-SAVEFIG-SKIP {#notebook-savefig-skip}

**What happened.** Cash was re-running earlier statements to rebuild what an
edited cell depends on, and the plan contained a bare `plt.savefig(path)` whose
figure is *not* being redrawn in the same pass. It dropped that write from the
plan instead of performing it. The file on disk is untouched.

<!-- claim: cash/notebook/upstream/reexecution_planner.py:ReexecutionPlanner._warn_orphaned_figure_write @f5ce1b7c -->
**Why it matters.** The refusal is the protection. `plt.savefig(path)` saves
pyplot's *current* figure, which it looks up in a process-global registry — it
has no link to any variable, so there is nothing for Cash to follow back to the
figure you meant. Running it with the drawing statement absent would make
`plt.gcf()` invent a fresh empty figure and flush it over your chart: measured,
a 960x540 chart became a 640x480 blank, which is matplotlib's default figure
geometry and nothing else. That is a wrong answer written to disk with no signal
anywhere in the notebook, so Cash declines to write rather than risk it.

**What to do.** If the image should be rewritten, re-run the cell that draws the
plot — that schedules the drawing and the save together, the figure is rebuilt
coherently, and the refusal does not arise. To stop it arising at all, save
through the figure object rather than through pyplot:

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    fig.savefig("chart.png")

`fig.savefig` names the figure it is saving, so Cash can tell that the object
being written is the one just drawn, and reconstruction handles it normally.

**When it is safe to ignore.** Almost always, because the refusal only happens
when the figure is not being rebuilt — nothing about the picture has changed, so
the file already on disk is the one you want. All you are being told is that a
file was not rewritten. It matters only when you changed something that *should*
have changed the image and the statement that draws it was restored from cache
rather than re-run; re-running the plotting cell settles that either way.

## RANDOM-REPLAYED {#random-replayed}

**What happened.** A cached value that came from an unseeded random source has
just been restored, and what you are looking at is the draw from an earlier run
rather than a fresh one. Two situations reach this warning and the message tells
you which you have: it either names a random call and the line it is on —
`np.random.normal()`, `rng.choice()`, `random.random()` — or it names an
estimator variable and says `random_state=None`, meaning a `.fit()` whose
randomness lives inside the library where no source scan can see it.

This is the restore-time twin of [RANDOM-UNSEEDED](#random-unseeded). That one
fires when the value is computed and is advice about the code; this one is
raised only after a restore has actually succeeded, and it is a statement about
the number in front of you. It *is* a replay, not a "may be".

**Why it matters.** Re-running the cell will not change the value. If you are
re-running precisely to see how much the answer moves — a different train/test
split, another bootstrap sample, a second fit from a different initialisation —
you will get the same number every time, and it is easy to read that as
stability. A correct cache producing a wrong conclusion.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker._opts_out_of_rng_rewind @00e6c078 -->
**What to do.** Decide what you wanted from that statement.

- Genuinely fresh every run: `# @cash:no-cache` on a comment line of its own
  above the statement. The spelling matters — see
  [the note under `RANDOM-UNSEEDED`](#random-unseeded).
- Reproducible rather than fresh: seed the source — `random_state=42` on the
  estimator, a seed on the generator in the cell that creates it — and the
  cached value now has something behind it that anyone can reproduce.
- Frozen on purpose: `# @cash:allow-random` silences this and changes nothing
  else.

**When it is safe to ignore.** When holding the value steady is exactly why you
cached it: a split you want fixed while you work on the cells below it, a sample
the whole session should agree on. There the warning is confirming what you
asked for. It is not safe to ignore when you are estimating variance, repeating
an experiment, or checking that a finding is not an artefact of one particular
draw — those questions need `# @cash:no-cache` on its own line, and a replay
answers a different question without saying so.

## RANDOM-SEED-NONE {#random-seed-none}

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._warn_entropy_reseed @a794f6b6 -->
**What happened.** A statement called `seed(None)` — `np.random.seed(None)`,
`random.seed()` with no argument, or the same on another supported module. That
asks for a different, entropy-derived random stream on every run, and Cash
cannot make cached values below it fresh to match.

Do not confuse this with [RANDOM-UNSEEDED](#random-unseeded), which is about
code that never seeded anything. This one is about code that seeded
*deliberately with no seed* — a request Cash cannot honour below the cache.

**Why it matters.** You get exactly what you asked the RNG for, a new stream,
while values computed from the old one are served from cache unchanged. The two
disagree on screen with nothing marking which is which. The sharpest version is
a model fitted in place: the fit is restored from the run that first computed
it, and every statement reading that model then describes a stream that no
longer exists.

No cache key resolves this, which is why it is a warning rather than a bug
waiting to be fixed. Keying the downstream values on the fresh entropy would
make them recompute on every run, and then they never converge: each re-run
mints another answer instead of agreeing with the last.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker._opts_out_of_rng_rewind @00e6c078 -->
**What to do.** Pick one of the two things `seed(None)` sits between.

- If those values must reflect the new stream, mark them `# @cash:no-cache` on
  a comment line of its own above the statement. That switches off caching *and*
  the RNG rewind, and it is the only thing that makes a draw genuinely fresh
  each run. Written trailing on the code line it turns caching off and leaves
  the rewind on, which redraws the identical number — see
  [`RANDOM-UNSEEDED`](#random-unseeded).
- If you want them reproducible, seed with a fixed integer instead —
  `np.random.seed(42)`. The cached value and the stream then agree, and they
  agree the same way on anyone else's machine.

**When it is safe to ignore.** When nothing cached derives from the stream it
resets. A `seed(None)` written to undo a fixed seed set earlier in an
exploratory notebook, with the draws below it already carrying an own-line
`# @cash:no-cache`, is doing what you meant and the warning has nothing to bite
on. So is a single cold top-to-bottom run: there is nothing stored yet to
disagree with the new stream. It matters in the loop Cash exists for — edit a
cell, re-run it, read the numbers below — because that is where the frozen value
and the fresh stream end up on screen together.

## RANDOM-UNSEEDED {#random-unseeded}

**What happened.** Cash found a draw from an unseeded random source in something
it is about to cache, and is telling you what that means: the first result is
stored and replayed from then on. The RNG is not consulted again, so the value
is frozen. The message either names the call and its line — `np.random.normal()`,
`rng.choice()` — or names an estimator you fitted with `random_state=None`,
where the randomness is inside the library and only the live object reveals it.

A third form names a **seed parameter**: `def simulate(params, seed=None)` with
`np.random.default_rng(seed)` inside. The source looks seeded, and is — when the
caller passes a seed. Leave it out, or pass `None`, and the generator draws from
OS entropy, so this warning fires for *that call*. It is the shape Monte Carlo
replicates usually take, and caching turns `[simulate(p) for _ in range(R)]`
into R copies of one draw. Pass `seed=i` per replicate: each one is then
reproducible *and* cacheable. The same check covers a seed read from a settings
object, a dict or a module global — `default_rng(settings.seed)`,
`default_rng(opts["seed"])` — when that field is `None` in the call.

**This is Cash working as designed, not a defect.** Worth being blunt about,
because the instinctive reaction — decide the cache is broken and turn caching
off — is the worst outcome available here. Unseeded randomness is everywhere in
the notebooks Cash is built for: `train_test_split` with no `random_state`,
sklearn defaults, dropout, bootstrap resampling. If Cash redrew those on every
run, everything computed from them would invalidate, the whole chain below would
recompute, and the cache would deliver essentially nothing. Freezing is what
makes the rest of it worth having.

**Why it matters.** Frozen is not the same as reproducible. The value is fixed
at one arbitrary draw from an unseeded stream; clear the cache, or run on
another machine, and you get a *different* arbitrary value, which is then fixed
in its turn. Nobody else can obtain your number, and neither can you once the
entry is gone.

Worth knowing, because it surprises people: caching is not the only thing that
freezes a draw. To keep a re-executed statement in the right place in the random
stream, Cash rewinds the RNG to where the cell started — so a cheap draw such as
`r = random.random()` can genuinely re-execute, never having been stored at all,
and still hand back the same number. See "A value can be frozen without being
cached" in [Annotations](annotations.md).

**What to do.** Choose the outcome you actually want.

- **Reproducible:** seed the source — `random_state=42` on the estimator,
  `np.random.default_rng(0)`, `random.seed(0)`. The value stays stable, and
  stays stable for everybody.
- **Genuinely fresh every run:** in a notebook, `# @cash:no-cache` on a comment
  line of its own above the statement. It switches off the RNG rewind as well as
  the caching, which is why merely not caching is not enough. Under the
  decorator, leave the function undecorated.
- **Frozen, deliberately:** `# @cash:allow-random` on the statement, or
  `@cash.cache(allow_random=True)`. Both silence the warning and change nothing
  else — the value was frozen before and stays frozen.

<!-- claim: cash/notebook/upstream/checker.py:UpstreamChecker._opts_out_of_rng_rewind @00e6c078 -->
!!! warning "Write `# @cash:no-cache` on a line of its own"

    For the rewind — and only for the rewind — the placement is load-bearing,
    and getting it wrong fails silently. Cash decides whether to skip the rewind
    by reading the cell's **comment lines**: a line that does not start with `#`
    once stripped is not examined at all. So
    `x = random.random()  # @cash:no-cache` switches caching off, exactly as the
    directive is documented to do everywhere else, and leaves the rewind
    switched **on** — the statement re-executes, lands on the same position, and
    hands back the identical number you were trying to get away from. Measured:
    own line → the rewind is skipped; trailing → it is not, while the annotation
    itself still parses as `no_cache`.

    The check also runs over the **whole cell**, not one statement. One own-line
    `# @cash:no-cache` anywhere in a cell — above the draw or below it — turns
    the rewind off for every draw in that cell. Caching is still decided per
    statement; only the rewind is cell-wide.

Whichever you pick, pick it per statement or per function. Switching caching off
across the board to "fix" this trades a known frozen value for a slow notebook
and gains nothing.

<!-- claim: cash/core.py:Cash._warn_unseeded_randomness @b0a31134 -->
The decorator form is checked when the decorator is applied rather than when the
function runs, so it appears at import time, before the function has been called
once, and once per decorated function. It reads that function's source alone: a
`random.seed(0)` elsewhere in your program does not silence it, and should not —
a process-wide seed only makes the first draw reproducible for as long as
nothing before it consumes the stream differently, and one added call is enough
to break that.

**When it is safe to ignore.** When a fixed arbitrary value is fine for what you
are doing, which is the common case and the reason this is the default: an
exploratory split you want held still while you work below it, a shuffle whose
exact permutation does not matter, a demo, a smoke test. Not safe to ignore when
the number is going into a paper, a report or a test assertion — frozen will not
survive a cleared cache, and seeding costs one argument. Not safe either when
you are measuring how much a result varies between runs; see
[RANDOM-REPLAYED](#random-replayed), which is the same situation seen from the
other end.

## REMOTE-FRESHNESS-COST {#remote-freshness-cost}

**What happened.** This is a measurement. Before serving a cached result Cash
checks whether the files it depends on have changed, and for a remote object —
`s3://`, `gs://`, `https://` — each of those checks is a network round trip. It
has added up what the trips cost for this function and compared that against how
long the function's own body takes, and the checking is the expensive half. The
message gives both figures and how many sources were checked.

<!-- claim: cash/remote_source.py:VALIDATION_WARN_RATIO == 0.5, cash/remote_source.py:VALIDATION_WARN_FLOOR_SECONDS == 0.25, cash/remote_source.py:VALIDATION_WARN_ABSOLUTE_SECONDS == 2.0 -->
**Why it matters.** This is the one overhead you cannot see. It lands on the
*hit* path, where the badge reports a saving and nothing reports what
establishing that saving cost. Cash speaks up on either of two grounds: the
checks took more than two seconds outright, or they cost more than half of the
compute they protect — with a floor of a quarter of a second, so a 100 ms
function is not flagged over 60 ms of checking.

**What to do.** It turns on whether the object can actually change.

- If it cannot, say so: `RemoteFileDataSource(url, immutable=True)` resolves the
  token once and every later check is free. A version-pinned URL
  (`?versionId=`, `#generation=`) is treated as immutable automatically and
  needs no request at all, because the pin *is* the token.
- If it can change but rarely, widen the window with
  `cash.configure(remote_revalidate_max_age_seconds=...)`. Be deliberate about
  the number: for the length of that window a change goes unnoticed. This is
  also the only lever for reads Cash tracked automatically, where there is no
  `DataSource` in your code to annotate.

**When it is safe to ignore.** When the check is the thing you are buying. If
the object really can change under you and a stale answer would be a problem,
seconds spent proving freshness are seconds well spent, and Cash measures
latency and nothing else. Take it seriously in the opposite case — a write-once
export, a dated partition, a version-pinned artefact — because there you are
paying, on every hit, for a question that could not have come back positive.

## REMOTE-SIZE-ONLY {#remote-size-only}

**What happened.** Cash is tracking a remote object so it knows when to
invalidate, and the store offered nothing solid to track it by: no ETag, no
version id, no last-modified time. The only thing left to compare is the
object's size in bytes, so that is what went into the cache key. The message
names the URL and which validators were missing.

<!-- claim: cash/remote_source.py:_warn_weak_token @bce362c5 -->
**Why it matters.** A size only catches edits that change the byte count.
Correct a value in a fixed-width column, rewrite a row to the same length,
replace the object with a different one that happens to be the same size, and
the token does not move — so the key does not move, and the old result is served
as though nothing had happened. That is a silent stale hit, which is the one
failure a cache must not have.

**What to do.** Give Cash something stronger to track.

- If the store supports versioning, pin the URL: `s3://bucket/key?versionId=...`
  or `gs://bucket/key#generation=...`. A pin is the token, so nothing weak is
  left and there is no request to make.
- Otherwise write a small `DataSource` whose token is something you control —
  the run id of the pipeline that produced the object, a digest from a manifest,
  a version written next to it. See [Data sources](api/data_sources.md).
- If the object is yours to publish, adding an `ETag` or `Last-Modified` at the
  source fixes it for every consumer at once.

**When it is safe to ignore.** When the object is append-only or write-once:
logs that only grow, dated partitions such as `.../date=2026-09-06/part-0.parquet`,
anything content-addressed. Either every change moves the length or the URL
itself changes, and a size is then a perfectly good validator. Do not ignore it
for anything overwritten in place — a re-run that rewrites yesterday's export, a
corrected file, a hand-edited CSV — because that is precisely the shape a size
cannot see.

## REMOTE-STATE-UNREADABLE {#remote-state-unreadable}

**What happened.** Cash tried to read a remote object's state to decide whether
the cached result is still fresh, and the request failed. The message names the
URL and the exception. Rather than serve a result whose freshness it could not
check, Cash produced a token it has never seen before, which forces a recompute.
The answer you received is a real one.

<!-- claim: cash/remote_source.py:RemoteFileDataSource._warn_failure @ed90c553 -->
**Why it matters.** This fails closed on purpose: an outage costs you the
speedup, never the correctness. What it costs while it lasts is the cache
itself. Every call to that function recomputes, and each failed check leaves
behind an entry no later call can reach, so the cache directory grows while
nothing in it is reused. It warns once per URL per kind of exception, so seeing
it a single time does not mean it happened a single time.

**What to do.** Read the exception; it usually names which of three this is.
Expired or missing credentials are the commonest (`NoCredentialsError`,
`AccessDenied`, an HTTP 403), then a permissions change on the object or its
bucket, then ordinary network or DNS trouble. A 404 means the object is not
there — check the URL, and check whether something upstream deletes and
recreates it. Nothing needs resetting once access is restored: the next check
reads the state and caching resumes on its own. A missing client library is
reported differently, as a `DependencyNotFoundError` rather than a warning,
because that one is not transient and silently recomputing forever would hide
it.

**When it is safe to ignore.** When it is a blip you can account for — a VPN
reconnect, a credential refresh, a rate limit — and the function is cheap enough
that a few extra recomputes do not hurt. Stop ignoring it as soon as you have
reason to think it is persistent, because the symptom is silence: the results
stay correct, so nothing else is going to tell you that this function's entries
are never being reused.

## STORE-CHUNK-FAILED {#store-chunk-failed}

**What happened.** Your function returned an iterator large enough for Cash to
store in chunks, and one of those chunks failed to write. The message names the
chunk, the backend and the exception. The rest of the entry — including the
manifest that records how many chunks there should be — was written anyway.

<!-- claim: cash/core.py:Cash._chunks_are_intact @769a0a1e -->
**Why it matters.** It costs you the caching, not the correctness. Before
serving a chunked entry Cash probes every chunk the manifest claims and treats a
manifest with a hole as *absent*, so the call misses and recomputes — the same
way every other `STORE-` failure degrades. The cost is
that the entry is written and permanently unreachable: every call recomputes,
and the wasted write repeats, until a write finally succeeds and completes the
entry. Measured: three calls after the failure ran the body three times and each
returned all ten items; the call after the write succeeded was the last one to
run the body.

<!-- claim: cash/core.py:Cash._compute_with_lock @b47c9e4c -->
Both read paths run that probe, including the double-checked re-read taken
inside the lock when `use_locking=True`. Until 2026-09-06 the locking path
skipped it and served the broken entry as a *short* iterator — three of ten
items, or none at all when the missing chunk was the first, with no recompute
and no error. If you are on an older version and use locking, this warning is a
correctness signal there and not only a cost one.

**What to do.** Get rid of the entry. `f.cache_clear()` on the decorated
function is the blunt version and always works, and it is what stops the
recompute-every-call state as well as the truncation risk under locking. Then
fix the write itself — the message names the exception, and the usual causes are
a full disk, a permissions problem, or an item in the iterator that cannot be
serialised.

**When it is safe to ignore.** Never for long. You are paying full compute on
every call to that function while the entry sits there unreadable, so it is a
performance bug that will not fix itself — and the write that failed will keep
failing until you fix its cause.

## STORE-FAILED {#store-failed}

**What happened.** Your function ran and returned its result. Writing that
result to the cache failed. The message names the backend and the exception.
Nothing was stored.

<!-- claim: cash/core.py:Cash._store_in_cache @691903b5 -->
**Why it matters.** The result you received is correct — the failure is on the
storage side only, and Cash deliberately reports it rather than raising it into
your code. If this happens once, it costs one recompute. If it happens on every
call, the cache is doing nothing at all while continuing to look like it works,
and every run pays full price.

**What to do.** Read the exception. The common causes are a full disk, a
`cache_dir` you do not have write permission to, a value that cannot be pickled
(a socket, a file handle, a lambda hiding in the result), and — on Windows — a
file another process is holding open, with antivirus and file-sync clients the
usual suspects. Serialisation failures are fixed in the function: return the
data, not the handle that produced it.

**When it is safe to ignore.** When it is a one-off and the compute is cheap.
The failure is contained, nothing on disk is corrupt, and the next call simply
writes the entry again. Stop ignoring it the moment it repeats — a persistently
failing write means you are paying the full cost of a cache and getting none of
the benefit.

## STORE-INPUT-CHANGED {#store-input-changed}

<!-- claim: cash/core.py:Cash._inputs_moved_during_call @4e525f0b, cash/notebook/file_tracker.py:FileAccessTracker.inputs_changed_since_read @faa17b34 -->
**What happened.** A file the cached function read changed before the function
returned — its size or timestamps moved between the moment it was read and the
moment the result was about to be stored. The warning names the file. The
result was returned to you, but it was **not cached**.

**Why it matters.** Cash fingerprints a function's input files to know later
whether they changed. If it fingerprinted the file *after* the change, the entry
would match the new file while holding a result computed from the old one, and
every later call would be a cache hit with the old answer — silently, for as
long as the entry lived. Not caching is the only honest option: nothing can say
which version of the file the result came from.

Two shapes produce it:

* **Something else writes the file while your function runs** — a sync job, a
  download, another process. The next call reads the settled file and caches
  normally, so this usually needs no action.
* **An outer cached function**, whose inner cached call read the file, and the
  file changed between that inner call finishing and the outer one returning.
  Same outcome: the outer result is returned and not stored.

**What to do.** Usually nothing. If it fires on every run, your function is most
likely **writing a file it also reads** — appending to a log it parsed, updating
a state file in place. Split the read and the write, so the cached part only
reads:

<!-- test:skip reason="illustrative: the point is the split, not a value" -->
```python
@cash.cache
def summarise(path):
    return build_summary(pd.read_csv(path))      # reads only

summary = summarise("state.csv")
summary.to_csv("state.csv")                      # the write happens outside
```

**When it is safe to ignore.** When the writer is something you expected to be
running, and it has finished by the next call. The warning fires once per
function per session.

## STORE-LOCK-FAILED {#store-lock-failed}

**What happened.** Before computing a miss, Cash takes a lock on that cache key
so two callers asking for the same thing at the same moment do not both compute
it. Acquiring the lock failed, and Cash went ahead without it rather than
failing your call. The message names the exception.

**Why it matters.** The lock is an efficiency device, not a correctness one.
Without it, concurrent calls with the same arguments can each do the same work
and each write the result. Nothing becomes wrong; it just stops being
deduplicated, which is expensive when the work is expensive.

**What to do.** The exception usually points at the backend rather than at your
code: a Redis timeout under contention, a dropped connection, a stale lock file
left behind by a process that was killed, a full disk, or a `cache_dir` on a
filesystem where locking does not work properly — some network mounts do not.
Fix that and locking resumes on its own; nothing needs to be reset.

<!-- claim: cash/backends/_base.py:CacheBackend.lock @03560a4b -->
**When it is safe to ignore.** When nothing is concurrent. A single-threaded
script or a single notebook kernel has no second caller to race with, so the
lock was never doing anything for you and its absence changes nothing. It
matters when several **threads** in one process share one cache and the work is
expensive — there, this warning is telling you your duplicate-work protection is
switched off. Note what the default lock is and is not: an in-process per-key
`RLock`, so it never deduplicated work across separate processes or worker
machines in the first place. Cross-process single-flight needs the Redis
backend, which replaces it with a distributed lock.

## STORE-METADATA-INVALID {#store-metadata-invalid}

**What happened.** Cash found an entry for this call but could not read the
bookkeeping stored alongside it — the record holding the timestamp, the TTL and
which serialiser wrote the value. It treated the entry as absent and
recomputed. The message names the exception.

<!-- claim: cash/core.py:Cash._warn_metadata_invalid @f1d92b39 -->
**Why it matters.** Mostly it does not, and that is worth saying plainly. The
fallback is the right one: an unreadable entry is ignored rather than
half-trusted, so you get a fresh, correct result. What it costs is one
recompute per affected entry, plus a little dead space on disk.

**What to do.** Nothing, for a one-off. If it keeps appearing, clear that
function's entries with `f.cache_clear()` so the unreadable records are
replaced. The two usual causes are an entry written by an older version of
Cash, and a write interrupted partway — a killed process, a machine that lost
power mid-write.

**When it is safe to ignore.** Often. This is a genuinely low-severity one:
right after upgrading Cash, seeing it a few times while old entries are
replaced is expected and needs no action at all. It is worth investigating only
if it persists after a `cache_clear()`, which would point at something in the
storage layer rather than at leftover data.
