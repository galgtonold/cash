# Warnings

Every warning Cash emits carries a code in square brackets and a link to its
section here. To look one up, search this page for the code.

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

If a warning scrolled past, most of what the decorator raises about a function
is also kept in a rolling log on the function itself — `f.cache_info()`
returns a dict whose `"warnings"` entry holds the last twenty, with the message
and a timestamp. See
[Debugging and monitoring](tutorials/feature-guides/debugging-and-monitoring.md).

For the class hierarchy, see [Exceptions & warnings](api/exceptions.md).

## CACHE-ASYNC-GENERATOR {#cache-async-generator}

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

**Why it matters.** Cash treats a predicate that raises as "do not cache", so
that function is now not caching at all: every call recomputes. It is a bug in
your predicate rather than a limitation of Cash, and until it is fixed the
decorator is doing nothing.

**What to do.** Read the exception. The usual cause is a predicate that assumes
a shape the result does not always have — `lambda r: len(r) > 0` meeting a
`None`, or `lambda r: r["rows"]` meeting an error payload. Make the predicate
total, or drop `cache_if=` if you no longer need the gate.

**When it is safe to ignore.** Only if you were about to remove `cache_if=`
anyway. Note that this fires once per function, not once per call: seeing it a
single time does not mean it happened a single time. Until the predicate is
fixed, every call is paying full compute.

## CACHE-LOOP-GROWTH {#cache-loop-growth}

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

**Why it matters.** `@cash.cache` on this function is making your program
slower, and not on some pathological call — on every call. The usual cause is a
large argument whose fingerprint is computed from its actual bytes. A 153 MiB
DataFrame passed to a function that sums one column measured 390 ms to
fingerprint against 11 ms of work: 34 times slower, every time. Cash only
speaks up once the loss has accumulated past a couple of real seconds *and* its
per-call overhead exceeds even the largest body time it has seen, so a function
that is usually fast but occasionally very slow will not be flagged.

**What to do.** The first move is to keep the caching and make the key cheap,
by registering a hasher for the expensive argument's type:

    cash.register_hasher(pd.DataFrame, lambda df: df.attrs["version"], override=True)

`override=True` is not decoration. For the types Cash fingerprints itself —
numpy arrays, pandas / polars / PyArrow / modin frames, dask collections — its
own content hasher runs first, and a plain registration for one of those types
is rejected outright rather than silently ignored. Whatever your hasher returns
becomes the entire identity of that value, so return something that genuinely
changes when the data changes: a version, a content id, an immutable
fingerprint. If there is no such handle to be had, remove the decorator from
this function. It is not the right tool for this shape of work.

**When it is safe to ignore.** When speed is not why the decorator is there.
Caching to avoid a metered API call, to hold a result steady across a session,
or to stop a nondeterministic step from re-running are all real reasons, and
Cash measures seconds and nothing else — so its verdict is accurate and beside
the point. If you added the decorator to make something faster, this is the
warning on this page most worth reading, and the last one to filter.

## CACHE-THRASH {#cache-thrash}

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

**What happened.** A single value is larger than a safe fraction of every
persistent tier's cap, so Cash kept it in memory instead of writing it to disk.
Writing it would immediately push the cache past its cap and evict it again.
The message names the value's size.

**Why it matters.** Inside this process the cache works normally and you get
every hit you would expect. But the value lives in RAM only, so it dies with
the process: the next script run, or the next kernel restart, recomputes it
from scratch.

**What to do.** Raise `max_cache_size` to a comfortable multiple of the value's
size. If that room is not available, cache something smaller — the aggregate,
the sample, or the columns you actually use rather than the whole object.

**When it is safe to ignore.** When the reuse you care about happens inside one
run. A script that computes the value once and consumes it ten times gets all
ten hits, and the warning costs you nothing. Do not ignore it if you are
relying on the cache surviving restarts, which is usually the reason for
caching something this large in the first place.

## KEY-BOOL-STATE-TOKEN {#key-bool-state-token}

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

## KEY-DEPENDS-ON-OPAQUE {#key-depends-on-opaque}

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

## KEY-INSTANCE-STATE {#key-instance-state}

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

**What happened.** A function, a class, a `functools.partial`, or an object
whose class carries code reached a cached call — as an argument you passed, or
as a parameter default you never typed — and Cash could not fingerprint its
body. Cash normally folds the code of such things into the key, so that editing
them invalidates. This one it could not.

**Why it matters.** The result depends on what that callable does, and the key
does not. Edit the callable, run again, and you get the old answer back with
nothing to indicate anything is wrong. This is the residue where Cash genuinely
cannot work the answer out, which is why it says so rather than staying quiet.

**What to do.** If the result really does depend on that implementation, name
it explicitly with `@cash.cache(depends_on=[the_callable])`. If it does not —
the callable is a stable third-party helper, or its identity is already covered
by another argument — say so deliberately: `cash.mark_opaque(TheType)`, or
`@cash.opaque` on a class you own. Both record the decision in the code, which
is what makes them better than a warning filter.

**When it is safe to ignore.** When the callable is not yours to edit —
`json.dumps`, an `operator.itemgetter`, a C-extension function from a pinned
library. Nothing will change under you, so the missing fingerprint costs
nothing. It is never safe to ignore when it is your own code and you are
actively editing it: that is precisely the "why is my cache serving me the old
version" problem, and this warning is the only notice you get.

## KEY-UNHASHABLE-ARG {#key-unhashable-arg}

**What happened.** One of the arguments could not be turned into a stable
fingerprint, so Cash could not build a cache key at all. The message names the
type when it can identify one; when the offending value is nested inside a
container it says so instead, because it cannot see which element is to blame.
The call ran and returned normally.

**Why it matters.** That call did not cache, and calls like it will not cache
either — this is not first-call warm-up. Every call passing that argument pays
full compute. Nothing can go stale, because nothing is being stored.

**What to do.** Register a hasher for the type:

    cash.register_hasher(DatabaseSession, lambda s: s.database_url)

Or pass something Cash can fingerprint in its place: the connection string
rather than the connection, the path rather than the open file handle. When
Cash could not name the type, the culprit is nested — a list of custom objects,
a dict holding a live handle — and the same two fixes apply once you find it.

**When it is safe to ignore.** When you do not need that call path to be fast.
There is no correctness risk here whatsoever: an unbuildable key means no entry
is written and none is read. What you lose is the caching, completely, for
every call that passes that argument — so ignore it only if you have decided
that is fine.

## KEY-UNHASHABLE-DEFAULT {#key-unhashable-default}

**What happened.** One of the function's *parameter defaults* — a value in the
`def` line, not something a caller passed — could not be fingerprinted, so Cash
declined to cache the call. The message names the type.

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

## STORE-CHUNK-FAILED {#store-chunk-failed}

**What happened.** Your function returned an iterator large enough for Cash to
store in chunks, and one of those chunks failed to write. The message names the
chunk, the backend and the exception. The rest of the entry — including the
manifest that records how many chunks there should be — was written anyway.

**Why it matters.** This is the one warning on this page that can hand you
wrong data. A later call that hits this entry reads chunks in order and stops
at the first one that is missing: you get a *truncated* iterator, not an error
and not a recompute. Ten thousand rows come back as four thousand, and nothing
says so.

**What to do.** Get rid of the entry before anything reads it. `f.cache_clear()`
on the decorated function is the blunt version and always works. Then fix the
write itself — the message names the exception, and the usual causes are a full
disk, a permissions problem, or an item in the iterator that cannot be
serialised.

**When it is safe to ignore.** Never. Every other `STORE-` failure degrades to
a recompute; this one degrades to a short answer. Clear the entry.

## STORE-FAILED {#store-failed}

**What happened.** Your function ran and returned its result. Writing that
result to the cache failed. The message names the backend and the exception.
Nothing was stored.

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

**When it is safe to ignore.** When nothing is concurrent. A single-threaded
script or a single notebook kernel has no second caller to race with, so the
lock was never doing anything for you and its absence changes nothing. It
matters when several threads, processes or workers share one cache and the work
is expensive — there, this warning is telling you your duplicate-work
protection is switched off.

## STORE-METADATA-INVALID {#store-metadata-invalid}

**What happened.** Cash found an entry for this call but could not read the
bookkeeping stored alongside it — the record holding the timestamp, the TTL and
which serialiser wrote the value. It treated the entry as absent and
recomputed. The message names the exception.

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
