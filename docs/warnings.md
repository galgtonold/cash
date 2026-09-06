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
returns a dict whose `"warnings"` entry holds the last twenty, each with the
code, the message and a timestamp. See
[Debugging and monitoring](tutorials/feature-guides/debugging-and-monitoring.md).

For the class hierarchy, see [Exceptions & warnings](api/exceptions.md).

## ANNOT-TTL-INVALID {#annot-ttl-invalid}

**What happened.** You put `# @cash:ttl=` on a statement in a notebook and the
value after the `=` is not a whole number of seconds, so Cash ignored that
annotation entirely and the statement keeps whatever caching it would have had
without it. There is no unit suffix and no decimal point: five minutes is
`ttl=300`, not `ttl=5m` and not `ttl=300.0`. Only ASCII digits count, so a
superscript or a full-width digit pasted in from elsewhere looks right in the
cell and is still rejected.

**Why it matters.** The statement is still cached — it simply has no expiry now.
If you set a TTL because the value goes out of date on a clock rather than
because of anything Cash can watch change — a query against a table that reloads
overnight, a rate-limited API response — then nothing is going to expire it, and
the stored value will be served for as long as the entry survives. Meanwhile the
line sits in your notebook looking like it works.

**What to do.** Rewrite the value as a bare count of seconds: `# @cash:ttl=300`
for five minutes, `3600` for an hour, `86400` for a day. Only this one line was
dropped — any other `@cash:` annotations on the same statement, a
`# @cash:persist` above it included, were parsed normally and still apply. See
[Annotations](annotations.md).

**When it is safe to ignore.** When something else already invalidates that
statement. A TTL added out of habit on top of a value derived from a file or a
`DataSource` Cash tracks is belt-and-braces: the entry invalidates when the data
changes, which is what you wanted, and losing the timer costs nothing. It is not
safe to ignore when the TTL was the *only* thing that would ever have
invalidated the statement — there the clock is the whole mechanism, and it is
switched off.

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
  built a line earlier is not flagged.
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

**What to do.** Go down the list one line at a time and fix the ones that are
real. For each one you have read and decided is fine, put `# @cash:assume-safe`
on that line:

    def build_report(rows):
        print(f"building {len(rows)} rows")  # @cash:assume-safe
        return summarise(rows)

On a line of its own the comment waives the statement below it as well as
itself; on the `def` line it waives the findings that belong to the whole body
rather than to any single line, which is where a `mutable_global` lands.

Reach for `@cash.cache(assume_safe=True)` only when you mean the whole function
for good. It silences the check for everything in the body *including code added
to it later* — audit the function today, add a `session.post(...)` next month,
and nothing says a word. A comment written next to the statement cannot do that:
new code arrives unannotated and is reported, and the scope of the exemption is
visible in the diff that granted it.

**When it is safe to ignore.** When every line it names is a `print`, a
`logging` call or a progress bar. That is far and away the commonest reason this
fires, and it is as harmless as it looks: you lose the printout on cache hits
and nothing else. Even then, the per-line comment is a better response than a
warning filter, because it leaves the rest of the function watched. Do not
ignore a `mutable_global` or a `dynamic_pattern` line — those two are the
stale-result kinds, and nothing else will tell you when they bite.

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

The warning repeats when the break changes and stays quiet while it does not, so
running other cells with the same broken cell in place will not spam you; fixing
it and later breaking it again will warn again.

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

**What to do.** Decide what you wanted from that statement.

- Genuinely fresh every run: `# @cash:no-cache` on it.
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
draw — those questions need `# @cash:no-cache`, and a replay answers a different
question without saying so.

## RANDOM-SEED-NONE {#random-seed-none}

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

**What to do.** Pick one of the two things `seed(None)` sits between.

- If those values must reflect the new stream, mark them `# @cash:no-cache`.
  That switches off caching *and* the RNG rewind for the statement, and it is
  the only thing that makes a draw genuinely fresh each run.
- If you want them reproducible, seed with a fixed integer instead —
  `np.random.seed(42)`. The cached value and the stream then agree, and they
  agree the same way on anyone else's machine.

**When it is safe to ignore.** When nothing cached derives from the stream it
resets. A `seed(None)` written to undo a fixed seed set earlier in an
exploratory notebook, with the draws below it already marked
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
- **Genuinely fresh every run:** in a notebook, `# @cash:no-cache` on the
  statement. It switches off the RNG rewind as well as the caching, which is
  why merely not caching is not enough. Under the decorator, leave the function
  undecorated.
- **Frozen, deliberately:** `# @cash:allow-random` on the statement, or
  `@cash.cache(allow_random=True)`. Both silence the warning and change nothing
  else — the value was frozen before and stays frozen.

Whichever you pick, pick it per statement or per function. Switching caching off
across the board to "fix" this trades a known frozen value for a slow notebook
and gains nothing.

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
