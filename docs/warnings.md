---
search:
  boost: 2
---

# Warnings

!!! info "Applies to: both paths"
    Every warning code cash emits, for `@cash.cache` users and notebook users. Each code says which path it comes from.

<!-- claim: cash/diagnostics.py:DIAGNOSTIC_CODES @bc60d43c -->
Every cash warning starts with a code in square brackets, such as
`[CACHE-THRASH]`, and ends with a link to that code's section below.

## How much a warning says {#warning-detail}

<!-- claim: cash/diagnostics.py:short_diagnostic @e965c47a, cash/decorator/reporting.py:Notices._displayed @edbc85c6 -->
By default a warning is one line: the code, what happened, the fix,
and where to read more. A list of findings or effects
is cut to how many there are and the first one. The rest is in the code's
section below, which the link opens:

```text title="Output"
model.py:10: CashImpurityWarning: [IMPURE-OBSERVED-EFFECTS] @cash.cache on model.backtest: the first call had effects static analysis did not see, which a cache hit will not repeat: 3 effects, e.g. file write: .../__pycache__/helpers.kern-2.py313.nbi.tmp.cc1b, at model.py:5 Fix: split the function if an effect is part of the result -- an 'argument mutation' line means an object the CALLER still holds stops being changed. If it is incidental, put `# @cash:assume-safe` on the line named (any line of yours on the way to it counts) or run it inside `with cash.assume_safe():`; @cash.cache(assume_safe=True) waives the whole function instead, including effects added later. (backtest.cache_info()['warnings']; https://cash-lib.readthedocs.io/en/latest/warnings/#impure-observed-effects)
```

For a decorated function, `f.cache_info()["warnings"]` keeps the full text
(every finding), which the line names. When a second function
gets a code the first already showed, its line only says so:
`[IMPURE-OBSERVED-EFFECTS] also on backtest2 (backtest2.cache_info()['warnings'])`.

Set [`warning_detail`](getting-started/configuration.md#output-and-switches)
to `"full"` (`CASH_WARNING_DETAIL=full`) to print the full text instead:

```text title="Output"
[CACHE-THRASH] the cache is full at its 500 MB cap.
  Fix: raise max_cache_size, or cache fewer values.
  https://cash-lib.readthedocs.io/en/latest/warnings/#cache-thrash
```

Logging (when your program configures it) always gets the full text.

## Silencing one code {#silencing-one-code}

Filter on the code. The message always starts with it, so this silences one
code and nothing else:

```python
import warnings

warnings.filterwarnings("ignore", message=r"\[CACHE-THRASH\]")
```

<!-- claim: cash/diagnostics.py:warn_diagnostic @c9245473 -->
Every warning also carries its code on `.code`, so a handler can test
`w.message.code == "CACHE-THRASH"` instead of matching the wording. Filtering
by class catches many codes at once (each code's section names its class);
the class recipes are in
[Exceptions and warnings](api/exceptions.md#filtering-warnings).
For a decorated function, `f.cache_info()["warnings"]` keeps the last twenty
warnings it raised, in case one scrolled past.

For the warnings about what a cached function does, three waivers act on
the code instead of the message. Each of those codes ends with a **Silencing
it** line that says which of them it honours.

- `# @cash:assume-safe` on a line waives that one finding and keeps the rest
  of the function checked.
- `with cash.assume_safe():` waives what the comment waives, for every line
  in the block, and the effects observed while the block runs.
- `@cash.cache(assume_safe=True)` waives every finding in the function,
  including code added to it later.

## Codes by family {#index}

Codes share a prefix with the others of their kind. Each family below
starts with a table of its codes; each code's section names its path and
its warning class.

| Family | Prefix | Codes | What it reports |
|---|---|---:|---|
| [Annotations](#annot-codes) | `ANNOT-` | 2 | A `# @cash:` comment that cash could not honour. |
| [Caching](#cache-codes) | `CACHE-` | 19 | Caching happened, or refused to, and it is worth saying. |
| [Configuration](#config-codes) | `CONFIG-` | 3 | A setting cash found but could not act on. |
| [Side effects](#impure-codes) | `IMPURE-` | 4 | The function does something a cache hit will not repeat. |
| [Cache keys](#key-codes) | `KEY-` | 26 | Something the result depends on may not be in the cache key. |
| [Notebook](#notebook-codes) | `NOTEBOOK-` | 16 | Notebook-wide machinery rather than one statement. |
| [Randomness](#random-codes) | `RANDOM-` | 4 | A cached value that randomness makes non-reproducible. |
| [Remote files](#remote-codes) | `REMOTE-` | 3 | Checking whether a remote file changed. |
| [Storing results](#store-codes) | `STORE-` | 8 | The call succeeded, but its result was not written. |

## Annotations {#annot-codes}

A `# @cash:` comment that cash could not honour. Every code here starts `ANNOT-`.

| Code | Applies to | Meaning |
|---|---|---|
| [ANNOT-TTL-INVALID](#annot-ttl-invalid) | notebook | `# @cash:ttl=` is not whole seconds; ignored |
| [ANNOT-UNKNOWN-DIRECTIVE](#annot-unknown-directive) | both | `# @cash:<name>` is not a directive; ignored |

### ANNOT-TTL-INVALID {#annot-ttl-invalid}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/analysis/annotations.py:parse_annotation_line @58513e3e -->
**What happened.** The value after `# @cash:ttl=` is not a whole number of
seconds (`ttl=5m`, `ttl=300.0`, `ttl=-5`, or no `=`), so cash ignored the
annotation.

**Why it matters.** The statement is still cached, with no expiry of its own.
If the TTL was the only thing meant to refresh it, the stored value is served
until something else invalidates it.

<!-- claim: cash/analysis/annotations.py:ANNOTATION_PATTERN @412c3ce1, cash/analysis/annotations.py:parse_annotation_line @58513e3e -->
**What to do.** Write plain seconds: `# @cash:ttl=300` for five minutes. See
[Annotations](annotations.md).

**When it is safe to ignore.** When cash already tracks what the value depends
on (a file, a `DataSource`) and the TTL was only a backup.

### ANNOT-UNKNOWN-DIRECTIVE {#annot-unknown-directive}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/analysis/annotations.py:KNOWN_DIRECTIVES @91e03db7, cash/analysis/annotations.py:_warn_unknown_directive @24d9ef6e -->
**What happened.** A comment starts `# @cash:` but the word after it is not a
directive, so cash ignored it. The known directives are `no-cache`, `persist`,
`ttl=N`, `allow-random`, `no-cache-calls` and `assume-safe`. When
your spelling is close to one, the message suggests it.

**Why it matters.** The code behaves as if the comment were not there. A
misspelled `no-cache` means the statement is cached anyway.

**What to do.** Fix the spelling and run the cell again. See
[Annotations](annotations.md).

**When it is safe to ignore.** When the comment was never meant as a
directive. Reword it so it does not start `# @cash:`.

## Caching {#cache-codes}

Caching happened, or refused to, and it is worth saying. Every code here starts `CACHE-`.

| Code | Applies to | Meaning |
|---|---|---|
| [CACHE-ASYNC-GENERATOR](#cache-async-generator) | decorator | async generators are not cached |
| [CACHE-CLEAR-INCOMPLETE](#cache-clear-incomplete) | decorator | `cache_clear()` could not remove some entries; they are still served |
| [CACHE-DIR-NEW](#cache-dir-new) | decorator | cash created its default cache folder, and says where |
| [CACHE-DIR-UNWRITABLE](#cache-dir-unwritable) | both | the cache directory cannot be written |
| [CACHE-EVICTED-RECOMPUTE](#cache-evicted-recompute) | both | the size cap had evicted a costly result, so it was computed again |
| [CACHE-FRESHNESS-COST](#cache-freshness-cost) | decorator | checking tracked files costs much of what a hit saves |
| [CACHE-IDENTITY-COUPLED](#cache-identity-coupled) | decorator | a live matplotlib figure is never stored |
| [CACHE-IF-BYPASSED](#cache-if-bypassed) | decorator | a chunked result was stored without running `cache_if` |
| [CACHE-IF-RAISED](#cache-if-raised) | decorator | the `cache_if` predicate raised |
| [CACHE-LOOP-GROWTH](#cache-loop-growth) | notebook | a loop stores every state of a growing value |
| [CACHE-NET-LOSS](#cache-net-loss) | decorator | caching this function costs more time than it saves |
| [CACHE-NONDETERMINISTIC](#cache-nondeterministic) | decorator | computed again, a result differs from the one stored before |
| [CACHE-NOT-WORTH-BYTES](#cache-not-worth-bytes) | notebook | a big, cheap value was not written to disk |
| [CACHE-PER-ITEM-RERUN](#cache-per-item-rerun) | notebook | a slow step ran again over a whole collection, though most of it is unchanged |
| [CACHE-RETURNS-AWAITABLE](#cache-returns-awaitable) | decorator | a sync function returned a coroutine, which is not stored |
| [CACHE-RESULT-SHARED](#cache-result-shared) | decorator | the result shares state with the caller's object |
| [CACHE-THRASH](#cache-thrash) | both | the cache is full and evicts what it just stored |
| [CACHE-VALUE-TOO-BIG](#cache-value-too-big) | both | a value is bigger than the disk cap |
| [CACHE-WRITE-ABANDONED](#cache-write-abandoned) | both | the process exited before its writes finished |

### CACHE-ASYNC-GENERATOR {#cache-async-generator}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/core.py:Cash.cache @470582df -->
**What happened.** You put `@cash.cache` on an async generator (an
`async def` that `yield`s). cash does not cache those, so it returned your
function unwrapped.

**Why it matters.** Every call runs the whole body.

**What to do.** If the results fit in memory, move the work into a plain
`async def` that returns a list and cache that. Otherwise cache the expensive
step inside the generator.

**When it is safe to ignore.** When the generator is cheap and you do not
need it cached.

### CACHE-CLEAR-INCOMPLETE {#cache-clear-incomplete}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/maintenance.py:Maintenance.delete_function_entries @a3f2bae5, cash/backends/file_eviction.py:FileEvictor.remove_path @76e08300 -->
**What happened.** `f.cache_clear()` could not remove some of the function's
entries. The message says how many.

**Why it matters.** Those entries are still stored, so the next call with
their arguments is served the result you meant to clear.

**What to do.** On Windows, a file cannot be removed while another process has
it open: a reader, a virus scanner, a search indexer. cash retries briefly
first. Close whatever holds the cache folder and clear again, or run
`cash clear --function NAME` once it has let go.

**When it is safe to ignore.** Never when you cleared to get rid of a wrong
result. For freeing space only, the entries go at the next clear.

### CACHE-DIR-NEW {#cache-dir-new}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/_location.py:announce_new_cache_dir @5260de8d, cash/_location.py:_existing_cache_near @e592579d, cash/backends/factory.py:build_backend_from_config @ef844e4c -->
**What happened.** cash created its cache folder for the first time, in the
default place: the project root above the running script, or beside the
script when no project is found above it. The message names the folder. When
a cache cash made already exists in the current directory or a folder above
it, or above the new folder, the message names that one too. It is shown
once per folder and process, not in IPython, and not when you
chose the folder with `CASH_CACHE_DIR`, `[tool.cash] cache_dir` or
`Cash(cache_dir=...)`.

**Why it matters.** A new folder is an empty cache, so this run computes
everything. That is expected the first time. If the code already has a cache
somewhere else, this run is not using it.

**What to do.** Nothing, if the folder named is where you want the cache. To
use another one, set `CASH_CACHE_DIR` for the run, or `cache_dir` under
`[tool.cash]` in the project's `pyproject.toml`. A script in a folder with no
project marker uses a `.cash` in a folder above it, so moving the script into
the folder that holds the cache works too. See
[Where the cache folder is](how-it-works/storage.md#where-the-cache-folder-is).

**When it is safe to ignore.** When the folder named is the one you meant.

### CACHE-DIR-UNWRITABLE {#cache-dir-unwritable}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/backends/cache_dir.py:warn_if_unwritable @2d37cc7a -->
**What happened.** cash could not create a file in its cache directory: a
read-only mount, missing permissions, or a path that no longer exists. The
message names the directory and the OS error. On Windows without long paths
it is also a directory whose path is too long for an entry: entry names are
70 characters, so a cache directory of about 188 characters or more takes
entry paths past the 260-character limit, and the message says so.

**Why it matters.** Nothing is written to disk for the rest of the run, so
every new process recomputes. Results are still correct, and in-process repeats
still hit the RAM tier.

**What to do.** Point cash at a writable directory, or give this account write
permission on the one named:

<!-- test:skip reason="repoints the cache at a fixed absolute path; running it would send every later fence on this page to a directory that outlives the test" -->
```python
# or set CASH_CACHE_DIR=/var/tmp/cash in the environment
cash.configure(cache_dir="/var/tmp/cash")
```

In a container, check the cache path is on a writable volume. On Windows,
use a shorter path or enable long paths (`LongPathsEnabled`).

**When it is safe to ignore.** When the directory is read-only on purpose: a
shared cache you only read from.

### CACHE-EVICTED-RECOMPUTE {#cache-evicted-recompute}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

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

<!-- claim: cash/decorator/explain.py:MissHistory.absent_entry_reason @27204896 -->
**When it is safe to ignore.** When the result is rarely needed and a
recompute now and then is cheaper than the disk it would take. It is said once
per function or statement per process. Cheaper recomputes are not warned
about, but `f.explain()`, the per-call log and a notebook badge still say
"evicted to make room" for them.

### CACHE-FRESHNESS-COST {#cache-freshness-cost}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/file_deps.py:FileDeps._warn_if_local_validation_is_expensive @45f85758, cash/remote_source.py:validation_is_expensive @18292cc6 -->
**What happened.** Before serving a hit, cash checks every file the call read.
Here that check cost more than half of the compute it saved, or more than two
seconds. The result was correct.

**Why it matters.** A hit should be nearly free. The usual causes are many
files, very large files, or a slow filesystem such as a network share.
File dependencies also pass up to cached callers, so a wide set is checked on
every caller's hit too.

**What to do.** Make the entry depend on less. Split the function so each
input is read by its own cached loader and the aggregate depends on their
results.

**When it is safe to ignore.** When the numbers in the message are still a good
trade, such as half a second of checking against a five-minute job. It fires
once per function per process.

### CACHE-IDENTITY-COUPLED {#cache-identity-coupled}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.refuses_identity_coupled @9a87665b -->
**What happened.** The function returned a live matplotlib `Figure` or `Axes`,
or a container holding one, and cash did not store it.

**Why it matters.** A restored copy would not be the figure pyplot tracks, so
`plt.savefig()` could write a blank image. Refusing keeps your plot correct;
the function simply runs every time.

**What to do.** If the function does real work before it plots, split it:
cache the part that computes the numbers, and draw in an uncached function.

**When it is safe to ignore.** Almost always, when the function only draws.

### CACHE-RETURNS-AWAITABLE {#cache-returns-awaitable}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/store.py:ResultStore._refuse_awaitable @71438815 -->
**What happened.** The cached function is a plain (sync) function that
returned a coroutine or another awaitable, and cash did not store it. This is
what `@cash.cache` sees over an ordinary wrapper, such as a retry or timing
decorator, around an `async def`: the wrapper is not itself `async`, so it
hands back the coroutine and the caller awaits it after cash has returned.

**Why it matters.** The function's value does not exist yet when cash would
store it, so the call is never cached: the body runs on every call.

**What to do.** Put `@cash.cache` directly on the `async def`, under the
wrapper:

<!-- test:skip reason="illustrative: retry is the reader's own wrapper" -->
```python
@retry
@cash.cache
async def fetch(x):
    ...
```

**When it is safe to ignore.** When the function is not meant to be cached;
then remove `@cash.cache`. It fires once per function per process.

### CACHE-IF-BYPASSED {#cache-if-bypassed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

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

### CACHE-IF-RAISED {#cache-if-raised}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/reporting.py:Notices.cache_if_raised @6298df6f -->
**What happened.** Your `cache_if=` predicate raised when called with the
function's result. The call returned normally; only storing was skipped.

**Why it matters.** Every result that makes the predicate raise is recomputed
on every call. The warning appears once per function, not once per call.

**What to do.** Read the exception and make the predicate handle every result
shape, for example `lambda r: r is not None and len(r) > 0`.

**When it is safe to ignore.** When those results are cheap to recompute.

### CACHE-LOOP-GROWTH {#cache-loop-growth}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

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

### CACHE-NET-LOSS {#cache-net-loss}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/effectiveness.py:CUMULATIVE_WASTE_SECONDS == 2.0 -->
**What happened.** cash times what it spends on each call (building the key,
the lookup, storing the result) and compares it with the function's own run
time. For this function, caching has lost more than two seconds in total so
far, and a hit costs more than the function's longest run. The message gives
the numbers and names the costliest argument. It waits for the function's
first hit in the process: before one, nothing could have been saved, and what
a hit costs is not known.

<!-- claim: cash/effectiveness.py:EffectivenessLedger.final_verdicts @d94c91a3 -->
**Why it matters.** The decorator makes your program slower. The usual cause
is a large argument, such as a big DataFrame, being hashed on every call.
The check also runs at exit, so a script that calls each function once is
covered, and several small losers are named together.

<!-- claim: cash/core.py:Cash.register_hasher @f8a61573 -->
**What to do.** If a cached function produced the argument and nothing
changes it afterwards, mark the producer `@cash.cache(frozen=True)`: the
argument is then keyed by the call that made it. If the argument is an object
of your own class holding the large data, give the class a
[`__cash_key__`](tutorials/feature-guides/custom-hashers.md#cash-key) method
that returns a version or id. Otherwise register a cheap hasher for its type
that returns a version or content id. For numpy and dataframe types this needs
`override=True`, and what it returns becomes the value's whole identity. If
none fits, remove the decorator. See
[`frozen=` and large arguments](decorator.md#frozen-and-large-arguments).

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
cash.register_hasher(
    pd.DataFrame, lambda df: df.attrs["version"], override=True
)
```

**When it is safe to ignore.** When speed is not why you cache, for example to
avoid a paid API call or to hold a result steady.

### CACHE-NONDETERMINISTIC {#cache-nondeterministic}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashRandomnessWarning</span>

<!-- claim: cash/decorator/store.py:ResultStore._warn_if_result_changed @c19fb467 -->
**What happened.** A `frozen=True` function's result (a DataFrame, an array
or a table) was stored, then evicted or cleared, and computed again in the
same process for the same arguments and the same files. The new result's
content differs from the one stored before.

**Why it matters.** Something in the call is not reproducible: an unseeded
random draw (often from a generator the code makes itself, such as
`np.random.default_rng()`), the clock, or an input cash does not see. The value
a hit returns is one draw of it, and another run would have cached another.

**What to do.** Seed what draws (`random_state=0`, `np.random.default_rng(0)`),
or pass the changing input in as an argument. If any one of the results will
do, pass `@cash.cache(allow_random=True)`.

**When it is safe to ignore.** When the difference is meant: the function
reads something that changes on purpose. Then give it a `ttl`, which this
check leaves alone, or pass the source with `dynamic_depends_on=`. Only
results cash already hashes are compared, so a miss costs no extra hashing,
and a function whose results are not compared is not warned about.

### CACHE-NOT-WORTH-BYTES {#cache-not-worth-bytes}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/backends/value_policy.py:worth_its_bytes, cash/backends/store_notices.py:StoreNotices.not_worth_bytes @04581c7b -->
**What happened.** A value over 8 MiB was cheap enough to rebuild that storing
it would cost more than 128 MiB of disk per second of compute saved, so it was
not written to disk. The message names the size and compute time, and divides
one by the other: "215 MiB of cache per second saved, against the 128 MiB"
means a 430 MiB value that takes 2 s to compute, above the 128 MiB a second
cash spends. It is shown once per statement in a session; a new kernel
recomputes the value, so it shows the warning again. A slow call that cash
caches inside a statement faces the same rule unless a statement stores a
reference to it (`b = load(p)`); the message then names the function, as in
`load(...)`.

<!-- claim: cash/notebook/statement/store.py:StatementStore._with_decorator_ref @9edb344d, cash/backends/persistence_policy.py:PersistencePolicy.decide @e3aa6741 -->
The result of a `@cash.cache` function is not judged by this rule: the
decorator writes every result to disk, and a statement that only binds it
(`df = mylib.build_features(raw)`) refers to that entry instead of storing
it again.

**Why it matters.** A new kernel recomputes it, which is usually faster than
you would notice. The rule keeps big, cheap values from filling the disk.

**What to do.** Usually nothing. To store it anyway, put `# @cash:persist` on
the statement ([Cost model](cost-model.md)), or `@cash.cache` on the function
that computes it. Better still, store something smaller: the aggregate or the
columns you use. When the statement calls a `@cash.cache` function
(`b = load(p) * 2`), the message names it instead of advising the decorator:
that function's own entry is stored as usual, and what was refused is the
value the statement builds from its result.

**When it is safe to ignore.** Usually. Act on it when the message names a
recompute time you can feel on every restart.

### CACHE-PER-ITEM-RERUN {#cache-per-item-rerun}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/notebook/per_item_reruns.py:PerItemReruns.after_run @3109d431, cash/notebook/per_item_reruns.py:MIN_COST_S == 1.0, cash/notebook/per_item_reruns.py:MIN_UNCHANGED == 0.5, cash/notebook/per_item_reruns.py:HASH_SHARE == 0.03, cash/notebook/per_item_reruns.py:STEP_SHARE == 0.1 -->
**What happened.** A statement that took a second or more ran again over a
whole collection, although most of it is what it was when the statement last
ran: 400 images were added to 3,000, one batch in the middle was exported
again, or a season was appended to fifty. The message names the statement,
the input, how many of its rows differ and where:

```text
[CACHE-PER-ITEM-RERUN] `clean = F.preprocess(imgs, sigma=1.0)`
re-ran over all 3,400 rows of `imgs` (26 s) but only 400 rows differ
from the last run (rows 3000-3399). Fix: do the work in a @cash.cache
function that takes one chunk of rows ...
```

A row is an item of the first axis of an array, a row of a pandas or polars
frame or series, or an array in a list of arrays. Cash compares the input
with the last run's in blocks of rows, so the rows it names are whole blocks:
a change to one row names the block of rows around it. It is shown once per
statement per session, and not when every row changed (most of a per-item
helper would run again too) or when the input is the same (the statement ran
for another reason, such as an edit to a function it calls), nor when
another input changed too (a parameter with a new value).

**Why it matters.** The input is one value, and it changed, so cash cannot
reuse any of the last run: the work on the unchanged rows is done again.

**What to do.** Do the work through a `@cash.cache` function that takes one
item or one chunk of rows, and combine the results. Each call is cached on
its own chunk, so after new data arrives only the chunks that changed run:

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
@cash.cache
def prep_batch(batch):
    return np.stack([prep_one(img) for img in batch])

starts = range(0, len(imgs), 250)
clean = np.concatenate([prep_batch(imgs[s:s + 250]) for s in starts])
```

A frame splits the same way, on a column that stays put, such as the
batch or the season:

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
parts = [summarize(g) for _, g in df.groupby("batch")]
```

A cached call costs about a millisecond, a little more the first time, so
give each call a chunk worth a tenth of a second or more of work rather
than one tiny item. Cut the chunks on something that does not move when data
is added (a batch column, a fixed row count): rows inserted at the front
shift every fixed-size chunk after them. When each step needs what the one
before it left (a rating carried from season to season), pass that in and
return it: `hist, ratings = run_season(season, ratings)`, season by season,
reuses every season up to the first one that changed.

You do not need to write catch-up logic of your own (work out what is new,
compute only that, merge it with a saved result): the cached helper does that
part, and stays correct when an old row is corrected.

**When it is safe to ignore.** When the work is not separable by rows (a
model fitted on all of them, a statistic over the whole column), or when the
step is fast enough as it is.

<!-- claim: cash/row_blocks.py:fingerprint @9ffab9b5, cash/notebook/per_item_reruns.py:HASH_SHARE == 0.03, cash/notebook/per_item_reruns.py:STEP_SHARE == 0.1 -->
**What it costs.** Cash keeps a digest per block of rows, never the values,
with the statement's other records, so the comparison holds across a kernel
restart. It hashes an input only after a statement that took a second or
more, once per value however many statements read it. All of it stays
within 3% of the time the session's slow statements took, and within 10%
of any one statement's; an input that would take longer is skipped. A
large frame of text columns hashes at about 0.2 s per million rows.

### CACHE-RESULT-SHARED {#cache-result-shared}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.warn_shared_result @8c38fe43, cash/decorator/purity_checks.py:PurityChecks._shared_with @edceb490 -->
**What happened.** The result shares state with an object the caller still
holds: it is an argument, holds one, is a view of an array argument, or is a
module global. On the first run, a write through one shows in the other. A
cache hit returns a separate copy, so from then on it does not. A module
global the function names and whose type compares by identity (a plain class
without `__eq__`, such as a sentinel) is not reported: a hit hands back that
global itself.

**Why it matters.** A caller that writes through the result, such as filling
a preallocated array, works on the first run and silently stops working on
later runs.

**What to do.** Return a value of its own: `.copy()`, `list(...)`,
`dict(...)`. If the sharing is the point, do not cache the function.

**When it is safe to ignore.** When callers only read the result.

**Silencing it.** Whole function: `@cash.cache(assume_safe=True)`. No line
carries this finding, so neither `# @cash:assume-safe` nor
`with cash.assume_safe():` applies. See [Silencing one
code](#silencing-one-code).

### CACHE-THRASH {#cache-thrash}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/backends/file_eviction.py:FileEvictor.warn_thrash @9c41f109 -->
**What happened.** The cache is at its size cap and evicts entries within a
few writes of storing them. The message names the cap and how much room the
volume has.

**Why it matters.** You pay for computing and for writing, and almost nothing
survives to be reused.

**What to do.** Raise `max_cache_size` if the volume has room. If not, cache
smaller values, or move `cache_dir` to a bigger volume.

**When it is safe to ignore.** Never: it always costs time.

### CACHE-VALUE-TOO-BIG {#cache-value-too-big}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/backends/store_notices.py:StoreNotices.too_big @0b79929c, cash/backends/file_backend.py:FileBackend.promotion_size_cap @db165d55 -->
**What happened.** One value, serialized, is bigger than every disk tier's
whole cap, so it was not written to disk. The message names its size and the
cap.

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend._evict_to_byte_cap @2e17ed9e, cash/backends/memory_backend.py:InMemoryBackend.set @8416cd51 -->
**Why it matters.** It is offered to the RAM tier instead, but the RAM cap is
usually smaller, so usually nothing is cached at all.

**What to do.** Raise `max_cache_size` well above the size named, or cache
something smaller.

**When it is safe to ignore.** When you did not need that value cached. For a
decorated function, `f.cache_info()` shows whether you get any hits.

### CACHE-WRITE-ABANDONED {#cache-write-abandoned}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/config/schema.py:CashConfig.shutdown_write_timeout @5ac9f606, cash/backends/_writes.py:PendingWrites.shutdown @8bf24fea, cash/backends/_writes.py:_DaemonWriterPool.shutdown @814ad0be, cash/backends/_writes.py:PendingWrites._warn_abandoned_writes @e502ed89 -->
**What happened.** At exit, cash waited for its background writes (60 s by
default) and some were still running, so the process exited without them. The
message says how many. With `shutdown_write_timeout=0`, which asks not to
wait, it is not shown: a write still running is only logged.

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

## Configuration {#config-codes}

A setting cash found but could not act on. Every code here starts `CONFIG-`.

| Code | Applies to | Meaning |
|---|---|---|
| [CONFIG-FILE-MISSING](#config-file-missing) | both | `Cash(config_path=...)` names a missing file |
| [CONFIG-INVALID](#config-invalid) | both | a setting or config file cash cannot use |
| [CONFIG-UNKNOWN-KEY](#config-unknown-key) | both | a config key that is not a setting |

### CONFIG-FILE-MISSING {#config-file-missing}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/config/resolve.py:_resolve_config @75ccfd52, cash/config/resolve.py:_warn_named_file_missing @fa72ccc3 -->
**What happened.** Your code passed `Cash(config_path=...)` naming a file that
does not exist. cash used the other configuration layers.

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

### CONFIG-INVALID {#config-invalid}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/config/resolve.py:_validated_layer @ea2b7c73, cash/config/sources.py:_warn_toml_malformed @7aa948bd, cash/config/sources.py:load_toml_layer @b8dfea05, cash/config/resolve.py:_build_tiers @d17e1cb5, cash/config/schema.py:TierConfig.__post_init__ @099eecbf -->
**What happened.** cash could not use part of its configuration:

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

### CONFIG-UNKNOWN-KEY {#config-unknown-key}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/config/resolve.py:_validated_layer @ea2b7c73, cash/config/resolve.py:_unknown_key @6098acf3 -->
**What happened.** A `[tool.cash]` table, a `[cash]` table or a
`CASH_TIER_<N>_*` variable sets a key that is not a cash setting. The message
names the closest real setting:

```text title="Output"
[CONFIG-UNKNOWN-KEY] …/pyproject.toml sets `max_cache_siz`,
which is not a cash setting, so it does nothing. Did you mean
`max_cache_size`?
```

Plain `CASH_*` variables that are not settings are not reported, because
other tools use that prefix. The same key given in code, to `Cash(...)` or
`cash.configure(...)`, raises `ValueError` instead.

**Why it matters.** The setting you meant is not in effect.

**What to do.** Rename or remove the key. `cash info` lists every setting in
effect and where it came from.

**When it is safe to ignore.** Never for long: the key does nothing.

## Side effects {#impure-codes}

The function does something a cache hit will not repeat. Every code here starts `IMPURE-`.

| Code | Applies to | Meaning |
|---|---|---|
| [IMPURE-HANDED-CALLABLE](#impure-handed-callable) | decorator | an argument is a function of yours that changes state |
| [IMPURE-OBSERVED-EFFECTS](#impure-observed-effects) | decorator | the first call wrote, sent or changed something |
| [IMPURE-SCOPE-MUTATION](#impure-scope-mutation) | decorator | the function rewrites a global it reads |
| [IMPURE-SIDE-EFFECTS](#impure-side-effects) | decorator | the source has likely side effects |

### IMPURE-HANDED-CALLABLE {#impure-handed-callable}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/runtime.py:KeyBuilder._hands_a_state_changing_callable @d02c3941, cash/analysis/handed_callables.py:hands_a_state_changing_callable @4c3c2d42 -->
**What happened.** The call handed the cached function a function of yours
that changes a global or a variable it closes over when it is called. It may
be passed directly (`run(record, 3)`), as a bound method, inside a
`functools.partial`, or in a list or object that the body calls into. An
object of your class counts when a special method or property of its class
changes a global (`__exit__` of a timer the body enters with `with cm:`, a
`__getitem__` or a property the body reads). A
function whose source cannot be read counts too, because cash cannot tell what
it changes. (A change to an object you passed in is seen after the call
instead, and reported as an argument mutation under
[IMPURE-OBSERVED-EFFECTS](#impure-observed-effects).)

**Why it matters.** A hit returns the stored value without running the body,
so the function you passed would not run and its change would be skipped.
That is why such a call runs uncached, every time, as plain Python runs it.
The warning is given once per cached function. Under `strict=True` the call
raises `CashImpureFunctionError` instead.

**What to do.** Have the function you pass return its result instead of
changing state, or call it outside the cached function and pass in what it
returns.

**When it is safe to ignore.** When the change is bookkeeping that nobody
reads back. The call is still correct; it is just not cached.

**Silencing it.** A `# @cash:assume-safe` line or a `with cash.assume_safe():`
block in the body does not reach it, because the caller hands the function in.
Whole function: `@cash.cache(assume_safe=True)` caches such calls again and
skips the change on a hit. See [Silencing one code](#silencing-one-code).

### IMPURE-OBSERVED-EFFECTS {#impure-observed-effects}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

**What happened.** cash watched the first (missing) call and saw it reach
outside its return value: a `file write`, a `subprocess`, or an `argument
mutation` (an object you passed in changed). A connection to a server is a read
the key cannot see, reported as [KEY-NETWORK-READ](#key-network-read).
Each effect names the path or address and the line of your code that led to it; the one-line form shows how many there are and the first, and `f.cache_info()["warnings"]` lists them all.

<!-- claim: cash/tracking/read_classification.py:is_runtime_cache_path @303bdc01 -->
Writes into a runtime's own cache are not reported: anything under a
`__pycache__` folder or under `NUMBA_CACHE_DIR`, and bytecode and numba cache
files (`.pyc`, `.nbi`, `.nbc`, and the `.nbi.tmp.<hex>` file numba writes
first). A hit that skips one only leaves the runtime to compile again.

**Why it matters.** A hit runs none of the body, so these effects happened
once and will not happen again. If the next step relies on them, later runs
behave differently from the first.

<!-- claim: cash/decorator/store.py:ResultStore.refusal @cf16cee7, cash/decorator/purity_checks.py:PurityChecks.check_argument_mutation @a43210ad -->
<!-- claim: cash/decorator/purity_checks.py:PurityChecks.argument_identities @dda2b327, cash/_plain_data.py:identity_changed @c64ac6c0 -->
A call that changes an argument is **not stored**, so it runs every time. The
arguments are checked on every miss, whatever their size, except those
[left out of the key with `ignore=`](decorator.md#leaving-arguments-out-of-the-key),
which are not checked at all. A plain list or tuple (of numbers, strings,
dates, and lists and tuples of them) is checked by the identity of every item
it holds, at every level, which costs a fraction of a hash; when every argument
is either such a list or a value that cannot change (a number, a string, a tuple
of them), nothing is hashed a second time. Any other argument is hashed again
after the call, and one that cannot be hashed again counts as changed. So does a draw from a random
generator held by an argument that `__cash_key__` or a registered hasher keys,
which that key does not see move. Past about 50 ms of hashing,
the message may say "an argument" instead of naming which one.

**What to do.** If the effect is part of the job, split the function: cache the
computation and do the writing in an uncached caller
([example](decorator.md#side-effects)). For an argument
mutation, return a modified copy instead. If the effect is incidental (a log
file, a temp file), put `# @cash:assume-safe` on the line the message names, or
run the code in `with cash.assume_safe():`.

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.report_observed_effects @2d8839cd -->
**When it is safe to ignore.** When everything listed is bookkeeping nobody
reads back. Only the path this call took was watched, so an empty report does
not prove the function is pure.

**Silencing it.** Per line: `# @cash:assume-safe` on the line the message
names (an argument mutation names no line). Per block:
`with cash.assume_safe():` around the code that performs the effect. Whole
function: `@cash.cache(assume_safe=True)`. See [Silencing one
code](#silencing-one-code).

### IMPURE-SCOPE-MUTATION {#impure-scope-mutation}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.learn_mutating_captures @6c1ea7cb -->
**What happened.** The function reads a module global or captured variable,
and calling the function changed it. The message names the variable and the
line that changes it, which may be in a helper. A callable object that changes
what it holds when called (an instance memoising into `self`, a library
wrapper filling its cache) is not reported: cash just stops folding what it
holds, after one extra miss. Nor is a random generator the body draws from:
it stays in the key, and a hit moves it on as the computed call did.

**Why it matters.** A hit skips the change, so a counter stops counting. cash
also stops folding that variable into the key, so a change you make to it
elsewhere no longer invalidates the entry. While other threads are running,
cash cannot tell the call's change from another thread's, so that call's
result is not stored either
([STORE-STATE-CHANGED](#store-state-changed)).

**What to do.** Pass the value in as an argument and return the new value. If
updating shared state is the function's job, cache only the expensive part.

**When it is safe to ignore.** When the variable is the function's own memo,
or a log you are happy for a hit to skip.

**Silencing it.** Per line: `# @cash:assume-safe` on the line the message
names. Per block: `with cash.assume_safe():` around that line.
`assume_safe=True` on the decorator does not silence this code; filter it by
code if you must. See [Silencing one code](#silencing-one-code).

### IMPURE-SIDE-EFFECTS {#impure-side-effects}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @c395765d -->
**What happened.** Before the first call, cash read the source of the
function and its helpers and found shapes that make a cached result doubtful.
Each finding has a line number and a label:

<!-- claim: cash/decorator/purity_checks.py:PurityChecks._mutable_global_is_keyed @f2ffd435 -->
<!-- claim: cash/analysis/purity_flow.py:_FreshFlow._check_insertion @70874c70, cash/analysis/purity_flow.py:_FreshFlow._loop_targets @bd0ebd1c, cash/analysis/purity_flow.py:is_log_line @90d04d4b, cash/effects.py:is_read_only_sql @94e903d1 -->
| Label | What it flags | Reported as |
|---|---|---|
| `impure_call` | A call made for its effect: `print` to stdout, `open(..., "w")`, `os.remove`, `subprocess.run`, `requests.post`, a write method such as `df.to_csv` on an object the function did not create, an in-place method such as `rows.sort()` on one (a `sort` whose result is used, as polars' `df.sort("ts")` is, returns a new object and is not reported), or a `@stateful` function; also reading the keyboard or standard input (`input()`, `sys.stdin.read()`, `for line in sys.stdin`) | this code |
| `scope_mutation` | `global` / `nonlocal`, or assigning to another object's attribute or item. An object the function made is its own: a literal, a copy, or what a numpy, pandas or polars function returns (`load = np.bincount(...); load[c] -= d`, `counts = pd.crosstab(...); counts.columns = [...]`), except a view of an argument (`np.reshape`, `np.asarray`, `out=`) | this code |
| `discarded_call` | A call whose return value is thrown away, except on an object the function made itself (`d = deque(xs); d.popleft()`, `m = LinearRegression(); m.fit(X, y)`) | this code |
| `mutable_global` | A module global that other code in the module reassigns, and that the key does not fold | this code |
| `dynamic_pattern` | A callable picked at run time from a runtime namespace: `globals()[name]` or `vars(mod)[name]` called through (`globals()[name].fit(x)`), a local bound to one (`ns = globals(); ns[name]()`), or a table holding `eval` / `exec` | this code |
| `ambient_read` | The clock, a fresh UUID, an environment variable named at run time | [KEY-AMBIENT-READ](#key-ambient-read) |
| `network_read` | A GET request or a read-only SQL query | [KEY-NETWORK-READ](#key-network-read) |
| `untrackable_dep` | `eval` / `exec` / `compile`, a call to a function that `exec` / `eval` built from a string into a namespace of no module (`ns = {}; exec(rules_text, ns)`), `getattr(obj, name)()`, `globals()[name]()`, `globals().get(name)()` or `vars(mod)[name]()`, `operator.attrgetter(name)` or `methodcaller(name)` with a run-time name, `importlib.import_module`, `sys.modules[name]` | [KEY-UNTRACKABLE-DEP](#key-untrackable-dep) |

Log lines (`logging`, `print(..., file=sys.stderr)`) are not reported. A
table of functions is not reported either, wherever it lives: a module-level
one (`HANDLERS[kind]()`) is hashed as a global, one built in the body
(`t = {...}; t[kind]()`) names its functions there, and one an argument holds
(`router.table[key]()`) is keyed with the argument; see
[Dispatch tables](decorator-limitations.md#dispatch-tables). Nor is a lookup
whose name is written out
(`sys.modules["helper"].g`, `getattr(helper, "g")`, `attrgetter("g")(helper)`):
cash follows the code it names.

**Why it matters.** `impure_call`, `scope_mutation` and `discarded_call` mean
a hit will not repeat the effect. `mutable_global` and `dynamic_pattern` are
worse: something the result depends on is not in the key, so editing it does
not invalidate the entry.

**What to do.** Fix the findings that are real. For each line you have checked
and accept, add `# @cash:assume-safe`:

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
def build_report(rows):
    print(f"building {len(rows)} rows")  # @cash:assume-safe
    return summarise(rows)
```

On the `def` line, the comment waives findings about the whole body. For
several lines in a row, wrap them in `with cash.assume_safe():` instead. A line
that changes an argument in place is better fixed than waived: return a
modified copy.

<!-- claim: cash/decorator/reporting.py:Notices._not_shown_before @7e3fc337, cash/decorator/reporting.py:Notices.warn_once @213ba8ef -->
**When it is safe to ignore.** When every line is a `print` to stdout or a
progress bar: you only lose the printout on hits. Never ignore
`mutable_global` or `dynamic_pattern`. This warning is shown once per cache,
not once per process. An edit that only moves the lines it names (a line added
above the function) does not show it again; one that changes the findings
does. A run whose warning filters ignore it, or turn it into
an error, does not count as having shown it, so a CI job that fails on it
fails on every run.

**Silencing it.** Per line: `# @cash:assume-safe`. Per block:
`with cash.assume_safe():`. Whole function: `@cash.cache(assume_safe=True)`.
See [Silencing one code](#silencing-one-code).

## Cache keys {#key-codes}

Something the result depends on may not be in the cache key. Every code here starts `KEY-`.

| Code | Applies to | Meaning |
|---|---|---|
| [KEY-AMBIENT-READ](#key-ambient-read) | both | the body reads the clock or a fresh UUID |
| [KEY-BOOL-STATE-TOKEN](#key-bool-state-token) | decorator | `state_token()` returned a bool |
| [KEY-BUILD-FAILED](#key-build-failed) | decorator | building the key raised; the call ran uncached |
| [KEY-CALLABLE-HASHER](#key-callable-hasher) | decorator | a hasher registered for every function |
| [KEY-DEPENDS-ON-OPAQUE](#key-depends-on-opaque) | decorator | a `depends_on=` target has no source or bytecode to fingerprint |
| [KEY-DYNAMIC-DEP-FAILED](#key-dynamic-dep-failed) | decorator | a `dynamic_depends_on` resolver failed |
| [KEY-DYNAMIC-DEPENDENCY](#key-dynamic-dependency) | decorator | code in an argument picks what it calls at run time |
| [KEY-FROZEN-MUTATED](#key-frozen-mutated) | decorator | a `frozen=True` result was modified |
| [KEY-FROZEN-NO-EFFECT](#key-frozen-no-effect) | decorator | `frozen=True` cannot mark this result |
| [KEY-FUNCTION-IMPURE](#key-function-impure) | decorator | the `key=` function reads something besides its arguments |
| [KEY-FUNCTION-RAISED](#key-function-raised) | decorator | the `key=` function raised; the call ran uncached |
| [KEY-HELPERS-UNWALKABLE](#key-helpers-unwalkable) | both | the code a function runs cannot all be found; not cached |
| [KEY-INSTANCE-STATE](#key-instance-state) | decorator | a bound method's instance cannot be hashed |
| [KEY-ITERATOR-CONSUMED](#key-iterator-consumed) | decorator | building the key read an iterator argument; the call ran uncached |
| [KEY-MEMO-BEFORE-IMPORT](#key-memo-before-import) | both | a `functools` memo already held results when cash was imported |
| [KEY-LOOP-VAR-CHANGED](#key-loop-var-changed) | notebook | a loop body changes its loop variable before work cached for each pass |
| [KEY-NETWORK-READ](#key-network-read) | decorator | the body reads from a server or database |
| [KEY-OPAQUE-CALLABLE](#key-opaque-callable) | decorator | a callable's code cannot be hashed |
| [KEY-SOURCE-CHANGED](#key-source-changed) | decorator | a code file changed after import |
| [KEY-STALE-CASH-KEY](#key-stale-cash-key) | decorator | one `__cash_key__` stands for two different contents |
| [KEY-UNHASHABLE-ARG](#key-unhashable-arg) | decorator | an argument cannot be hashed; not cached |
| [KEY-UNHASHABLE-CAPTURE](#key-unhashable-capture) | decorator | a value a closure reads cannot be hashed; not cached |
| [KEY-UNHASHABLE-DEFAULT](#key-unhashable-default) | decorator | a parameter default cannot be hashed; not cached |
| [KEY-UNHASHABLE-GLOBAL](#key-unhashable-global) | decorator | a global the function reads cannot be hashed |
| [KEY-UNRESOLVED-CALL](#key-unresolved-call) | decorator | a call goes through a property or `__getattr__`; what it reaches is not in the key |
| [KEY-UNTRACKABLE-DEP](#key-untrackable-dep) | decorator | the body picks code by a run-time name; not cached |

### KEY-AMBIENT-READ {#key-ambient-read}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/effects.py:MODULE_CALLS @91df07f5, cash/analysis/purity_visitor.py:PurityVisitor.visit_Subscript @3b13759e -->
<!-- claim: cash/analysis/ambient_reads.py:ambient_call @00d7cd08, cash/effects.py:_canonical_names @e0692d46 -->
<!-- claim: cash/effects.py:CLOCK_WHEN_ARGS_OMITTED @3c78d511, cash/effects.py:_reads_clock_when_omitted @b543a896 -->
**What happened.** The function reads the clock, a fresh UUID or random
bytes from the system (`datetime.now()`, `date.today()`, `time.time()`,
`uuid.uuid4()`, `os.urandom()`, `secrets.token_hex()`, `pd.Timestamp.now()`), an environment variable whose name is only known at
run time (`os.getenv(name)`), or the whole environment (`os.environ.copy()`,
`.items()`, `dict(os.environ)`). A helper whose body only returns one of
these reads is reported where it is called, however it is called: `now()`,
`clocks.now()`, `Clock.now()` or `self.stamp()`.
The same read in code the call runs without naming it is reported too, naming
that code: a function held in a table or a list (`HANDLERS["stamp"](x)`),
passed as an argument, and a property or dunder method (`settings.stamp`,
`settings["KEY"]`, `with settings as mode:`) of an object the function reads
or is given.

<!-- claim: cash/effects.py:environment_input @cc5c0e0f, cash/decorator/environment_fold.py:EnvironmentFold.fold_environment @0398e851 -->
<!-- claim: cash/analysis/purity_flow.py:is_log_helper @6bf250bd, cash/analysis/ambient_reads.py:log_helper_names @39516f3a -->
<!-- claim: cash/analysis/ambient_reads.py:clock_helper_read @300df3f0 -->
An environment read with the name written out (`os.getenv("TENANT")`,
`"DEBUG" in os.environ`) or held in a module constant named in capitals
(`os.getenv(TENANT_VAR)`), what a standard-library helper reads for you
(`os.path.expandvars("$DATA_DIR/x")`, `os.path.expanduser`, `Path.home()`,
`tempfile.gettempdir()`, `shutil.which`), a read of the working directory
(`os.getcwd()`, `Path.cwd()`, `os.path.abspath(p)` or `Path(p).resolve()` on
a path that may be relative) and a read of the command line (`sys.argv`, or
`parser.parse_args()` with no list of its own) are not reported: their values are folded into
the key. A
reading that only goes into a log line is not reported either.

**Why it matters.** The value is an input the key cannot see. The first call's
value is stored and returned to every later call, in every later process.

<!-- claim: cash/analysis/cacheability_decision.py:_helper_ambient_reasons @5edc872e -->
In a notebook, a statement that calls a function of yours making such a read
(`s = stamp(1)`, where `stamp` returns `time.time()`), there or in a helper it
calls, is not cached: it runs every time, as it would without cash, and the
warning names the function and the read once per session. A clock that
measures time passing (`time.perf_counter()`, `time.monotonic()`) does not
count there.

**What to do.** Pass the value in, so it reaches the key:

```python
from datetime import date

import cash

# as_of is an argument, so it is in the key
@cash.cache
def report(rows, as_of):
    return sum(rows), as_of

report([1, 2, 3], as_of=date.today())
```

For an environment variable, write its name out.

**When it is safe to ignore.** When freezing the value is the point, such as
a timestamp of when the result was computed. Like
[IMPURE-SIDE-EFFECTS](#impure-side-effects), it is shown once per cache, and
again only when an edit changes the findings.

**Silencing it.** Per line: `# @cash:assume-safe`. Per block:
`with cash.assume_safe():`. Whole function: `@cash.cache(assume_safe=True)`.
See [Silencing one code](#silencing-one-code).

### KEY-BOOL-STATE-TOKEN {#key-bool-state-token}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/data_source.py:state_token_of @60e6073c -->
**What happened.** Your `DataSource.state_token()` returned `True` or
`False`.

**Why it matters.** A bool cannot say "the data is different now", so entries
never invalidate when the source changes.

**What to do.** Return something that changes with the data: a version, a
digest, an mtime, an ETag. See [Data sources](api/data_sources.md).

**When it is safe to ignore.** Never, if the source can change while your
program runs.

### KEY-BUILD-FAILED {#key-build-failed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/runtime.py:KeyBuilder.resolve @a39468e5 -->
**What happened.** Something raised while cash built the cache key. The
message names the exception and, when it can, the argument type. The call ran
and returned its real result, uncached.

**Why it matters.** That call is not cached. Nothing stale can be served.

**What to do.** If the exception points at a type of yours, register a hasher
for it with `cash.register_hasher`. If it does not, please report it as a bug
with the traceback.

**When it is safe to ignore.** When the function is cheap enough to run every
time.

### KEY-CALLABLE-HASHER {#key-callable-hasher}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/core.py:Cash.register_hasher @f8a61573 -->
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

### KEY-DEPENDS-ON-OPAQUE {#key-depends-on-opaque}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/registry.py:FunctionRegistry._register_declared_callable_dep @cdef7cb8, cash/decorator/registry.py:warn_inert_dependency @6242cef6 -->
**What happened.** A callable in `depends_on=` has no source and no Python
bytecode (a builtin, a NumPy ufunc, or an installed compiled extension). cash
can key it only by its name, so the declaration does next to nothing. An
extension built inside your project (`build_ext --inplace`, an editable
install) does not warn: it is keyed by the content of its built file.

**Why it matters.** Changing that callable will not invalidate the entry.

**What to do.** For an extension you build and install yourself, pass its
version as an argument, or use a `DataSource` whose token is the build id.

**When it is safe to ignore.** Usually: a stdlib or pinned third-party builtin,
such as `depends_on=[math.sqrt]`, will not change between runs.

### KEY-DYNAMIC-DEP-FAILED {#key-dynamic-dep-failed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/registry.py:resolve_dynamic_dependencies @379962b9 -->
**What happened.** A `dynamic_depends_on=` resolver raised, or returned
something that is not a `DataSource`, a list of them, or `None`. The call ran
uncached, and so did any cached function that called it: its result depends
on sources nothing recorded.

**Why it matters.** Every such call recomputes. Nothing stale is served.

**What to do.** Fix the resolver. It receives the same arguments as your
function. Wrap a raw value in a `DataSource`, and return `None` for calls with
no dependency.

**When it is safe to ignore.** When the function is cheap. If the dependency
no longer changes, remove `dynamic_depends_on=`.

### KEY-DYNAMIC-DEPENDENCY {#key-dynamic-dependency}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/code_args.py:CodeArgs._warn_untrackable_in_carrier_once @34d65ca5 -->
**What happened.** An object you passed to a cached function carries code, and
that code picks what it calls at run time: `getattr(module, name)()` with
`name` in a variable, `eval`, a dynamic import. The message names the method,
line and argument.

**Why it matters.** cash cannot follow the call, so editing the function it
lands on will not invalidate the entry. (The same line in the cached
function's own body runs uncached instead:
[KEY-UNTRACKABLE-DEP](#key-untrackable-dep).)

**What to do.** Call the function by name where you can. Otherwise list the
candidates with `depends_on=[...]`.

**When it is safe to ignore.** When the functions it can pick never change.

**Silencing it.** Per line: `# @cash:assume-safe` on the line the message
names. Per block: `with cash.assume_safe():` around that line.
`assume_safe=True` on the decorator does not silence this code, because the
line is in an argument's code, not the function's. See [Silencing one
code](#silencing-one-code).

### KEY-FROZEN-MUTATED {#key-frozen-mutated}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/frozen.py:FrozenResults.audit @2571147a -->
**What happened.** A function marked `@cash.cache(frozen=True)` promised its
result is not modified, and a later check found one of its results modified.

**Why it matters.** Until the check, cached functions receiving that object
could be served results computed for the unmodified object. From the check on,
it is keyed by content, so later calls are correct. Checks run at the 8th use
and every 64th after that, or every use under `CASH_DEBUG=1`.

**What to do.** Find what modifies it (often `model.fit(...)`), then either
remove `frozen=True` or modify a copy.

**When it is safe to ignore.** Never.

**Silencing it.** No waiver silences this code: not `# @cash:assume-safe`,
`with cash.assume_safe():` or `assume_safe=True`. See [Silencing one
code](#silencing-one-code).

### KEY-FROZEN-NO-EFFECT {#key-frozen-no-effect}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

**What happened.** A `frozen=True` function returned a value cash cannot mark:
a `set`, or an object that takes no new attributes (`__slots__`, many C
types).

**Why it matters.** `frozen=True` does nothing for this result; cached
functions receiving it still hash it in full.

**What to do.** Return a numpy array, a dataframe, a list, tuple or dict, or an
object that takes attributes. Or remove `frozen=True`.

**When it is safe to ignore.** When the result is small.

### KEY-FUNCTION-IMPURE {#key-function-impure}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/arg_key.py:key_function_impurities @784f8639, cash/decorator/runtime.py:KeyBuilder.key_arguments @316e84b5 -->
**What happened.** The function passed as `key=` reads something besides its
arguments: a file, the clock, a random number, the network, a database, a
subprocess or the environment. Reading its code finds most of these when the
function is decorated; a file read through a reader cash does not name is
caught on the first call.

**Why it matters.** The key decides which stored result a call gets. A key
that depends on something else can give equal arguments different keys (no
hits), or give two calls whose results differ the same key (a wrong result
served).

**What to do.** Compute the key from the arguments alone. Pass anything else
it needs as an argument of the cached function, or let the cached function
read it, where cash tracks it.

**When it is safe to ignore.** Never, unless what it reads cannot change while
the cache is in use.

**Silencing it.** Whole function: `@cash.cache(assume_safe=True)`.
`strict=True` raises `CashImpureFunctionError` instead. See [Silencing one
code](#silencing-one-code).

### KEY-FUNCTION-RAISED {#key-function-raised}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/arg_key.py:keyed_arguments @d9fd1022 -->
**What happened.** The function passed as `key=` raised when called with the
call's arguments. The call ran and returned its real result, uncached.

**Why it matters.** Every call that makes the key function raise runs the
body. The warning appears once per function, not once per call.

**What to do.** Read the exception and make the key function handle every
call the cached function accepts, defaults included.

**When it is safe to ignore.** When those calls are cheap to recompute.

### KEY-HELPERS-UNWALKABLE {#key-helpers-unwalkable}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

**What happened.** cash keys a function by the code of every helper it can
reach, however deep, and so the code an argument's code reaches and every
function a decorated callable wraps. Here that code could not all be found.
Either it never ran out: something hands out a new function each time it is
read (a module `__getattr__` that builds one, for example), so the walk passed
thousands of them. Or finding it failed: reading the function's helpers raised,
or an object it runs could not be looked into (the message names the error).
The function ran uncached.

In a notebook, cash follows every function a statement calls, however deep, to
find one that writes a file. When that walk passes thousands of functions, cash
cannot tell, so the statement runs every time, as if it wrote a file.

**Why it matters.** That function is never cached. Nothing stale can be served:
cash does not cache under a key that leaves code out.

**What to do.** Stop creating the helper on every read (build it once and keep
it). If the message names an error that is not your code's, report it as a
bug with that error.

**When it is safe to ignore.** When the function is cheap enough to run every
time.

### KEY-INSTANCE-STATE {#key-instance-state}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/closure_fold.py:ClosureFold.fold_bound_self @06de34b5 -->
**What happened.** You cached a bound method (`c.cache(obj.method)`) and the
instance could not be hashed, so cash keyed on the object's in-memory identity.

**Why it matters.** Identity means nothing in a new process, so every new run
recomputes.

**What to do.** Register a hasher for the class that returns what the result
depends on:

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
cash.register_hasher(Config, lambda c: c.fingerprint)
```

**When it is safe to ignore.** In a long-running process with one instance.

### KEY-ITERATOR-CONSUMED {#key-iterator-consumed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/runtime.py:KeyBuilder._check_iterators_unread @dd5b4fdd -->
**What happened.** An argument is an iterator (a generator, `map(...)`,
`iter(rows)`), and building the key read it: a `key=` function such as
`key=lambda rows: tuple(rows)`, or a hasher registered for its type. An
iterator can be read once, so the body got what was left of it. The call
ran uncached. When cash cannot tell how far an iterator was read and a
`key=` function or a registered hasher could have read it, it says so too.

**Why it matters.** The body's result was computed from an emptied
iterator. Stored, it would be served to every later call with the same
key, a list of the same rows included.

**What to do.** Pass a list or a tuple instead of the iterator, or key the
call by something that does not read it.

**When it is safe to ignore.** When the message says the iterator *may* have
been read and you know the key did not read it. When it says it was read,
the result of that call came from an emptied iterator.

### KEY-MEMO-BEFORE-IMPORT {#key-memo-before-import}

<span class="md-tag cash-warning-path">both</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/tracking/memos_before_import.py:find_memos_filled_before_import @e09c920f, cash/tracking/memos_before_import.py:warn_memos_filled_before_import @f66dd90f -->
**What happened.** When `import cash` ran, a function of yours memoised
with `functools.lru_cache` or `functools.cache` already held results: it
was imported and called ahead of cash. The message names the memo (up to
three of them) and points at its definition. It is shown once, at the
import. Only module-level memos in your own modules are looked at.

**Why it matters.** cash watches file reads from `import cash` on. A file
the memo read before that was read where cash could not see it, and later
calls are served from the memo and read nothing. A cached function that
gets its data from the memo records no file, so an edit to that file
returns the old result with no other warning.

**What to do.** Import `cash` before the modules that call the memo, or
pass the file's path as an argument (`load_cfg("config.yaml")`) from inside
the cached function.

**When it is safe to ignore.** When the memo reads no file (it computes
from its arguments), or no cached function uses it.

### KEY-LOOP-VAR-CHANGED {#key-loop-var-changed}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/control_structures/loop_target_change.py:target_changed_before_cached_work @1d8505cf, cash/notebook/control_structures/for_handler.py:ForLoopHandler._warn_target_changed @e307b4b5 -->
**What happened.** A `for` loop that cash runs pass by pass changes its loop
variable in place (`q.append(...)`, `q[0] = ...`, `q += ...`, `q.n = ...`)
before a statement of the same pass that cash caches: a call of yours or of a
library, or a binding that reads the variable. The message names the cell, the
line that changes the variable and the line of the cached work. It is shown
once per loop, when the loop first runs, and read from the loop's text alone.

**Why it matters.** Each pass is keyed on the loop variable as the `for`
binds it. The change comes after that, so it is not in the key: two passes
whose values were equal when bound share one cached result, and the second
gets the first's answer, on the first Run All too.

**What to do.** Make the value that tells the passes apart part of the `for`
target (`for i, q in enumerate(items):`): only names the `for` binds reach
the key of each pass. If there is no such value, put `# @cash:no-cache` on a
line of its own above the statement the message names.

**When it is safe to ignore.** When the passes never start out equal (each
`q` differs when bound), or the cached work does not depend on the change.

### KEY-NETWORK-READ {#key-network-read}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/purity_checks.py:PurityChecks.surface_purity @c395765d, cash/analysis/purity_policy.py:DECORATOR_POLICY @44b8bc03, cash/effects.py:MODULE_CALLS @91df07f5 -->
<!-- claim: cash/analysis/purity_visitor.py:_opens_tracked_database @0e105084 -->
**What happened.** The function fetches from a server (`requests.get`,
`httpx.get`, `urlopen(url)`) or queries a database (`cur.execute("SELECT
...")`, `pd.read_sql`). A query over a SQLite file the function opens itself
is not reported: that file is tracked. Writes (`requests.post`, `INSERT`) are
[IMPURE-SIDE-EFFECTS](#impure-side-effects) instead.

<!-- claim: cash/decorator/purity_checks.py:PurityChecks._report_observed_network @23e868ea -->
A fetch the source does not name (`requests.Session().get(url)`, `from
requests import get`, a client library) is caught when the first call runs:
the message then lists the `socket connect` and the line of yours that led to
it. A remote file a reader opened by URL (`pd.read_csv("s3://...")`) is not
reported: it is tracked by its ETag.

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
you pass, such as a fetch by pinned version.

**Silencing it.** Per line: `# @cash:assume-safe`. Per block:
`with cash.assume_safe():`. Whole function: `@cash.cache(assume_safe=True)`,
or a `ttl=`. See [Silencing one
code](#silencing-one-code).

### KEY-OPAQUE-CALLABLE {#key-opaque-callable}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/code_args.py:is_user_code_carrier @16845be0, cash/install_paths.py:is_user_module @a54a6d7a, cash/decorator/user_code.py:is_user_code_object @07d4f95a -->
**What happened.** A function, class or object from your own code reached a
cached call (as an argument or a default), and cash could not hash its code.
The typical case is a class of yours whose behaviour comes from a compiled
function, such as `__call__ = staticmethod(abs)`. Library callables do not
trigger it, and neither do compiled library objects and their methods
(`re.compile(p).match`, a lock, a stream), wherever they are reached.

**Why it matters.** Editing that code will not invalidate the entry.

**What to do.** If the result depends on it, name the wrapped function with
`depends_on=[my_fn]`. If it does not, mark the type with
`cash.opaque(TheType)` (or `@cash.opaque` on a class you own), which silences
this code for every object of that type.

**When it is safe to ignore.** When the compiled code it runs cannot change
under you, as with a builtin such as `abs`.

**Silencing it.** No waiver silences this code: not `# @cash:assume-safe`,
`with cash.assume_safe():` or `assume_safe=True`. `cash.opaque(TheType)`
does. See [Silencing one
code](#silencing-one-code).

### KEY-SOURCE-CHANGED {#key-source-changed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/loaded_code.py:loaded_code_matches_disk @f7d13d9d, cash/decorator/function_identity.py:warn_source_changed_since_load @4f566032 -->
<!-- claim: cash/loaded_code.py:_pyc_proves_unchanged @42e99e55 -->
**What happened.** A file holding a cached function or a helper was edited
after this process imported it, or the import loaded bytecode compiled from an
earlier save of it. The process runs the old code.

<!-- claim: cash/decorator/function_identity.py:OwnSourcePins.pin_own_source @fb1cadae -->
**Why it matters.** cash keys that code by what is actually running, so
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

### KEY-STALE-CASH-KEY {#key-stale-cash-key}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/cash_key.py:KeyCheck._check @3770f723, cash/decorator/cash_key.py:content_digest @85326417 -->
**What happened.** A class's
[`__cash_key__`](tutorials/feature-guides/custom-hashers.md#cash-key)
returned the same key for two objects holding different data. cash reads an
object's content in the background the first time it is keyed by
`__cash_key__` in a process, and compares it with what the same key stood for
before, in this run or an earlier one. The warning points at the
`__cash_key__` method.

**Why it matters.** The key is the object's whole identity, so results
computed from the first object's data are served for the second. That is a
wrong answer, not a slow one.

**What to do.** Make `__cash_key__` return something that changes whenever
the data does: a version you bump, a content id, or the source file's path
with its modification time.

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
class Dataset:
    def __cash_key__(self):
        return (self.path, os.path.getmtime(self.path))
```

Then clear the results stored under the old key with `f.cache_clear()` on the
cached functions that took the object.

<!-- claim: cash/decorator/cash_key.py:content_digest @85326417 -->
**When it is safe to ignore.** When the difference is state that never
changes a result: a lazily filled memo, a load timestamp, a handle. cash
compares every attribute it can pickle, so such an attribute set before the
first cached call differs between runs. Turn the check off with
`check_cash_keys=False` if that is the case for your class.

### KEY-UNHASHABLE-ARG {#key-unhashable-arg}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/runtime.py:KeyBuilder.resolve @a39468e5 -->
**What happened.** An argument could not be hashed, so no key could be built.
The message names the type, says the value is nested in a container, or says
it is nested too deeply to key: deeper than pickle follows, such as a long
linked list. A value a cached `functools.partial` binds counts as an argument
too. The call ran uncached.

**Why it matters.** Every call with that argument recomputes. Nothing stale
is served.

**What to do.** For a class of yours, give it a
[`__cash_key__`](tutorials/feature-guides/custom-hashers.md#cash-key) method
that returns what identifies an instance. If the message names
`__cash_key__()` itself, that method raised or returned something cash cannot
hash; make it return a string, a number or a tuple of them. For a type you
don't own, register a hasher, or pass something hashable in its place (a
connection string, not a connection):

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
class DatabaseSession:
    def __cash_key__(self):
        return self.database_url

# or, for a class you can't change:
cash.register_hasher(DatabaseSession, lambda s: s.database_url)
```

For a `functools.partial` of a closure or `lambda`, or a library's closure,
pass a module-level function and give the captured values as arguments
([Code you pass as an argument](decorator-limitations.md#code-you-pass-as-an-argument)).

**When it is safe to ignore.** When you do not need that call cached.

### KEY-UNHASHABLE-CAPTURE {#key-unhashable-capture}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/closure_fold.py:unhashable_capture @b432a5b6, cash/decorator/closure_fold.py:ClosureFold.key_function_part @e027b066 -->
**What happened.** The function, a helper it calls, or its `key=` function
is a closure that reads a captured variable whose value could not be hashed
-- an object holding a lock, a socket or a file handle -- or the `key=`
function is a callable object whose state could not be hashed. The call was
not cached. The message names the variable or the object's type.

**Why it matters.** The function is not cached while it captures that value.
Keyed without it, a change to what the value carries (a setting on a config
object) would have served the old result.

**What to do.** Pass what the function needs as an argument, capture only the
plain values it reads (`weight = settings.weight` in the factory), or register
a hasher for the type.

**When it is safe to ignore.** When you do not need that call cached.

### KEY-UNHASHABLE-DEFAULT {#key-unhashable-default}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/closure_fold.py:ClosureFold._defaults_unhashable @0a6de304, cash/decorator/closure_fold.py:HelperIdentity.identity @e54e2652, cash/decorator/code_surface.py:CodeSurface._unpicklable_identity @70e59364 -->
**What happened.** A parameter default of the function, of a helper it
calls, or of a function or class passed to it, could not be hashed, so the
call was not cached. The message names the type.

**Why it matters.** The function is not cached for any caller, including
those that pass the argument.

**What to do.** Move the value out of the signature (build it in the body, or
require it), or register a hasher for its type. `def load(session=Session())`
is the classic case. When the parameter does not change the result (a
connection, a lock, a logger), leave it out of the key with `ignore=` or
`cash.Ignore`: its default is left out with it, as are all defaults under
`key=`.

**When it is safe to ignore.** When you do not need the function cached.

### KEY-UNHASHABLE-GLOBAL {#key-unhashable-global}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/decorator/globals_fold.py:GlobalsFold.fold_read_globals @8944d903 -->
**What happened.** The function (or a helper) reads a module global that
could not be hashed, so it was left out of the key.

**Why it matters.** Changing that global will not invalidate the entry.

**What to do.** Have the function read what the result depends on (a
connection string, not a connection), pass the value as an argument, or
register a hasher for its type.

**When it is safe to ignore.** When the global is set once at import and never
changed: a client, a logger, a compiled pattern.

**Silencing it.** Per line: `# @cash:assume-safe` on the line that reads the
global. Per block: `with cash.assume_safe():` around that line.
`assume_safe=True` on the decorator does not silence this code; filter it by
code if you must. See [Silencing one code](#silencing-one-code).

### KEY-UNRESOLVED-CALL {#key-unresolved-call}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashImpurityWarning</span>

<!-- claim: cash/analysis/code_analyzer.py:CodeAnalyzer.find_called_functions_and_gaps @f44cc346 -->
**What happened.** The function calls something through an object of your own
code whose attribute is a property, or whose class answers missing names with
`__getattr__`: `api.inner(x)`. cash never runs that code to find what it
returns. It reads the getter's body instead, and keys every cached function the
getter names, or failing that every cached function called `inner`. Here it
found none, or only a plain function, so the function the call reaches is not
in the key.

**Why it matters.** Editing that function does not invalidate the cached
results: the old result is served.

**What to do.** Call the function by its own name (`impl.inner(x)`), or name
it on the decorator: `@cash.cache(depends_on=[impl.inner])`.

**When it is safe to ignore.** When the property hands out library code, or
code you do not edit while results are cached.

**Silencing it.** No waiver silences this code: not `# @cash:assume-safe`,
not `with cash.assume_safe():`, not `assume_safe=True`. Filter it by code if
you must. See [Silencing one code](#silencing-one-code).

### KEY-UNTRACKABLE-DEP {#key-untrackable-dep}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/purity_checks.py:PurityChecks._warn_untrackable @90bfa6af, cash/decorator/registry.py:FunctionRegistry.refresh_helper_bindings @006fceab -->
**What happened.** The function, or a helper it calls, picks the code it runs
by a name known only at run time: `getattr(module, name)()`,
`globals()[name]()`, `vars(module)[name]()`,
`importlib.import_module(name)`, `sys.modules[name]`,
`operator.attrgetter(name)`, `eval`, `exec` or `compile`. The message names
the line. The function runs uncached, and so does every cached function that
calls it.

**Why it matters.** cash cannot tell which code that name lands on, so an edit
to it, or to a global it reads, would not change the key. Nothing stale is served: the call is not
cached.

**What to do.** Call the function by name, or pass the function itself in as
an argument, where its code reaches the key. A module-level table of
functions (`STRATEGIES[name](data)`) is keyed too; see
[Dispatch tables](decorator-limitations.md#dispatch-tables).

**When it is safe to ignore.** When the function is cheap enough to run every
time.

**Silencing it.** Per line: `# @cash:assume-safe` on the line the message
names. Per block: `with cash.assume_safe():`. Whole function:
`@cash.cache(assume_safe=True)`. Each of these caches the function anyway, and
an edit to the code the name lands on is then not noticed. `strict=True`
raises `CashImpureFunctionError` instead. See [Silencing one
code](#silencing-one-code).

## Notebook {#notebook-codes}

Notebook-wide machinery rather than one statement. Every code here starts `NOTEBOOK-`.

| Code | Applies to | Meaning |
|---|---|---|
| [NOTEBOOK-ANALYSIS-BUDGET](#notebook-analysis-budget) | notebook | working out how to key a statement took longer than caching it could save; it ran uncached |
| [NOTEBOOK-ANALYSIS-FAILED](#notebook-analysis-failed) | notebook | a safety check raised; the statement ran uncached |
| [NOTEBOOK-BAILOUT](#notebook-bailout) | notebook | an internal error; the cell ran uncached |
| [NOTEBOOK-CELL-SYNTAX](#notebook-cell-syntax) | notebook | an earlier cell does not parse |
| [NOTEBOOK-CHANGED-BELOW](#notebook-changed-below) | notebook | a cell re-run on its own reads a value rebuilt without the in-place change a cell below made |
| [NOTEBOOK-CLASS-STATE-KEPT](#notebook-class-state-kept) | notebook | a class's own state a cell changes is kept as it is for a re-run, as another cell changes it too |
| [NOTEBOOK-FAILED-CELL](#notebook-failed-cell) | notebook | a cell raised part way, and a name it changed is not what a plain run leaves |
| [NOTEBOOK-FILE-ROUNDTRIP](#notebook-file-roundtrip) | notebook | a file written in this session is read back, so the read reruns after every write |
| [NOTEBOOK-FORWARD-READ](#notebook-forward-read) | notebook | the cell reads a name only a cell below binds, so it runs uncached |
| [NOTEBOOK-MAGIC-STALE](#notebook-magic-stale) | notebook | a name a shell command or magic bound is kept, though it would bind it differently now |
| [NOTEBOOK-NO-LIVE-READER](#notebook-no-live-reader) | notebook | cells above are read from the saved file, so an unsaved edit there is not seen |
| [NOTEBOOK-NOT-FOUND](#notebook-not-found) | notebook | the notebook's cells cannot be read; cells above the one you run are not re-checked |
| [NOTEBOOK-RELOAD-STATE](#notebook-reload-state) | notebook | a reload or restart dropped state cells set on a module, and cash cannot rebuild it |
| [NOTEBOOK-SAVEFIG-SKIP](#notebook-savefig-skip) | notebook | a `plt.savefig` was not re-run |
| [NOTEBOOK-SHARE-UNCHECKED](#notebook-share-unchecked) | notebook | not every variable sharing an object with a statement's outputs was found in time |
| [NOTEBOOK-SHARED-KEPT](#notebook-shared-kept) | notebook | a value a variable below shares is kept as it is for a cell re-run on its own |
| [NOTEBOOK-THREAD-ALIVE](#notebook-thread-alive) | notebook | a thread the cell started still runs after the cell ended |

### NOTEBOOK-ANALYSIS-BUDGET {#notebook-analysis-budget}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/analysis/analysis_budget.py:analysis_budget_seconds @0a25bb39, cash/notebook/statement/processor.py:StatementProcessor._tell_past_budget @aa3c92a2, cash/notebook/statement/processor.py:StatementProcessor._settle_budget @ffbe1a5a -->
**What happened.** To key a statement, cash reads what it can run: the
functions and methods of the objects it hands to a call, however deep in
them, and, after it ran, which other variables share its outputs. Over a big
graph of your own objects (20,000 agents each holding a strategy function)
that reading took longer than the statement's budget: four times what the
statement took the last time it ran, at least half a second. cash stopped
reading. The message names how long the statement ran. It is shown once per
statement, and only for a statement that took a tenth of a second or more:
for a quicker one, caching would not have mattered.

**Why it matters.** The statement runs every time and is not stored. What it
binds is keyed by its value, as a `# @cash:no-cache` statement's is: a cell
below that reads it is served from the cache only when the value is the
same. What its calls were handed could change any local module's state or
the process's working directory or environment; cash takes the modules as
changed and watches the process. The result is correct.

**What to do.** Nothing, usually: the statement measures itself as it
runs, and its next budget follows. To cache it, hand the call the function
and the values it needs rather than an object graph that holds them, or
move the work into a function decorated with `@cash.cache`.

**When it is safe to ignore.** When the statement is quick enough to run
every time.

### NOTEBOOK-ANALYSIS-FAILED {#notebook-analysis-failed}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/analysis/cacheability_decision.py:analysis_failed @3e0e56fc -->
**What happened.** A check that decides whether a statement may be cached
raised an error: the scan for calls it must not cache, or the analysis of the
functions it calls and what they write. The message names the check and the
exception.

**Why it matters.** cash cannot tell whether a cache hit would be safe, so the
statement runs every time and is not stored. The result is correct.

**What to do.** Nothing in your code needs to change. Please report the
exception as a bug. The warning is shown once per check and error type.

**When it is safe to ignore.** When the statement is cheap to run.

### NOTEBOOK-BAILOUT {#notebook-bailout}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

**What happened.** cash hit an internal error while processing the cell,
stepped aside, and let IPython run it normally. The message names the
exception.

**Why it matters.** The result is correct, but nothing from this cell was
stored, and cells below lose the lineage they would have inherited.

**What to do.** Re-running is safe. Please report the exception as a bug.
Restarting the kernel often clears it.

**When it is safe to ignore.** When it happens once. If it repeats on the same
cell, that cell is never cached.

### NOTEBOOK-CELL-SYNTAX {#notebook-cell-syntax}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashUpstreamSyntaxWarning</span>

<!-- claim: cash/notebook/upstream/notebook_vetting.py:NotebookVetter._warn_broken_upstream_cells @9f38dbd3 -->
**What happened.** An earlier cell in the notebook has a syntax error. The
message gives its number (counting code cells from the top, not the `[7]`
execution count) and quotes its first line.

**Why it matters.** cash skips that cell, so cells that use its output are no
longer invalidated when it changes.

**What to do.** Fix the cell, then re-run it and the cells below that use it.
If it is not code, delete it or make it a markdown cell.

**When it is safe to ignore.** When nothing below uses that cell.

### NOTEBOOK-CHANGED-BELOW {#notebook-changed-below}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/upstream/mismatch_classifier.py:MismatchClassifier._reject_read_only_input @d536a0ab, cash/notebook/upstream/changed_below.py:warn_changed_below @bbf25145 -->
**What happened.** You re-ran a cell on its own, and a cell below it had
changed one of its inputs in place (`s.iloc[0] = 1e9`, `rows.append(r)`).
cash gives the re-run the input as the cells above leave it, as a run from the
top does: it rebuilt the value, so the cell computed without the change below,
and the variable no longer holds that change. The message names the variable,
the cell below and its statement. It is shown once per cell, variable and
statement.

**Why it matters.** Plain Jupyter would run the cell on the changed value.
Here the result is the one a run from the top gives, and the change the cell
below made is gone until that cell runs again, so a cell further down that
expects it reads the value without it.

**What to do.** Run the cell below again to apply its change. For the cell
you re-ran to see the change, move the change into a cell above it.

**When it is safe to ignore.** When you want the value as a run from the top
leaves it, and will run the cells below again.

### NOTEBOOK-CLASS-STATE-KEPT {#notebook-class-state-kept}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/upstream/stale_values.py:StaleValueGuard._warn_class_state_kept @8d9db6f6, cash/analysis/mutation_effects.py:_object_protocol_effects @33c97c7b -->
**What happened.** You re-ran on its own a cell whose constructor or method
calls change an attribute of the class itself (`Widget.count += 1` in
`__init__`, a registry the class appends to), and another cell changes that
class too. cash normally re-runs the class definition before such a re-run,
so the change is not applied twice. Here that would also undo the other
cell's change, so cash kept the class as it is: the cell's change applies
again on top of the one it made last time, as in a plain re-run. The message
names the cell and the class. It is shown once per cell and class.

**Why it matters.** The class holds both runs' changes (a counter counts the
cell's objects twice), which no run of the notebook from the top gives.

**What to do.** For the value a run from the top gives, run the notebook from
the top, or from the cell that defines the class.

**When it is safe to ignore.** When the class's state is meant to accumulate,
or applying the change twice does not change the result (adding to a set).

### NOTEBOOK-FAILED-CELL {#notebook-failed-cell}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/upstream/simulator.py:NotebookSimulator._warn_failed_run_left @961fa468, cash/notebook/ipython/cell_executor.py:_owed_by_skips @fb2193e4 -->
**What happened.** A cell raised part way through, and a name it changed is
not what a run of the cells from the top gives. The message names the cell
and the variables. It comes in two cases.

A cell above raised part way, say in a loop filling `tot` or in a function
it calls that fills `tot` (`proc.fill(tot, data)`), and the cell you
ran reads `tot`, or something built from it since (`final = {... tot ...}`).
No run of the cells gives those values: the lines before the error leave
`tot` empty, and the whole cell raises. So cash keeps them as they are, as a
plain kernel does: it does not run the lines before the error again, re-run
the cells between or restore them from the cache. A name built from them that
is not in memory (after a restart) stays undefined.

Or the cell you ran raised. Running it, cash had skipped or restored some of
its lines up front, by the values the whole cell ends with. When a line
before the error sets a name that a line at or after it sets again, cash runs
that earlier line once more after the error, with the lines it reads from, so
the name holds what a plain run leaves. Here it could not: the line is a loop
or branch, or it reads a value from before the cell that a later line of the
cell replaced. Those names hold the values the end of the cell gives them
until something rebuilds them; a cell below that reads them rebuilds them
first.

**Why it matters.** The cell you run sees values that a run from the top
would not give it. They are the values the kernel holds, which is what you
see without cash, but a result built from them is not the notebook's result.

**What to do.** Fix the error and run the cell that raised again, then the
cells below it.

**When it is safe to ignore.** While you are still looking into the error.

### NOTEBOOK-FILE-ROUNDTRIP {#notebook-file-roundtrip}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/notebook/file_round_trips.py:FileRoundTrips._find_round_trips @4bda0d68, cash/notebook/file_round_trips.py:MIN_COST_S == 1.0 -->
**What happened.** A statement read a file that another statement wrote
earlier in this kernel session, and it ran again because that file changed
since it last read it. One cell writes `report.xlsx`, a later one opens it
again, formats it, saves it and reads it back to check it. The message names
the file, the statement that wrote it and the one that read it back. It is
shown once per file per session, and only when the read and the statements
that ran on what it read took a second or more.

**Why it matters.** Every write gives the file new bytes, even when the data
is the same (an Excel file stores the time it was saved), so the statement
that reads it runs again after every write, and so does everything computed
from what it read. Nothing from the read down can come from the cache.

**What to do.** Keep the data in a variable (a DataFrame) and pass that to
the next step instead of reading the file back; write the file once, as the
last step. For a formatting pass, format while writing (through the
`ExcelWriter`'s workbook) instead of loading the saved file again.

**When it is safe to ignore.** When reading the file back is the point,
such as a check that the saved file opens. A file an earlier session wrote,
such as a checkpoint read after a restart, never gives this warning.

### NOTEBOOK-FORWARD-READ {#notebook-forward-read}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/notebook/upstream/notebook_vetting.py:NotebookVetter._refuse_forward_references @a173133d, cash/notebook/ipython/upstream_phase.py:UpstreamResolution._step_aside @90bc587e -->
**What happened.** The cell reads, at module level, a name that only a later
cell binds (`y = x + 1` above `x = 10`). It works now because that cell already
ran, so the cell runs as plain Jupyter runs it, on the value in memory, but
uncached. The message names the variable and the cell that binds it. A name
used inside a `def` does not count; a decorator, default argument or base class
does.

**Why it matters.** A run from the top raises `NameError` at this cell, and a
cached result would stand for an order of cells the notebook does not have.

**What to do.** Move the binding above this cell, or move this cell below it.

**When it is safe to ignore.** Never for long: the notebook does not run from
the top.

### NOTEBOOK-MAGIC-STALE {#notebook-magic-stale}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/upstream/simulator.py:NotebookSimulator._warn_stale_magic @777a3665 -->
**What happened.** The cell reads a name that a shell command or a line magic
above binds or changes (`files = !ls`, `res["out"] = !cmd`, `%sx cmd`), and
that command has not run since something it reads changed, or the name is not
in memory at all (after a restart). The message names the variable and quotes
the command's line. A `%time`, `%timeit` or `%prun` line is not one of them:
it runs Python, and cash runs it again with the Python above it, as a run from
the top does.

**Why it matters.** cash rebuilds what a cell above would compute, but it never
runs a shell command or any other magic for you, and rebuilding the name from
the Python before it would drop what the command did. So the cell runs on the
value the name holds now, which is not the one a run from the top would give
it; a name that is gone raises NameError.

**What to do.** Re-run the cell the message names, then this cell.

**When it is safe to ignore.** When the change above does not affect what the
command computes.

### NOTEBOOK-NO-LIVE-READER {#notebook-no-live-reader}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/server_discovery.py:warn_no_live_reader_once @675af5d0 -->
**What happened.** Before a cell runs, cash checks the cells above it. It
read them from the saved notebook file, because no live reader sends it the
cells as they are in the editor: neither cash's JupyterLab extension, nor VS
Code's backup, nor Colab. That is the case in the classic Notebook, PyCharm,
or a JupyterLab whose server does not have the extension installed.

**Why it matters.** An edit you have not saved is checked as the code saved
before it. Run a cell below an unsaved edit and cash answers for the old
code, the way it would if you had not edited at all.

**What to do.** Save (`Ctrl+S`) before running a cell below an edit. Or use
JupyterLab with the `cash-live-cells` extension (installed with `pip install
cash-lib` in the environment that runs JupyterLab), VS Code or Colab.

**When it is safe to ignore.** When you save before running, or run cells
top to bottom. It is shown once per session, for the first cell you run after
a pause; a Run All, papermill or nbconvert run does not show it.

### NOTEBOOK-NOT-FOUND {#notebook-not-found}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/server_discovery.py:warn_notebook_not_found_once @b331a2de -->
**What happened.** cash could not read this notebook's cells, so before a
cell runs it does not re-check the cells above it. Each statement is still
cached and restored by its own code and inputs.

**Why it matters.** If you edit a cell and then run a cell below it without
re-running the edited one, cash does not run the edited cell first. Re-running
the edited cell, and then the cells after it, recomputes what the edit changed,
as without the warning.

**What to do.** Nothing under papermill, nbconvert or CI: there is no Jupyter
server to ask. In JupyterLab or VS Code, restart the kernel; if it persists,
close and reopen the notebook.

**When it is safe to ignore.** In a run that executes the notebook top to
bottom once. It is shown once per session.

### NOTEBOOK-RELOAD-STATE {#notebook-reload-state}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/module_invalidator.py:ModuleInvalidator._schedule_state_rebuild @c5b71672, cash/notebook/upstream/checker.py:_warn_state_not_rebuilt @a1d9ad27, cash/notebook/upstream/checker.py:_warn_unheld_state @93fa96ae -->
**What happened.** You edited one of your modules, so cash reloaded it, or
you restarted the kernel. Either way the module holds its file's values, not
what cells set on it (`mylib.K = 7`, `mylib.CONFIG["k"] = 7`,
`mylib.set_k(7)`). cash rebuilds that state at the next cell you run, from
the statements that set it, as it rebuilds a variable. Here it cannot: no
name in the notebook sees the module, neither the module itself nor
something imported from it (the cells set it through another module:
`outer.mylib.set_k(7)`), or, after a reload, cash does not know the
notebook's cells ([NOTEBOOK-NOT-FOUND](#notebook-not-found)) or where the
cell you ran is among them. The message names the module, and after a
restart the statement.

**Why it matters.** The module holds the file's values for what those
statements set, not the ones the notebook set, until they run again.

**What to do.** Run the cells that set it again. To have cash rebuild it
next time, `import mylib` in a cell above the statements that set it.

**When it is safe to ignore.** When you removed what the statements set on
purpose and no cell reads it any more.

### NOTEBOOK-SAVEFIG-SKIP {#notebook-savefig-skip}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/upstream/figure_writes.py:FigureWriteGuard._warn_orphaned_write @74b1d56c -->
**What happened.** While re-running earlier statements, cash skipped a
`plt.savefig(path)` whose figure was not being redrawn. The file on disk is
unchanged.

**Why it matters.** Running it without the drawing would save a blank figure
over your chart.

**What to do.** To rewrite the image, re-run the cell that draws the plot. To
avoid this, save through the figure object:

<!-- test:skip reason="fragment: names a type or helper the page does not define" -->
```python
fig, ax = plt.subplots()
ax.plot(xs, ys)
fig.savefig("chart.png")
```

**When it is safe to ignore.** Almost always: the figure did not change, so the
file on disk is the one you want.

### NOTEBOOK-SHARE-UNCHECKED {#notebook-share-unchecked}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/statement/processor.py:StatementProcessor._record_shared_object_edges @b3fd57ed, cash/notebook/shared_objects.py:referring_names @5602736f, cash/notebook/statement/processor.py:StatementProcessor.record_structure_share_edges @4fe4d23f -->
**What happened.** After a statement ran, cash looked for the other variables
that hold an object of its outputs, or are held in one (`data = raw`,
`config = {"features": features}`), so that a later change through either
name re-runs the cells reading the other. The search takes at most twice
what the statement took (at least a second; for what a loop, `if` or `try`
block changed, checked once when the block ends, at least a tenth of a
second), and it ran out of time, or past a holder that is not a variable (a
function default, a library object) met more objects than it follows,
before it was sure it had found them all. The message names the outputs. It
is shown once per statement.

**Why it matters.** A later change made in place through a variable it
missed (`raw.append(x)`) does not re-run the cells that read the outputs,
or the other way round, and they keep their cached results.

**What to do.** If a later cell changes these objects in place through
another name, rebind instead (`raw = raw + [x]`), or re-run the cells that
read them yourself.

**When it is safe to ignore.** When nothing below changes these objects in
place, through any name.

### NOTEBOOK-SHARED-KEPT {#notebook-shared-kept}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/upstream/stale_values.py:StaleValueGuard._kept_for_partner_below @bebe6e19 -->
**What happened.** You re-ran on its own a cell that changes a value in place
(`history.append(x)`), and a cell below makes another name share that value
(`log = history`, `cfg = {"features": features}`). cash normally rebuilds the
value as the cells above leave it before such a re-run, so the change is not
applied twice. Here it kept the value as it is: the cell's change applies again
on top of the one it made last time, as in a plain re-run.

**Why it matters.** A rebuilt value would be a new object under the cell's own
name only. The name below would keep the old one, and from then on a change
made through either name would no longer reach the other, which no run of the
notebook from the top does.

**What to do.** For the value a run from the top gives, run the notebook from
the top, or from the cell that creates the value.

**When it is safe to ignore.** When the cell's change is meant to accumulate,
or applying it twice does not change the result (`s.add(x)` on a set).

### NOTEBOOK-THREAD-ALIVE {#notebook-thread-alive}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashWarning</span>

<!-- claim: cash/notebook/background_threads.py:BackgroundThreads.cell_ends @b6d8c938, cash/notebook/background_threads.py:user_code_of @310cd8c4 -->
**What happened.** A cell started a thread and finished while the thread
was still running. The message names the cell, the thread and the function
it runs. Only a thread that runs your own code counts: a pool's idle
workers or a library's helper thread run the library's code and are left
out. It is shown once per cell.

**Why it matters.** What the thread changes after the cell finished (a list
it fills, a file it writes) is outside cash's view. The statements that read
those values were keyed and stored on what they held when the cell ended,
and a cell above that is re-run can see the changed values.

**What to do.** Join the thread before the cell ends (`t.join()`), or have
it hand back its result (`future.result()` with `concurrent.futures`) in the
cell that uses it.

**When it is safe to ignore.** When the thread changes nothing a cell reads,
such as a progress display or a server you query from outside the notebook.

## Randomness {#random-codes}

A cached value that randomness makes non-reproducible. Every code here starts `RANDOM-`.

| Code | Applies to | Meaning |
|---|---|---|
| [RANDOM-HELD-GENERATOR](#random-held-generator) | notebook | a cell re-run on its own draws from a generator held by an object, from where it stands |
| [RANDOM-REPLAYED](#random-replayed) | notebook | a restored value is an earlier random draw |
| [RANDOM-SEED-NONE](#random-seed-none) | notebook | `seed(None)` cannot refresh cached values |
| [RANDOM-UNSEEDED](#random-unseeded) | both | a draw from an unseeded source is not reproducible |

### RANDOM-HELD-GENERATOR {#random-held-generator}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashRandomnessWarning</span>

<!-- claim: cash/notebook/statement/held_generators.py:HeldGenerators.before_run @57d870a1, cash/tracking/randomness/detect.py:attribute_draws @9a5150b6 -->
**What happened.** You ran a cell on its own (again, or not right after the
cell above it), and a statement in it draws from a generator reached through
an attribute: `c.rng.normal()` with `rng` a class attribute, `mod.rng`,
`f.rng`, or `self.rng.normal()` in a method of yours the statement calls. The
generator is the one the statement drew from last time, and it has moved on
since. cash does not put such a generator back where a run from the top has
it, so the draw continues from where the last run left it. The message names
the cell, the statement and the generator, once per statement.

**Why it matters.** The numbers differ from the ones a run from the top
gives. A generator held in a variable (`rng.normal()`), or by an object the
re-run rebuilds (`self.rng` set in `__init__`), is put back, and does not warn.

**What to do.** Draw from such a generator in the cell that creates it, or
hold it in a variable and draw from that. For the numbers a run from the top
gives, run the notebook from the top.

**When it is safe to ignore.** When the draw only needs to be random, not the
same as a run from the top.

### RANDOM-REPLAYED {#random-replayed}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashRandomnessWarning</span>

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

### RANDOM-SEED-NONE {#random-seed-none}

<span class="md-tag cash-warning-path">notebook</span> <span class="md-tag cash-warning-class">CashRandomnessWarning</span>

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

### RANDOM-UNSEEDED {#random-unseeded}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashRandomnessWarning</span>

**What happened.** A statement or a cached function draws from an unseeded
random source.

**Why it matters.** The value is not reproducible. A cached draw is frozen: the
first result is returned from then on, and clearing the cache or running on
another machine gives a different frozen value.

<!-- claim: cash/tracking/randomness/state.py:capture_rng_state @421bfe05, cash/tracking/randomness/detect.py:RandomnessDetector.analyze_code @4890c3ec -->
In a notebook, a draw too cheap to cache is frozen too when it comes from the
`random`, `numpy.random` or `torch` stream, because cash rewinds those streams
before a re-run. A cheap draw from a generator held in a variable
(`rng = np.random.default_rng()`) is not rewound, so it changes on every run.
The message for a generator draw says both.

**What to do.** Seed the source (`random_state=42`, `np.random.default_rng(0)`)
for a stable, reproducible value. Or accept the frozen value and silence the
warning, as below.

<!-- claim: cash/notebook/upstream/rng_rewind.py:RngRewind._opts_out_of_rng_rewind @494a3663, cash/notebook/statement/randomness.py:StatementRandomness.warn_unseeded @0a570907 -->
In a notebook: for a fresh draw every run, put `# @cash:no-cache` on the
statement, on the line above it or at the end of its line. It turns off the
rewind as well as caching, and the statement no longer raises this warning.
`# @cash:allow-random` only silences the warning.

<!-- claim: cash/decorator/rng.py:RngWatch.warn_unseeded_randomness @37dc56c2 -->
With `@cash.cache`: the check runs when the decorator is applied, once per
function, and reads only that function's source. A draw from the global
`random` or `numpy.random` stream is not reported when a seed of that stream
is already in force, wherever it was set (module level, before `import cash`);
a seed set later, before the call, is keyed too, and the message says so. A `seed=None` parameter passed on to the
generator warns for calls that leave it out; pass `seed=i` per replicate. To
keep the frozen value, use `@cash.cache(allow_random=True)`. For a fresh draw,
do not cache the function.

<!-- claim: cash/decorator/rng.py:unseeded_library_calls @54a3c86b -->
A library call that draws inside its own compiled code warns too when the
function passes it no seed: `train_test_split(X)`, `KFold(shuffle=True)`,
`make_classification()`, an estimator such as `SGDClassifier()` or
`RandomForestClassifier()`, and `df.sample(3)`. Pass `random_state=0`
(`seed=0` in polars). A `.sample()` on one of the function's parameters
(`seed.sample(xs, k)` with a `random.Random` passed in) draws from the
argument, which is keyed by its state, so it does not warn; a DataFrame passed
in that draws from numpy's global stream is reported when the call runs. An estimator that uses its seed only with some settings
warns only when the call picks them: `Ridge()` is silent, `Ridge(solver='sag')`
warns. When cash cannot read the setting (a variable, or `PCA()`, whose
`svd_solver='auto'` picks by the data's shape), the message says what the draw
depends on rather than that one happened.

<!-- claim: cash/notebook/statement/randomness.py:StatementRandomness._warn_hidden_unseeded @f80cb855 -->
In a notebook the same library calls warn when a statement makes them without
a seed (`df.sample(100)`, `train_test_split(X)`), as does a statement that
calls one of your functions which builds its own unseeded generator
(`rng = make_rng()` with `return np.random.default_rng()`) and an anonymous
generator (`z = np.random.default_rng().normal()`). A pandas or scikit-learn
call left at `random_state=None` draws from numpy's global stream, so after
`np.random.seed(0)` it is reproducible and does not warn; a polars call does.

<!-- claim: cash/decorator/rng.py:RngWatch.warn_observed_draw @4bad64b0, cash/notebook/statement/randomness.py:StatementRandomness.flag_observed_hidden_draw @71b1e7ff, cash/tracking/randomness/lineage.py:unseeded_streams @5abf622f -->
A draw that neither the function's source nor the statement's shows is caught
as it happens: cash compares the global streams (Python's `random`, numpy's
`np.random`, and torch's default generator when torch is imported) before and
after a computed call or statement. If one moved and no seed stands behind it,
the message says, for example, that the call "drew from numpy's global random
stream without a seed". That covers a helper or a library deep inside the call:
a scikit-learn transformer left at `random_state=None` inside a
`ColumnTransformer` a helper builds, which subsamples large data from numpy's
stream. A seed counts when it was set before the call, by the caller, or inside
the call before the draw (`np.random.seed(0)`, `random.seed(0)`,
`torch.manual_seed(0)`); `seed(None)` does not. Some library code takes a
number from the stream and then does not use it (scikit-learn's `SVC()` and
`Lasso()` take one for settings they are not using), so the message says the
value is frozen if the result depends on that draw. A returned estimator left
at `random_state=None` is judged by its own settings, as above. With
`@cash.cache` this is said on a computed call, once per function. A generator the code makes for itself
without a seed (`np.random.default_rng()` in a helper) moves no global stream,
so this check cannot see it; for that, see
[CACHE-NONDETERMINISTIC](#cache-nondeterministic).

**When it is safe to ignore.** When any fixed value will do: an exploratory
split, a demo, a smoke test. Not when the number goes into a report or a test
assertion.

## Remote files {#remote-codes}

Checking whether a remote file changed. Every code here starts `REMOTE-`.

| Code | Applies to | Meaning |
|---|---|---|
| [REMOTE-FRESHNESS-COST](#remote-freshness-cost) | both | checking remote files costs more than it protects |
| [REMOTE-SIZE-ONLY](#remote-size-only) | both | a remote file is tracked by size alone |
| [REMOTE-STATE-UNREADABLE](#remote-state-unreadable) | both | a remote file's state could not be read |

### REMOTE-FRESHNESS-COST {#remote-freshness-cost}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

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

### REMOTE-SIZE-ONLY {#remote-size-only}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/remote_source.py:_warn_weak_token @d860c31b -->
**What happened.** The store gave no ETag, version id or last-modified time
for a remote file, or for some of the objects under a prefix or glob you read,
so cash tracks them by size alone.

**Why it matters.** A same-size edit does not change the key, so the old
result is served.

**What to do.** Pin a version in the URL (`?versionId=...`,
`#generation=...`), or write a `DataSource` whose token you control (see
[Data sources](api/data_sources.md)).

<!-- claim: cash/remote_source.py:_listing_token @fe6b3d29 -->
**When it is safe to ignore.** When each object is written once and never
rewritten in place. For a prefix or a glob, such as a folder of dated
partitions, the objects' names are tracked too, so a new or removed partition
still changes the key.

### REMOTE-STATE-UNREADABLE {#remote-state-unreadable}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/remote_source.py:RemoteFileDataSource._warn_failure @bc945f69 -->
**What happened.** Reading a remote file's state failed. The message names
the URL and the exception. cash recomputed rather than serve an unchecked
result.

**Why it matters.** While it lasts, every call recomputes and leaves an entry
no later call can reach. It warns once per URL and error type.

**What to do.** Read the exception: usually expired credentials, a permissions
change, or network trouble. A 404 means the URL is wrong or the object was
removed. Caching resumes by itself once access works.

<!-- claim: cash/remote_source.py:addressing_options @3d914602, cash/remote_source.py:read_options @269a4b73 -->
The check goes to the store the read did: a reader's `storage_options=` (or
an fsspec `filesystem=`) is used for it. Only the part that names the store
(endpoint, region, profile, `anon`) is written into the entry; credentials are
kept in memory. So in a later process whose credentials come only from
`storage_options`, the first check can fail with this warning, and the call
recomputes once in each new process. Put the credentials in the environment or
a profile to avoid that.

**When it is safe to ignore.** For a short blip on a cheap function.

## Storing results {#store-codes}

The call succeeded, but its result was not written. Every code here starts `STORE-`.

| Code | Applies to | Meaning |
|---|---|---|
| [STORE-CHUNK-FAILED](#store-chunk-failed) | decorator | one chunk of an iterator failed to write |
| [STORE-CODE-CHANGED](#store-code-changed) | decorator | code changed on disk during the call; not stored |
| [STORE-FAILED](#store-failed) | both | the backend refused the write |
| [STORE-INPUT-CHANGED](#store-input-changed) | decorator | an input file changed during the call; not stored |
| [STORE-LOCK-FAILED](#store-lock-failed) | decorator | the per-key lock failed; ran without it |
| [STORE-METADATA-INVALID](#store-metadata-invalid) | decorator | a stored entry's metadata is unreadable |
| [STORE-STATE-CHANGED](#store-state-changed) | decorator | a global the key read changed during the call; not stored |
| [STORE-UNTRACKED-SOURCE](#store-untracked-source) | decorator | a cached function it calls has a dynamic source that cannot be pickled; kept in RAM only |

### STORE-CHUNK-FAILED {#store-chunk-failed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/runtime.py:CallRunner._chunks_are_intact @49566c4c -->
<!-- claim: cash/decorator/runtime.py:CallRunner.compute_with_lock @b4c5c8c2 -->
**What happened.** A chunk of a large iterator result failed to write. The
message names the chunk, the backend and the exception.

**Why it matters.** cash treats an entry with a missing chunk as absent, so
every call recomputes until a write succeeds.

**What to do.** Call `f.cache_clear()`, then fix the cause: a full disk,
permissions, or an item that cannot be serialized.

**When it is safe to ignore.** Not for long: the function recomputes on every
call.

### STORE-CODE-CHANGED {#store-code-changed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/file_deps.py:FileDeps.code_moved_since_keyed @b425daba, cash/decorator/registry.py:FunctionRegistry.code_functions @f4899cd2 -->
**What happened.** A file holding the function, a helper, or a cached function
it depends on changed on disk during the call, in code this call runs. The
result was returned but not stored.

**Why it matters.** Worker processes started now import the new code, so the
result may mix old and new code. Storing it could serve a wrong result later.

**What to do.** Restart the process. If you are not editing the code,
something is replacing files under a running job, such as a deploy.

**When it is safe to ignore.** Once, while you edit a helper during a run.

### STORE-FAILED {#store-failed}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/store.py:ResultStore.store @e289b59d -->
**What happened.** The result was computed, but writing it to the cache
failed. The message names the backend and the exception. Whatever the
exception, the call returns its result; a failed write never fails the call.

**Why it matters.** Your result is correct. If it happens every call, the cache
does nothing.

**What to do.** Read the exception. Common causes: a full disk, no write
permission on `cache_dir`, a value that cannot be pickled (a socket, a file
handle), or on Windows a file held open by another process.

**When it is safe to ignore.** A one-off on a cheap function.

### STORE-INPUT-CHANGED {#store-input-changed}

<span class="md-tag cash-warning-path">both paths</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/file_deps.py:FileDeps.inputs_moved_during_call @7d20722f, cash/tracking/file_tracker.py:FileAccessTracker.inputs_changed_since_read @efc7273e -->
<!-- claim: cash/tracking/file_tracker.py:FileAccessTracker._digest_now @57e71e71, cash/tracking/file_dep_snapshot.py:snapshot_file_deps @538d7a29 -->
<!-- claim: cash/notebook/statement/store.py:StatementStore.inputs_moved @6c593ab7, cash/notebook/statement/processor.py:StatementProcessor._post_execute @95b3940e -->
=== "Decorator"

    **What happened.** A file the function read changed before it returned.
    The result was returned but not stored.

=== "Notebook"

    **What happened.** A file the statement read changed before it finished,
    another process replacing it, say. The statement's value is kept for this
    run but not stored, and the statements below it that use the value are
    not served from what they stored for it; the next run reads the settled
    file. The warning comes once per statement.

**Why it matters.** Nobody can say which version of the file the result came
from, so storing it could serve a stale result later.

<!-- claim: cash/tracking/file_tracker.py:FileAccessTracker.created_by_block @edb2ba0e, cash/tracking/file_tracker.py:FileAccessTracker.get_accessed_files @5c8d904e, cash/tracking/read_events.py:_on_mkdir @309b5321 -->
A file the function creates itself (opened for writing, in a directory it made
with `tempfile.TemporaryDirectory`, `mkdtemp` or `os.mkdir`, or by
`tempfile.mkstemp`) is its own output while the function runs: writing a
scratch file, reading it back and removing it caches normally. A file it
created and read that is still there when it returns is an input from then on,
checked against its content at that moment: a "download if missing, then read"
loader runs again when the file is replaced or deleted. A folder that was already
there is not one the function made, so `os.makedirs(d, exist_ok=True)` leaves
the files under `d` inputs.

<!-- claim: cash/tracking/file_tracker.py:FileAccessTracker.note_entry_changed @dbbf9b49 -->
A folder the function or statement listed (`os.listdir`, `glob`,
`Path.iterdir`, `os.walk`) counts too: a file landing in it or leaving it
before the call returns means the listing was of a folder that is no longer
there. A folder in which the call itself writes, makes, removes or moves an
entry is left out, as its own output moves the folder too and cash cannot tell
the two apart; a file arriving from outside while such a call runs is not
caught.

**What to do.** Usually nothing: the next call or run caches normally. If it
fires on every run, the function or statement probably reads a file and then
writes to it. Split the read from the write:

<!-- test:skip reason="illustrative: the point is the split, not a value" -->
```python
@cash.cache
def summarise(path):
    return build_summary(pd.read_csv(path))  # reads only

summary = summarise("state.csv")
summary.to_csv("state.csv")  # the write happens outside
```

**When it is safe to ignore.** When another process was writing the file and
has finished.

### STORE-LOCK-FAILED {#store-lock-failed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/backends/_base.py:CacheBackend.lock @2c1d7483 -->
**What happened.** cash could not take the per-key lock that stops two
callers computing the same thing at once, and went ahead without it.

**Why it matters.** Concurrent calls with the same arguments may each compute
the result. Nothing becomes wrong.

**What to do.** Read the exception: a Redis timeout, a stale lock file, a full
disk, or a filesystem where locking does not work.

**When it is safe to ignore.** When nothing runs concurrently.

### STORE-METADATA-INVALID {#store-metadata-invalid}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheIneffectiveWarning</span>

<!-- claim: cash/decorator/reporting.py:Notices.metadata_invalid @4b14c4a0 -->
**What happened.** cash found an entry but could not read its metadata, so it
treated the entry as missing and recomputed.

**Why it matters.** Little: one recompute per entry.

**What to do.** Nothing for a one-off. If it keeps appearing, run
`f.cache_clear()`. Usual causes: an entry from another cash version, or an
interrupted write.

**When it is safe to ignore.** Usually. Look into it if it persists after
`cache_clear()`.

### STORE-STATE-CHANGED {#store-state-changed}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/runtime.py:CallRunner._inputs_moved_check @2dd33f3d, cash/decorator/runtime.py:KeyBuilder.ledger_now @355ab3b5 -->
**What happened.** A global, a class attribute or other program state the
function's key read changed while the call was running: another thread
reloaded a config, say. The result was returned but not stored. The check
runs only while other threads are running (or for a coroutine function).

**Why it matters.** The key was built from the old value and the body may have
used the new one, so storing it could serve the wrong result to every later
call with the old value.

**What to do.** Nothing, if the state was changed on purpose: the next call
keys on the new value and caches normally. If calls overlap a writer all the
time, pass the value in as an argument.

**When it is safe to ignore.** Once, after a reload.

### STORE-UNTRACKED-SOURCE {#store-untracked-source}

<span class="md-tag cash-warning-path">decorator</span> <span class="md-tag cash-warning-class">CashCacheStoreFailedWarning</span>

<!-- claim: cash/decorator/store.py:ResultStore._unpicklable_source_refusal @176a0ddd, cash/decorator/dynamic_sources.py:recorded_sources @d682f3a6 -->
**What happened.** The function calls a cached function whose
`dynamic_depends_on=` returned a `DataSource` that cannot be pickled. This
function's entry keeps such sources so a lookup can ask them whether they
changed, and a later process could not. So the result is kept in memory for
this process only, and not written to disk. With a backend that has no RAM
tier, it was returned but not cached.

**Why it matters.** The next process recomputes this function (with no RAM
tier, every call does); the cached function it calls is still served from its
own entries. Nothing stale is served. A source that pickles does not warn: its
copy is kept in the cache, in the entry or beside it
([Cached functions that call it](tutorials/feature-guides/dynamic-dependencies.md#cached-functions-that-call-it)).

**What to do.** Make the source picklable: keep no open connection, lock or
client on it, and open what `state_token()` needs inside it. Or call that
function outside this one and pass its result in as an argument.

**When it is safe to ignore.** When this function is cheap next to the one it
calls, or the script runs once.

## Related

- [Exceptions and warnings](api/exceptions.md): the warning classes, and
  filtering many codes at once by class.
- [Seeing what cash did](how-it-works/inspecting.md): `explain()` and the
  other ways to ask why a call hit or missed.
- [Annotations](annotations.md): `# @cash:assume-safe` and the other
  `# @cash:` directives.
- [Configuration](getting-started/configuration.md): the settings the
  `CONFIG-` codes are about.
