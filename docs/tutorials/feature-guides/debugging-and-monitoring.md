# Debugging

!!! info "Applies to: notebook"
    Notebooks with `%cash_on`. How to find out why a cell ran, did not run, or
    was not cached. For `@cash.cache` functions, see
    [Seeing what cash did](../../decorator.md#seeing-what-cash-did).

Start with the badge; most questions end there. When it is not enough, turn on
`%cash_debug`. `%cash_stats` tells you whether caching pays off over the
session, and the `cash` command line shows what is on disk.

## 1. Read the badge

Every statement's row says what happened, and when a statement ran again or was
not stored, why. Open the row for its detail, and open the **upstream context**
to see what cash re-ran in earlier cells. [Reading the badge](../../badges.md)
lists every status and reason. For headless runs, `%cash_badge print` gives the
same as text:

```text
[Cash] EXECUTED (0.21s)
  EXECUTED: result = featurize(df)  (0.20s) -> RAM+DISK
    sub-call featurize(df): 0/1 hit - changed: df
```

## 2. Turn on `%cash_debug`

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_debug @29ba1a1b -->
`%cash_debug on` logs each step as cells run; `%cash_debug off` stops it.
`%cash_debug json` logs one JSON object per record, and `%cash_debug file PATH`
also appends them to a file. The output is long, so run the one cell you are
investigating and switch it off again.

Each line starts with the logger name and a tag. The tags that answer "why did
this run?":

| Tag | What it tells you |
|---|---|
| `[CACHE_KEY]` | The statement's key and what went into it: the code hash, each input and the lineage hash it resolved to, the functions it calls. Compare two runs to see which input moved. |
| `[CACHE DEBUG]` | The key looked up, `Cache hit: True` or `False`, and what happened next: `Executing (cache miss)`, `Stored in cache`, `Restored from cache`. |
| `[CACHE_HIT_DEBUG]` | On a hit, the input lineages the stored entry was matched on. |
| `[SIZE_AWARE]` | Why a result was not stored, for example `below 10ms floor`. |
| `[STORAGE]` | Where a result was written: `Stored in: RAM, DISK`. |
| `[UPSTREAM_DEBUG]` | The check of earlier cells: which inputs the cell needs and what was re-run. |

Other tags (`TIMING`, `TIMING_PROXY`, `CELL_ID`, `ENSURE_STATE_DEBUG`) trace
timing and cell identification.

Here `df` was changed upstream, so `result = featurize(df)` ran again
(lines shortened):

```text
[cash.notebook.cache_key] [CACHE_KEY] Input 'df' resolved to: 70e3a0edf794526b...
[cash.notebook.cache_key] [CACHE_KEY] Input 'featurize' resolved to: 8ca7ecb57d08c0ae...
[cash.notebook.cache_key] [CACHE_KEY] Code: result = featurize(df)... | source_hash: 6c5677ce1711... | input_hashes: ['70e3a0edf794...', '8ca7ecb57d08...'] | ... | cache_key: stmt:ffd3d255253e...
[cash.notebook.statement.processor] [CACHE DEBUG] Cache hit: False
[cash.notebook.statement.processor] [CACHE DEBUG] Executing (cache miss)
[cash.backends.tiered_backend] [STORAGE] Stored in: RAM, DISK
```

On the previous run, `Input 'df'` had resolved to a different hash: that is the
input that moved.

## 3. Check the session with `%cash_stats`

<!-- claim: cash/notebook/ipython/admin.py:CashAdminMagicsMixin.cash_stats @711be826 -->
`%cash_stats` summarises this kernel session: cells run, statements computed,
restored and skipped, the hit rate, and time saved.

```text
Cash Session Statistics
  (since this kernel started; a restart resets them)
----------------------------------------
  Cells executed:      4
  Statements computed: 6
  Statements restored: 0
  Statements skipped:  0
  Cache hit rate:      0.0%  (0/4 statements worth caching)
                       0.0% counting all 6 statements -- the other 2 were too
                       cheap to cache, so cash never tried: not misses.
  Compute time:        402.2ms
  Gross time saved:    0us  (estimated)
  Cash overhead:       167.0ms  (measured)
  Net time saved:      -167.0ms  (cash cost you 167.0ms this session)
  Tracked variables:   4
```

- **Net time saved** is the number to trust. It counts only savings backed by a
  measurement (this session recomputed the statement, or an earlier kernel on
  this machine did) and subtracts cash's own overhead. A session where cash cost
  more than it saved says so.
- **Discarded writes**, when shown, counts results cash failed to write, with
  the first cause. Those recompute every run, and no other counter shows it.
  Read this line first.
- `%cash_stats json` returns the same numbers as a dict. `%cash_stats reset`
  zeroes the counters and forgets the stored measurements; it keeps the
  discarded-writes line.

A notebook of sub-second cells with no restarts may show a net loss: cash's
bookkeeping can cost more than it saves there.

## 4. Look at the cache on disk

From a terminal in the notebook's folder, or a cell starting with `!`:

```bash
cash info                         # where the cache is, its size and settings
cash inspect                      # entries by size, with the time each saves
cash inspect --function NAME      # one function's entries
cash clear --all                  # delete the whole cache
```

After a `cash clear`, restart the kernel. See the [CLI reference](../../cli.md).

## Common symptoms

### A cell I did not change runs again

Open the row. `FUNC CHANGED`, `MODULE RELOADED` and "file changed" name the
cause. "Input lineage changed" means an upstream statement ran again: open that
row to see why. An `unstable key` row means something upstream runs on every
run. If the badge names nothing, compare the `[CACHE_KEY]` lines of two runs.

### A cell I changed still shows the old result

Cash cannot see the change. The usual causes, each with a fix, are in
[Writing cache-safe cells](../../known-limitations.md): an unsaved edit, a file
read through a loader cash does not watch, a helper that reads the clock, a
change made through another name, a function defined below the one that calls
it. To start over, run `!cash clear --all` and restart the kernel.

### Nothing seems to be cached

- The work is in the `%cash_on` cell. Nothing there is cached; move it down.
- The statements are under 10 ms. They show plain `EXECUTED` rows and appear in
  `%cash_stats` as "too cheap to cache".
- The rows say `NOT CACHED`: the reason names a side effect or an in-place
  change. See [What gets cached](../../notebook_caching_api.md#what-gets-cached).
- `%cash_stats` shows discarded writes.
- `CASH_DISABLE=1` is set; `%cash_on` then says caching is disabled.

### The cache is large

`cash inspect` lists entries by size with the time each one saves, so a large
entry that saves little stands out. Remove it with `cash clear --function NAME`
or `cash clear --entry ID`. [Where your cache lives](../../how-it-works/storage.md#where-the-cache-folder-is)
explains the size caps.

## Scripted access

<!-- claim: cash/notebook/ipython/magics.py:CashMagics.cash_status @c4c64c7d -->
`status = %cash_status dict` returns the last cell's statements (status, code,
times) and the session's tracking state, for tools and agents. It uses the enum
names `COMPUTED` (the badge's EXECUTED) and `RESTORED` (CACHED).
`%cash_provenance NAME` shows how a variable was computed: the code, its inputs
and its history. See [Magic commands](../../magics.md).

<!-- claim: cash/analytics.py:AnalyticsManager.__init__ @a038a9da -->
Cash also records per-session hit and miss events in a small `analytics.db` in
your user cache folder, which `cash.show_stats()` reads. Deleting it is always
safe. `CASH_ANALYTICS=0` turns it off.
