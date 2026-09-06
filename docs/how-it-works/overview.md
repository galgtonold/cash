# How Cash works — overview

Cash watches your code run, fingerprints each computation and its inputs, and
restores a saved result instead of recomputing whenever nothing relevant has
changed. This section is the guided tour: start here, then follow the journey
through cache keys, invalidation, safety, the two caching paths, storage, and
the tools that let you verify what Cash did.

## The core loop

Every computation Cash touches — a notebook statement or a decorated function call — passes through the same five steps. Cash **analyzes** which variables and files the code reads and writes, **keys** the computation by fingerprinting the code together with its current inputs, **checks** the backend to see whether that exact fingerprint is already stored, then either **executes** the code fresh or **restores** the saved result, and finally **tracks** lineage so that anything downstream knows what it depends on.

<!-- claim: cash/notebook/file_dep_snapshot.py:file_dep_is_fresh @5f35e472, cash/notebook/function_tracker.py:FunctionTracker @1a485c19 broad="the trust thesis names two whole mechanisms - file freshness and function-source tracking - not one function each" -->
The trust thesis is simple: Cash recomputes whenever something relevant changed, and refuses to cache when replaying a snapshot would be wrong. If your code reads a file that was modified or calls a function whose source changed, Cash will not serve you the old answer.

<!-- claim: cash/notebook/randomness.py:capture_rng_state @4bddf256, cash/notebook/randomness.py:restore_rng_state @ccba2493 -->
Non-determinism is the one case where "recompute" is not the safe answer, and Cash treats it separately: an unseeded random draw is **frozen**, not blocked. The first value you drew is the value you keep, so the notebook stays reproducible — see [knowing when to recompute](invalidation.md#randomness-re-seeding-invalidates-the-draws-below-it). Change the seed and every draw below it does recompute.

<div class="cash-coreloop" aria-hidden="true">
  <span class="cash-coreloop-step">analyze</span>
  <span class="cash-coreloop-step">key</span>
  <span class="cash-coreloop-step">check</span>
  <span class="cash-coreloop-step">execute / restore</span>
  <span class="cash-coreloop-step">track</span>
</div>

<div class="cash-arch" role="img" aria-label="How a computation flows through Cash. Your notebook statements and decorated calls enter CashMagics, the coordinator. For each statement or call it runs four jobs — analyze, safety checks, key and look up, and track lineage — and reads from or writes to the tiered cache backend.">
  <div class="cash-arch-node cash-arch-entry">
    <span class="cash-arch-title">Your code</span>
    <span class="cash-arch-sub"><code>%cash_on</code> notebook statements &middot; <code>@cash.cache</code> functions</span>
  </div>
  <div class="cash-arch-arrow" aria-hidden="true"></div>
  <div class="cash-arch-node cash-arch-hub">
    <span class="cash-arch-title">CashMagics &mdash; the coordinator</span>
    <span class="cash-arch-sub">Intercepts every statement and call &middot; tracks variables &middot; records provenance for the badge</span>
  </div>
  <div class="cash-arch-arrow" aria-hidden="true"></div>
  <div class="cash-arch-grid">
    <div class="cash-arch-node">
      <span class="cash-arch-title">Analyze</span>
      <span class="cash-arch-sub">Parse the AST to find which files and variables the code reads and writes.</span>
    </div>
    <div class="cash-arch-node">
      <span class="cash-arch-title">Safety checks</span>
      <span class="cash-arch-sub">Mutation, side-effect and purity detectors veto anything unsound to cache; the randomness detector warns and freezes.</span>
    </div>
    <div class="cash-arch-node">
      <span class="cash-arch-title">Key &amp; look up</span>
      <span class="cash-arch-sub">Fingerprint the code with its inputs, then check the backend &mdash; execute fresh or restore.</span>
    </div>
    <div class="cash-arch-node">
      <span class="cash-arch-title">Track lineage</span>
      <span class="cash-arch-sub">Record what each result depends on, and fold decorator hits into the badge.</span>
    </div>
  </div>
  <div class="cash-arch-arrow" aria-hidden="true"></div>
  <!-- claim: cash/backends/tiered_backend.py:TieredBackend @cf873702, cash/backends/sqlite_backend.py:SQLiteBackend, cash/backends/redis_backend.py:RedisBackend, cash/backends/s3_backend.py:S3Backend broad="TieredBackend's tier ordering is a property of the class as a whole; the other three are existence claims" -->
  <div class="cash-arch-node cash-arch-backend">
    <span class="cash-arch-title">Cache backend</span>
    <span class="cash-arch-sub">TieredBackend &mdash; L1 in-memory &rarr; L2 on disk &middot; pluggable: SQLite, Redis, S3</span>
  </div>
</div>

## One core, two front-ends

<!-- claim: cash/notebook/ipython/cell_executor.py:CellExecutor._execute_cell_statements @5f82719d, cash/core.py:Cash.cache @8a01c3c8 -->
Under the surface, Cash is one engine: a shared set of cache-key computation rules, a lineage and dependency-hashing layer, a family of invalidation policies, and a pluggable storage tier. What varies is only how you drive it. The **notebook path** (`%cash_on`) operates statement by statement — Cash intercepts each line of a cell via IPython hooks and decides independently whether to execute or restore. The **decorator path** (`@cash.cache`) wraps a Python function — Cash intercepts each call and caches the return value.

The next pages explain the shared foundation first, then each path's specifics. If you read from top to bottom you will understand both paths by the time you reach the storage and inspection pages.

??? question "Why statement-level, not whole-cell?"
    <!-- claim: cash/notebook/statement/processor.py:StatementProcessor.process_statement @728813ad -->
    Cash caches each statement in a cell independently rather than the cell as a
    unit. If a 3-statement cell changes only its first line, statements 2 and 3
    still restore from cache; a one-line edit never throws away a cell full of
    expensive work; and individual loop iterations can be cached separately. The
    cost is a more involved implementation (per-statement AST parsing and lineage)
    and a per-statement rather than per-cell overhead — a trade Cash makes
    deliberately, because real notebooks pile several operations into one cell.
    You do not have to take the trade on faith: the badge breaks the cell's
    wall time down into Cash's own overhead (upstream, cache, badge, and the
    rest) versus the statements themselves — see
    [seeing what Cash did](inspecting.md).

## Where to go next

<div class="cash-pain-grid" markdown="1">
<div class="cash-pain-card" markdown="1">
**[Cache keys, lineage & hashing](cache-keys-and-lineage.md)** — how Cash fingerprints a computation.
</div>
<div class="cash-pain-card" markdown="1">
**[Knowing when to recompute](invalidation.md)** — how a change ripples downstream.
</div>
<div class="cash-pain-card" markdown="1">
**[Knowing when not to cache](safety.md)** — the safety net.
</div>
<div class="cash-pain-card" markdown="1">
**[The notebook path](notebook-path.md)** — `%cash_on`, statement by statement.
</div>
<div class="cash-pain-card" markdown="1">
**[The decorator path](decorator-path.md)** — `@cash.cache` on functions.
</div>
<div class="cash-pain-card" markdown="1">
**[Where your cache lives](storage.md)** — tiers, promotion, serialization.
</div>
<div class="cash-pain-card" markdown="1">
**[Seeing what Cash did](inspecting.md)** — badges, provenance, audit logs.
</div>
</div>
