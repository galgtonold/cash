# FAQ

Short answers to the questions that come up most. For the "is this for my
workflow?" case, see [Why Cash?](why-cash.md); for the vocabulary, see the
[Glossary](glossary.md).

## Getting started

??? question "Do I need to change my code?"
    No. `%cash_on` caches your existing cells as-is — no decorators, no config
    file. The `@cash.cache` decorator is opt-in for plain scripts. See
    [Quick start](getting-started/quickstart.md).

??? question "Notebooks only, or scripts too?"
    Both. The `@cash.cache` decorator works in plain Python scripts; the
    notebook integration (`%cash_on`, statement-level caching) is the
    notebook-specific layer over the same engine. See [API reference](api/index.md).

??? question "How do I turn it on for every notebook automatically?"
    `cash autoload on` installs a one-time IPython startup hook so every new
    kernel imports cash and runs `%cash_on`. See [CLI reference](cli.md).

## Correctness

??? question "What if cash returns a stale value?"
    Cash invalidates a cached result whenever the code that produced it changes
    *or* any of its inputs change. Inputs are tracked by [lineage](glossary.md#lineage)
    hash, so a change three cells upstream still propagates, and a changed data
    file is detected by content hash. See
    [Knowing when to recompute](how-it-works/invalidation.md).

??? question "What about in-place mutations like `df['x'] = 0`?"
    Cash uses AST-based [mutation detection](glossary.md#mutation-detection) to
    flag in-place mutations so cached objects are invalidated correctly. See
    [the mutation pattern in the data-science tutorial](tutorials/use-cases/data-science.md).

??? question "What about unseeded randomness?"
    A draw with no fixed seed can't be reproduced, so cash warns and freezes the
    drawn value by design. Opt in with `@cash:allow-random`, or seed the RNG for
    normal caching. See [Known limitations](known-limitations.md).

??? question "Why did `@cash.cache` raise `CashImpureFunctionError`?"
    The function resolves a dependency from a runtime value cash can't track —
    `eval`/`exec`, dynamic dispatch via `getattr(obj, name)()`, or
    `importlib.import_module` — so a cached result could go silently stale, and
    cash refuses to cache it by default. Pass `@cash.cache(assume_safe=True)` to
    cache anyway, or refactor to a statically-named call. See
    [the decorator guide](decorator.md).

## Coverage

??? question "Does it work with pandas / numpy / polars / torch / duckdb?"
    Native built-in hashers cover pandas, numpy, polars, PyArrow, modin, and
    dask. For anything else — torch tensors, duckdb relations, custom domain
    types — register a hasher with `cash.register_hasher`. See
    [API reference](api/cash.md#cash.Cash).

## Performance

??? question "How much does cash slow down a cold run?"
    ~5–30 ms per cached statement on a cold run (lineage computation, cache key,
    write). For most real work the overhead is dwarfed by what's being cached.
    Run #1 is a net cost — the win is on iteration and restart. See
    [Cost model](cost-model.md).

??? question "How much faster will it actually make things?"
    It depends on the ratio of compute cost to result size — the range is wide
    (a heavy loop body can see ~190×; a naive big-frame ETL ~1.2×, and Run #1 is
    slower). `%cash_stats` tells you which case you're in and will say so plainly
    when cash cost you time. See [Cost model](cost-model.md) for the decision
    model and [Benchmarks](benchmarks.md) for measured numbers you can reproduce.

??? question "Will my cache still be valid after I upgrade cash?"
    Cache entries are not guaranteed to survive a version change — see
    [Versioning & compatibility](versioning.md) for what is and isn't promised,
    and clear the cache (`python -m cash clear`) if in doubt.

## Troubleshooting

??? question "A cell isn't caching — how do I find out why?"
    Read the [badge](glossary.md#badge): it says *skipped* and why. The usual
    reasons are (1) the statement is under the ~10 ms floor (too cheap to be
    worth caching), (2) the result is too large for the [cost
    model](glossary.md#cost-model) to persist economically, or (3) a side effect
    or unseeded draw made it unsafe. A long `for`-loop that appends into a list
    can also stop caching — see
    [known limitations](known-limitations.md#a-long-for-append-loop-can-stop-caching).
    Force caching with `# @cash:persist` when you know better than the heuristic.
    Full walkthrough: [Debugging and monitoring](tutorials/feature-guides/debugging-and-monitoring.md).

??? question "A cell recomputed when I expected a hit."
    Something in its [cache key](glossary.md#cache-key) changed: the code, an
    upstream variable's [lineage](glossary.md#lineage), or a tracked
    [file dependency](glossary.md#file-dependency). Use `%cash_status` for the
    last cell and `%cash_provenance` to see what a variable depends on. In a
    notebook, use `.explain()` rather than `cache_info()` on decorated functions
    — cash may rebuild the wrapper, so its counters can read zero even while
    caching works. See [Seeing what Cash did](how-it-works/inspecting.md).

??? question "How do I force a fresh run or clear the cache?"
    Three escape hatches for a single run: `@cash:no-cache` on one statement,
    `%cash_off` for a whole cell, or the `cash clear` CLI command. To wipe
    everything, delete `.cash/` or run `%cash_repair --full`. See
    [Annotations](annotations.md) and [CLI reference](cli.md).

## Production readiness

??? question "Is a 0.x release safe for real work?"
    Yes for notebook use; `0.1.1`, the first public release, is backed by
    thousands of integration tests (many derived from real-world bug reports).
    Treat it like any library you'd pin — this is a `0.x` release, so the API
    and cache format may change between minor versions (run `%cash_repair --full`
    after upgrading). The
    [CHANGELOG](https://github.com/galgtonold/cash/blob/main/CHANGELOG.md)
    documents breaking changes.

??? question "Is loading a cache safe?"
    A cache is executable: cash unpickles stored objects, so loading a cache is
    equivalent to running a Python script from whoever produced it. Only load
    caches you trust — your own local `.cash/` is as safe as the code that wrote
    it. See the trust model in [Backends](api/backends.md#security).

## Vs. alternatives

??? question "How is cash different from joblib, lru_cache, diskcache, jupyter-cache, %store?"
    The short version: those cache function calls, cells, or key→value pairs;
    cash caches **statements** and tracks the dependency graph between them, with
    file tracking, mutation detection, and kernel-restart persistence. See the
    full capability matrix and per-tool notes in [Why Cash?](why-cash.md#vs-alternatives).
