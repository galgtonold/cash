# FAQ

!!! info "Applies to: both paths"
    Short answers. Questions for one path are grouped under that path.

## Both paths

??? question "Which path should I use?"
    `@cash.cache` in scripts, services and libraries; `%cash_on` in a notebook.
    [Why cash?](why-cash.md) lists what each path does.

??? question "What if cash returns a stale value?"
    A result is recomputed when its code, a helper it calls, an input or a
    data file it read changes. Cash cannot see a database table or a web page
    change; give such results a `ttl`. See
    [Knowing when to recompute](how-it-works/invalidation.md).

??? question "Does it work with pandas, numpy, polars, torch or duckdb?"
    Built-in hashers cover pandas, numpy, polars, PyArrow, modin and dask. For
    anything else, register one with `cash.register_hasher`. See
    [Custom hashers](tutorials/feature-guides/custom-hashers.md).

??? question "How much faster will it make things?"
    It depends on compute time against reload time, so cash quotes no
    multiplier. The first run is slightly slower; later runs skip the work.
    [Benchmarks](benchmarks.md) shows what a restore costs and how to work out
    your own number.

??? question "Where is the cache, and how do I clear it?"
    In a `.cash/` folder, with a size cap. See
    [Where your cache lives](how-it-works/storage.md#where-the-cache-folder-is). To clear it, run
    `cash clear --all` or delete the folder.

??? question "How big can the cache get, and how do I limit it?"
    By default the disk cache may use a quarter of the free room on its
    volume, between 8 GiB and 100 GiB. When it is full, cash evicts the
    entries that are cheapest to recompute per byte. `cash info` prints the
    cap in use. To pin it, set `max_cache_size`, for example
    `CASH_MAX_CACHE_SIZE=20GB` or `max_cache_size = "20GB"` under
    `[tool.cash]` in `pyproject.toml`. See
    [Where your cache lives](how-it-works/storage.md#when-the-disk-fills-up).

??? question "Is my cache still valid after I upgrade cash?"
    Cash clears a local cache written in an older, incompatible format by
    itself, and treats any entry it cannot read as a miss. An upgrade can cost
    you a recompute, never a wrong value. `cash clear --all` only frees space.
    See [Versioning](versioning.md).

??? question "Is a 0.x release safe for real work?"
    The decorator is the simpler problem. The notebook path reasons about what
    your code reads and changes across cells, so surprises are more likely
    there; watch the badge. Known gaps are in
    [Writing cache-safe cells](known-limitations.md); the tests are described
    in [How cash is tested](how-it-works/testing.md). Pin the version: during
    0.x a minor release can change the API.

??? question "Is loading a cache safe?"
    Only if you trust whoever wrote it. Cash unpickles stored results, and
    unpickling can run code. See the trust model in
    [Backends](api/backends.md#security).

??? question "How is cash different from joblib, lru_cache, diskcache, jupyter-cache or %store?"
    See the [comparison](why-cash.md#compared-with-other-tools), and
    [Coming from other caches](migration_guide.md) for before-and-after code.

## Decorator

??? question "Why did `@cash.cache` raise `CashImpureFunctionError`?"
    The function picks what to run at runtime (`eval`, `exec`,
    `getattr(obj, name)()`, `importlib.import_module`), so cash cannot tell
    when a result goes stale. Put `# @cash:assume-safe` on that line, pass
    `assume_safe=True` for the whole function, or call the function by name.
    See [Side effects](decorator.md#side-effects).

??? question "Why does a cache hit not print anything?"
    A hit returns the stored value without running the function body, so its
    prints and other side effects do not happen again.

??? question "How do I force a fresh call?"
    `f.__wrapped__(...)` runs the body once without the cache, and
    `with cash.disabled():` does that for every cached call inside the block.
    Neither replaces the stored result: the next cached call returns it
    again. To replace it, run `f.cache_clear()` (or
    `cash clear --function f` in a terminal); the next call runs the body and
    stores the new result.

??? question "Why do I get `KEY-UNHASHABLE-ARG`?"
    An argument (a lock, a socket, an open file) cannot be hashed, so the call
    runs uncached. See [the warning](warnings.md#key-unhashable-arg).

??? question "How do I keep an argument such as a logger or a `verbose` flag out of the key?"
    There is no `ignore=` parameter: every argument is in the key. Call a
    cached core function from a thin wrapper that takes the extra arguments,
    or, for a type such as `logging.Logger`, register a hasher that returns a
    constant. See
    [An argument that does not change the result](decorator-limitations.md#an-argument-that-does-not-change-the-result).

## Notebook

??? question "Do I need to change my notebook?"
    No. Add `import cash` and `%cash_on` as the first cell; the cells below
    cache as they are.

??? question "Why is nothing in my first cell cached?"
    Statements in the cell that runs `%cash_on` are never cached: cash starts
    tracking after it. Do your work in the cells below it.

??? question "How do I turn it on for every notebook?"
    Run `cash autoload on` once in a terminal. See
    [Command-line interface](cli.md).

??? question "A statement isn't cached. Why?"
    The badge row reads `NOT CACHED` and names the reason: usually under
    10 ms, a side effect such as a file write, a change to an object from an
    earlier cell, or `# @cash:no-cache`. See [Reading the badge](badges.md).

??? question "A statement re-ran when I expected a hit."
    Something in its key changed: its code, an input variable, or a file it
    read. `%cash_status` shows the last cell and `%cash_provenance x` shows
    what `x` depends on. See [Reading the badge](badges.md).

??? question "What about unseeded randomness?"
    An unseeded draw is cached, so a restore replays the same draw. Seed the
    generator, or put `# @cash:no-cache` on the statement for a fresh draw.
    `# @cash:allow-random` only hides the warning. See
    [Writing cache-safe cells](known-limitations.md).

??? question "How do I force a fresh run?"
    Put `# @cash:no-cache` above one statement, or run `%cash_off` to stop
    caching until `%cash_on`. Either runs the code fresh for as long as it is
    in place, and leaves the stored result alone: remove the directive, or
    turn caching back on, and the stored result comes back. That is
    intended: once you allow caching again, an existing result is used.

    To replace the stored result, run the statement once with
    `# @cash:ttl=0` above it. That counts the stored result as expired, so
    the statement runs and stores the new value; then remove the directive.
    `cash clear --all` in a terminal empties the whole cache instead. See
    [Annotations](annotations.md).

??? question "`cache_info()` on a decorated function reads zero in my notebook."
    Re-running the defining cell creates a new wrapper with fresh counters.
    Use `f.explain(...)` or `%cash_stats` instead.
