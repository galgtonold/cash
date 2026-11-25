# Threads and processes

!!! info "Applies to: decorator"
    Code that calls cached functions from several threads, event-loop tasks or
    processes at once.

Cached functions are safe to call concurrently. Without locking, two callers
that miss the same key at the same moment both run the body, and the second
write replaces the first with an equal value. The answers are right, but you pay
twice, which matters for a 30-second fit or a paid API call. A global or class
attribute the function reads that another thread changes while a call runs (a
config reload) makes that call's result go uncached, with
[STORE-STATE-CHANGED](../../warnings.md#store-state-changed): it cannot be told
which value it was computed from.

## What threads share, and what each call keeps

A cached function runs in a process whose settings, random streams and
imports belong to every thread. This is what cash guarantees when several
threads call cached functions, and what it does not.

**Guaranteed:**

<!-- claim: cash/backends/memory_backend.py:InMemoryBackend._keep_handed @8705e6ef, cash/handed_copies.py:reuse @7f13b0dc -->
- **Each caller gets its own object.** Two threads that hit the same entry at
  the same moment each receive their own copy of the result. Changing one
  never changes the other, and a copy one thread is still changing is never
  handed to another. The exception is `frozen=True`, where you promise not to
  change the result and every hit hands back
  [the same object](../../decorator.md#frozen-and-large-arguments).
<!-- claim: cash/loaded_modules.py:loaded_module @b27d18ff -->
- **A call works while another thread imports a library.** While one thread
  is in the middle of `import numpy`, `import pandas` or `import torch`, a
  cached call on another thread treats that library as not loaded yet and
  runs normally. The importing thread itself sees the library as plain
  Python shows it.
<!-- claim: cash/tracking/trace_recorder.py:Unit.detached @c3bd08a0 -->
- **A call is judged by its own record.** cash records the code a call ran in
  order to check its entry later. Code that other threads run while the call
  finishes does not get into that record and does not make the call fail.
<!-- claim: cash/tracking/process_registries.py:changed_by @4e1146d2 -->
- **A setting another thread writes is not blamed on your function** when the
  write can be traced to its thread. That holds for the environment
  (`os.environ`, `os.putenv`), `atexit.register`, `codecs.register` and
  `sys.addaudithook`: a heartbeat thread that writes `os.environ` while your
  function runs leaves the function cached. A write the function makes itself
  still makes its calls run uncached
  ([IMPURE-OBSERVED-EFFECTS](../../warnings.md#impure-observed-effects)), and
  so does a write by a thread the function started.
- **A result is not stored when state its key read changed mid-call.** That
  is the [STORE-STATE-CHANGED](../../warnings.md#store-state-changed) rule
  above. It does not depend on who is blamed for the change.

**Decided on the second look:**

<!-- claim: cash/decorator/runtime.py:CallRunner._refuse_registries @631605a3 -->
- **Other process settings.** A logger's level, the warnings filters,
  `sys.path`, the working directory, an interpreter hook and matplotlib's
  `rcParams` carry no record of which thread changed them. If one changes
  while a call runs and other threads are running too, that call is not
  stored and cash warns once with
  [STORE-STATE-CHANGED](../../warnings.md#store-state-changed). If it changes
  again during the next call that runs the function, cash takes the change as
  the function's and its calls run uncached. With no other thread running,
  the first change already counts as the function's.
<!-- claim: cash/decorator/rng.py:RngWatch.maybe_another_threads @6ffb844e -->
- **The global random streams.** `random`, `np.random` and torch's default
  generator are shared by all threads, so a stream that moved during a call
  may have been moved by another thread. The first time this happens to a
  function that is not known to draw, the call is not stored and nothing else
  changes. If a stream moves again during the next call that runs the
  function, in this process or the next, cash treats the function as drawing
  random numbers: its key follows the seed, and an unseeded draw warns with
  [RANDOM-UNSEEDED](../../warnings.md#random-unseeded). A function whose
  own source shows the draw, or that has `allow_random=True`, is treated as
  drawing at once.

<!-- claim: cash/tracking/process_registries.py:beside_threads @58b500d3 -->
"Other threads" here means threads that may be doing your program's work.
cash's own threads and a Jupyter kernel's do not count. On Linux, neither
does a thread that used no processor time while the call ran, such as an idle
pool worker or a sleeping monitor: with only such threads around, a change
counts as the function's at once, as it does with no other thread. On macOS
and Windows every other live thread counts.

**Not guaranteed:**

- **Telling a thread that keeps changing shared state from your
  function.** A background thread that draws from the global streams all the
  time, or resets a logger's level again and again, changes them during two
  calls in a row, and the function is then treated as if it did so itself.
  Give such a thread its own generator (`random.Random()`,
  `np.random.default_rng()`), or cache a function that takes the generator as
  an argument.
- **Telling tasks on one thread apart.** What another `asyncio` task does
  between two awaits of a cached coroutine happens on the same thread, so
  cash counts it as the call's.
- **One computation per key.** Without locking, two threads that miss the
  same key both run the body. The next section turns that off.

## `use_locking=True`: compute once

<!-- claim: cash/decorator/runtime.py:CallRunner.compute_with_lock @b4c5c8c2, cash/core.py:Cash.__init__ @d6897bc2 -->
Turn on locking on the `Cash` instance. It is not a decorator parameter, and it
applies to every function registered through that instance:

```python
from cash import Cash

app = Cash(use_locking=True)

@app.cache
def train(n_trees):
    return {"trees": n_trees}

train(300)   # first call: computes
```

On a miss, the caller takes a lock for that cache key, looks again, and only
computes if the entry is still missing. The other callers wait for the lock,
then read the stored result. The lock is per key, so calls with different
arguments never wait for each other, and a hit takes no lock at all.

<!-- claim: cash/decorator/runtime.py:CallRunner.single_flight @dfba7161 -->
It works for `async def` functions too: concurrent awaits of one key, in any
event loop of the process, wait for the first one and read its result. If the
first caller stores nothing (it raised, or `cache_if` said no), the others
compute for themselves.

## Which lock spans what

<!-- claim: cash/backends/_base.py:CacheBackend.lock @2c1d7483, cash/backends/redis_backend.py:RedisBackend.lock @dcc9c1b1 -->
| Backend | Callers that share one computation |
|---|---|
| Any backend, including the default | threads and tasks in **one process** |
| `RedisBackend` on its own (`CASH_BACKEND=redis`) | processes and hosts that use that Redis |
| A tier stack with RAM in front of Redis | one process only |

Workers in separate processes (a `multiprocessing` pool, joblib, gunicorn, two
containers) that miss the same key at once each compute it, unless the backend
is a bare `RedisBackend`. The Redis lock expires after 60 seconds, in case its holder
dies, and a caller waits at most 10 seconds for it. See
[Deploying](deploying.md#sharing-a-cache-between-processes-and-hosts).

## What to expect

- **A lock that fails is skipped.** If taking the lock raises (Redis is down, a
  timeout), the call computes without it and warns once
  ([`STORE-LOCK-FAILED`](../../warnings.md#store-lock-failed)). A call never
  hangs or fails because of the lock.
- **The lock is held while the body runs.** That is the point, but a slow body
  holds it for its whole run.
- **A failing function turns a burst into a queue.** When the body raises,
  nothing is stored, so the next waiter takes the lock and computes, and fails,
  in turn. N callers can wait up to N times the body's run time. Async callers
  wait for one round, then compute together.
- **Recursion is safe.** A body that calls itself with the same key on the same
  thread re-enters the lock instead of deadlocking.

Leave locking off for single-threaded code and for cheap functions, where a
duplicate computation costs less than the lock.

## Cheap functions in a thread pool

Cash's own work for a call -- building the key, looking it up, storing a
result -- is Python code, and a Python process runs one thread's Python code
at a time. In a thread pool that work does not overlap: eight threads pay it
one after another. A hit costs about 0.2 ms of it and a miss 1 to 2 ms, so a
function whose body only waits a few milliseconds (a quick HTTP request, a
small database query) called from many threads at once runs slower cached
than uncached, on hits too. Below about 5 ms of waiting per call, leave such a
function uncached, cache the larger step that calls it, or run the calls in
processes (`ProcessPoolExecutor`, `multiprocessing.Pool`), where each worker
does its own share of cash's work in parallel.

## Across processes: Pool, ProcessPoolExecutor, joblib { #across-processes-pool-processpoolexecutor-joblib }

<!-- claim: cash/_paths.py:main_namespace_name @d870998d, cash/backends/_writes.py:in_multiprocessing_child @8a8d5f6a, cash/decorator/run_summary.py:RunSummary.print_at_exit @63645a47 -->
Worker processes use the cache folder of the process that started them, so what
one worker computes is a hit for the other workers, for the parent, and for the
next run. That holds whether you map the cached function itself or a plain
function that calls cached ones.

<!-- claim: cash/_active.py:publish_settings @035400df, cash/_active.py:_arrive @ea2f94f8 -->
Settings changed in code reach the workers too. A pool created inside
`with cash.disabled():` runs every cached call's body:

```python { title="sweep.py" }
from multiprocessing import Pool

import cash

@cash.cache
def simulate(seed):
    return seed * 2

if __name__ == "__main__":
    with Pool(4) as pool:
        # computed once; later runs and the other workers get hits
        results = pool.map(simulate, range(8))

    with cash.disabled():
        with Pool(4) as pool:
            # created inside the block: every body runs
            fresh = pool.map(simulate, range(8))
```

- `cash.configure(...)` before the pool starts reaches the workers the same
  way: `configure(cache_dir=...)` moves their cache with the parent's.
- This works for every start method: fork, spawn (the default on Windows and
  macOS), forkserver (the default on Linux from Python 3.14) and joblib's
  workers. `CASH_*` variables set before the run reach them as well.
- **A worker keeps the settings it started with.** A pool created before the
  change keeps its old settings, and so does a pool whose workers started
  inside a `disabled()` block and are used after it.
- joblib keeps its workers between `Parallel` calls, so a
  `with cash.disabled():` around the second call does not reach them. Create
  the pool inside the block, or set `CASH_DISABLE=1` for the whole run.

<!-- claim: cash/tracking/reader_patches.py:_patch_multiprocessing_pool @e9ba11bf -->
A file a worker reads for a cached call in the parent is a dependency of that
call, as if the parent had read it: editing it recomputes the call. This holds
for `ProcessPoolExecutor`, `multiprocessing.Pool` and joblib's workers. So
does a `dynamic_depends_on=` source of a cached function the worker calls; one
that cannot be pickled back to the parent keeps the parent's call from being
stored.

Each process keeps some things to itself:

- **RAM.** Each process has its own memory tier; only the disk is shared.
- **Writes.** A worker writes each result before its task returns, so a pool
  that shuts its workers down loses nothing. A worker forked while the parent
  is still writing a result does not wait for that write: it reads what is on
  disk, and recomputes if the entry is not there yet.
- **The summary.** `CASH_SUMMARY` prints one table per process that exits
  normally, labelled with its pid. `multiprocessing.Pool` and
  `ProcessPoolExecutor` stop their workers without a normal exit, so only the
  parent's table prints. joblib's workers exit normally and print a table
  each (`cash (pid 5682): ...`), so add the hits up across them to see
  whether a run was warm.
- **`frozen=True`.** A result a worker returns reaches the parent as a copy,
  so the parent keys it by its content, not by the worker's call
  ([`frozen=` and large arguments](../../decorator.md#frozen-and-large-arguments)).
- **Warnings.** Each process warns once for itself.
- **The disk cap.** Each process enforces `max_cache_size` on its own writes,
  so several workers together can overshoot it briefly.
- **Locking.** See the table above: only a bare `RedisBackend` locks across
  processes.

<!-- claim: cash/decorator/script_pickling.py:expose_script_function @fe18c928, cash/decorator/script_pickling.py:refuse_pickling_by_value @3a356e73 -->
For **joblib**, keep the script's work behind `if __name__ == "__main__":`.
cash sends a cached function from the running script to the workers by name,
and each worker imports the script to find it. Without the guard, that import
runs your script again, so cash stops with a message asking for the guard.

- A cached function defined in a module you import always works, and so does
  `Parallel(prefer="threads")`.
- A script run as `python -m pkg.mod` is not sent this way: move its cached
  functions into a module it imports.

## Related

- [Async functions](async-caching.md): locking for concurrent awaits.
- [Deploying](deploying.md): workers, shared backends and locking in
  production.
- [Choosing a backend](choosing-a-backend.md): which backend locks across
  processes.
