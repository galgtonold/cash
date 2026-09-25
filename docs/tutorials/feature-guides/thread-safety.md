# Threads and processes

!!! info "Applies to: decorator"
    Code that calls cached functions from several threads, event-loop tasks or
    processes at once.

Cached functions are safe to call concurrently. Without locking, two callers
that miss the same key at the same moment both run the body, and the second
write replaces the first with an equal value. The answers are right, but you pay
twice, which matters for a 30-second fit or a paid API call.

## `use_locking=True`: compute once

<!-- claim: cash/decorator/runtime.py:CallRunner.compute_with_lock @b4c5c8c2, cash/core.py:Cash.__init__ @e183e4ae -->
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

<!-- claim: cash/backends/_base.py:CacheBackend.lock @2c1d7483, cash/backends/redis_backend.py:RedisBackend.lock @cfdf2e01 -->
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

## Across processes: Pool, ProcessPoolExecutor, joblib { #across-processes-pool-processpoolexecutor-joblib }

<!-- claim: cash/_paths.py:resolve_main_module @fd6aef0f, cash/backends/_writes.py:in_multiprocessing_child @9bd4615e, cash/core.py:Cash._print_run_summary @f2a46f9f -->
Worker processes use the cache folder of the process that started them, so what
one worker computes is a hit for the other workers, for the parent, and for the
next run.

<!-- claim: cash/_active.py:publish_settings @035400df, cash/_active.py:_arrive @ea2f94f8 -->
That holds for settings changed in code, too. A worker started after
`cash.configure(...)`, or inside `with cash.disabled():`, runs with those
settings: `configure(cache_dir=...)` moves the workers' cache with the
parent's, and under `cash.disabled()` every cached call in a worker runs its
body. This works for every start method: fork, spawn (the default on Windows
and macOS), forkserver (the default on Linux from Python 3.14) and joblib's
workers. `CASH_*` variables set before the run reach them as well.

A worker keeps the settings it started with. A pool created before the change
keeps its workers' old settings, and so does a pool whose workers started
inside a `disabled()` block and are used after it. joblib keeps its workers
between `Parallel` calls, so a `with cash.disabled():` around the second call
does not reach them. Create the pool inside the block, or set `CASH_DISABLE=1`
for the whole run.

<!-- claim: cash/tracking/reader_patches.py:_patch_multiprocessing_pool @a7a12595 -->
A file a worker reads for a cached call in the parent is a dependency of that
call, as if the parent had read it: editing it recomputes the call. This holds
for `ProcessPoolExecutor`, `multiprocessing.Pool` and joblib's workers.

Each process keeps some things to itself:

- **RAM.** Each process has its own memory tier; only the disk is shared.
- **Writes.** A worker writes each result before its task returns, so a pool
  that shuts its workers down loses nothing. A worker forked while the parent
  is still writing a result does not wait for that write: it reads what is on
  disk, and recomputes if the entry is not there yet.
- **The summary.** `CASH_SUMMARY` prints one table per process that exits
  normally, labelled with its pid. `multiprocessing.Pool` terminates its
  workers, so only the parent's table prints.
- **Warnings.** Each process warns once for itself.
- **The disk cap.** Each process enforces `max_cache_size` on its own writes,
  so several workers together can overshoot it briefly.
- **Locking.** See the table above: only a bare `RedisBackend` locks across
  processes.

<!-- claim: cash/decorator/script_pickling.py:expose_script_function @f20865df, cash/core.py:Cash.__reduce__ @98bbf40f -->
For **joblib**, keep the script's work behind `if __name__ == "__main__":`.
Cash sends a cached function from the running script to the workers by name,
and each worker imports the script to find it. Without the guard, that import
runs your script again, so cash stops with a message asking for the guard. A
cached function defined in a module you import always works, and so does
`Parallel(prefer="threads")`. A script run as `python -m pkg.mod` is not sent
this way: move its cached functions into a module it imports.

## Related

- [Async functions](async-caching.md)
- [Deploying](deploying.md): workers, shared backends and locking in production.
- [Choosing a backend](choosing-a-backend.md)
