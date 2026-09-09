"""``use_locking=True`` must admit exactly ONE thread into the compute.

Round-15 gate finding. It admitted exactly two, deterministically, at 2, 4, 8
and 16 threads -- the other N-2 blocked correctly, so the lock plainly worked;
it just always leaked a second computer. `thread-safety.md` recommends the flag
for paid API calls and metered GPU jobs, which double-fire under that: twice the
spend, twice the rate limit, and twice the side effect for anything not
idempotent.

The lock was never the bug. Instrumenting the acquire path showed the two
threads holding *different* lock objects for what should have been one key --
because they had resolved two different cache KEYS for the same call::

    …heavy:eb95dea4…::a937c083…   <- 5 threads, one compute (locked correctly)
    …heavy:0b4392ad…::a937c083…   <- 1 thread,  one compute (its own key)

The middle component is the state hash, and it is built from what
``_analyze_dependencies`` populates -- helper source hashes, graph edges. That
analysis ran under a bare ``if func_name not in self._analyzed``, so whichever
threads arrived while it was still running keyed the call from a half-populated
state. Two keys, one flight each, "exactly two" at every thread count.

Proof before the fix, by measurement rather than argument: forcing the analysis
once in the main thread (``heavy.explain(...)``, which computes nothing)
collapsed six threads to one key and one execution.

The pre-analysis entry is also an entry nothing will ever read again: every
later run keys the call the analysed way. So this wasted a compute AND left a
permanent orphan in the cache directory, both of which are asserted below.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cash import Cash


def _helper(x):
    """A plain, undecorated helper.

    Load-bearing for every test here, and the second thing the first draft got
    wrong. The state hash in the cache key folds the source of the helpers a
    function calls -- so a function that calls NOTHING hashes the same before
    and after the analysis, and the race is invisible however wide its window
    is. The reporter's function called a module-level ``_work``; so does this.
    """
    return x * 2


def _instance(tmp_path, **kwargs):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, **kwargs)


def _widen_the_analysis_window(c, seconds=0.5):
    """Make the one-time dependency analysis take *seconds*, deterministically.

    The race window in the wild is however long it takes to read and hash a
    function's helpers -- wide enough to lose on a real workload (5 of 6 threads
    in the reporter's, every run), far too narrow to lose reliably on a test
    function whose analysis finishes in microseconds. Every test below passed on
    the BROKEN code without this: they measured the width of the window, not the
    lock around it.

    This changes duration only, nothing about semantics, and it targets exactly
    the step the fix serialises. On the unfixed code it splits the callers into
    two cache keys every time; on the fixed code they wait and share one.
    """
    real = c._analyze_dependencies

    def slow(func):
        time.sleep(seconds)
        return real(func)

    c._analyze_dependencies = slow


def _run_threads(fn, n):
    """Start *n* threads on *fn* at once and wait for them all."""
    ready = threading.Barrier(n)
    errors: list[BaseException] = []

    def body():
        try:
            ready.wait(timeout=30)
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=body) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    if errors:
        raise errors[0]
    assert not any(t.is_alive() for t in threads), "a worker never finished"


@pytest.mark.parametrize("n_threads", [2, 4, 8, 16])
def test_locking_admits_exactly_one_thread(tmp_path, n_threads):
    """Parametrised on purpose: a bug that leaks a FIXED number of extra
    computes is invisible at any single thread count."""
    c = _instance(tmp_path, use_locking=True)
    _widen_the_analysis_window(c)
    runs: list[str] = []

    @c.cache(assume_safe=True)
    def heavy(k):
        runs.append(threading.current_thread().name)
        time.sleep(0.4)          # long enough that the others really overlap
        return _helper(k)

    _run_threads(lambda: heavy(21), n_threads)

    assert len(runs) == 1, (
        f"{len(runs)} threads entered the body with use_locking=True: {runs}"
    )


def test_without_locking_every_thread_computes(tmp_path):
    """The reporter's control arm, kept: the test must not be able to pass by
    quietly serialising the callers."""
    c = _instance(tmp_path)
    runs: list[str] = []

    @c.cache(assume_safe=True)
    def heavy(k):
        runs.append(threading.current_thread().name)
        time.sleep(0.4)
        return _helper(k)

    _run_threads(lambda: heavy(21), 4)

    assert len(runs) == 4, f"expected 4 concurrent computes, got {len(runs)}"


def test_concurrent_first_calls_agree_on_one_cache_key(tmp_path):
    """The defect underneath, asserted directly -- and it is not about locking.

    Without the flag there is no single-flight to violate, but two keys for one
    call is still a wasted compute and an entry no later run will ever read.
    """
    c = _instance(tmp_path)
    _widen_the_analysis_window(c)
    keys: list[str] = []
    lock = threading.Lock()

    @c.cache(assume_safe=True)
    def heavy(k):
        time.sleep(0.2)
        return _helper(k)

    def call():
        result = heavy.explain(21)
        with lock:
            keys.append(result.cache_key)

    _run_threads(call, 8)

    assert len(set(keys)) == 1, (
        f"eight concurrent first calls resolved {len(set(keys))} different "
        f"cache keys for one call: {sorted(set(keys))}"
    )


def test_no_orphan_entry_is_left_behind(tmp_path):
    """The cost that outlives the run: an entry keyed the pre-analysis way.

    A later call keys the function the analysed way and never looks at it
    again, so it sits in the cache directory for ever.
    """
    cache_dir = tmp_path / ".cash"
    c = _instance(tmp_path, use_locking=True)
    _widen_the_analysis_window(c)

    @c.cache(assume_safe=True)
    def heavy(k):
        time.sleep(0.3)
        return _helper(k)

    _run_threads(lambda: heavy(21), 8)
    c.backend.shutdown()

    entries = sorted(p.name for p in cache_dir.glob("*.entry"))
    assert len(entries) == 1, f"one call left {len(entries)} entries: {entries}"


def test_async_single_flight_admits_one_coroutine(tmp_path):
    """The same shape on the async path, which has its own single-flight."""
    c = _instance(tmp_path, use_locking=True)
    _widen_the_analysis_window(c)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    async def heavy(k):
        runs.append(k)
        await asyncio.sleep(0.3)
        return _helper(k)

    async def main():
        return await asyncio.gather(*[heavy(21) for _ in range(8)])

    results = asyncio.run(main())

    assert results == [42] * 8
    assert len(runs) == 1, f"{len(runs)} coroutines entered the body: {runs}"


def test_a_second_call_still_hits(tmp_path):
    """The control for all of the above: it must still be a cache.

    Every assertion here counts executions, and every one of them passes if the
    function is simply never cached at all.
    """
    c = _instance(tmp_path, use_locking=True)
    runs: list[int] = []

    @c.cache(assume_safe=True)
    def heavy(k):
        runs.append(k)
        time.sleep(0.3)
        return _helper(k)

    assert heavy(21) == 42
    assert heavy(21) == 42
    assert len(runs) == 1
