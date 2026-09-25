"""Background cache writes: the per-backend write queue and its thread pool.

Backends whose writes touch a disk or a network (File, SQLite, Redis, S3)
serialize a value on the caller's thread and hand the bytes to a
`PendingWrites` queue, so a slow write never blocks the user's code.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import sys
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

from cash.diagnostics import warn_diagnostic
from cash.exceptions import CashCacheStoreFailedWarning
from cash.tracking.tracker_context import untracked

from ..config import get_config

logger = logging.getLogger(__name__)

__all__ = [
    "PendingWrites",
    "all_pending_writes",
    "discarded_writes",
    "in_multiprocessing_child",
    "reset_discarded_writes",
]

# Marks any thread currently running a PendingWrites task, for ANY instance.
# ``PendingWrites._tls.current_key`` answers "am I the worker for this key, on
# this instance?"; this answers "am I a write worker at all?", which a backend
# needs before blocking on a SIBLING backend's queue. Without it, two backends
# over one cache directory could each have their worker waiting on the other's
# future and deadlock.
_WORKER_THREAD = threading.local()

# Every live write queue in the process, whatever backend or Cash instance owns
# it. A notebook routinely has more than one (the ``%cash_on`` instance plus any
# ``Cash(...)`` built in a cell), and durability is a property of the kernel,
# so anything flushing "the pending writes" must reach all of them. Weak, so a
# backend going out of scope is not kept alive by this.
_LIVE_WRITE_QUEUES: weakref.WeakSet = weakref.WeakSet()


def all_pending_writes() -> list[PendingWrites]:
    """Every write queue still accepting work, across all backends."""
    return [q for q in _LIVE_WRITE_QUEUES if not q.is_shutdown()]


# Every cache write that failed and was discarded this process, as
# ``(key, "ExcType: message")``, appended the moment the worker raises. Unlike
# ``PendingWrites.failed_writes`` it survives the backend being collected and
# does not wait for the future to be looked at: it is the record a session
# summary quotes and a test asserts is empty. Never trimmed: a healthy process
# appends to it zero times.
_DISCARDED_WRITES: list[tuple[str, str]] = []
_DISCARDED_LOCK = threading.Lock()


def _reset_after_fork_in_child() -> None:
    """Make every write queue usable in a forked child (`PendingWrites._after_fork_in_child`)."""
    global _DISCARDED_LOCK
    _DISCARDED_LOCK = threading.Lock()
    for pending in list(_LIVE_WRITE_QUEUES):
        pending._after_fork_in_child()


if hasattr(os, "register_at_fork"):  # not on Windows, which cannot fork
    os.register_at_fork(after_in_child=_reset_after_fork_in_child)


def _record_discarded(key: str, exc: BaseException) -> None:
    with _DISCARDED_LOCK:
        _DISCARDED_WRITES.append((key, f"{type(exc).__name__}: {exc}"))


def discarded_writes() -> list[tuple[str, str]]:
    """Cache writes that failed and were discarded, oldest first.

    Each entry means: that key's work was recomputed rather than served, and
    nothing raised at the call site. A non-empty list is a cache that is
    quietly doing less than it appears to.
    """
    with _DISCARDED_LOCK:
        return list(_DISCARDED_WRITES)


def reset_discarded_writes(keep: int = 0) -> None:
    """Drop recorded failures past *keep* (test isolation; not for library use).

    ``keep`` rather than a bare clear so a test that induces failures on
    purpose can absorb exactly its own: the writes are asynchronous, so its
    failures often land after it has finished.
    """
    with _DISCARDED_LOCK:
        del _DISCARDED_WRITES[keep:]


#: Fallback for `_shutdown_write_timeout`; the configured default lives on
#: ``CashConfig.shutdown_write_timeout``.
_DEFAULT_SHUTDOWN_WRITE_TIMEOUT = 60.0


def _shutdown_write_timeout() -> float:
    """Seconds a finished process will wait for its cache writes.

    Read per call so a value set after import applies. Falls back to the
    default when the config cannot load: this runs from an ``atexit`` handler,
    where imports can fail during interpreter teardown.
    """
    try:
        value = float(get_config().shutdown_write_timeout)
    except Exception:  # noqa: BLE001 - teardown, or a config that cannot load
        return _DEFAULT_SHUTDOWN_WRITE_TIMEOUT
    return value if value > 0 else 0.0


#: ``(pid, answer)`` for `in_multiprocessing_child`. Keyed by pid because a
#: forked child inherits the parent's module state, answer included.
_CHILD_ANSWER: tuple[int, bool] = (-1, False)


def in_multiprocessing_child() -> bool:
    """Is this process a ``multiprocessing`` worker (Pool, Process, executor)?

    Never imports ``multiprocessing``: a process it did not start has no
    reason to have it loaded.
    """
    global _CHILD_ANSWER
    pid = os.getpid()
    if _CHILD_ANSWER[0] == pid:
        return _CHILD_ANSWER[1]
    mp = sys.modules.get("multiprocessing")
    try:
        answer = mp is not None and mp.parent_process() is not None
    except Exception:  # noqa: BLE001 - an odd embedding must not break writes
        answer = False
    _CHILD_ANSWER = (pid, answer)
    return answer


class _DaemonWriterPool:
    """The small thread pool `PendingWrites` runs its writes on.

    It exists for one property ``ThreadPoolExecutor`` cannot give: its threads
    are daemons, so a stuck write cannot keep a finished process alive.
    ``concurrent.futures.thread`` joins every worker it started at exit, with
    no timeout, so a write blocked in the OS (an unwritable or stalled cache
    directory) would hold the interpreter open after the work was done.

    A daemon thread is stopped wherever it is if the interpreter finalizes
    first. That is safe for the same reason a power cut is: entries are written
    to a temp file and renamed into place.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._max_workers = max(1, max_workers)
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.put((future, fn, args, kwargs))
            self._grow_if_needed()
        return future

    def _grow_if_needed(self) -> None:
        """Start another worker, lazily, up to ``max_workers``. Called under ``_lock``."""
        if len(self._threads) >= self._max_workers:
            return
        thread = threading.Thread(target=self._work, name="cash-cache-writer", daemon=True)
        self._threads.append(thread)
        thread.start()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # shutdown sentinel
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - reported via the future
                future.set_exception(exc)

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> bool:
        """Stop the workers. True if they all finished within *timeout*."""
        with self._lock:
            already = self._shutdown
            self._shutdown = True
            threads = list(self._threads)
        if not already:
            for _ in threads:
                self._queue.put(None)
        if not wait:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in threads:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        return True


class PendingWrites:
    """Per-backend background-write scheduler.

    * One worker per backend, so writes to that backend are serialized.
    * The caller serializes the value before submitting, so a mutation after
      ``set()`` cannot reach the cached bytes; this class only runs the write.
    * ``submit(key, fn, ...)`` records the future under *key*; a second submit
      for the same key waits for the first, so "set k=a; set k=b" lands in order.
    * ``wait(key)`` blocks until that key's write resolves. A failure is warned
      about (``CashCacheStoreFailedWarning``) and dropped, never re-raised: the
      value was computed, and a failed cache write must not destroy it.
    * ``shutdown()`` waits for in-flight writes only until
      ``config.shutdown_write_timeout``: it runs from ``atexit``, where an
      unbounded wait means a finished process that never exits.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._executor = _DaemonWriterPool(max_workers=max_workers)
        self._pending: dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        # ``current_key`` is set on the worker while it runs a task, so a
        # re-entrant ``drain``/``wait`` for that key from inside the task (e.g.
        # eviction deciding to drop the entry being written) does not wait on
        # itself.
        self._tls = threading.local()
        _LIVE_WRITE_QUEUES.add(self)

    def _after_fork_in_child(self) -> None:
        """Start this queue afresh in a forked child.

        A child inherits ``_pending`` with the parent's unfinished futures but
        not the writer thread that would finish them, so waiting on one --
        every read of the key, a second write of it, a listing, a clear --
        blocked the child forever. Those writes are the parent's to finish;
        the child reads what is on disk. The locks are new too: another thread
        may have held one at the moment of the fork, and nothing in the child
        would ever release it.
        """
        self._lock = threading.Lock()
        self._executor = _DaemonWriterPool(max_workers=self._executor._max_workers)
        self._pending = {}
        self._tls = threading.local()

    def is_shutdown(self) -> bool:
        """True once ``shutdown()`` has been called; no further work accepted."""
        with self._lock:
            return self._shutdown

    def _run_task(self, key: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        """Worker-side wrapper: marks ``current_key`` and records a failure as it happens."""
        self._tls.current_key = key
        prev_worker = getattr(_WORKER_THREAD, "active", False)
        _WORKER_THREAD.active = True
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            # Recorded here, not when someone next asks for the key: until
            # then a discarded write would leave no trace anyone can query.
            _record_discarded(key, exc)
            raise
        finally:
            self._tls.current_key = None
            _WORKER_THREAD.active = prev_worker

    @staticmethod
    def in_worker_thread() -> bool:
        """True when the calling thread is running a background write task.

        A write worker must never block on another backend's queue (see
        ``_WORKER_THREAD``).
        """
        return getattr(_WORKER_THREAD, "active", False)

    def current_worker_key(self) -> str | None:
        """The key of the write task this thread is running, or ``None`` off the worker."""
        return getattr(self._tls, "current_key", None)

    def submit(self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """Run ``fn(*args, **kwargs)`` in the background, tagged with *key*.

        Waits first for a previous submission for the same key, so repeated
        sets of one key land in order.
        """
        with self._lock:
            prev = self._pending.get(key)
        # Waited on outside the lock so other keys' submissions are not held up.
        if prev is not None and not prev.done():
            try:
                prev.result()
            except Exception:  # noqa: BLE001 - surfaces later via wait(key)
                pass
        if in_multiprocessing_child():
            return self._run_inline(key, fn, args, kwargs)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("PendingWrites: executor has been shut down")
            future = self._executor.submit(self._run_task, key, fn, args, kwargs)
            self._pending[key] = future
        # Registered outside the lock: the callback runs inline when the future
        # has already finished, and it takes ``_lock`` itself.
        future.add_done_callback(lambda f, k=key: self._forget_if_succeeded(k, f))
        return future

    def _run_inline(self, key: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> concurrent.futures.Future:
        """Write on the calling thread, as part of a ``multiprocessing`` task.

        A background write does not survive a pool worker: ``with Pool() as p:``
        terminates its workers on exit, and a forked worker leaves through
        ``os._exit``, which skips the exit-time drain. So each worker's last
        write was lost. Run untracked, as the writer thread would be, so the
        write's own directory scan does not become a dependency of the cached
        call that triggered it.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("PendingWrites: executor has been shut down")
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        try:
            with untracked():
                future.set_result(self._run_task(key, fn, args, kwargs))
        except BaseException as exc:  # noqa: BLE001 - reported via wait(), as below
            future.set_exception(exc)
            with self._lock:
                self._pending[key] = future
        return future

    def _forget_if_succeeded(self, key: str, future: concurrent.futures.Future) -> None:
        """Drop a finished, successful write from ``_pending``.

        Keeps ``_pending`` (and so ``wait_all``) proportional to the writes in
        flight rather than to every key ever written. Failures stay, so
        ``wait(key)`` and ``failed_writes()`` can still report them.
        """
        try:
            if future.cancelled() or future.exception() is not None:
                return
        except BaseException:  # noqa: BLE001 - a callback must never propagate
            return
        with self._lock:
            # Identity check: a later submit for the key may have replaced it.
            if self._pending.get(key) is future:
                del self._pending[key]

    def wait(self, key: str) -> None:
        """Block until the pending write for *key* (if any) finishes.

        A failed write is reported as ``CashCacheStoreFailedWarning`` and its
        future dropped -- the write did not land, so nothing is pending. It is
        not re-raised: ``wait`` runs on every lookup of the key, so a re-raise
        surfaced far from its cause, for as long as the future was kept, and on
        a cold cache killed a cell whose computation had already succeeded.

        Returns at once on the worker that is writing *key* itself.
        """
        if getattr(self._tls, "current_key", None) == key:
            return
        with self._lock:
            future = self._pending.get(key)
        if future is None:
            return
        try:
            future.result()
        except BaseException as exc:  # noqa: BLE001 - reported, never propagated
            with self._lock:
                # Only the future waited on: a later submit may have replaced it.
                if self._pending.get(key) is future:
                    del self._pending[key]

            warn_diagnostic(
                CashCacheStoreFailedWarning,
                "STORE-FAILED",
                f"cash could not store the result for {key!r} "
                f"({type(exc).__name__}: {exc}); compute succeeded, but the "
                f"entry is absent and the work will recompute.",
                "read the exception: a full disk, a cache_dir you cannot write "
                "to, or a value that cannot be pickled -- return the data, not "
                "the handle that produced it.",
            )

    def drain(self, key: str) -> None:
        """Wait for the pending write for *key* and forget it.

        Used by ``delete(key)`` so a slow write cannot land after the delete.
        On the worker writing *key* itself, forgets without waiting.
        """
        if getattr(self._tls, "current_key", None) == key:
            with self._lock:
                self._pending.pop(key, None)
            return
        with self._lock:
            future = self._pending.pop(key, None)
        if future is not None:
            try:
                future.result()
            except Exception:  # about to delete; the failure is moot
                logger.debug("Pending write for %r failed before its delete", key, exc_info=True)

    def has_pending(self, key: str) -> bool:
        """Is there an UNFINISHED write for *key* on this queue?

        Lets a caller on the worker avoid waiting for a write queued behind
        its own, which could never start. A finished-but-failed write does not
        count.
        """
        with self._lock:
            future = self._pending.get(key)
            return future is not None and not future.done()

    def pending_count(self) -> int:
        """How many writes are still in flight."""
        with self._lock:
            return sum(1 for f in self._pending.values() if not f.done())

    def wait_all(self) -> None:
        """Block until every pending write completes (skipped on the worker).

        Used by bulk reads and ``clear`` so they see every write already
        submitted. Failures are logged, not raised: one bad entry must not fail
        a listing, and ``wait(key)`` still reports it.
        """
        if getattr(self._tls, "current_key", None) is not None:
            return
        with self._lock:
            futures = list(self._pending.values())
        for f in futures:
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001 - re-raising would punish the wrong caller
                logger.debug("Pending write failed (surfaced via wait(key)): %s", exc)

    def failed_writes(self) -> list[tuple[str, BaseException]]:
        """Keys whose write raised and was never observed by a ``wait(key)``."""
        with self._lock:
            items = list(self._pending.items())
        failed = []
        for key, future in items:
            if not future.done():
                continue
            try:
                exc = future.exception(timeout=0)
            except concurrent.futures.CancelledError:
                continue
            if exc is not None:
                failed.append((key, exc))
        return failed

    def _report_failed_writes(self) -> None:
        """Log any write failure nobody observed. Never raises (runs at exit)."""
        failed = self.failed_writes()
        if not failed:
            return
        logger.warning(
            "Cash: %d cache write(s) failed and were discarded; those entries "
            "are absent, so the work will be recomputed. First failure: %s: %s",
            len(failed),
            type(failed[0][1]).__name__,
            failed[0][1],
        )
        for key, exc in failed:
            logger.debug("  failed write key=%r: %s: %s", key, type(exc).__name__, exc)

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop accepting writes; when *wait*, give the in-flight ones until
        *timeout* seconds (default ``config.shutdown_write_timeout``).

        Bounded, because this runs from ``atexit``: losing a cache entry costs
        a recompute, a job that never exits is an outage. An expired deadline
        is reported (``CACHE-WRITE-ABANDONED``) with how many writes it dropped.
        """
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        if timeout is None:
            timeout = _shutdown_write_timeout()
        finished = self._executor.shutdown(wait=wait, timeout=timeout)
        if wait:
            if not finished:
                self._warn_abandoned_writes(timeout)
            self._report_failed_writes()

    def _warn_abandoned_writes(self, timeout: float) -> None:
        """Say that the exit deadline expired with writes still running."""
        with self._lock:
            unfinished = sum(1 for f in self._pending.values() if not f.done())

        try:
            warn_diagnostic(
                CashCacheStoreFailedWarning,
                "CACHE-WRITE-ABANDONED",
                f"cash gave up waiting for {unfinished} cache write(s) after "
                f"{timeout:g}s and let the process exit; those entries were not "
                f"stored, so that work will be recomputed next run.",
                "a write this slow usually means the cache directory is "
                "unwritable or on a stalled mount -- check the path in "
                "cash.configure(cache_dir=...) or CASH_CACHE_DIR. Raise the "
                "deadline with CASH_SHUTDOWN_WRITE_TIMEOUT=<seconds> if the "
                "storage really is that slow.",
            )
        except Exception:  # noqa: BLE001 - never raise out of an atexit path
            logger.warning(
                "Cash abandoned %d cache write(s) after %gs at shutdown.",
                unfinished,
                timeout,
            )
