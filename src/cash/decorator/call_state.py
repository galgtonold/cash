"""What one cached call carries from its lookup to its store, and the
per-call context the wrappers set."""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from ..backends import CacheMetadata

# Sentinel object used by wrapper helpers to signal a cache miss without
# conflicting with any legitimate cached value (including None).
CACHE_MISS = object()


class KeyBuildFailed(Exception):
    """Building a key met something it cannot key, and says what to tell the user.

    Raised from inside a key build; `_resolve_cache_key` warns once with
    *code*, *message* and *fix*, and the call runs uncached -- never keyed
    without the part that failed, which would serve a stale result silently.
    """

    def __init__(self, code: str, message: str, fix: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix


class UnhashableDefault(Exception):
    """A parameter default could not be hashed; `_fold_defaults` has warned."""


class UnhashableArgs(Exception):
    """The call's arguments could not be hashed."""


class BuiltKey(NamedTuple):
    """What `Cash._build_key` built: the key, two of its segments, and the
    canonicalised arguments explain() reads frozen producers off."""

    cache_key: str
    state_hash: str
    args_hash: str
    normalized_args: tuple[tuple, dict]


class Call:
    """One call's state, from the lookup (`Cash._lookup`) to the store
    (`Cash._finish_miss`), shared by the sync and async wrappers."""

    __slots__ = (
        "args",
        "kwargs",
        "call_start",
        "ttl",
        "recompute",
        "capture_watch",
        "cache_key",
        "state_hash",
        "args_hash",
        "metadata",
        "cash_overhead",
        "outcome",
    )

    def __init__(self, args: tuple, kwargs: dict) -> None:
        self.args = args
        self.kwargs = kwargs
        self.metadata: CacheMetadata | None = None
        self.cash_overhead = 0.0
        self.outcome: Any = CACHE_MISS


class BodyRun:
    """What `Cash._body_scope` observed while the body ran, and what it returned."""

    __slots__ = ("tracker", "observer", "rng_pre", "res", "body_seconds", "saves_seconds", "rng_new")


def run_to_completion(make_coroutine: Callable[[], Any]) -> Any:
    """Run a coroutine to completion from synchronous code, and return its result.

    On a thread of its own with a fresh event loop, because the caller may be
    inside a running loop already (a cached iterator being read in async code),
    where ``asyncio.run`` refuses. Used to recompute an async function's
    iterator when a stored chunk has gone.
    """
    # Local: asyncio adds ~76ms to `import cash`, and only async callers need it.
    import asyncio

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coroutine())).result()


#: When this process started: keys the stored-key record holds from before it
#: were written by earlier runs.
PROCESS_STARTED = time.time()


#: The stats wrapper's slot for the call it is running: `CallLog.log`
#: puts the call's entry there, and the stats wrapper counts it once the call
#: returns or raises. Per context (thread or asyncio task), and set afresh by
#: every cached call, so a nested call fills its own slot and never the
#: caller's.
CALL_ENTRY: "contextvars.ContextVar[list | None]" = contextvars.ContextVar("_cash_call_entry", default=None)


#: The capture watch of the key being built: {name: (pre-call hash, scope,
#: owner_globals, owner)} for every provisional capture folded into it. Both
#: `_fold_closure` and `_fold_read_globals` add to it; `_resolve_cache_key`
#: sets a fresh one per key and hands it back with the key, so two threads, or
#: a cached call nested in another's key build, never share one.
#:
#: `owner_globals` is the mapping the pre-call hash was taken FROM, and it is
#: not always the decorated function's own. `_fold_read_globals` also runs on
#: behalf of module-bounded HELPERS, so a global read by a helper in another
#: module lands here under a bare name that does not exist in
#: `func.__globals__` at all. Re-reading it there found None, hashed that, and
#: reported every such global as mutated by the call -- a provider registry
#: read by a client helper warned on every first call. Carrying the owning
#: mapping is what makes the after-hash look at the same variable the
#: before-hash did. Unset (None) outside a real call's key build, so
#: `explain()` records nothing.
CAPTURE_WATCH: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar("_cash_capture_watch", default=None)


#: `_store_refusal` was not handed a capture watch (the streaming path).
NO_WATCH = object()


#: The seconds cash spent inside the body of the cached call in progress, on
#: its nested cached calls: their keys, lookups, stores. They belong to those
#: calls, not to this body -- counted in, an outer function's "saved" was 4-9x
#: what running it uncached costs. A one-element list, so a nested
#: call adds to its caller's without resetting anything.
NESTED_CASH_SECONDS: contextvars.ContextVar[list | None] = contextvars.ContextVar("_cash_nested_seconds", default=None)


#: Threads inside a cached call right now, and how deep each is. A hit's saving
#: is the body time it stood in for, and sixteen 0.5 s hits on eight threads
#: stood in for 1 s of waiting, not 8 s: the summary divides a
#: hit's saving by how many threads were running cached calls with it.
_CALL_DEPTH = threading.local()
THREADS_IN_CALLS = [0]
_THREADS_IN_CALLS_LOCK = threading.Lock()


def enter_cached_call() -> None:
    depth = getattr(_CALL_DEPTH, "depth", 0)
    _CALL_DEPTH.depth = depth + 1
    if depth == 0:
        with _THREADS_IN_CALLS_LOCK:
            THREADS_IN_CALLS[0] += 1


def exit_cached_call() -> None:
    depth = getattr(_CALL_DEPTH, "depth", 1) - 1
    _CALL_DEPTH.depth = depth
    if depth == 0:
        with _THREADS_IN_CALLS_LOCK:
            THREADS_IN_CALLS[0] -= 1
