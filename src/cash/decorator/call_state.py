"""What one cached call carries from its lookup to its store, and the
per-call context the wrappers set."""

from __future__ import annotations

import concurrent.futures
import contextvars
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from .cache_metadata import CacheMetadata

# Sentinel object used by wrapper helpers to signal a cache miss without
# conflicting with any legitimate cached value (including None).
CACHE_MISS = object()


class KeyBuildFailed(Exception):
    """Building a key met something it cannot key, and says what to tell the user.

    Raised from inside a key build; `KeyBuilder.resolve` warns once with
    *code*, *message* and *fix*, and the call runs uncached -- never keyed
    without the part that failed, which would serve a stale result silently.
    """

    def __init__(self, code: str, message: str, fix: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix


class UnhashableDefault(Exception):
    """A parameter default could not be hashed; `ClosureFold.fold_defaults` has warned."""


class UnhashableArgs(Exception):
    """The call's arguments could not be hashed. *keyed*, when set, is the
    ``(args, kwargs)`` the key was built from (``key=``, ``ignore=``), for
    the warning to name an argument that was actually hashed."""

    def __init__(self, *args: Any, keyed: tuple[tuple, dict] | None = None) -> None:
        super().__init__(*args)
        self.keyed = keyed


class BuiltKey(NamedTuple):
    """What `KeyBuilder.build` built: the key, two of its segments, and the
    canonicalised arguments explain() reads frozen producers off.

    *call_args_hash* is the hash of every argument the call was made with.
    It is *args_hash* unless ``key=`` or ignored parameters decide the
    argument part of the key; then it is None when they cannot all be hashed.
    """

    cache_key: str
    state_hash: str
    args_hash: str
    normalized_args: tuple[tuple, dict]
    call_args_hash: str | None = None
    #: The shortest ttl among the cached functions the key reached outside
    #: the dependency graph (`REACHED_TTLS`), or None.
    reached_ttl: int | None = None
    #: Its body is recorded and the trace of what ran stored with the entry,
    #: which is served only while that trace holds (`code_trace`).
    traced: bool = False
    #: The call picks code by a run-time name: without a trace it is not
    #: stored, and an entry without one is not served.
    needs_trace: bool = False
    #: The code the key folded (`cash.code_swaps.FoldedCode`), compared
    #: again after the body; None for a key not built for a call.
    folded_code: Any = None
    #: The modules not loaded yet the code the key reached imports, whose
    #: top level cash cannot read before they are (UNREAD_MODULE): a call
    #: that loads one is not stored (`CallRunner._loaded_unread`).
    unread: frozenset = frozenset()
    #: The key's trace was asked for only by a lazy import of such a module
    #: in code an argument carries: a call that did not load it is stored
    #: by its key when the trace cannot be built.
    trace_for_unread: bool = False
    #: Built with the classes of module data held by their surface
    #: (`HELD_BY_NAME`): stored only with a trace of what ran, or else with
    #: a check of their code in full (`CallRunner.finish_miss`).
    narrowed: bool = False


class Call:
    """One call's state, from the lookup (`CallRunner.lookup`) to the store
    (`CallRunner.finish_miss`), shared by the sync and async wrappers."""

    __slots__ = (
        "args",
        "kwargs",
        "call_start",
        "ttl",
        "recompute",
        "capture_watch",
        "watch_identities",
        "cache_key",
        "state_hash",
        "ledger",
        "args_hash",
        "call_args_hash",
        "checked",
        "metadata",
        "cash_overhead",
        "outcome",
        "handed",
        "polars_args",
        "keyed_parts",
        "content_keyed",
        "by_content",
        "stored_in",
        "traced",
        "needs_trace",
        "trace_unit",
        "ran_unit",
        "folded_code",
        "full_folded",
        "full_fold_gate",
        "unread",
        "trace_for_unread",
        "narrowed",
        "closure_check",
        "refusal_probe",
        "refusal_key",
        "args_fingerprinted",
        "fill_mark",
    )

    def __init__(self, args: tuple, kwargs: dict) -> None:
        self.args = args
        self.kwargs = kwargs
        self.metadata: CacheMetadata | None = None
        self.call_args_hash: str | None = None
        #: The arguments the in-place-change check hashes, when they are not
        #: the body's: an iterator split for the key (`split_for_key`) stands
        #: as a placeholder, since the body reading its copy is no change.
        self.checked: tuple[tuple, dict] | None = None
        #: What the key's state segment was made of, by name (`STATE_LEDGER`).
        self.ledger: dict | None = None
        #: When the call started, for the ``cached_property`` slots its
        #: body fills on Enum members (`class_data.note_getter_fills`).
        self.fill_mark = sys.maxsize
        #: What the watched globals' attributes were bound to just before the
        #: body ran (`purity_checks.watched_identities`).
        self.watch_identities: dict | None = None
        self.cash_overhead = 0.0
        self.outcome: Any = CACHE_MISS
        #: The notebook statement's hand-off this call claimed (`cash.handed_digests`).
        self.handed: Any = None
        #: The key's digests of the big polars arguments (`CallPolarsDigests`),
        #: for the in-place-change check after the body.
        self.polars_args: Any = None
        #: What each list or tuple argument was keyed by (`arg_hashing.KEYED_PARTS`).
        self.keyed_parts: dict | None = None
        #: Which arguments the key holds by all they hold
        #: (`arg_hashing.ContentKeyed`).
        self.content_keyed: Any = None
        #: ``call_args_hash`` was taken by the arguments' content, not by
        #: the statement tag the key holds one of them by
        #: (`arg_hashing.content_only`): the check after the body hashes
        #: them the same way.
        self.by_content = False
        #: The tiers the store put the result in (``storage``), once stored.
        self.stored_in: list[str] | None = None
        #: The key's `BuiltKey.traced` and ``needs_trace``, and what recorded
        #: the body (`code_trace`).
        self.traced = False
        self.needs_trace = False
        self.trace_unit: Any = None
        #: That record once the trace is built from it, for what the code
        #: that ran reads (`argument_sharing.reads_of`).
        self.ran_unit: Any = None
        #: The key's `BuiltKey.folded_code`.
        self.folded_code: Any = None
        #: For a call that needs its trace and whose key was narrowed: the
        #: code its key would fold unnarrowed (`KeyBuilder.full_key_folded`),
        #: which its strict trace accepts.
        self.full_folded: Any = None
        #: Else, for such a call: what the world was before the body
        #: (`CallRunner._full_fold_gate`), so that the strict trace builds
        #: that code after the body only when it refuses what ran against
        #: the narrowed key's, and only while nothing it reads has moved.
        self.full_fold_gate: Any = None
        #: The key's `BuiltKey.unread`.
        self.unread: frozenset = frozenset()
        self.trace_for_unread = False
        #: The key's `BuiltKey.narrowed`, and the trace an entry of a call
        #: whose own trace is missing is stored with in its place: a check of
        #: its state with the classes of module data keyed in full.
        self.narrowed = False
        self.closure_check: list | None = None
        #: Whether this call keeps or drops a refusal verdict once it is
        #: checked (`CallRunner._known_refusal`), and its `RefusalVerdicts`
        #: key when the lookup made one.
        self.refusal_probe = False
        self.refusal_key: str | None = None
        #: Whether its arguments were fingerprinted before the body ran
        #: (`CachedFunction.changes_arguments` at lookup): a refusal for
        #: changing them then is not the first-seen one the next call stores.
        self.args_fingerprinted = False


class BodyRun:
    """What `CallRunner.body_scope` observed while the body ran, and what it returned."""

    __slots__ = (
        "tracker",
        "observer",
        "rng_pre",
        "carriers_pre",
        "carriers_moved",
        "arg_carriers_pre",
        "arg_carriers_moved",
        "res",
        "body_seconds",
        "nested_cash_seconds",
        "saves_seconds",
        "first_call",
        "rng_new",
        "rng_drew",
        "env_reads",
        "lib_reads",
        "warned",
        "args_before",
        "args_reads",
        "registries",
    )


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
#: `ClosureFold.fold_closure` and `GlobalsFold.fold_read_globals` add to it; `KeyBuilder.resolve`
#: sets a fresh one per key and hands it back with the key, so two threads, or
#: a cached call nested in another's key build, never share one.
#:
#: `owner_globals` is the mapping the pre-call hash was taken FROM, and it is
#: not always the decorated function's own. `GlobalsFold.fold_read_globals` also runs on
#: behalf of module-bounded HELPERS, so a global read by a helper in another
#: module lands here under a bare name that does not exist in
#: `func.__globals__` at all. Re-reading it there found None, hashed that, and
#: reported every such global as mutated by the call -- a provider registry
#: read by a client helper warned on every first call. Carrying the owning
#: mapping is what makes the after-hash look at the same variable the
#: before-hash did. Unset (None) outside a real call's key build, so
#: `explain()` records nothing.
CAPTURE_WATCH: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar("_cash_capture_watch", default=None)


#: The effective ttls of the cached functions a key build reached without a
#: graph edge of its own registry -- passed as an argument, held in a dict, a
#: list or a closure, on another instance (`KeyBuilder.callee_state`). The
#: caller refreshes at least as often as each. None outside a key build.
REACHED_TTLS: "contextvars.ContextVar[list | None]" = contextvars.ContextVar("_cash_reached_ttls", default=None)


#: The run-time lookups by name (KEY-UNTRACKABLE-DEP findings) in code the
#: values a key build reaches carry -- an argument, a data global, a capture
#: (`CodeArgs.carrier_lookups`) -- which no key part covers: set by
#: `KeyBuilder.resolve`, None otherwise. The call is then served only on a
#: trace of what it ran, or runs uncached, as the same line in the cached
#: function's own body does. A remembered fold (`state_memo`,
#: `helper_fold_memo`, a held table) leaves the lookups it found here again
#: on reuse, as it leaves its watch and ledger.
CARRIER_LOOKUPS: "contextvars.ContextVar[list | None]" = contextvars.ContextVar("_cash_carrier_lookups", default=None)


#: `ResultStore.refusal` was not handed a capture watch (the streaming path).
NO_WATCH = object()


#: Is the key being built for a call whose body the code trace records
#: (`code_trace.on`, a plain function)? Then a class of the user's whose
#: objects module data holds -- an instance read as ``lib.SVC``, a table of
#: classes -- is keyed by its surface (`trace_items.class_surface_digest`):
#: its name, its bases, what it holds, its methods by name. The code of the
#: methods that run is checked by the trace's code items, so an edit to
#: one no call runs (``Minimizer.shgo`` beside the ``leastsq`` a fit runs)
#: keeps the key. None outside a key build: each build sets it, and the
#: state of a cached callee built inside one follows it.
HELD_BY_NAME: contextvars.ContextVar[bool | None] = contextvars.ContextVar("_cash_held_by_name", default=None)


def held_by_name() -> bool:
    """`HELD_BY_NAME`, False outside a key build."""
    return bool(HELD_BY_NAME.get())


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
