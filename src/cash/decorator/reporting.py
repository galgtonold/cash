"""What the decorator tells the user: coded warnings once per function, and
the per-call log and debug lines."""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._active import EXPLAINING as _EXPLAINING
from .._clock import perf_counter as _perf_counter
from ..backends import adaptive_caps, budget_notices
from ..diagnostics import format_diagnostic, warn_diagnostic, warn_diagnostic_message
from ..exceptions import CashCacheIneffectiveWarning
from .cached_function import WARNINGS_MAX
from .call_state import CALL_ENTRY, NESTED_CASH_SECONDS
from .explain import MissKind, MissReason, entry_id_of, is_sampled_dep, same_file_key

if TYPE_CHECKING:
    from ..config import CashConfig
    from ..effectiveness import EffectivenessLedger
    from .backend_slot import BackendSlot
    from .cached_function import CachedFunction
    from .explain import MissHistory
    from .stored_keys import StoredKeyRecord

#: One line per decorated call -- hit or miss, and why -- when `debug=True` /
#: `CASH_DEBUG=1` or `verbose=True` asks for it.
calls_logger = logging.getLogger("cash.calls")

#: How many call events a `CallLog` holds. The notebook drains them after
#: every statement; nothing drains them in a script or a service, so the log
#: keeps only the most recent calls rather than one entry per call forever.
CALL_LOG_MAX = 10_000


def describe_call(entry: dict[str, Any]) -> str:
    """One line for the per-call log: what happened, and on a miss, why."""
    name = entry["func_name"]
    # The id `cash inspect` lists and `cash clear --entry` takes, so a log
    # line can be matched to an entry on disk.
    key = entry.get("cache_key") or ""
    tag = f"  [{entry_id_of(key)}]" if key else ""
    if entry["cache_hit"]:
        saved = entry.get("time_saved") or 0.0
        lookup = entry.get("execution_time") or 0.0
        # What the hit cost, when it is not small: the summary said "time
        # saved" while warm runs were 9x slower than uncached.
        if lookup >= 0.01 and lookup >= 0.1 * saved:
            verdict = "; a net loss" if lookup > saved else ""
            line = f"HIT  {name}{tag}  (saved {saved:.2f}s; the lookup took {lookup:.2f}s{verdict})"
        else:
            line = f"HIT  {name}{tag}  (saved {saved:.2f}s)"
        sampled = entry.get("sampled_files")
        if sampled:
            # Larger than file_hash_full_max_bytes: the HIT rests on the
            # timestamps, and "when it does not recompute I need to be sure
            # it was right not to" had no way to see that.
            shown = ", ".join(os.path.basename(p) for p in sampled[:3])
            more = f" and {len(sampled) - 3} more" if len(sampled) > 3 else ""
            line += f"  -- trusts the timestamps of {shown}{more} (sampled: larger than file_hash_full_max_bytes)"
        return line
    missed = entry.get("miss_reason") or MissReason(MissKind.FIRST)
    if missed.kind is MissKind.RAISED:
        return f"RAISE {name}  {missed.text}; nothing stored  (ran {entry['execution_time']:.2f}s)"
    line = f"MISS {name}{tag}  {missed}"
    # The body's own time: the persistence floor named beside it is judged
    # on that, and the call's time -- key, analysis, lookup -- made "ran
    # 0.20s ... under the 0.1s floor" read as a contradiction.
    ran = entry.get("body_seconds")
    line += f"  (ran {entry['execution_time'] if ran is None else ran:.2f}s"
    if entry.get("not_stored"):
        line += f"; not stored: {entry['not_stored']}"
    elif entry.get("not_persisted"):
        line += f"; kept in RAM only -- {entry['not_persisted']} -- so another process will recompute it"
    return line + ")"


class Notices:
    """The coded warnings of one `Cash`: each shown at most once per function,
    and kept in that function's ``cache_info()['warnings']``."""

    def __init__(
        self,
        cached: dict[str, CachedFunction],
        functions: dict[str, Callable[..., Any]],
        stored_keys: StoredKeyRecord,
        backend_slot: BackendSlot,
    ) -> None:
        self._cached = cached
        self._functions = functions
        self._stored_keys = stored_keys
        self._backend_slot = backend_slot
        #: Guards the seen set and every function's warnings log.
        self.lock = threading.Lock()
        # What `warn_once` has shown: (category, func_name, arg_type_name, code).
        self._seen: set[tuple[type[Warning], str, str, str]] = set()

    def warn_once(
        self,
        category: type[Warning],
        func_name: str,
        arg_type_name: str,
        message: str,
        *,
        code: str,
        fix: str,
        once_per_version: bool = False,
    ) -> None:
        """Emit a coded diagnostic at most once per
        ``(category, func_name, arg_type_name, code)`` for this Cash instance.

        ``once_per_version``: and once per CACHE for the same text -- which
        names the lines and the code it found them in -- so a later process
        records it in ``cache_info()['warnings']`` without printing it. For
        the static findings a source reading makes, which were the same 32
        lines in a nightly job's log every night; an edit that
        changes what they say shows them again.

        ``message`` is one sentence of *what happened*; ``fix`` is one
        imperative sentence; ``code`` is the diagnostic code from
        ``cash.diagnostics`` that names the section of ``docs/warnings.md``
        expanding both. The three are rendered together by
        :func:`~cash.diagnostics.format_diagnostic`, and the rendered text is
        what reaches BOTH stderr and ``cache_info()['warnings']`` -- the log
        and the terminal must not drift apart, since the log is where people
        look once the stderr line has scrolled away.

        ``arg_type_name`` is the empty string for warnings that do not
        attach to a specific arg type (e.g. store-failed). The seen-set
        key still distinguishes by func_name.

        The warning blames the nearest frame outside ``cash/``, found at emit
        time (see :func:`~cash.diagnostics._stacklevel_of_first_user_frame`):
        one helper is reached at different call depths, so no fixed
        ``stacklevel`` could name the user's line.
        """
        if _EXPLAINING.get():
            return
        rendered = format_diagnostic(code, message, fix)  # raises on a bad code
        # The code is part of the key: warnings sharing a category, function
        # and (often empty) arg type are still different warnings, and one
        # must not silence another.
        key = (category, func_name, arg_type_name, code)
        with self.lock:
            if key in self._seen:
                return
            self._seen.add(key)
            # Also record in per-function rolling log so the warning is
            # discoverable after the fact via ``f.cache_info()['warnings']``
            # - even if the user missed the stderr emission. The code goes in
            # as its own field as well as inside the text, so a reader of the
            # log can branch on it the way a warning handler branches on
            # ``w.message.code``.
            entry = {
                "category": category.__name__,
                "code": code,
                "message": rendered,
                "timestamp": time.time(),
            }
            cf = self._cached.get(func_name)
            if cf is not None:
                cf.warnings.append(entry)
                del cf.warnings[:-WARNINGS_MAX]
        if once_per_version and not self._first_showing(func_name, rendered):
            entry["shown_by_an_earlier_run"] = True
            return
        warn_diagnostic_message(category, code, rendered, fallback=self._definition_site(func_name))

    def has_warned(self, key: tuple[type[Warning], str, str, str]) -> bool:
        """Has `warn_once` already shown the warning *key* names?"""
        return key in self._seen

    def log_of(self, cf: CachedFunction) -> list[dict[str, Any]]:
        """A copy of *cf*'s recent warnings, for ``cache_info()``."""
        with self.lock:
            return list(cf.warnings)

    def forget(self, cf: CachedFunction) -> None:
        """Drop *cf*'s warnings log and what it was shown, so its next
        misbehavior warns again instead of staying silent."""
        with self.lock:
            cf.warnings.clear()
            self._seen = {k for k in self._seen if k[1] != cf.name}

    def _first_showing(self, func_name: str, rendered: str) -> bool:
        """Has no earlier run on this cache shown *rendered*? Records that one
        has. True whenever the cache keeps no record -- when in doubt, show."""
        try:
            self._backend_slot.backend  # the first call is about to build it for its lookup anyway
        except Exception:  # noqa: BLE001 - no backend, no record: show it
            return True
        if self._stored_keys.path(func_name) is None:
            return True
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        if digest in self._stored_keys.read(func_name)["warned"]:
            return False
        self._stored_keys.note_warning_shown(func_name, digest)
        return True

    def _definition_site(self, func_name: str) -> tuple[str, int] | None:
        """Where *func_name* is defined: what a warning blames when the call
        runs on a pool thread, whose stack holds nothing of the user's."""
        fn = self._functions.get(func_name)
        try:
            code = getattr(inspect.unwrap(fn), "__code__", None) if fn is not None else None
        except ValueError:  # a wrapper chain that loops
            return None
        return (code.co_filename, code.co_firstlineno) if code is not None else None

    def cache_if_raised(
        self,
        func_name: str,
        error: BaseException,
    ) -> None:
        """Surface a raised ``cache_if`` predicate as a user-visible warning.

        A one-shot `CashCacheIneffectiveWarning` rather than a log line, so a
        buggy predicate is diagnosed instead of silently disabling the cache.
        """
        self.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "cache_if",
            f"@cash.cache on {func_name}: cache_if predicate raised "
            f"{type(error).__name__} ({error}), so the result is returned "
            f"un-cached and every later call recomputes.",
            code="CACHE-IF-RAISED",
            fix="make the predicate total -- it must handle every shape the "
            "result can take -- or drop cache_if= to restore caching.",
        )

    def metadata_invalid(
        self,
        func_name: str,
        error: BaseException,
    ) -> None:
        """Surface a malformed cache-metadata read as a user-visible warning.

        Happens when a backend returns a metadata dict missing the
        expected keys (e.g. a partially-written entry from an older
        cash version, or a corrupted file on disk). The call falls
        through to recompute - but the user should know.
        """
        self.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "metadata_invalid",
            f"@cash.cache on {func_name}: a stored cache entry's metadata "
            f"could not be validated ({type(error).__name__}: {error}), so "
            f"cash treated the entry as absent and recomputed.",
            code="STORE-METADATA-INVALID",
            fix="nothing, for a one-off; if it keeps appearing, run "
            "f.cache_clear() so the unreadable records are replaced.",
        )

    def evicted_recompute(self, func_name: str, seconds: float) -> None:
        """A result the disk cap had evicted was computed again, and it took
        long enough to matter (CACHE-EVICTED-RECOMPUTE). Once per function."""
        if seconds < budget_notices.EVICTED_RECOMPUTE_WARN_SECONDS:
            return
        backend = self._backend_slot.built
        try:
            budget = backend.disk_budget() if backend is not None else None
        except Exception:  # noqa: BLE001 - the warning must not fail the call
            budget = None
        free = adaptive_caps.free_bytes_on_volume(budget.cache_dir) if budget is not None else 0
        message, fix = budget_notices.evicted_recompute_warning(
            f"@cash.cache on {func_name}: the result for these arguments", seconds, budget, free
        )
        self.warn_once(
            CashCacheIneffectiveWarning, func_name, "evicted", message, code="CACHE-EVICTED-RECOMPUTE", fix=fix
        )

    def lock_failed(
        self,
        func_name: str,
        error: BaseException,
    ) -> None:
        """Surface a backend-locking failure as a user-visible warning.

        A CashCacheIneffectiveWarning rather than a log line, which default
        logging settings hide, so the user notices the implicit race risk.
        """
        self.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "lock_failed",
            f"@cash.cache on {func_name}: backend lock acquisition failed "
            f"({type(error).__name__}: {error}), so cash proceeded without the "
            f"lock and concurrent calls with the same args may compute "
            f"redundantly.",
            code="STORE-LOCK-FAILED",
            fix="investigate the backend the exception names -- a full disk, a "
            "stale lock file, or a cache_dir on a filesystem where locking "
            "does not work.",
        )


class CallLog:
    """One event per decorated call -- for the notebook's badge, for
    ``cache_info()`` and for the per-call log line -- and the running account
    of what caching cost against what it saved."""

    def __init__(
        self,
        config: CashConfig,
        cached: dict[str, CachedFunction],
        misses: MissHistory,
        effectiveness: EffectivenessLedger,
    ) -> None:
        self._config = config
        self._cached = cached
        self._misses = misses
        #: What caching cost vs what it saved, per function. It only ever
        #: informs: the decorator caches because the user asked it to.
        self.effectiveness = effectiveness
        #: The recent call events, oldest first. The notebook statement
        #: processor drains them after each statement (`drain`) for its badge.
        #: Bounded: outside a notebook nothing drains them, and a long-running
        #: process must not keep every call.
        self.entries: deque[dict[str, Any]] = deque(maxlen=CALL_LOG_MAX)
        self._lock = threading.Lock()

    def log(
        self,
        func_name: str,
        cache_hit: bool,
        execution_time: float,
        args_hash: str,
        cache_key: str,
        time_saved: float = 0.0,
        miss: MissReason | None = None,
        body_seconds: float | None = None,
        cash_seconds: float | None = None,
        file_deps: dict | None = None,
    ) -> None:
        """Record a decorator call event for notebook integration.

        Thread-safe: uses a lock to protect concurrent appends.
        The notebook ``StatementProcessor`` drains this log after each
        statement execution to include decorator call metrics in the badge;
        it keeps the last ``CALL_LOG_MAX`` events, since nothing drains it
        outside a notebook. The entry also goes to the running call's
        `CALL_ENTRY` slot, which is what ``cache_info()`` counts.

        ``execution_time`` is the wall-time of *this* operation - a lookup on a
        hit, the compute on a miss. ``time_saved`` is the compute a hit
        *avoided* (the originally-measured execution time stored with the
        cached entry), and 0.0 on a miss. They are distinct: a hit's
        ``execution_time`` is microseconds, but its ``time_saved`` is the full
        compute it stood in for. ``cache_info()['total_time_saved']`` sums the
        latter.

        *miss* is the reason for a miss that had no lookup (no key, or the
        body raised); a looked-up miss takes the one `MissHistory.note_miss` held.
        """
        # What cash spent on this call rather than the body: the whole of a
        # hit, and what the miss path measured around a body. Added to the
        # caller's tally when this call is nested in another cached call's
        # body (`NESTED_CASH_SECONDS`).
        if cash_seconds is None:
            cash_seconds = execution_time if cache_hit else 0.0
        nested = NESTED_CASH_SECONDS.get()
        if nested is not None:
            nested[0] += cash_seconds
        entry = {
            "func_name": func_name,
            "cache_hit": cache_hit,
            "execution_time": execution_time,
            "body_seconds": body_seconds,
            "time_saved": time_saved,
            "cash_seconds": cash_seconds,
            "args_hash": args_hash,
            "cache_key": cache_key,
            "timestamp": time.time(),
        }
        outcome: dict[str, Any] = {}
        if not cache_hit:
            entry["miss_reason"] = miss or self._misses.take_pending(cache_key) or MissReason(MissKind.FIRST)
            outcome = self._misses.outcome(cache_key) or {}
            # Only this call's own outcome. A streamed result is logged before
            # it is stored, and must not borrow the previous call's verdict.
            if outcome.get("at", 0) < entry["timestamp"] - execution_time:
                outcome = {}
            entry["not_persisted"] = outcome.get("not_persisted")
            entry["not_stored"] = outcome.get("not_stored")
        with self._lock:
            self.entries.append(entry)
        slot = CALL_ENTRY.get()
        if slot is not None:
            slot[0] = entry
        if self.per_call_lines():
            if file_deps:
                # Only for the line: a hit pays nothing for it otherwise.
                # One name per file: the tracker can record a file under both
                # the relative and the absolute path it was opened by.
                sampled: dict[str, str] = {}
                for path, rec in file_deps.items():
                    if is_sampled_dep(rec):
                        sampled.setdefault(same_file_key(path), path)
                entry["sampled_files"] = tuple(sampled.values())
            calls_logger.info("%s", describe_call(entry))

    def per_call_lines(self) -> bool:
        """Is the one-line-per-call log on? Asked for, not merely permitted: an
        application that turned the `cash` logger up to INFO did not ask for a
        line per call."""
        return bool(self._config.verbose or self._config.debug) and calls_logger.isEnabledFor(logging.INFO)

    def log_raised(self, func_name: str, exc: BaseException, call_start: float) -> None:
        """Record a call whose body raised: nothing is stored, and it counts.

        Such a call produced no line at all, and a run that crashed half-way
        summarised as "5 of 5 calls restored".
        """
        self.log(
            func_name,
            cache_hit=False,
            execution_time=_perf_counter() - call_start,
            args_hash="",
            cache_key="",
            miss=MissReason(MissKind.RAISED, f"{type(exc).__name__}: {str(exc)[:80]}"),
        )

    def drain(self) -> list[dict[str, Any]]:
        """Return and clear the recorded events, atomically."""
        with self._lock:
            calls = list(self.entries)
            self.entries.clear()
        return calls

    def note_effectiveness(
        self,
        func_name: str,
        overhead_seconds: float,
        *,
        body_seconds: float | None,
        was_hit: bool,
    ) -> None:
        """Account for one call and warn if caching has become a net loss.

        Informational only: the decorator caches because the user asked it
        to, and deciding otherwise is the notebook cost model's job, not
        this one. See ``cash.effectiveness`` for when it speaks up.

        ``overhead_seconds`` is everything cash did around the body: the key and
        lookup, and on a miss the store as well: the store is not a once-per-key
        cost, since a result kept in RAM only is copied in every process that
        computes it.
        """
        cf = self._cached.get(func_name)
        culprit = cf.arg_cost if cf is not None else None
        producer = self._cached.get(culprit[3]) if culprit is not None and culprit[3] else None
        if producer is not None and producer.frozen:
            # Already frozen: advising frozen=True on it is noise.
            culprit = (*culprit[:3], None, *culprit[4:])
        try:
            verdict = self.effectiveness.record(
                func_name,
                overhead_seconds=overhead_seconds,
                body_seconds=body_seconds,
                was_hit=was_hit,
                culprit=culprit,
            )
        except Exception:  # noqa: BLE001 - accounting must never break a call
            return
        # The warn is deliberately OUTSIDE that guard. Under ``-W error`` it
        # raises into the caller: the user configured warnings-as-errors and
        # is entitled to have that honoured. This is unlike the backend, which
        # never raises over a failure the user did not ask to hear about.
        # Swallowing it would silently override their filter.
        if verdict:
            what, fix = verdict
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "CACHE-NET-LOSS",
                what,
                fix,
            )
