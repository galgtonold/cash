"""The end-of-run table: per cached function, how its calls went.

Printed at exit under the ``summary`` setting, and by ``show_stats`` outside
a notebook. Reads the stats each `CachedFunction` keeps; never builds a
backend or a cache directory, since it runs from an ``atexit`` handler.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .. import _log
from ..backends._writes import in_multiprocessing_child
from ..backends.budget_notices import DiskBudget, cap_text
from .explain import MissKind

if TYPE_CHECKING:
    from ..config import CashConfig
    from .backend_slot import BackendSlot
    from .registry import FunctionRegistry

#: Widest a function name column gets; longer names keep their tail.
_NAME_WIDTH = 44


class RunSummary:
    """The hit/miss table of one `Cash` instance's cached functions."""

    def __init__(self, config: Callable[[], CashConfig], registry: FunctionRegistry, backend_slot: BackendSlot) -> None:
        self._config = config
        self._registry = registry
        self._backend_slot = backend_slot

    def text(self) -> str:
        """A per-function hit/miss table for this process, or ``""``.

        Empty when no cached function was ever called, so a caller can print
        this unconditionally without emitting a header over nothing.
        """
        stats = [(name, cf.stats) for name, cf in self._registry.cached.items()]
        rows = [(name, s) for name, s in stats if s["hits"] or s["misses"]]
        bypassed = sum(s.get("bypassed", 0) for _, s in stats)
        disabled_line = (
            f"cash: caching disabled (disable=True / CASH_DISABLE) -- {bypassed} "
            f"call{'' if bypassed == 1 else 's'} ran uncached"
            if bypassed
            else ""
        )
        if not rows:
            return disabled_line
        rows.sort(key=lambda r: r[1]["total_time_saved"], reverse=True)

        lines = [_head(rows)]
        where = self._cache_dir()
        if where:
            # Which directory this ran against. A script user has no badge and
            # no other place to see it, and every question that starts "why is
            # nothing cached" is answered or excluded by this one line: a
            # scheduled job's cwd-relative cache, a path typed with one
            # backslash too few, a container volume that is not the one they
            # meant.
            lines.append(f"  cache: {where}{self._budget()}")
        width = min(_NAME_WIDTH, max(len(name) for name, _ in rows))
        for name, stat in rows:
            lines.append(_row(name, stat, width))
            lines.extend(self.reasons(stat))
        if disabled_line:
            lines.append("  " + disabled_line)
        return "\n".join(lines)

    @staticmethod
    def reasons(stat: dict[str, Any]) -> list[str]:
        """The indented lines under a summary row: why it missed, what stayed.

        A bare "1 miss" would hide the common surprise: a result computed in
        0.05 s is never written to disk, so every new process misses it. The
        run that CAUSES that is the one that can say so.
        """
        out = []
        reasons = stat.get("miss_reasons") or {}
        if reasons:
            out.append(
                "      missed: " + ", ".join(f"{n} {kind}" for kind, n in sorted(reasons.items(), key=lambda r: -r[1]))
            )
        for what, n in (stat.get("changed") or {}).items():
            out.append(f"      {MissKind.CODE} ({n}x): {what}")
        for why, n in (stat.get("not_stored") or {}).items():
            out.append(f"      not stored ({n}x): {why}")
        for why, n in (stat.get("not_persisted") or {}).items():
            out.append(f"      kept in RAM only ({n}x): {why}; a new process recomputes it")
        return out

    def _cache_dir(self) -> str | None:
        """The cache directory this instance is using, for the summary header.

        Reads the ALREADY-BUILT backend when there is one and falls back to the
        configured path otherwise: the summary must never be the thing that
        creates a cache directory, and it runs from an ``atexit`` handler where
        building one is worse than saying nothing.
        """
        path = self._backend_slot.local_dir()
        if path:
            return path
        configured = getattr(self._config(), "cache_dir", None)
        return configured if isinstance(configured, str) and configured else None

    def _budget(self) -> str:
        """``", up to 26.0 GiB (a quarter of the free disk space)"`` for the
        summary's cache line, or ``""``. From the built backend only."""
        backend = self._backend_slot.built
        try:
            budget = backend.disk_budget() if backend is not None else None
        except Exception:  # noqa: BLE001 - the summary runs at exit and must not fail
            return ""
        if not isinstance(budget, DiskBudget):
            return ""
        return f", up to {cap_text(budget.cap, budget.why is not None)} ({budget.why or 'set by max_cache_size'})"

    def print_at_exit(self) -> None:
        """``atexit`` hook for ``summary=True``. Must never raise.

        Interpreter shutdown tears modules down underneath handlers, so a
        diagnostic that explodes here would turn a finished run into a
        traceback the user cannot act on.
        """
        try:
            text = self.text()
            if text:
                if in_multiprocessing_child():
                    # One table per worker process: say whose it is.
                    text = text.replace("cash:", f"cash (pid {os.getpid()}):", 1)
                _emit(text)
        except Exception:  # noqa: BLE001 - a summary must not fail a finished run
            pass


def _cash_seconds(stat: dict[str, Any]) -> float:
    """What cash cost one function: the hits' lookups AND the misses' keys,
    checks and stores. Counting the lookups alone reported a 24 s loss as
    14 s, and a function that never hit as costing nothing."""
    return stat.get("lookup_seconds", 0.0) + stat.get("miss_overhead_seconds", 0.0)


def _head(rows: list[tuple[str, dict[str, Any]]]) -> str:
    """The summary's first line: calls restored, time saved, time cash spent."""
    hits = sum(s["hits"] for _, s in rows)
    calls = hits + sum(s["misses"] for _, s in rows)
    saved = sum(s["total_time_saved"] for _, s in rows)
    spent = sum(_cash_seconds(s) for _, s in rows)
    head = f"cash: {hits} of {calls} calls restored, {saved:.1f}s saved"
    if spent >= 0.1 and spent >= 0.1 * saved:
        # Saved is the compute the hits stood in for; cash itself cost
        # this much. Left out, a run that got 9x SLOWER read as a win.
        head += f" -- and {spent:.1f}s spent by cash on keys, lookups and stores"
        if spent > saved:
            head += f", a net loss of {spent - saved:.1f}s"
    return head


def _fit(name: str, width: int) -> str:
    """*name* cut to *width*, keeping the TAIL.

    A name is ``module.qualname``, so for anything nested -- a closure, a
    method, a test helper -- the part that identifies it is at the end, and
    truncating from the right threw away the function name and kept the
    package path.
    """
    return name if len(name) <= width else "..." + name[-(width - 3) :]


def _row(name: str, stat: dict[str, Any], width: int) -> str:
    """One function's line of the table."""
    # Pad the whole "N hits," token, not the word: padding the word
    # puts the space before the comma ("1 hit ,").
    hit_col = f"{stat['hits']} {'hit' if stat['hits'] == 1 else 'hits'},"
    miss_col = f"{stat['misses']} {'miss' if stat['misses'] == 1 else 'misses'}"
    saved_col = f"{stat['total_time_saved']:.1f}s saved" if stat["total_time_saved"] else "-"
    cost = _cash_seconds(stat)
    if cost >= 0.1 and cost >= 0.1 * stat["total_time_saved"]:
        saved_col += f", {cost:.1f}s spent by cash"
    return f"  {_fit(name, width):<{width}}  {hit_col:<10}{miss_col:<12}{saved_col}"


def _emit(text: str) -> None:
    """Hand the summary *text* to the application's log, or to stderr.

    stderr, not stdout: stdout is the program's output -- a report, a pipe, a
    JSON response -- and a summary landing in it broke all three. ONE write:
    pool workers exiting together interleave separate writes mid-line.

    Through the application's handlers whenever it has one that takes INFO --
    past the level filters, as CASH_DEBUG's lines are: CASH_SUMMARY asked for
    it, and a service whose output goes through dictConfig must see it. Gated
    on the levels, the block would come through the app's formatter at INFO
    and raw at WARNING, two shapes for a log shipper to parse. Never both:
    with cash's own handler passing it on as well, that prints it three times.
    """
    cash_logger = logging.getLogger("cash")
    if any(h.level <= logging.INFO for h in _log.application_handlers(cash_logger)):
        summary_logger = logging.getLogger("cash.summary")
        summary_logger.handle(
            summary_logger.makeRecord(summary_logger.name, logging.INFO, "(cash summary)", 0, "%s", (text,), None)
        )
    else:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
