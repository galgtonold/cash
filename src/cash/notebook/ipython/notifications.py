"""The badge rows a cell gets besides its statements' own: what changed
before it ran (modules reloaded, functions redefined, opaque calls), a
notebook file proven stale, and cache writes that failed.

Each is built here and only appended by the cell executor or the magics.
Every builder is a diagnostic: none may raise into the cell.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...backends._writes import discarded_writes
from ..cache_status import CacheStatus
from ..statement import ProcessResult

__all__ = [
    "discarded_writes_notification",
    "function_change_rows",
    "module_reloaded_row",
    "opaque_call_rows",
    "stale_notebook_rows",
    "staleness_notification",
]

logger = logging.getLogger(__name__)


def staleness_notification(tracker) -> dict | None:
    """Badge row for a notebook file cash has PROVEN is out of date.

    Returns None unless there is proof. This is deliberately quiet: a warning
    that appears when cash is merely unsure is a warning users learn to skip,
    and this one needs to be believed the once it matters.

    ASCII only: `code` is written into the saved .ipynb and may be read back
    by a different process (nbconvert, a log scraper, an agent) on a console
    whose codepage cash cannot know. See `cash.notebook.staleness._to_ascii`,
    which sanitises `hint()` for the same reason before it ever gets here.

    Message order is load-bearing, not stylistic. `%cash_badge print` renders
    `code` through `renderers.text._row_line`, which hard-truncates a row's
    first line at `theme.HEADER_MAX_LEN` (80 chars) -- there is no tooltip or
    drawer in that mode to hold the rest, unlike HTML. The fact of staleness
    and the remedy ("Save and re-run") are what make the row actionable, so
    they go FIRST, comfortably inside the cap; the save time and the cell
    hint are supporting evidence, appended after, and may be silently cut off
    in print mode. Keep the essential clause short enough that it plus a
    small margin stays under 80 chars even after the RNG suffix a future
    change might add.
    """
    if not tracker.is_stale():
        return None
    saved = tracker.saved_at()
    when = time.strftime("%H:%M:%S", time.localtime(saved)) if saved else "an earlier time"
    hint = tracker.hint()
    where = f" '{hint}' differs from the saved copy." if hint else ""
    return {
        "status": CacheStatus.WARNING,
        "code": (
            f"[!] Notebook file is stale -- Save (Ctrl+S) and re-run to be sure. "
            f"Upstream check used the copy saved at {when}.{where} "
            f"Other cells may have changed too."
        ),
        "is_upstream": True,
        "total_time": 0.0,
        "execution_time": 0.0,
        "outputs": [],
    }


def discarded_writes_notification(seen_before: int) -> tuple[dict | None, int]:
    """Badge row said when a cache write failed and was thrown away.

    Returns ``(row_or_None, new_total)`` so the caller can carry the watermark
    to the next cell.

    A discarded write is the one failure the rest of the badge cannot express.
    It is not a miss -- a miss is a row that says EXECUTED and tells you so. It
    is a hit that never got the chance to exist: the entry is absent, the work
    recomputes every run, and every counter on the badge looks healthy. Windows
    spent an unknown period doing exactly this on every run (fixed in 0.4.1),
    and the only report was a logger warning at kernel shutdown, which in a
    notebook means never.

    Loud on every occurrence: this reports work being lost right now, and a
    second occurrence is a second lost result rather than a repeat of the same
    news.

    ASCII only and short, for the reasons `staleness_notification` gives -- the
    print renderer caps a row at 80 characters.
    """
    try:
        total = len(discarded_writes())
    except Exception:  # noqa: BLE001 - a diagnostic must never break a cell
        return None, seen_before
    if total <= seen_before:
        return None, total

    new = total - seen_before
    plural = "s" if new != 1 else ""
    return {
        "status": CacheStatus.WARNING,
        "code": (f"[!] {new} cache write{plural} failed -- not cached, will recompute. See %cash_stats."),
        "is_upstream": False,
        "total_time": 0.0,
        "execution_time": 0.0,
        "outputs": [],
    }, total


def module_reloaded_row(changed_modules: dict[str, str]) -> ProcessResult:
    """The badge row for tracked modules cash just reloaded."""
    mod_names = ", ".join(sorted(changed_modules.keys()))
    return {
        "status": CacheStatus.MODULE_RELOADED,
        # No glyph: this text reaches `%cash_badge print`, whose readers are
        # often cp1252 consoles. The label says it already.
        "code": f"Module{'s' if len(changed_modules) > 1 else ''} reloaded: {mod_names}",
        "is_upstream": True,
        "total_time": 0.0,
        "execution_time": 0.0,
        "saved_time": 0.0,
        "error": None,
        "restored_vars": [],
        "uncacheable_reasons": [],
        "outputs": [],
        "changed_modules": dict(changed_modules.items()),
    }


def function_change_rows(function_tracker: Any, user_ns: dict) -> list[ProcessResult]:
    """Return notification metrics for any user-defined functions that changed source."""
    try:
        changed_funcs = function_tracker.detect_changed_functions(user_ns)
        if not changed_funcs:
            return []
        func_names = ", ".join(sorted(changed_funcs))
        logger.debug("[FUNCTION_CHANGE] Detected changed functions: %s", func_names)
        return [
            {
                "status": CacheStatus.FUNCTION_CHANGED,
                "code": f"Function{'s' if len(changed_funcs) > 1 else ''} changed: {func_names}",
                "is_upstream": True,
                "execution_time": 0.0,
                "total_time": 0.0,
                "saved_time": 0.0,
                "error": None,
                "restored_vars": [],
                "uncacheable_reasons": [],
                "outputs": [],
                "changed_functions": sorted(changed_funcs),
            }
        ]
    except (AttributeError, TypeError, OSError) as exc:
        logger.debug("Failed to check function changes: %s", exc)
        return []


def opaque_call_rows(function_tracker: Any, raw_cell: str, user_ns: dict) -> list[ProcessResult]:
    """Return WARNING metrics for opaque call patterns detected in raw_cell."""
    try:
        opaque_warnings = function_tracker.detect_opaque_call_patterns(raw_cell, user_ns)
        if not opaque_warnings:
            return []
        for w in opaque_warnings:
            logger.debug("[OPAQUE_CALL] %s", w)
        return [
            {
                "status": CacheStatus.WARNING,
                "code": f"⚠️ {msg}",
                "is_upstream": True,
                "execution_time": 0.0,
                "total_time": 0.0,
                "saved_time": 0.0,
                "error": None,
                "restored_vars": [],
                "uncacheable_reasons": [],
                "outputs": [],
            }
            for msg in opaque_warnings
        ]
    except (AttributeError, TypeError, SyntaxError, ValueError) as exc:
        logger.debug("Failed to detect opaque call patterns: %s", exc)
        return []


def stale_notebook_rows(upstream_checker: Any) -> list[ProcessResult]:
    """Return the WARNING notification for a proven-stale notebook file.

    Only proof is reported. A once-per-session "cash cannot see unsaved
    edits here" row used to join it whenever cash read the saved file; it
    fired on every fresh kernel of every headless run, where nothing can be
    unsaved, and no user could act on it. Where edits CAN be
    unsaved -- JupyterLab with the extension, VS Code, Colab -- cash reads
    the live cells. Guarded
    like `function_change_rows` / `opaque_call_rows`
    above. Nothing in `StalenessTracker`'s current implementation raises,
    but this is a diagnostic nicety layered on top of upstream resolution
    (which must already have succeeded to reach this point) -- its failure
    must never be able to take down a user's cell execution over what is,
    at worst, a missed warning. Broader than the siblings' exception tuples
    on purpose: unlike theirs, there is no specific failure mode to name
    here, so the guarantee has to be unconditional.
    """
    try:
        notifications = []
        stale = staleness_notification(upstream_checker.staleness)
        if stale is not None:
            notifications.append(stale)
        return notifications
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never break execution
        logger.debug("Failed to check notebook staleness: %s", exc)
        return []
