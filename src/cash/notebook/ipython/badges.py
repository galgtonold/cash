"""The badge a cell shows while it runs and when it is done.

:class:`BadgePresenter` is the one place cash draws on a cell's output: the
RUNNING badge a cell opens with, the progress updates while its statements
run, the final DONE badge (or the plain-text summary in ``print`` mode) and
the clean traceback of a failed statement. :class:`CellExecutor` reports to it
as a cell goes through the pipeline, and :class:`CashMagics` for the final
badge and for ``%cash_badge``, which sets its :attr:`~BadgePresenter.mode`.
"""

from __future__ import annotations

import ast
import logging
import sys
import threading
import time
from typing import Any

from IPython.display import HTML, display

from ... import __version__
from .. import badge_renderer as _badge
from ..server_discovery import get_notebook_cells
from ..statement import ProcessResult
from .error_display import show_clean_error

__all__ = ["BadgePresenter"]

logger = logging.getLogger(__name__)


class BadgePresenter:
    """Draws a cell's badge, throttling progress updates, and its errors.

    *cash* is read for its backend at each render, so a backend swapped in
    mid-session shows on the next badge.
    """

    #: The least time between two progress renders, and how long a statement
    #: runs before its progress badge is published.
    MIN_RENDER_INTERVAL = 0.3

    def __init__(self, shell: Any, cash: Any) -> None:
        self.shell = shell
        self._cash = cash
        #: 'html' (interactive badge), 'print' (text summary) or 'off'.
        self.mode = "html"
        self._last_render_time = 0.0
        self._progress_timer: threading.Timer | None = None
        # Guards the race between `cancel_progress` (main thread) and a timer's
        # `fire()` (background thread): `_progress_generation` is bumped on
        # every cancel, and `fire()` re-checks it after acquiring
        # `_progress_lock`, so a cancel either lands before `fire()` starts
        # (generation mismatch -> no-op) or blocks until an in-flight `fire()`
        # finishes (so the caller's own render, e.g. DONE, is always the one
        # published last).
        self._progress_lock = threading.Lock()
        self._progress_generation = 0
        # The shell's REAL display publisher, remembered while no capture is
        # installed -- see `_uncaptured_display_pub`.
        self._display_pub: Any = None
        #: ``(execution_count, context)`` of the last bug-report context read.
        self._bug_report_context_cache: tuple[Any, dict] | None = None

    def start_cell(self, display_id: str) -> None:
        """Open a cell's badge: RUNNING, under *display_id*, with the progress
        throttle reset so the cell's first progress update is shown."""
        self._last_render_time = 0.0
        if self.mode == "html":
            self.render([], display_id=display_id, status="RUNNING", update_existing=False)

    def close(self, display_id: str, status: str = "DONE") -> None:
        """End a cell's badge with no rows, for a cell cash did not run."""
        self.cancel_progress()
        self.render([], display_id=display_id, status=status)

    def finish(
        self,
        metrics: list[ProcessResult],
        display_id: str,
        cell_total_time: float,
        timing_breakdown: dict[str, float] | None = None,
    ) -> None:
        """The cell's final badge, in the current mode.

        Any pending progress update is cancelled first, so a late one cannot
        overwrite the final badge with a stale RUNNING one.
        """
        self.cancel_progress()
        if self.mode == "html":
            self.render(
                metrics,
                display_id=display_id,
                cell_total_time=cell_total_time,
                timing_breakdown=timing_breakdown,
            )
        elif self.mode == "print":
            self.print_text(metrics, cell_total_time=cell_total_time)

    def _throttle_allows(self) -> bool:
        """Check if enough time has passed to render a progress badge update.

        Badge throttling policy:
        Enforce a minimum interval between badge renders to prevent flicker
        when many fast statements execute in quick succession.  The first
        render after cell start is always allowed so the user immediately
        sees what is running.

        Returns True if a badge update should be rendered now.
        """
        now = time.time()

        # Throttle: skip if we rendered very recently
        if now - self._last_render_time < self.MIN_RENDER_INTERVAL:
            return False

        self._last_render_time = now
        return True

    def maybe_progress(
        self,
        metrics: list[ProcessResult],
        display_id: str,
        step: int,
        total: int,
        code: str | None,
    ) -> None:
        """Render a RUNNING badge update if throttle allows.

        Consolidates the throttle check + render into a single call so the
        execution loop reads as: ``execute → maybe_progress(...)`` rather
        than the repeated ``if mode == 'html' and _throttle_allows()`` pattern.
        """
        if self.mode == "html" and self._throttle_allows():
            self.render(
                metrics,
                display_id=display_id,
                status="RUNNING",
                current_step=step,
                total_steps=total,
                current_code=code,
            )

    def arm_progress(
        self, metrics: list[ProcessResult], display_id: str, step: int, total: int, code: str | None
    ) -> None:
        """Publish a RUNNING badge only if this statement is still running.

        Rendering BEFORE the statement published once per statement no matter
        how fast it was -- 68 progress badges in a ten-cell run. Throttling the
        leading edge instead would be worse than the traffic: it drops the
        render that says "now running the slow one", so a long statement shows
        the PREVIOUS one for its whole duration.

        Deferring inverts that. A statement faster than the interval publishes
        nothing; a slower one publishes once, naming itself.
        """
        self.cancel_progress()
        if self.mode != "html":
            return

        # Resolve the display publisher HERE, on the main thread, while no
        # statement -- and therefore no output capture -- is running. By the
        # time `fire()` runs, `shell.display_pub` has been swapped out for a
        # capturing stand-in and asking for it then loses the badge. See
        # `_uncaptured_display_pub`.
        publisher = self._uncaptured_display_pub()

        with self._progress_lock:
            generation = self._progress_generation

        def fire() -> None:
            # Build OUTSIDE the lock. This is the expensive half of a render:
            # `_build_html` calls `_get_bug_report_context`, which can
            # hit disk, poll for an in-flight notebook save, or (on a cache
            # miss) make a bounded network call to the Jupyter server.
            # Holding `_progress_lock` across that is what used to make
            # `cancel_progress` block the main thread for as long as
            # the render took (measured: ~400ms for a 400ms render). Only the
            # generation re-check + the actual publish need the lock.
            #
            # `metrics` is also the live `all_metrics` list the main thread
            # keeps mutating via `.append()` (cell_executor.py's
            # `_handle_regular_stmt_metrics` / `_collect_ctrl_outputs`), and
            # reading it here holds no lock either -- a deliberately accepted
            # race, since the window is narrow, a stress run of hundreds of
            # concurrent appends raised nothing, and `_build_html`'s own
            # blanket `except Exception` swallows whatever would.
            html = self._build_html(
                metrics,
                status="RUNNING",
                current_step=step,
                total_steps=total,
                current_code=code,
            )
            if not html:
                return

            # Re-check under the SAME lock `cancel_progress` takes.
            # `Timer.cancel()` alone cannot stop a timer whose `run()` has
            # already passed its internal `is_set()` check -- at that point
            # the callback WILL execute no matter what the main thread does.
            # The generation check makes that execution a no-op if a cancel
            # happened anytime before we got the lock; if a cancel is
            # concurrently in flight (already past ITS own is_set check on
            # the timer, now waiting on this lock), it blocks until we
            # release it, so the cancel's caller (about to publish its own,
            # e.g. DONE, badge) never runs ahead of us -- this PUBLISH is
            # always fully finished before that one starts. The lock no
            # longer guards the build, so that blocking window is now
            # bounded by a publish call (a ZMQ send), not by a render.
            with self._progress_lock:
                if generation != self._progress_generation:
                    return
                # `_publish_html` already swallows everything: a
                # badge must never break a cell, and this runs on a timer
                # thread where a raise would be lost anyway.
                self._publish_html(html, display_id=display_id, _from_thread=True, publisher=publisher)

        try:
            timer = threading.Timer(self.MIN_RENDER_INTERVAL, fire)
            timer.daemon = True
            timer.start()
        except Exception as e:  # noqa: BLE001 - a badge must never break a cell; degrade to none
            logger.debug("[BADGE ARM ERROR] %s", e, exc_info=True)
            return
        self._progress_timer = timer

    def cancel_progress(self) -> None:
        """Stop a pending progress badge. Safe to call when none is armed.

        Bumps the generation counter FIRST (under the lock a concurrent
        `fire()` also holds), so either `fire()` hasn't reached its own
        lock yet -- and will see the mismatch and no-op -- or it is already
        inside the lock publishing, and this call blocks until that publish
        finishes. `fire()` builds the HTML for that publish BEFORE taking the
        lock, so this can only ever block for a publish (a ZMQ send), never
        for a full render. Either way, nothing this function's caller does
        next (typically publishing a DONE badge) can be overtaken by a stale
        RUNNING render.
        """
        with self._progress_lock:
            self._progress_generation += 1
            timer = self._progress_timer
            if timer is not None:
                timer.cancel()
                self._progress_timer = None

    @staticmethod
    def _is_capturing_display_pub(pub: Any) -> bool:
        """True if ``pub`` is IPython's capture-time stand-in publisher."""
        try:
            from IPython.core.displaypub import CapturingDisplayPublisher
        except Exception:  # noqa: BLE001 - a badge must never break a cell
            return False
        return isinstance(pub, CapturingDisplayPublisher)

    def _uncaptured_display_pub(self) -> Any:
        """The shell's REAL display publisher, not whatever a capture installed.

        Every statement executes inside
        ``IPython.utils.capture.capture_output(display=True)`` (see
        ``StatementProcessor._make_capture_ctx``), which swaps
        ``shell.display_pub`` for a ``CapturingDisplayPublisher``. That swap is
        PROCESS-wide, not thread-local -- so a progress badge published from
        the timer thread while a slow statement is running never reaches the
        frontend at all. It is swallowed into that STATEMENT's captured
        outputs, which cash then replays with
        ``publish_display_data(data, metadata)`` -- dropping both ``transient``
        and ``update``. The badge lands as a brand-new ``display_data`` with no
        display id, so no later update can ever reach it: a frozen RUNNING
        badge stored for good beside the cell's real DONE badge, and (because
        captured outputs become ``metrics['rich_outputs']``, part of the
        statement's cache payload) replayed again on every future cache hit.

        Resolving the publisher on the MAIN thread at arm time -- before the
        statement, and so before its capture, starts -- hands ``fire()`` the
        real one to publish through. The last non-capturing publisher is
        remembered so that an arm which somehow does land inside a capture
        still has one. Returns ``None`` when there is no shell to ask, or on
        the session's very first arm if it lands inside a capture before any
        real publisher has ever been remembered -- in which case
        :meth:`_publish_html` skips the publish rather than falling
        back to IPython's module-level ``publish_display_data``, which would
        land inside the same capture and swallow the badge right back in.
        """
        try:
            pub = getattr(self.shell, "display_pub", None) if self.shell is not None else None
            if pub is not None and not self._is_capturing_display_pub(pub):
                self._display_pub = pub
        except Exception as e:  # noqa: BLE001 - a badge must never break a cell
            logger.debug("[BADGE PUBLISHER ERROR] %s", e)
        return self._display_pub

    def render(
        self,
        metrics_list: list[ProcessResult],
        display_id: str | None = None,
        status: str = "DONE",
        current_step: int = 0,
        total_steps: int = 0,
        current_code: str | None = None,
        update_existing: bool = True,
        cell_total_time: float | None = None,
        timing_breakdown: dict[str, float] | None = None,
        _from_thread: bool = False,
    ) -> None:
        """Render a clickable interactive badge with detailed execution history.

        Delegates HTML generation to :func:`badge_renderer.render_interactive_badge`
        and handles the IPython display / publish lifecycle.

        The badge is a diagnostic overlay drawn AROUND the user's cell — it is
        rendered before each statement runs (see CellExecutor) and again at the
        end. So a failure to BUILD or DISPLAY it must never propagate: if it
        did, it would abort the statement loop before the user's code ran and
        swallow the cell's output entirely. Degrade to "no badge" instead.
        (A broken renderer once shipped that raised at import on Python 3.11,
        which is exactly how this manifested: every cell went blank.)

        Split into :meth:`_build_html` (expensive) and
        :meth:`_publish_html` (cheap) so a caller that must not hold a
        lock across the build -- ``arm_progress``'s ``fire()`` -- can
        call them separately, taking a lock around only the publish half.
        Every other caller (all of them, other than ``fire()``) goes through
        this method and sees identical behaviour to before the split: both
        halves always run in sequence, and both still swallow every
        exception.
        """
        html = self._build_html(
            metrics_list,
            status=status,
            current_step=current_step,
            total_steps=total_steps,
            current_code=current_code,
            cell_total_time=cell_total_time,
            timing_breakdown=timing_breakdown,
        )
        if not html:
            return
        self._publish_html(
            html,
            display_id=display_id,
            update_existing=update_existing,
            _from_thread=_from_thread,
        )

    def _build_html(
        self,
        metrics_list: list[ProcessResult],
        status: str = "DONE",
        current_step: int = 0,
        total_steps: int = 0,
        current_code: str | None = None,
        cell_total_time: float | None = None,
        timing_breakdown: dict[str, float] | None = None,
    ) -> str | None:
        """Build badge HTML without publishing it. The expensive half of a render.

        Delegates to :func:`badge_renderer.render_interactive_badge`, which
        calls :meth:`_get_bug_report_context` -- that can hit disk, poll for
        an in-flight notebook save, or (on a cache miss) make a bounded
        network call to the Jupyter server. Callers that must not block a
        lock for that long -- see ``arm_progress``'s ``fire()`` -- call
        this OUTSIDE the lock and only take one around :meth:`_publish_html`.

        Returns ``None`` (never raises) on failure or empty markup: see
        :meth:`render` for why a badge must never break a
        cell.
        """
        try:
            html = _badge.render_interactive_badge(
                metrics_list=metrics_list,
                badge_mode=self.mode,
                status=status,
                current_step=current_step,
                total_steps=total_steps,
                current_code=current_code,
                cell_total_time=cell_total_time,
                timing_breakdown=timing_breakdown,
                bug_report_context=self._get_bug_report_context(),
                configured_tiers=self._configured_tier_labels(),
            )
            return html or None
        except Exception as e:  # noqa: BLE001 — intentionally broad; see render
            logger.debug("[BADGE RENDER ERROR] %s", e, exc_info=True)
            return None

    def _publish_html(
        self,
        html: str,
        display_id: str | None = None,
        update_existing: bool = True,
        _from_thread: bool = False,
        publisher: Any = None,
    ) -> None:
        """Publish already-built badge HTML. The cheap half of a render.

        Just an IPython display / publish_display_data call (a ZMQ send) --
        nothing here touches disk or the notebook server, so it is safe to
        call while holding ``_progress_lock``.

        Never raises: see :meth:`render` for why a badge
        must never break a cell.

        ``publisher``, when given, is the shell's real display publisher as
        resolved by :meth:`_uncaptured_display_pub` on the main thread. A
        background-thread publish MUST go through it rather than through
        IPython's module-level ``publish_display_data``: that helper resolves
        ``shell.display_pub`` at call time, and while a statement is running
        that is a ``CapturingDisplayPublisher`` which swallows the badge into
        the statement's own output. When no publisher was resolved, this
        publishes nothing at all rather than risk that fallback -- see the
        comment at the call site.
        """
        try:
            if _from_thread and display_id:
                # From a background thread, publish an ``update_display_data``
                # message directly. This avoids display()'s bookkeeping which
                # can create duplicate output areas when called from non-main
                # threads.
                if publisher is not None:
                    publisher.publish(
                        {"text/html": html},
                        metadata={},
                        transient={"display_id": display_id},
                        update=True,
                    )
                # else: publish NOTHING. `_uncaptured_display_pub` returns
                # None only on the session's very first arm, if it lands
                # inside a capture before any real publisher has ever been
                # remembered. Falling back to module-level
                # ``publish_display_data`` here would resolve
                # ``shell.display_pub`` at call time -- mid-statement, that is
                # the ``CapturingDisplayPublisher`` this whole publisher
                # resolution dance exists to avoid -- and re-swallow the badge
                # into the statement's captured outputs and its cache entry,
                # the exact defect this branch fixed. A missing progress
                # badge beats a RUNNING one frozen into the cache forever.
            elif display_id:
                display(HTML(html), display_id=display_id, update=update_existing)
            else:
                display(HTML(html))
        except Exception as e:  # noqa: BLE001 — intentionally broad; see render
            logger.debug("[BADGE RENDER ERROR] %s", e, exc_info=True)

    def print_text(self, metrics_list: list[ProcessResult], cell_total_time: float | None = None) -> None:
        """Print a plain-text summary of the cell execution (for 'print' badge mode).

        Delegates to :func:`badge_renderer.print_text_badge`.
        """
        # As with the HTML badge: a failure to render this diagnostic summary
        # must never abort the cell or swallow the user's output.
        try:
            _badge.print_text_badge(metrics_list, cell_total_time=cell_total_time)
        except Exception as e:  # noqa: BLE001 — the badge is never worth breaking a cell
            logger.debug("[BADGE RENDER ERROR] %s", e, exc_info=True)

    def show_error(
        self,
        exc: Exception,
        raw_cell: str,
        node: ast.AST,
    ) -> None:
        """Display an exception with a clean traceback pointing to the user's cell.

        Delegates to :func:`error_display.show_error`.
        """
        show_clean_error(exc, raw_cell, node, self.shell)

    def _get_bug_report_context(self) -> dict:
        """Collect runtime environment info for the pre-filled bug report URL.

        Once per cell run. Every badge render asked, progress renders
        included, and each read the whole notebook again: 20 reads, 0.9-1.9 s,
        per cell of r24s4. The notebook's source cannot change while the cell
        runs, so the first answer stands until the next cell.
        """
        count = getattr(self.shell, "execution_count", None)
        cached = self._bug_report_context_cache
        if cached is not None and cached[0] == count:
            return cached[1]
        context = self._collect_bug_report_context()
        self._bug_report_context_cache = (count, context)
        return context

    def _collect_bug_report_context(self) -> dict:
        backend = getattr(self._cash, "backend", None)
        backend_name = type(backend).__name__ if backend else "unknown"
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # --- Execution history (what IPython actually ran) ---
        # Prefer input_hist_raw (untransformed magics like %cash_on) over
        # input_hist_parsed / In (which transforms magics to get_ipython() calls).
        # input_hist_raw lives on history_manager, not directly on the shell.
        hm = getattr(self.shell, "history_manager", None)
        in_history = getattr(hm, "input_hist_raw", None) if hm else None
        if in_history is None:
            in_history = getattr(self.shell, "user_ns", {}).get("In", [])
        # Filter empty strings and the 'pass' pseudo-cells that cash injects,
        # then deduplicate consecutive identical cells (from re-running).
        filtered: list[str] = []
        for c in in_history:
            if not c.strip() or c.strip() == "pass":
                continue
            if filtered and c == filtered[-1]:
                continue
            filtered.append(c)
        exec_history = filtered[-6:]

        # --- Notebook source (actual .ipynb cell contents on disk) ---
        notebook_cells: list[str] = []
        try:
            notebook_cells = get_notebook_cells() or []
        except Exception:
            pass

        return {
            "version": __version__,
            "python_version": python_version,
            "backend": backend_name,
            "notebook_history": exec_history,
            "notebook_source": notebook_cells,
        }

    def _configured_tier_labels(self) -> tuple[str, ...]:
        """Snapshot the active backend's tier list for the badge renderer.

        Each render reads it fresh so a user reconfiguring the backend
        mid-session (e.g. swapping in a Redis tier) sees the new layout
        on the next cell run.
        """
        backend = getattr(self._cash, "backend", None)
        if backend is None:
            return ()
        try:
            return tuple(backend.tier_labels())
        except Exception:  # noqa: BLE001 — best-effort: never break the badge over a backend quirk
            return ()
