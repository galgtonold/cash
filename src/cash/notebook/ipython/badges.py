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
        """Whether a progress update may render now: at most one per
        :attr:`MIN_RENDER_INTERVAL`, so fast statements do not flicker. The
        first one after :meth:`start_cell` always may."""
        now = time.time()
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
        """Render a RUNNING badge update, if the throttle allows one."""
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
        """Publish a RUNNING badge naming this statement, if it is still
        running after :attr:`MIN_RENDER_INTERVAL`.

        Deferred rather than throttled: a statement faster than the interval
        publishes nothing, and a slow one is always named while it runs
        (throttling the leading edge would leave the previous statement on
        screen for the whole of a slow one).
        """
        self.cancel_progress()
        if self.mode != "html":
            return

        # Resolved here, before the statement's output capture starts; see
        # `_uncaptured_display_pub`.
        publisher = self._uncaptured_display_pub()

        with self._progress_lock:
            generation = self._progress_generation

        def fire() -> None:
            # Build OUTSIDE the lock: the build can hit disk or the notebook
            # server, and `cancel_progress` on the main thread would block for
            # as long. `metrics` is the executor's live list, read unlocked; a
            # racing append at worst costs this one render, which swallows
            # its own errors.
            html = self._build_html(
                metrics,
                status="RUNNING",
                current_step=step,
                total_steps=total,
                current_code=code,
            )
            if not html:
                return

            # `Timer.cancel()` cannot stop a timer already running its
            # callback. Under the lock `cancel_progress` takes, a cancel made
            # before this point turns the publish into a no-op, and a cancel
            # made during it waits for it, so the caller's own DONE badge is
            # always published last.
            with self._progress_lock:
                if generation != self._progress_generation:
                    return
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

        Once this returns, no RUNNING render can overtake what the caller
        publishes next: it blocks, at most for one publish, while a timer is
        publishing (see ``arm_progress``).
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
        """The shell's real display publisher, not a capture's stand-in.

        A statement runs under ``capture_output(display=True)``, which swaps
        ``shell.display_pub`` process-wide. A progress badge published through
        the stand-in from the timer thread would land in the statement's
        captured outputs -- and so in its cache entry, replayed on every hit
        as a frozen RUNNING badge. The last real publisher seen is remembered;
        ``None`` until one has been.
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
        """Render the interactive badge: :meth:`_build_html`, then
        :meth:`_publish_html`.

        Never raises. The badge is drawn around the user's cell, and a failure
        to build or show it must not abort the statements or swallow their
        output; the cell goes without a badge instead.
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
        """The badge's HTML, or ``None`` on failure or empty markup. The slow
        half of a render: the bug-report context can read the notebook from
        disk or the Jupyter server."""
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
        """Publish built badge HTML: one display message, so it is safe under
        ``_progress_lock``. Never raises.

        From the timer thread (*_from_thread*) it goes through *publisher*,
        the real publisher :meth:`_uncaptured_display_pub` resolved, as an
        update of *display_id*; with no publisher it publishes nothing, since
        the module-level ``publish_display_data`` would find the capture's
        stand-in.
        """
        try:
            if _from_thread and display_id:
                # Directly: display()'s bookkeeping can create duplicate output
                # areas when called off the main thread.
                if publisher is not None:
                    publisher.publish(
                        {"text/html": html},
                        metadata={},
                        transient={"display_id": display_id},
                        update=True,
                    )
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

        Delegates to :func:`error_display.show_clean_error`.
        """
        show_clean_error(exc, raw_cell, node, self.shell)

    def _get_bug_report_context(self) -> dict:
        """Runtime environment info for the pre-filled bug report URL.

        Collected once per cell (per execution count): it reads the whole
        notebook, and every render asks for it.
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
