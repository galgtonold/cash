"""Phase 4 of the cell pipeline: bring the cell's inputs up to date.

Before a cell's statements run, every input it reads must be in the
namespace and current: a missing one is restored from the cache, and the
upstream check re-runs or restores the cells above whose code or inputs
changed. When that check fails, the cell executor steps aside and the hook
runs :class:`RunInstead`'s source instead of the cell.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..._clock import perf_counter as _perf_counter
from ...analysis.code_analyzer import CodeAnalyzer
from ...diagnostics import warn_diagnostic
from ...exceptions import (
    AmbiguousCellError,
    CashCacheIneffectiveWarning,
    ForwardReferenceError,
    UpstreamStateError,
)
from ._types import RunInstead

if TYPE_CHECKING:
    from .._protocols import ShellProtocol
    from ..control_structures import ControlStructureProcessor
    from ..statement import ProcessResult, StatementProcessor
    from ..upstream import UpstreamChecker
    from ._types import TimingBreakdown
    from .badges import BadgePresenter

__all__ = ["UpstreamResolution"]

logger = logging.getLogger(__name__)


def _pyplot_open_fignums() -> set[int]:
    """Open matplotlib figure numbers, or an empty set if pyplot isn't loaded.

    Only inspects an already-imported ``matplotlib.pyplot`` — never imports it,
    so it stays a no-op for notebooks that don't plot.
    """
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return set()
    try:
        return set(plt.get_fignums())
    except Exception:  # noqa: BLE001 - a broken backend must not break execution
        return set()


def _close_pyplot_figures(nums: set[int]) -> None:
    """Close the given matplotlib figures, removing them from pyplot's registry.

    Used after upstream re-execution: a figure that reconstruction OPENED (to
    rebuild a ``fig``/``ax`` a downstream cell needs) would otherwise be flushed
    by the inline backend's post-execute hook into the DOWNSTREAM cell's output —
    a stray plot. A normally-run cell closes its figures on flush anyway, so
    closing the reconstructed ones matches that end state. The Figure/Axes
    objects stay valid (``fig.savefig`` / ``ax.*`` still work) for the cell that
    asked for them.
    """
    if not nums:
        return
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for num in nums:
        try:
            plt.close(num)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


class UpstreamResolution:
    """Makes the cell's inputs available and current before it runs.

    Has the :class:`UpstreamChecker` re-run or restore what is missing or
    changed above the cell, as a top-to-bottom run would have it.
    Its one entry is :meth:`resolve`.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        badges: BadgePresenter,
        statement_processor: StatementProcessor,
        upstream_checker: UpstreamChecker,
        control_structure_processor: ControlStructureProcessor,
    ) -> None:
        self.shell = shell
        self._badges = badges
        self._statement_processor = statement_processor
        self._upstream_checker = upstream_checker
        self._control_structure_processor = control_structure_processor

    def _check_and_reexecute_upstream_cells(
        self,
        cell_code: str,
        required_inputs: set,
        progress_callback: Callable[..., None] | None = None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> tuple[list[ProcessResult], float, float]:
        """Delegate to ``UpstreamChecker``.

        Returns a list of metrics for any executed or restored upstream
        statements, plus the total restore and execution times.
        """
        return self._upstream_checker.check_and_reexecute(
            cell_code,
            required_inputs,
            self._statement_processor.process_statement,
            ttl,
            cell_id=cell_id,
            progress_callback=progress_callback,
            control_structure_callback=self._control_structure_processor.process,
        )

    def _ensure_state_for_inputs(
        self,
        cell_code: str,
        progress_callback: Callable[..., None] | None = None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> tuple[list[ProcessResult], float, float]:
        """Ensure all required inputs are available in ``user_ns``.

        Through :meth:`_check_and_reexecute_upstream_cells`: an input is
        brought back only when a top-to-bottom run would hold it here, so a
        name deleted above the cell stays deleted.
        """
        # Reconstructing an upstream PLOT cell (to rebuild a fig/ax a downstream
        # cell needs) opens a matplotlib figure. The inline backend's
        # post-execute hook would then flush that figure into THIS (downstream)
        # cell's output — a stray plot. Close any figure reconstruction opens so
        # the downstream cell only shows its own output (a normally-run cell
        # closes its figures on flush anyway).
        figs_before = _pyplot_open_fignums()
        try:
            inputs, outputs = CodeAnalyzer.analyze_code_block(cell_code)
            logger.debug("[ENSURE_STATE_DEBUG] Cell %.50r: inputs %s, outputs %s", cell_code, inputs, outputs)

            total_restore_time = 0.0
            upstream_metrics: list[ProcessResult] = []

            reexec_metrics, upstream_restore_time, total_execution_time = self._check_and_reexecute_upstream_cells(
                cell_code,
                inputs,
                progress_callback=progress_callback,
                ttl=ttl,
                cell_id=cell_id,
            )
            total_restore_time += upstream_restore_time
            upstream_metrics.extend(reexec_metrics)

        except (RuntimeError, SyntaxError, AmbiguousCellError, ForwardReferenceError):
            raise
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.debug("[STATE] Error in state restoration logic: %s", e)
            raise
        finally:
            _close_pyplot_figures(_pyplot_open_fignums() - figs_before)

        return upstream_metrics, total_restore_time, total_execution_time

    def resolve(
        self,
        raw_cell: str,
        pre_upstream_metrics: list[ProcessResult],
        badge_display_id: str,
        timing_breakdown: "TimingBreakdown",
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> tuple[list[ProcessResult], float, float] | RunInstead:
        """Run upstream dependency checking and state restoration.

        On error, step aside (:meth:`_step_aside`).
        """
        t_ensure = _perf_counter()

        def _upstream_progress_cb(
            upstream_metrics_so_far: list,
            current_stmt_code: str,
            current_step: int | None = None,
            total_steps: int | None = None,
        ) -> None:
            combined = pre_upstream_metrics + upstream_metrics_so_far
            upstream_label = f"↑ {current_stmt_code}" if current_stmt_code else current_stmt_code
            self._badges.maybe_progress(
                combined,
                display_id=badge_display_id,
                step=current_step if current_step is not None else len(combined),
                total=total_steps or 0,
                code=upstream_label,
            )

        caught: Exception | None = None
        try:
            upstream_metrics, total_restore_time, total_execution_time = self._ensure_state_for_inputs(
                raw_cell,
                progress_callback=_upstream_progress_cb,
                ttl=ttl,
                cell_id=cell_id,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - broad fallback for upstream simulation failures
            # Do NOT dispatch original_run_cell (or render the badge) from inside
            # this suite: it is a LIVE except block, so sys.exc_info() is set to
            # this internal exception.  Any exception IPython raises while
            # surfacing the user's error would then be implicitly chained onto it
            # via __context__, leaking cash's own frames plus a spurious "During
            # handling of the above exception, another exception occurred" banner
            # into the user's traceback.  Capture here and dispatch
            # AFTER the block exits, when sys.exc_info() is clear.
            caught = e

        if caught is not None:
            return self._step_aside(caught, raw_cell, badge_display_id)

        timing_breakdown["upstream_check_raw"] = _perf_counter() - t_ensure
        timing_breakdown["total_restore_time"] = total_restore_time
        timing_breakdown["total_execution_time"] = total_execution_time
        timing_breakdown["upstream_check"] = (_perf_counter() - t_ensure) - total_restore_time - total_execution_time

        if logger.isEnabledFor(logging.DEBUG):
            ensure = _perf_counter() - t_ensure
            logger.debug(
                "[TIMING_PROXY] Ensure state: %.2fms (restore %.2fms, execution %.2fms, overhead %.2fms)",
                ensure * 1000,
                total_restore_time * 1000,
                total_execution_time * 1000,
                (ensure - total_restore_time - total_execution_time) * 1000,
            )

        return upstream_metrics, total_restore_time, total_execution_time

    def _step_aside(
        self,
        caught: Exception,
        raw_cell: str,
        badge_display_id: str,
    ) -> RunInstead:
        """What the hook runs instead of the cell when the upstream check failed.

        Called AFTER :meth:`resolve`'s try/except has exited,
        and the hook runs the returned source after the executor returned, so
        ``sys.exc_info()`` is clear by then. Running it inside the live
        ``except`` block would chain whatever IPython raises onto cash's
        internal exception via ``__context__``, and the user's traceback would
        show cash's own frames and a "During handling of the above exception"
        banner.

        - SyntaxError (a cell above does not parse): run the cell as written.
        - RuntimeError / AmbiguousCellError / UpstreamStateError /
          ForwardReferenceError: a fresh raise of the same error, run as the
          cell, so the cell fails loudly and IPython attributes it there.
        - anything else: an internal failure; warn and run the cell uncached.
        """
        if isinstance(caught, SyntaxError):
            self._badges.close(badge_display_id)
            return RunInstead(raw_cell, caught)
        if isinstance(caught, (RuntimeError, AmbiguousCellError, UpstreamStateError, ForwardReferenceError)):
            # Re-raise inside the user's cell so IPython renders the traceback
            # as if the cell itself raised.  Import the exception class
            # explicitly because the user's namespace may not have it.  The
            # trailing ``from None`` suppresses any ambient context so the
            # synthesised raise carries only the message, never a chain back
            # into cash's internals.
            cls = type(caught)
            # repr(), not a triple-quoted literal. Python quotes names in its
            # own messages -- "No such file or directory: 'side.txt'" -- and a
            # message ending in a quote closed the literal early, so the user's
            # cell died with `SyntaxError: unterminated string literal` from
            # code cash wrote, with the real failure nowhere in sight. repr()
            # also handles the newlines this message routinely carries.
            error_code = f"from {cls.__module__} import {cls.__name__}; raise {cls.__name__}({str(caught)!r}) from None"
            self._badges.close(badge_display_id)
            return RunInstead(error_code, caught)
        # An internal failure, and the cell is about to run UNCACHED. This used
        # to be logger.error only -- invisible in a notebook, where nobody is
        # watching the kernel log -- so the sole trace was an empty badge, which
        # itself then read as "EXECUTED 0.00s". A user hitting this saw a cell
        # produce nothing and had no way to learn why. Warn where they are.
        # With the traceback: this message asks the user to report the failure,
        # and "ModuleNotFoundError: No module named 'openpyxl'" on its own says
        # nothing about where in cash it came from.
        logger.error("Cash auto-caching failed: %s. Falling back to normal execution.", caught, exc_info=caught)
        try:
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "NOTEBOOK-BAILOUT",
                what=(
                    f"cash hit an internal error and stepped aside: "
                    f"{type(caught).__name__}: {caught}. This cell ran normally "
                    f"but was NOT cached, and neither were its results."
                ),
                fix=(
                    "Nothing in your code caused this and re-running is safe -- "
                    "the cell's result is correct, just uncached. Please report "
                    "it with the message above."
                ),
            )
        except Exception:  # noqa: BLE001 - a diagnostic must never break a cell
            pass
        self._badges.close(badge_display_id, status="BYPASSED")
        return RunInstead(raw_cell, caught)
