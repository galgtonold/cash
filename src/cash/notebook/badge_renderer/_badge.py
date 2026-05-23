"""Public HTML badge entry points.

Both functions are thin shims over the BadgeView pipeline:

* :func:`render_interactive_badge` is the multi-section expandable badge.
  Implementation: :func:`.view_builder.build_interactive_badge` then
  :func:`.renderers.html.render_html`.
* :func:`render_status_badge` is the compact non-interactive pill.
  Implementation kept inline because its signature accepts loose
  ``str | list | None`` values from many call sites and predates the IR.
"""

from __future__ import annotations

from typing import Any

from cash.notebook.cache_status import CacheStatus

from .theme import FONT_MONO, RAIL_CACHED, RAIL_EXEC


def render_interactive_badge(
    metrics_list: list[dict[str, Any]],
    badge_mode: str,
    status: str = "DONE",
    current_step: int = 0,
    total_steps: int = 0,
    current_code: str | None = None,
    cell_total_time: float | None = None,
    timing_breakdown: dict[str, float] | None = None,
    bug_report_context: dict | None = None,
    configured_tiers: tuple[str, ...] = (),
) -> str:
    """Build the interactive HTML badge string for cell execution results.

    Pipeline: ``metrics_list → view_builder.build_interactive_badge → BadgeView
    → renderers.html.render_html → HTML string``.

    ``configured_tiers`` is the backend-declared tier list
    (``cash.backend.tier_labels()``); it drives the per-row dot indicator
    so a three-tier configuration shows three dots, not the legacy two.
    """
    if badge_mode != "html":
        return ""

    from .renderers.html import render_html
    from .view_builder import build_interactive_badge

    view = build_interactive_badge(
        metrics_list or [],
        status=status,
        current_step=current_step,
        total_steps=total_steps,
        current_code=current_code,
        cell_total_time=cell_total_time,
        timing_breakdown=timing_breakdown,
        bug_report_context=bug_report_context,
        configured_tiers=configured_tiers,
    )
    return render_html(view)


def render_status_badge(
    status: str,
    execution_time: float = 0.0,
    time_saved: float = 0.0,
    source: str | None = None,
    storage: Any = None,
) -> str:
    """Render a simple colored status badge as an HTML string.

    Parameters
    ----------
    status:
        Cache status value (e.g. ``'COMPUTED'``, ``'RESTORED'``).
    execution_time:
        Wall-clock execution time in seconds.
    time_saved:
        Time saved by cache restore in seconds.
    source:
        Cache backend source label (e.g. ``'memory'``, ``'disk'``).
    storage:
        Cache backend storage label(s).
    """
    color = RAIL_CACHED if status == CacheStatus.RESTORED else RAIL_EXEC
    parts = [f"<b>{status}</b>"]
    if source:
        parts.append(f"FROM: {source}")
    if storage:
        storage_str = "+".join(storage) if isinstance(storage, list) else str(storage)
        parts.append(f"IN: {storage_str}")
    if status == CacheStatus.RESTORED and time_saved > 0:
        parts.append(f"Saved {time_saved:.2f}s")
    elif execution_time > 0:
        parts.append(f"{execution_time:.2f}s")
    text = " | ".join(parts)
    return f"""
    <div style="
        display: inline-block;
        padding: 4px 8px;
        background-color: {color};
        color: white;
        border-radius: 4px;
        font-family: {FONT_MONO};
        font-size: 11px;
        margin-bottom: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    ">{text}</div>
    """
