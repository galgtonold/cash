"""Public HTML badge entry point.

Thin shim over the BadgeView pipeline:
:func:`render_interactive_badge` builds the multi-section expandable badge
via :func:`.view_builder.build_interactive_badge` then
:func:`.renderers.html.render_html`.
"""

from __future__ import annotations

from typing import Any


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
