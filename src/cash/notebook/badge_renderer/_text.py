"""Text-mode badge entry point.

Counterpart of :mod:`._badge` for plain-text output. Builds the
:class:`BadgeView` and prints it via :mod:`.renderers.text`.
"""

from __future__ import annotations

from typing import Any

from cash.utils import safe_text


def print_text_badge(
    metrics_list: list[dict[str, Any]],
    cell_total_time: float | None = None,
) -> None:
    """Print a plain-text summary of cell execution (for ``print`` badge mode).

    Counterpart of :func:`render_interactive_badge` for non-HTML output.
    """
    metrics_list = metrics_list or []
    if not metrics_list:
        return

    from .renderers.text import render_text
    from .view_builder import build_interactive_badge

    view = build_interactive_badge(
        metrics_list,
        cell_total_time=cell_total_time,
    )
    print(safe_text(render_text(view)))
