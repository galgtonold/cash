"""Badge rendering helpers for notebook cell execution output.

This package contains the static rendering functions that were extracted from
:class:`~cash.notebook.magics.CashMagics` to reduce the size of the god-class
and isolate HTML badge generation from core caching logic.

All functions are pure (no side-effects) — they accept metrics data and return
HTML strings or transformed data structures.
"""

from ._badge import render_interactive_badge, render_status_badge
from ._components import (
    render_control_body_html,
    render_control_group,
    render_control_group_single,
    render_decorator_calls,
    render_for_loop_group,
    render_loop_stmt_row,
)
from ._grouping import create_loop_group, format_loop_var, group_loop_iterations
from ._text import print_text_badge
from ._types import (
    ControlGroup,
    ControlGroupSingle,
    ForLoopGroup,
    GroupedBadgeItem,
    LoopGroup,
    SingleMetric,
)

__all__ = [
    "format_loop_var",
    "render_control_body_html",
    "group_loop_iterations",
    "create_loop_group",
    "render_for_loop_group",
    "render_loop_stmt_row",
    "render_control_group",
    "render_control_group_single",
    "render_decorator_calls",
    "render_interactive_badge",
    "render_status_badge",
    "print_text_badge",
    # TypedDicts (for backward compatibility)
    "LoopGroup",
    "ForLoopGroup",
    "ControlGroup",
    "ControlGroupSingle",
    "SingleMetric",
    "GroupedBadgeItem",
]
