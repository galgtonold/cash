"""Badge rendering package.

The pipeline:

    metrics_list  ──►  view_builder.build_interactive_badge  ──►  BadgeView
                                                                       │
                                                                       ▼
                              renderers.html.render_html  /  renderers.text.render_text
                                                                       │
                                                                       ▼
                                                       HTML string  /  plain-text string

Public entry points (used by ``magics.py``):

* :func:`render_interactive_badge` — interactive HTML badge for one cell.
* :func:`render_status_badge` — compact non-interactive status pill.
* :func:`print_text_badge` — plain-text counterpart of the interactive badge.

See :mod:`.view` for node types, :mod:`.view_builder` for translation,
:mod:`.theme` for design tokens, and :mod:`.renderers` for the output adapters.
"""

from ._badge import render_interactive_badge, render_status_badge
from ._text import print_text_badge

__all__ = [
    "render_interactive_badge",
    "render_status_badge",
    "print_text_badge",
]
