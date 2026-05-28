"""Renderers consume a :class:`~..view.InteractiveBadge` IR tree and emit output.

Today: :mod:`.html` (the only consumer in slice 4).
Future: :mod:`.text` (slice 6), and a React-mounting renderer.
"""

from .html import render_html
from .text import render_text

__all__ = ["render_html", "render_text"]
