"""Renderers consume a :class:`~..view.InteractiveBadge` IR tree and emit output.

:mod:`.html` renders the interactive badge; :mod:`.text` renders the plain-text
form for the ``print`` badge mode (``%cash_badge print``).
"""

from .html import render_html
from .text import render_text

__all__ = ["render_html", "render_text"]
