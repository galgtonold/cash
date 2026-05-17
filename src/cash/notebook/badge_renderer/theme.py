"""Design tokens for badge rendering.

Single entry point for the visual vocabulary used by every badge renderer.
All renderers (HTML today; Text and React future) import their colors,
fonts, spacing, and display limits from here so a visual refresh is a
one-file change.

The :class:`BadgeView` IR (see :mod:`.view`) carries only *semantic* props
(``status='restored'``). Mapping from semantics to (icon, color, label) is
the renderer's job, performed via the helpers in this module.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Typography — system-native font stacks. Cash never ships a webfont; the
# host (Jupyter / browser) provides the typography. See the Cash design
# system's colors_and_type.css for the canonical token values.
# ---------------------------------------------------------------------------
FONT_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, '
    'sans-serif, "Apple Color Emoji", "Segoe UI Emoji"'
)
FONT_MONO = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)

# ---------------------------------------------------------------------------
# Display limits
# ---------------------------------------------------------------------------
CODE_SNIPPET_MAX_LEN = 120   # chars before truncating code snippets
HEADER_MAX_LEN = 80          # chars before truncating header lines
MIN_TIME_DISPLAY_S = 0.01    # seconds threshold for displaying a time value
MIN_TIME_DISPLAY_MS = 0.001  # milliseconds threshold (overhead breakdown)

# ---------------------------------------------------------------------------
# Status palette — semantic colors keyed by cache status. Renderers map
# BadgeView ``status`` props through here rather than inlining hex codes.
#
# Today only two values are exposed because the legacy ``render_status_badge``
# uses only this binary distinction. The full palette (one color per status
# enum value) is consolidated in slice 4, once the HTML renderer is the
# single owner of all status-conditional styling.
# ---------------------------------------------------------------------------
BADGE_COLOR_RESTORED = "#006644"
BADGE_COLOR_DEFAULT = "#996300"
