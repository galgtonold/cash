"""Design tokens for badge rendering.

Single entry point for the visual vocabulary used by every badge renderer.
All renderers (HTML v3 today; Text and a future React renderer) import
their colors, fonts, spacing, and display limits from here so a visual
refresh is a one-file change.

The :class:`BadgeView` IR (see :mod:`.view`) carries only *semantic* props
(``status='restored'``). Mapping from semantics to (icon, color, label,
kind) is the renderer's job, performed via the helpers in this module.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Typography — system-native font stacks. Cash never ships a webfont; the
# host (Jupyter / browser) provides the typography.
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
CODE_SNIPPET_MAX_LEN = 120
HEADER_MAX_LEN = 80
MIN_TIME_DISPLAY_S = 0.01
MIN_TIME_DISPLAY_MS = 0.001

# ---------------------------------------------------------------------------
# v3 state palette — single source of truth for the row-level look.
# Mirrors design/notebook-badges/badge.css → Badge v3.
#
# Three "kinds" cover every status:
#   - cached  : RESTORED, SKIPPED       (green family)
#   - exec    : COMPUTED, MIXED         (ochre family)
#   - warn    : WARNING, ERROR,
#               FUNCTION_CHANGED,
#               MODULE_RELOADED         (red/amber family)
# ---------------------------------------------------------------------------

# Rail / accent / state-color (used on left rails and text accents)
RAIL_CACHED = "#006644"
RAIL_EXEC   = "#996300"
RAIL_MIXED  = "#336699"
RAIL_WARN   = "#c0392b"
RAIL_NEUTRAL = "#888"

# Timing-bar fills + state-tinted time-chip backgrounds (softer, pastel).
BAR_CACHED = "#7ec19f"
BAR_EXEC   = "#d3a76b"
BAR_WARN   = "#e8a397"

CHIP_BG_CACHED = "#f0f8f4"
CHIP_BG_EXEC   = "#fdfaf0"
CHIP_BG_WARN   = "#fdf3f0"
CHIP_FG_CACHED = "#00563a"
CHIP_FG_EXEC   = "#7a5310"
CHIP_FG_WARN   = "#a02c1a"

# Summary chip background tints (lighter than chip-bg).
SUMMARY_BG_CACHED = "#f1faf6"
SUMMARY_BG_EXEC   = "#fdfaf0"

# Neutrals — borders, dim text, table tints.
INK         = "#222"
INK_2       = "#444"
INK_3       = "#666"
INK_4       = "#888"
INK_5       = "#999"
RULE        = "#e6e6e6"
RULE_SOFT   = "#f4f3ef"
BG_PANEL    = "#fdfdfc"
BG_UPSTREAM = "#f6f7f8"
BG_HOVER    = "#faf9f5"
BG_DETAIL   = "#fafbfc"

# Bug-report pill (subtle tertiary link in v3).
BUG_FG = "#888"
BUG_FG_HOVER = "#444"

# Legacy aliases kept for back-compat with render_status_badge.
BADGE_COLOR_RESTORED = RAIL_CACHED
BADGE_COLOR_DEFAULT  = RAIL_EXEC


# ---------------------------------------------------------------------------
# Semantic mappers
# ---------------------------------------------------------------------------

def kind_of(status: str) -> str:
    """Return ``'cached' | 'exec' | 'warn'`` for any :class:`BadgeStatus` value."""
    if status in ("restored", "skipped"):
        return "cached"
    if status in ("warning", "error", "function_changed", "module_reloaded"):
        return "warn"
    return "exec"  # computed, mixed, unknown


def rail_color(status: str) -> str:
    """Color of the left rail for one row."""
    if status == "restored" or status == "skipped":
        return RAIL_CACHED
    if status == "mixed":
        return RAIL_MIXED
    if status in ("warning", "error", "function_changed", "module_reloaded"):
        return RAIL_WARN
    return RAIL_EXEC


def bar_color(kind: str) -> str:
    if kind == "cached":
        return BAR_CACHED
    if kind == "warn":
        return BAR_WARN
    return BAR_EXEC


def chip_bg(kind: str) -> str:
    if kind == "cached":
        return CHIP_BG_CACHED
    if kind == "warn":
        return CHIP_BG_WARN
    return CHIP_BG_EXEC


def chip_fg(kind: str) -> str:
    if kind == "cached":
        return CHIP_FG_CACHED
    if kind == "warn":
        return CHIP_FG_WARN
    return CHIP_FG_EXEC
