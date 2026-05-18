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
# State palette — single source of truth for the row-level look.
#
# Every row maps to ONE of three "kinds". Each kind has ONE underlying hue
# (PRIMARY) — every other shade in the family is the same hue at a
# different lightness/alpha, so a row reads as a single visual family
# even though rail / bar / chip-bg / chip-fg are slightly different
# values. Picking distinct hex values per usage (not pure alpha) lets
# the colours stay vivid against Jupyter / VS Code backgrounds without
# bleeding through to neighbour rows.
#
# Mapping status → kind:
#   - cached  : RESTORED, SKIPPED                                 (green)
#   - exec    : COMPUTED, MIXED                                   (amber)
#   - warn    : WARNING, ERROR, FUNCTION_CHANGED, MODULE_RELOADED (red)
#
# Family-shade ladder (consistent across families):
#   RAIL / FG       — saturated, used on accents (left rail, chip text)
#   BAR             — mid-tone, used on the per-row timing bar fill
#   CHIP_BG         — very pale wash, used on chip + summary backgrounds
# ---------------------------------------------------------------------------

# CACHED — single green hue family. Rail/FG = saturated; BAR = mid; BG = pale.
RAIL_CACHED    = "#2a8a5e"
BAR_CACHED     = "#86bfa3"
CHIP_BG_CACHED = "#ecf6f0"
CHIP_FG_CACHED = "#1f6646"

# EXEC — single amber hue family.
RAIL_EXEC      = "#a87530"
BAR_EXEC       = "#d3a76b"
CHIP_BG_EXEC   = "#fbf3e5"
CHIP_FG_EXEC   = "#7a5310"

# WARN — single red hue family.
RAIL_WARN      = "#b53d29"
BAR_WARN       = "#e8a397"
CHIP_BG_WARN   = "#fbeae5"
CHIP_FG_WARN   = "#8a2916"

# MIXED rail: blue accent used only for the LEFT RAIL when a single row
# combines cached + computed iterations. The bar/chip for a mixed row
# still uses the EXEC family — the rail is the lone signal that this is
# a hybrid state, so we don't introduce a fourth full kind.
RAIL_MIXED     = "#336699"

# Summary chip background equals CHIP_BG so the summary header and the
# row chip read as the same wash. (Two near-identical near-white shades
# used to live here; consolidated into one per family.)
SUMMARY_BG_CACHED = CHIP_BG_CACHED
SUMMARY_BG_EXEC   = CHIP_BG_EXEC

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
