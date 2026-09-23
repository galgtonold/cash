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

from .view import BadgeStatus

# ---------------------------------------------------------------------------
# Typography — system-native font stacks. Cash never ships a webfont; the
# host (Jupyter / browser) provides the typography.
# ---------------------------------------------------------------------------
FONT_SANS = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"'
)
FONT_MONO = 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'

# ---------------------------------------------------------------------------
# Display limits
# ---------------------------------------------------------------------------
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
RAIL_CACHED = "#2a8a5e"
BAR_CACHED = "#86bfa3"
CHIP_BG_CACHED = "#ecf6f0"
CHIP_FG_CACHED = "#1f6646"

# EXEC — single amber hue family.
RAIL_EXEC = "#a87530"
BAR_EXEC = "#d3a76b"
CHIP_BG_EXEC = "#fbf3e5"
CHIP_FG_EXEC = "#7a5310"

# WARN — single red hue family.
RAIL_WARN = "#b53d29"
BAR_WARN = "#e8a397"
CHIP_BG_WARN = "#fbeae5"
CHIP_FG_WARN = "#8a2916"

# MIXED rail: blue accent used only for the LEFT RAIL when a single row
# combines cached + computed iterations. The bar/chip for a mixed row
# still uses the EXEC family — the rail is the lone signal that this is
# a hybrid state, so we don't introduce a fourth full kind.
RAIL_MIXED = "#336699"

# Summary chip background equals CHIP_BG so the summary header and the
# row chip read as the same wash. (Two near-identical near-white shades
# used to live here; consolidated into one per family.)
SUMMARY_BG_CACHED = CHIP_BG_CACHED
SUMMARY_BG_EXEC = CHIP_BG_EXEC

# Neutrals — borders, dim text, table tints.
INK = "#222"
INK_2 = "#444"
INK_3 = "#666"
INK_4 = "#888"
INK_5 = "#999"
RULE = "#e6e6e6"
RULE_SOFT = "#f4f3ef"
BG_PANEL = "#fdfdfc"
BG_UPSTREAM = "#f6f7f8"
BG_HOVER = "#faf9f5"
BG_DETAIL = "#fafbfc"

# Bug-report pill (subtle tertiary link in v3).
BUG_FG = "#888"
BUG_FG_HOVER = "#444"
BG_BUG_HOVER = "#f5f5f5"

# Surfaces and rules the stylesheet uses in one place each.
BG_SURFACE = "#fff"
BG_ROWTIP = "#fbfaf5"
BG_TRACK = "#f2efea"
BG_ITER_TRACK = "#ececec"
BG_CTRL_BODY = "#fbfaf3"
BG_OVERHEAD = "#fbfbfa"
RULE_SECTION = "#ececec"
RULE_SUMMARY = "#efece4"
RULE_SPARK = "#e8e5dc"
RULE_CHIP = "#e2e2e0"
RULE_ROWTIP = "#e6e1d2"
RULE_TIME_CHIP = "#f0eee8"
RULE_WARN_PILL = "#f1c8c1"
RULE_DETAIL = "#f0f0ef"
RULE_CTRL_BODY = "#e8e1ce"
ACCENT_ROWTIP = "#b69a4d"
DOT_EMPTY_BORDER = "#d2d4d8"
CACHE_TAG_FG = "#1a73e8"
CACHE_TAG_BG = "#e8f0fe"
SCROLLBAR_THUMB = "#c5c1b5"
SCROLLBAR_THUMB_HOVER = "#a8a496"

# Syntax highlighting of the code in a row.
SYNTAX_KEYWORD = "#cf222e"
SYNTAX_STRING = "#0a3069"
SYNTAX_COMMENT = "#6e7781"
SYNTAX_NUMBER = "#0550ae"


# ---------------------------------------------------------------------------
# CSS custom properties
# ---------------------------------------------------------------------------

# Every token ``badge.css`` reads, as ``var(--c3-<name>)`` with ``_`` -> ``-``.
_CSS_TOKENS = (
    "FONT_SANS",
    "FONT_MONO",
    "RAIL_CACHED",
    "BAR_CACHED",
    "CHIP_BG_CACHED",
    "CHIP_FG_CACHED",
    "RAIL_EXEC",
    "BAR_EXEC",
    "CHIP_BG_EXEC",
    "CHIP_FG_EXEC",
    "RAIL_WARN",
    "BAR_WARN",
    "CHIP_BG_WARN",
    "CHIP_FG_WARN",
    "RAIL_MIXED",
    "SUMMARY_BG_CACHED",
    "SUMMARY_BG_EXEC",
    "INK",
    "INK_2",
    "INK_3",
    "INK_4",
    "INK_5",
    "RULE",
    "RULE_SOFT",
    "BG_PANEL",
    "BG_UPSTREAM",
    "BG_HOVER",
    "BG_DETAIL",
    "BUG_FG",
    "BUG_FG_HOVER",
    "BG_BUG_HOVER",
    "BG_SURFACE",
    "BG_ROWTIP",
    "BG_TRACK",
    "BG_ITER_TRACK",
    "BG_CTRL_BODY",
    "BG_OVERHEAD",
    "RULE_SECTION",
    "RULE_SUMMARY",
    "RULE_SPARK",
    "RULE_CHIP",
    "RULE_ROWTIP",
    "RULE_TIME_CHIP",
    "RULE_WARN_PILL",
    "RULE_DETAIL",
    "RULE_CTRL_BODY",
    "ACCENT_ROWTIP",
    "DOT_EMPTY_BORDER",
    "CACHE_TAG_FG",
    "CACHE_TAG_BG",
    "SYNTAX_KEYWORD",
    "SYNTAX_STRING",
    "SYNTAX_COMMENT",
    "SYNTAX_NUMBER",
)

# The scrollbar rules style the badge's scrolling ANCESTORS, which sit
# outside ``.c3-wrap`` and so need these declared on themselves.
_SCROLLBAR_TOKENS = ("SCROLLBAR_THUMB", "SCROLLBAR_THUMB_HOVER")


def _declarations(names: tuple[str, ...]) -> str:
    return "".join(f"  --c3-{n.lower().replace('_', '-')}: {globals()[n]};\n" for n in names)


def css_custom_properties() -> str:
    """The token declarations ``badge.css`` is written against."""
    return (
        f".c3-wrap {{\n{_declarations(_CSS_TOKENS)}}}\n"
        f":has(> .c3-wrap),\n:has(.c3-wrap) {{\n{_declarations(_SCROLLBAR_TOKENS)}}}\n"
    )


# ---------------------------------------------------------------------------
# Semantic mappers
# ---------------------------------------------------------------------------

# One row per status: its kind (``cached`` / ``exec`` / ``warn``) and the word
# the badge shows. The words are the reader's, not cash's: a notebook user
# "executes" a cell and a result is "cached", so ``restored`` and ``computed``
# never reach the page, at any level.
_STATUS_STYLE: dict[BadgeStatus, tuple[str, str]] = {
    BadgeStatus.RESTORED: ("cached", "CACHED"),
    BadgeStatus.SKIPPED: ("cached", "SKIPPED"),
    BadgeStatus.COMPUTED: ("exec", "EXECUTED"),
    BadgeStatus.MIXED: ("exec", "MIXED"),
    BadgeStatus.RUNNING: ("exec", "RUNNING"),
    BadgeStatus.BYPASSED: ("exec", "BYPASSED"),
    BadgeStatus.ERROR: ("warn", "ERROR"),
    BadgeStatus.FUNCTION_CHANGED: ("warn", "FUNC CHANGED"),
    BadgeStatus.MODULE_RELOADED: ("warn", "MODULE RELOADED"),
    BadgeStatus.WARNING: ("warn", "WARNING"),
}

# What an uncacheable row says instead of EXECUTED. It ran *and* it will run
# again every time, which is the part worth the reader's attention.
LABEL_UNCACHEABLE = "NOT CACHED"


def kind_of(status: BadgeStatus) -> str:
    """``'cached' | 'exec' | 'warn'`` for a status."""
    return _STATUS_STYLE[status][0]


def label_of(status: BadgeStatus) -> str:
    """The word the badge shows for a status."""
    return _STATUS_STYLE[status][1]


def rail_color(status: BadgeStatus) -> str:
    """Color of the left rail for one row."""
    if status is BadgeStatus.MIXED:
        return RAIL_MIXED
    return {"cached": RAIL_CACHED, "warn": RAIL_WARN}.get(kind_of(status), RAIL_EXEC)


def bar_color(kind: str) -> str:
    return {"cached": BAR_CACHED, "warn": BAR_WARN}.get(kind, BAR_EXEC)


def chip_bg(kind: str) -> str:
    return {"cached": CHIP_BG_CACHED, "warn": CHIP_BG_WARN}.get(kind, CHIP_BG_EXEC)


def chip_fg(kind: str) -> str:
    return {"cached": CHIP_FG_CACHED, "warn": CHIP_FG_WARN}.get(kind, CHIP_FG_EXEC)
