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

# Detail palette for individual rows. Renderers map from BadgeStatus through
# :func:`row_visual` rather than inlining hex codes.
_COLOR_RESTORED = "#006644"
_COLOR_COMPUTED = "#996300"
_COLOR_SKIPPED = "#006644"
_COLOR_NOTIFY = "#e65100"      # function changed / module reloaded
_COLOR_NOTIFY_DARK = "#bf360c"
_COLOR_WARNING = "#f57f17"
_COLOR_DECORATOR = "#1a73e8"
_COLOR_MUTED = "#666"
_COLOR_DIM = "#999"

# Background fills for the summary pill, keyed by the overall cell outcome.
SUMMARY_BG_CACHED = "#e6fffa"
SUMMARY_BG_EXECUTED = "#fffbe6"
SUMMARY_BG_RUNNING = "#f0f0f0"
SUMMARY_BORDER_RUNNING = "#ccc"
SUMMARY_COLOR_RUNNING = "#666"


def row_visual(status: str) -> tuple[str, str, str]:
    """Return ``(icon, color, label)`` for a per-row status.

    Argument is the *value* of a :class:`~.view.BadgeStatus` (the str enum
    decays cleanly into ``str``). Single source of truth for the row-level
    palette; previously this mapping was duplicated across
    ``_render_upstream_badge_row``, ``_render_current_badge_row``,
    ``_ctrl_stmt_icon_color``, ``render_decorator_calls``, and inline
    f-strings throughout :mod:`._badge` and :mod:`._components`.
    """
    if status == "restored":
        return "⚡", _COLOR_RESTORED, "Restored"
    if status == "computed":
        return "⚙️", _COLOR_COMPUTED, "Executed"
    if status == "skipped":
        return "⏩", _COLOR_SKIPPED, "Skipped"
    if status == "function_changed":
        return "\U0001f504", _COLOR_NOTIFY, "Changed"
    if status == "module_reloaded":
        return "\U0001f504", _COLOR_NOTIFY, "Reloaded"
    if status == "warning":
        return "⚠️", _COLOR_WARNING, "Warning"
    if status == "error":
        return "❌", _COLOR_NOTIFY_DARK, "Error"
    if status == "mixed":
        return "⚡", _COLOR_RESTORED, "Partial"
    return "❓", _COLOR_DIM, "Unknown"


def upstream_row_visual(status: str) -> tuple[str, str, str]:
    """Like :func:`row_visual` but uses the upstream arrow icon for restored/computed.

    Upstream rows historically use ``⬆️`` (the up-arrow) regardless
    of status to signal "from a prior cell"; only the colour and label change.
    """
    if status == "restored":
        return "⬆️ Restored", _COLOR_RESTORED, "Restored"
    if status == "computed":
        return "⬆️ Executed", _COLOR_COMPUTED, "Executed"
    icon, color, label = row_visual(status)
    return icon, color, label

