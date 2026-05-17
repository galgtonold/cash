"""HTML renderer for the BadgeView IR — v3 "code-first" design.

Walks an :class:`InteractiveBadge` tree and emits the v3 visual language
from ``design/notebook-badges/Badge.v3.jsx``:

* code-line foreground (syntax-highlighted, monospace)
* 5px state rail on the left of every row
* tier dots (RAM / DISK) instead of arrows + text
* per-row timing bar scaled to the cell-max
* state-tinted time chip on the right
* loop bodies render an inline mini-histogram of iteration times
* expansions use ``<details>`` (works in classic Jupyter *and* VS Code)
* hover detail is pure CSS :hover (no inline JS, survives notebook strip)

Interactive layers that require JS (filter chips, fixed-position
tooltips, keyboard nav) are deliberately omitted — they would silently
break in VS Code's sanitized renderer. The visual language is preserved.

Style is hoisted into a single per-badge ``<style>`` block so HTML output
stays compact and the CSS class names form a stable contract for tests.
"""

from __future__ import annotations

import threading

from .. import theme
from ..view import (
    BadgeHeader,
    BadgeStatus,
    BugReportLink,
    ControlBody,
    ControlGroup,
    ControlGroupSingle,
    DecoratorCall,
    DecoratorCallGroup,
    ForLoopGroup,
    InteractiveBadge,
    IterationRow,
    LoopStatement,
    OverheadBreakdown,
    Section,
    SectionItem,
    SectionKind,
    SkippedBucket,
    StatementRow,
    StatusBadge,
)
from ._pytoken import highlight_python

# Threshold for the loop-body expansion view: at or below this many
# iterations we show the full per-iteration drill-down (bullet · key ·
# bar · time, one row per iteration); above it we fall back to the
# condensed summary panel with counts. ~25 fits in a notebook cell
# without forcing scroll on a typical screen.
_ITER_INLINE_LIMIT = 25

# ---------------------------------------------------------------------------
# CSS — emitted once per badge inside a <style> block. Class names are
# prefixed with ``c3-`` so they form a stable test contract and don't
# collide with notebook-host styles.
# ---------------------------------------------------------------------------

_CSS = f"""
/* Scoped scrollbar styling — applies only to scrollable elements that
   *contain* a Cash badge. Uses :has() (Chromium 105+, Safari 15.4+,
   Firefox 121+) so we don't repaint scrollbars in cells that don't
   show our badge. Firefox uses scrollbar-color / scrollbar-width;
   Chromium falls back to ::-webkit-scrollbar pseudo-elements. */
:has(> .c3-wrap),
:has(.c3-wrap) {{
  scrollbar-width: thin;
  scrollbar-color: #c5c1b5 transparent;
}}
:has(> .c3-wrap)::-webkit-scrollbar,
:has(.c3-wrap)::-webkit-scrollbar {{
  width: 10px;
  height: 10px;
  background: transparent;
}}
:has(> .c3-wrap)::-webkit-scrollbar-thumb,
:has(.c3-wrap)::-webkit-scrollbar-thumb {{
  background: #c5c1b5;
  border-radius: 5px;
  border: 2px solid transparent;
  background-clip: padding-box;
}}
:has(> .c3-wrap)::-webkit-scrollbar-thumb:hover,
:has(.c3-wrap)::-webkit-scrollbar-thumb:hover {{
  background: #a8a496;
  background-clip: padding-box;
}}
:has(> .c3-wrap)::-webkit-scrollbar-corner,
:has(.c3-wrap)::-webkit-scrollbar-corner {{ background: transparent; }}

/* Outer wrap is a thin positioning anchor only — no reserved padding,
   so the badge takes its natural width and doesn't trigger horizontal
   scroll. Hover detail panels render *in flow* under their row (see
   .c3-rowtip below), which sidesteps every ancestor-overflow trap. */
.c3-wrap {{
  display: inline-block;
  position: relative;
  margin-top: 5px;
  max-width: 100%;
}}
.c3-card {{
  display: inline-flex;
  flex-direction: column;
  background: #fff;
  border: 1px solid {theme.RULE};
  border-left: 3px solid;
  border-radius: 4px;
  overflow: visible;
  font-family: {theme.FONT_SANS};
  font-size: 12px;
  color: {theme.INK};
  max-width: 100%;
  position: relative;
}}
.c3-card[data-kind="cached"] {{ border-left-color: {theme.RAIL_CACHED}; }}
.c3-card[data-kind="exec"]   {{ border-left-color: {theme.RAIL_EXEC}; }}
.c3-card[data-kind="warn"]   {{ border-left-color: {theme.RAIL_WARN}; }}
.c3-card[data-kind="mixed"]  {{ border-left-color: {theme.RAIL_MIXED}; }}

/* Summary chip */
.c3-summary {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  cursor: pointer;
  user-select: none;
  font-size: 11px;
  list-style: none;
  outline: none;
  min-height: 22px;
}}
.c3-summary::-webkit-details-marker {{ display: none; }}
.c3-summary::marker {{ content: ""; }}
.c3-card[open] > .c3-summary {{ border-bottom: 1px solid #efece4; }}
.c3-card[data-kind="cached"][open] > .c3-summary {{ background: {theme.SUMMARY_BG_CACHED}; }}
.c3-card[data-kind="exec"][open]   > .c3-summary {{ background: {theme.SUMMARY_BG_EXEC}; }}

.c3-summary-label {{
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.c3-card[data-kind="cached"] .c3-summary-label {{ color: {theme.RAIL_CACHED}; }}
.c3-card[data-kind="exec"]   .c3-summary-label {{ color: {theme.RAIL_EXEC}; }}
.c3-card[data-kind="warn"]   .c3-summary-label {{ color: {theme.RAIL_WARN}; }}
.c3-card[data-kind="mixed"]  .c3-summary-label {{ color: {theme.RAIL_MIXED}; }}
.c3-summary-sep {{ color: {theme.INK_5}; font-size: 10px; }}
.c3-summary-sub {{
  font-family: {theme.FONT_MONO};
  font-size: 11px;
  color: {theme.INK_3};
}}
.c3-summary-spark {{
  display: inline-flex;
  align-items: flex-end;
  height: 18px;
  padding: 0 4px;
  border-left: 1px solid #e8e5dc;
  border-right: 1px solid #e8e5dc;
  margin-left: 4px;
}}
.c3-spark {{
  display: inline-flex;
  align-items: flex-end;
  height: 16px;
  gap: 1px;
}}
.c3-spark-bar {{
  width: 3px;
  min-height: 2px;
  border-radius: 0.5px;
}}
.c3-summary-chips {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
}}
.c3-fchip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px 2px 5px;
  background: #fff;
  border: 1px solid #e2e2e0;
  border-radius: 10px;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: {theme.INK_3};
}}
.c3-fchip-dot {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
.c3-fchip-exec   .c3-fchip-dot {{ background: {theme.BAR_EXEC}; }}
.c3-fchip-cached .c3-fchip-dot {{ background: {theme.BAR_CACHED}; }}
.c3-fchip-warn   .c3-fchip-dot {{ background: {theme.BAR_WARN}; }}
.c3-fchip-count  {{ color: {theme.INK}; font-variant-numeric: tabular-nums; }}
.c3-summary-caret {{
  width: 7px; height: 7px;
  border-right: 1.5px solid {theme.INK_4};
  border-bottom: 1.5px solid {theme.INK_4};
  transform: rotate(45deg) translate(-2px, -2px);
  margin-left: 6px;
  align-self: center;
  transition: transform 0.15s ease;
}}
.c3-card[open] > .c3-summary > .c3-summary-caret {{
  transform: rotate(-135deg) translate(-2px, -2px);
}}

/* Panel — overflow is visible so pure-CSS hover tooltips can escape;
   we trade per-badge scroll for the badge growing with its content. */
.c3-panel {{
  background: {theme.BG_PANEL};
  padding: 0;
  min-width: 0;
  overflow: visible;
}}

/* Upstream subsection — nested <details> */
.c3-upstream {{
  background: {theme.BG_UPSTREAM};
  border-bottom: 1px solid #ececec;
}}
.c3-upstream > summary {{
  display: flex; align-items: center; gap: 6px;
  padding: 6px 12px;
  cursor: pointer;
  user-select: none;
  font-size: 11px;
  color: {theme.INK_3};
  list-style: none;
}}
.c3-upstream > summary::-webkit-details-marker {{ display: none; }}
.c3-upstream > summary::marker {{ content: ""; }}
.c3-upstream > summary:hover {{ background: #f0f2f4; }}
.c3-upstream-label {{ font-weight: 600; letter-spacing: 0.02em; }}
.c3-upstream-meta  {{
  font-family: {theme.FONT_MONO};
  font-size: 10px;
  color: {theme.INK_4};
  margin-left: auto;
}}
.c3-upstream-body {{ border-top: 1px solid #ececec; padding: 2px 0; }}
.c3-upstream-caret {{
  display: inline-block;
  color: {theme.INK_5};
  font-size: 9px;
  width: 10px;
  margin-right: 2px;
}}
.c3-upstream[open] .c3-upstream-caret::after {{ content: "▾"; }}
.c3-upstream:not([open]) .c3-upstream-caret::after {{ content: "▸"; }}

/* Row grid */
.c3-row {{
  display: grid;
  grid-template-columns: 5px minmax(0, 1fr) 70px 80px 76px;
  align-items: center;
  border-bottom: 1px solid {theme.RULE_SOFT};
  min-height: 26px;
  position: relative;          /* tooltip anchor */
}}
.c3-row:last-child {{ border-bottom: 0; }}
.c3-row[data-clickable="true"] {{ cursor: pointer; }}
.c3-row[data-clickable="true"]:hover {{ background: {theme.BG_HOVER}; }}
.c3-row:hover {{ background: {theme.BG_HOVER}; }}

/* Click-to-expand row detail via the checkbox-hack pattern.
   <input type="checkbox" hidden> + <label for="..."> + sibling CSS.
   Bulletproof across every browser since IE9 — no <details>/<summary>
   quirks, no JS, no overflow traps, no layout shift on hover. */
.c3-rowx, .c3-loop-body {{ display: block; }}
.c3-rxtog {{
  position: absolute;
  opacity: 0;
  pointer-events: none;
  width: 0; height: 0;
  margin: 0;
}}
label.c3-row {{ cursor: pointer; }}
.c3-rowtip {{
  display: none;
  background: #f4efe1;
  border-top: 1px solid #c8c3b3;
  border-bottom: 1px solid #c8c3b3;
  padding: 12px 16px 14px 22px;
  font-family: {theme.FONT_SANS};
  font-size: 11px;
  color: {theme.INK};
  white-space: normal;
  box-shadow: inset 5px 0 0 #b69a4d, 0 1px 2px rgba(0,0,0,0.04);
}}
.c3-rxtog:checked ~ .c3-rowtip {{ display: block; }}
.c3-rxtog:checked ~ label.c3-row {{ background: {theme.BG_HOVER}; }}
/* Loop body expansion (per-iteration drill-down + tooltip) uses its
   own checkbox toggle but keeps the existing summary/drill-down
   markup inside the label. */
.c3-rxtog:checked ~ .c3-iter-table {{ display: flex; }}
.c3-iter-table {{ display: none; }}
.c3-rt-h {{
  display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;
}}
.c3-rt-status {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 3px;
}}
.c3-rt-time {{
  font-family: {theme.FONT_MONO};
  font-size: 11px;
  color: {theme.INK_2};
  margin-left: auto;
}}
.c3-rt-saved {{ color: {theme.RAIL_CACHED}; }}
.c3-rt-code {{
  margin: 0 0 8px;
  padding: 6px 8px;
  background: #f7f6f1;
  border-radius: 3px;
  font-family: {theme.FONT_MONO};
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 110px;
  overflow: hidden;
  color: {theme.INK};
}}
.c3-rt-dl {{
  display: grid;
  grid-template-columns: 100px 1fr;
  gap: 4px 10px;
  margin: 0;
  font-size: 10px;
}}
.c3-rt-dl dt {{
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {theme.INK_4};
  padding-top: 1px;
}}
.c3-rt-dl dd {{
  margin: 0;
  color: {theme.INK};
  font-family: {theme.FONT_MONO};
  font-size: 10px;
  word-break: break-word;
}}

.c3-rail {{ width: 5px; align-self: stretch; }}
.c3-rail-soft {{ opacity: 0.5; }}

.c3-code {{
  margin: 0;
  padding: 5px 10px;
  font-family: {theme.FONT_MONO};
  font-size: 12px;
  color: {theme.INK};
  white-space: pre;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4;
}}
.c3-code-body  {{ color: {theme.INK_2}; }}
.c3-code-group {{ font-style: italic; }}
.c3-caret {{
  display: inline-block;
  color: {theme.INK_5};
  font-size: 9px;
  width: 10px;
  margin-right: 2px;
}}

/* Python syntax tokens */
.c3-kw  {{ color: #cf222e; }}
.c3-str {{ color: #0a3069; }}
.c3-com {{ color: #6e7781; font-style: italic; }}
.c3-num {{ color: #0550ae; }}

/* Tier dots cell */
.c3-dots-cell {{ padding: 0 4px; text-align: left; }}
.c3-dots {{ display: inline-flex; align-items: center; gap: 2px; }}
.c3-dot {{
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
  border: 1.5px solid {theme.RULE};
  background: transparent;
  vertical-align: middle;
}}
.c3-dot-solid   {{ background: currentColor; border-color: currentColor; }}
.c3-dot-ring    {{ background: transparent;  border-color: currentColor; }}
.c3-dot-blocked {{ background: transparent;  border-color: currentColor; border-style: dashed; }}
.c3-dot-empty   {{ background: transparent;  border-color: #d2d4d8; }}
.c3-dots-cached  {{ color: {theme.RAIL_CACHED}; }}
.c3-dots-exec    {{ color: {theme.RAIL_EXEC}; }}
.c3-dots-warn    {{ color: {theme.RAIL_WARN}; }}

/* Timing bar */
.c3-tbar-cell {{ padding: 0 8px; }}
.c3-tbar {{
  display: block;
  height: 6px;
  background: #f2efea;
  border-radius: 3px;
  overflow: hidden;
  width: 100%;
}}
.c3-tbar-fill {{
  display: block;
  height: 100%;
  border-radius: 3px;
}}
.c3-tbar-fill-cached {{ background: {theme.BAR_CACHED}; }}
.c3-tbar-fill-exec   {{ background: {theme.BAR_EXEC}; }}
.c3-tbar-fill-warn   {{ background: {theme.BAR_WARN}; }}

/* Time chip */
.c3-time-chip {{
  padding: 3px 8px 3px 6px;
  text-align: right;
  display: block;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  font-family: {theme.FONT_MONO};
  line-height: 1.3;
  border-left: 1px solid #f0eee8;
}}
.c3-time-chip-exec   {{ color: {theme.CHIP_FG_EXEC};   background: {theme.CHIP_BG_EXEC}; }}
.c3-time-chip-cached {{ color: {theme.CHIP_FG_CACHED}; background: {theme.CHIP_BG_CACHED}; }}
.c3-time-chip-warn   {{ color: {theme.CHIP_FG_WARN};   background: {theme.CHIP_BG_WARN}; }}
.c3-time-sub {{
  display: block;
  font-size: 9px;
  color: {theme.RAIL_CACHED};
  margin-top: 1px;
}}
.c3-notif-pill {{
  display: inline-block;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: {theme.RAIL_WARN};
  padding: 1px 5px;
  background: #fff;
  border: 1px solid #f1c8c1;
  border-radius: 3px;
}}

/* Loop heading line */
.c3-loop-head .c3-code {{ font-weight: 500; }}
.c3-loop-meta {{
  font-family: {theme.FONT_MONO};
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: {theme.INK_3};
  padding: 0 6px;
  text-align: left;
}}

/* Inline iteration mini-histogram on loop body line */
.c3-iter-cell {{ padding: 0 4px; display: flex; align-items: center; }}
.c3-iter-strip {{
  display: inline-flex;
  align-items: flex-end;
  gap: 1px;
  height: 16px;
  max-width: 100%;
  overflow: hidden;
}}
.c3-iter-bar {{
  width: 4px;
  min-height: 3px;
  border-radius: 0.5px;
  flex-shrink: 0;
}}

/* Per-iteration drill-down — collapsed by default, revealed when the
   parent loop-body row's checkbox is checked. The display rule lives
   with the click-to-expand rules above; here we only set layout. */
.c3-iter-table {{
  background: {theme.BG_DETAIL};
  padding: 6px 12px 8px 30px;
  border-bottom: 1px solid {theme.RULE_SOFT};
  flex-direction: column;
  gap: 2px;
}}
.c3-iter-row {{
  display: grid;
  grid-template-columns: 12px 160px 1fr 70px;
  align-items: center;
  gap: 8px;
  font-size: 10px;
}}
.c3-iter-bullet {{
  width: 7px; height: 7px; border-radius: 50%; display: inline-block;
}}
.c3-iter-key {{
  color: {theme.INK_3};
  font-family: {theme.FONT_MONO};
}}
.c3-iter-key b {{ color: {theme.INK}; }}
.c3-iter-bar-track {{
  height: 4px;
  background: #ececec;
  border-radius: 2px;
  overflow: hidden;
}}
.c3-iter-bar-track > span {{ display: block; height: 100%; border-radius: 2px; }}
.c3-iter-time {{
  color: {theme.INK_2};
  text-align: right;
  font-family: {theme.FONT_MONO};
  font-variant-numeric: tabular-nums;
}}

/* Decorator inline detail */
.c3-detail {{
  background: {theme.BG_DETAIL};
  padding: 10px 14px 10px 30px;
  border-bottom: 1px solid #f0f0ef;
  font-size: 11px;
  color: {theme.INK_2};
}}
.c3-detail-h {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {theme.INK_3};
  margin-bottom: 6px;
}}
.c3-cache-tag {{
  display: inline-block;
  padding: 1px 5px;
  font-family: {theme.FONT_MONO};
  font-size: 9px;
  color: #1a73e8;
  background: #e8f0fe;
  border-radius: 3px;
  margin-right: 4px;
}}
.c3-deco-fn      {{ margin-top: 6px; }}
.c3-deco-fn-name {{
  font-family: {theme.FONT_MONO};
  font-size: 11px;
  color: {theme.INK_2};
  margin-bottom: 4px;
}}
.c3-deco-strip {{
  display: flex; align-items: flex-end; gap: 2px; height: 20px;
}}
.c3-deco-bar {{ width: 6px; border-radius: 1px; }}

/* Overhead row — collapsed single-row variant. Section label sits inline
   with the breakdown so we never spend 3 rows showing 0.00s sub-categories. */
.c3-ovh {{ background: #fbfbfa; }}
.c3-ovh-cell {{
  padding: 5px 10px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-family: {theme.FONT_SANS};
  font-size: 10px;
  color: {theme.INK_4};
}}
.c3-ovh-label {{
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 9px;
  color: {theme.INK_4};
  margin-right: 10px;
}}
.c3-ovh-parts {{ font-family: {theme.FONT_MONO}; font-size: 10px; }}
.c3-ovh-part  {{ white-space: nowrap; }}
.c3-ovh-time  {{ color: {theme.INK_3}; font-variant-numeric: tabular-nums; }}

/* Section divider (current cell, decorator cache) — kept low-key */
.c3-section {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {theme.INK_4};
  padding: 8px 12px 4px;
}}

/* Skipped intermediate-dependency bucket (collapsible <details>) */
.c3-skipped {{
  background: {theme.BG_UPSTREAM};
  border-top: 1px solid #ececec;
  border-bottom: 1px solid #ececec;
}}
.c3-skipped > summary {{
  display: flex; align-items: center; gap: 6px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 11px;
  color: {theme.INK_3};
  list-style: none;
}}
.c3-skipped > summary::-webkit-details-marker {{ display: none; }}
.c3-skipped > summary::marker {{ content: ""; }}
.c3-skipped-meta {{
  font-family: {theme.FONT_MONO};
  font-size: 10px;
  color: {theme.INK_4};
  margin-left: auto;
}}

/* Footer */
.c3-footer {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 12px;
  border-top: 1px solid #ececec;
  background: {theme.BG_PANEL};
}}
.c3-hint {{
  font-size: 9px;
  color: {theme.INK_5};
  font-family: {theme.FONT_MONO};
  letter-spacing: 0.02em;
}}
/* !important here because Jupyter classic's notebook.css sets
   a strong default anchor color that would repaint our subtle bug
   link bright blue otherwise. */
a.c3-bug,
a.c3-bug:link,
a.c3-bug:visited {{
  font-family: {theme.FONT_SANS};
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.04em;
  color: {theme.BUG_FG} !important;
  background: transparent !important;
  text-decoration: none !important;
  padding: 3px 6px;
  border-radius: 3px;
}}
a.c3-bug:hover {{
  color: {theme.BUG_FG_HOVER} !important;
  background: #f5f5f5 !important;
}}
.c3-bug-arrow {{
  margin-left: 2px;
  color: {theme.INK_5};
  display: inline-block;
}}
"""


# Emit the style block at most once per <details id>. Because each
# notebook output is a fresh DOM fragment we can re-emit it cheaply; the
# browser deduplicates rule sets.
_STYLE_BLOCK = f"<style>{_CSS}</style>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_id_counter = 0
_id_lock = threading.Lock()
# Per-process salt so checkbox-hack IDs don't collide across multiple
# badges in one notebook page (each render bumps the counter, the salt
# distinguishes the process / kernel session).
import secrets as _secrets  # noqa: E402
_ID_SALT = _secrets.token_hex(3)


def _reset_ids() -> None:
    global _id_counter
    with _id_lock:
        _id_counter = 0


def _uid(prefix: str = "id") -> str:
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return f"{prefix}-{_ID_SALT}-{_id_counter}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_time(t: float) -> str:
    return f"{t:.2f}s" if t >= theme.MIN_TIME_DISPLAY_S else "0.00s"


def _snippet(code: str) -> str:
    """First-line truncated snippet (raw, not yet escaped or highlighted)."""
    if not code:
        return ""
    line = code.splitlines()[0]
    if len(line) > theme.CODE_SNIPPET_MAX_LEN:
        line = line[: theme.CODE_SNIPPET_MAX_LEN] + "…"
    return line


def _code_html(code: str) -> str:
    """Highlighted single-line snippet for the inline row code cell.

    When the source spans multiple lines (``def``, ``class``, multi-line
    expressions), the row only shows the first line; we suffix a small,
    italic ``… +N lines`` hint so users know there's more (full body is
    visible in the hover tooltip).
    """
    if not code:
        return ""
    highlighted = highlight_python(_snippet(code))
    n_extra = max(0, len(code.splitlines()) - 1)
    if n_extra > 0:
        plural = "s" if n_extra != 1 else ""
        highlighted += (
            f'<span class="c3-com" style="margin-left:6px;">'
            f'… +{n_extra} line{plural}</span>'
        )
    return highlighted


# ---------------------------------------------------------------------------
# Tier-dots — semantic derivation from a StatementRow (or pseudo-row).
# ---------------------------------------------------------------------------

def _dots(
    *,
    status: BadgeStatus,
    storage_tiers: tuple[str, ...],
    source: str | None,
    uncacheable_reasons: tuple[str, ...],
) -> str:
    """Two-dot tier indicator: RAM (left) / DISK (right)."""
    if uncacheable_reasons:
        ram, disk, kind = "blocked", "blocked", "warn"
        title = "Uncacheable: " + ", ".join(uncacheable_reasons)
    elif status is BadgeStatus.RESTORED:
        from_ram = (source or "").upper() == "RAM"
        ram, disk = ("ring", "empty") if from_ram else ("empty", "ring")
        kind = "cached"
        title = f"Restored from {source or 'cache'}"
    elif status is BadgeStatus.SKIPPED:
        # 'SKIPPED' here means 'not re-executed' — the cache already had a
        # downstream value that didn't need this step's output. The value
        # was never produced this run, so neither RAM nor DISK has it now;
        # show empty dots in a neutral color to avoid suggesting otherwise.
        ram, disk, kind = "empty", "empty", "exec"
        title = "Not re-run — downstream value was satisfied by cache"
    elif status in (BadgeStatus.WARNING, BadgeStatus.FUNCTION_CHANGED,
                    BadgeStatus.MODULE_RELOADED, BadgeStatus.ERROR):
        ram, disk, kind = "empty", "empty", "warn"
        title = "—"
    elif storage_tiers:
        ram = "solid" if "RAM" in storage_tiers else "empty"
        disk = "solid" if "DISK" in storage_tiers else "empty"
        kind = "exec"
        title = "Cached to: " + "+".join(storage_tiers)
    else:
        ram, disk, kind = "empty", "empty", "exec"
        title = "no storage info"
    return (
        f'<span class="c3-dots-cell"><span class="c3-dots c3-dots-{kind}" '
        f'title="{_esc(title)}">'
        f'<span class="c3-dot c3-dot-{ram}"></span>'
        f'<span class="c3-dot c3-dot-{disk}"></span>'
        f"</span></span>"
    )


# ---------------------------------------------------------------------------
# Per-cell maximum time (for timing-bar scaling)
# ---------------------------------------------------------------------------

def _max_time(badge: InteractiveBadge) -> float:
    """Largest single row time across the whole badge, including overhead.

    Used as the denominator for sqrt-scaled timing bars so every row
    (including overhead entries) shares one scale.
    """
    max_t = 0.0
    for section in badge.sections:
        for item in section.items:
            if isinstance(item, OverheadBreakdown):
                max_t = max(max_t, *(e.time_s for e in item.entries), 0.0)
            else:
                max_t = max(max_t, _item_total_time(item))
    return max(max_t, 0.001)


def _item_total_time(item: SectionItem) -> float:
    if isinstance(item, StatementRow):
        return item.time_s
    if isinstance(item, ForLoopGroup):
        return sum(it.time_s for ls in item.stmts for it in ls.iterations)
    if isinstance(item, ControlGroup):
        return sum(r.time_s for r in item.rows)
    if isinstance(item, ControlGroupSingle):
        return item.row.time_s
    if isinstance(item, SkippedBucket):
        return item.total_saved_time_s
    return 0.0


# ---------------------------------------------------------------------------
# Timing bar + time chip
# ---------------------------------------------------------------------------

def _tbar(time_s: float, max_time: float, kind: str) -> str:
    """Render a per-row timing bar.

    Width uses square-root scaling against the cell-max so trivial rows
    stay visible when one row dominates — a 1% row becomes a 10% bar, a
    25% row becomes 50%, 100% stays 100%. Rows below the displayable
    time threshold render as an empty track (no fill at all) so the eye
    isn't fooled by a min-width clamp into thinking they're non-trivial.
    """
    if max_time <= 0 or time_s < theme.MIN_TIME_DISPLAY_S:
        return '<span class="c3-tbar-cell"><span class="c3-tbar"></span></span>'
    ratio = min(1.0, time_s / max_time)
    pct = (ratio ** 0.5) * 100  # square-root scaling
    return (
        f'<span class="c3-tbar-cell"><span class="c3-tbar">'
        f'<span class="c3-tbar-fill c3-tbar-fill-{kind}" '
        f'style="width:{pct:.1f}%;"></span></span></span>'
    )


def _time_chip(time_s: float, saved_s: float, kind: str) -> str:
    main = f"<span>{_fmt_time(time_s)}</span>"
    sub = (
        f'<span class="c3-time-sub">↑{saved_s:.2f}s</span>'
        if saved_s > theme.MIN_TIME_DISPLAY_S else ""
    )
    return f'<span class="c3-time-chip c3-time-chip-{kind}">{main}{sub}</span>'


def _notif_chip(label: str) -> str:
    return (
        '<span class="c3-time-chip c3-time-chip-warn">'
        f'<span class="c3-notif-pill">{_esc(label)}</span></span>'
    )


# ---------------------------------------------------------------------------
# Statement row
# ---------------------------------------------------------------------------

def _rowtip_html(row: StatementRow) -> str:
    """Pure-CSS hover tooltip body for a :class:`StatementRow`."""
    kind = theme.kind_of(row.status.value)
    status_pill = (
        f'<span class="c3-rt-status" '
        f'style="color:{theme.chip_fg(kind)};background:{theme.chip_bg(kind)};">'
        f"{_esc(row.status.value)}</span>"
    )
    is_notif = row.status in (
        BadgeStatus.WARNING, BadgeStatus.FUNCTION_CHANGED, BadgeStatus.MODULE_RELOADED,
    )
    time_html = "—" if is_notif else f"{row.time_s:.3f}s"
    if row.saved_time_s > theme.MIN_TIME_DISPLAY_S:
        time_html += f' <span class="c3-rt-saved">· saved {row.saved_time_s:.2f}s</span>'

    code_block = (
        f'<pre class="c3-rt-code">{highlight_python(row.code)}</pre>'
        if row.code else ""
    )

    dl_parts: list[str] = []

    # Storage — paired tier-dot indicator (matches the row's dots cell) plus
    # a human-readable label, mirroring the v3 design tooltip.
    tier_dot_pair = _dots(
        status=row.status,
        storage_tiers=row.storage_tiers,
        source=row.source,
        uncacheable_reasons=row.uncacheable_reasons,
    )
    if row.uncacheable_reasons:
        dl_parts.append(
            f"<dt>Storage</dt><dd>{tier_dot_pair} uncacheable</dd>"
            f"<dt>Reason</dt><dd>{_esc(', '.join(row.uncacheable_reasons))}</dd>"
        )
    elif row.status is BadgeStatus.RESTORED and row.source:
        dl_parts.append(
            f"<dt>Storage</dt><dd>{tier_dot_pair} ← {_esc(row.source)}</dd>"
        )
    elif row.storage_tiers:
        label = " · ".join(_esc(t) for t in row.storage_tiers)
        dl_parts.append(f"<dt>Storage</dt><dd>{tier_dot_pair} {label}</dd>")
    elif row.status is BadgeStatus.SKIPPED:
        dl_parts.append(
            f"<dt>Storage</dt><dd>{tier_dot_pair} not re-run (cache covered downstream)</dd>"
        )
    elif row.skipped_reason:
        dl_parts.append(f"<dt>Skipped</dt><dd>{_esc(row.skipped_reason)}</dd>")

    if row.restored_vars:
        dl_parts.append(f"<dt>Restored</dt><dd>{_esc(', '.join(row.restored_vars))}</dd>")
    elif row.output_vars:
        dl_parts.append(f"<dt>Produced</dt><dd>{_esc(', '.join(row.output_vars))}</dd>")

    if row.decorator_calls:
        hits = sum(1 for c in row.decorator_calls if c.status is BadgeStatus.RESTORED)
        n = len(row.decorator_calls)
        dl_parts.append(f"<dt>@cache</dt><dd>{hits}/{n} cache hits</dd>")
    if row.changed_functions:
        dl_parts.append(f"<dt>Fn changed</dt><dd>{_esc(', '.join(row.changed_functions))}</dd>")
    if row.changed_modules:
        dl_parts.append(f"<dt>Modules reloaded</dt><dd>{_esc(', '.join(row.changed_modules))}</dd>")

    dl = f'<dl class="c3-rt-dl">{"".join(dl_parts)}</dl>' if dl_parts else ""

    return (
        '<div class="c3-rowtip">'
        f'<div class="c3-rt-h">{status_pill}<span class="c3-rt-time">{time_html}</span></div>'
        f"{code_block}"
        f"{dl}"
        "</div>"
    )


def _statement_row_html(row: StatementRow, max_time: float) -> str:
    status = row.status
    kind = theme.kind_of(status.value)
    rail = theme.rail_color(status.value)
    rail_soft = " c3-rail-soft" if row.is_upstream else ""

    # Notification rows (WARNING / FUNCTION_CHANGED / MODULE_RELOADED) get a
    # text pill in the time chip and skip the timing bar.
    if status is BadgeStatus.FUNCTION_CHANGED:
        label = "changed"
        descriptor = ", ".join(row.changed_functions) or _snippet(row.code) or "—"
        code_html = (
            f'<pre class="c3-code"><span class="c3-com">function source changed: </span>'
            f"{_esc(descriptor)}</pre>"
        )
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip(label)
    elif status is BadgeStatus.MODULE_RELOADED:
        label = "reloaded"
        descriptor = ", ".join(row.changed_modules) or _snippet(row.code) or "—"
        code_html = (
            f'<pre class="c3-code"><span class="c3-com">module reloaded: </span>'
            f"{_esc(descriptor)}</pre>"
        )
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip(label)
    elif status is BadgeStatus.WARNING:
        label = "warn"
        descriptor = _snippet(row.code) or "—"
        code_html = f'<pre class="c3-code">{_code_html(descriptor)}</pre>'
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip(label)
    else:
        # Code line shows just the highlighted code. Variable info, source,
        # decorator stats, etc. all live in the hover tooltip rather than
        # crowding the row with a redundant `# var` prefix.
        code_html = f'<pre class="c3-code">{_code_html(row.code)}</pre>'
        bar = _tbar(row.time_s, max_time, kind)
        chip = _time_chip(row.time_s, row.saved_time_s, kind)

    dots = _dots(
        status=status,
        storage_tiers=row.storage_tiers,
        source=row.source,
        uncacheable_reasons=row.uncacheable_reasons,
    )
    drawer = _rowtip_html(row)

    rid = _uid("rx")
    return (
        f'<div class="c3-rowx">'
        f'<input type="checkbox" class="c3-rxtog" id="{rid}">'
        f'<label class="c3-row" for="{rid}" data-kind="{kind}" data-status="{status.value}">'
        f'<span class="c3-rail{rail_soft}" style="background:{rail};"></span>'
        f"{code_html}"
        f"{dots}"
        f"{bar}"
        f"{chip}"
        f"</label>"
        f"{drawer}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Iteration histogram (loop body marginalia)
# ---------------------------------------------------------------------------

def _iter_histogram_html(iterations: tuple[IterationRow, ...]) -> str:
    if not iterations:
        return '<span class="c3-iter-cell"></span>'
    max_t = max((it.time_s for it in iterations), default=0.001) or 0.001
    bars = []
    for it in iterations:
        h = max(3, int((it.time_s / max_t) * 16))
        kind = theme.kind_of(it.status.value)
        bar_color = theme.bar_color(kind)
        bindings = ", ".join(f"{name}={value!r}" for name, value in it.loop_bindings)
        title = f"{bindings} · {it.status.value} · {it.time_s:.3f}s"
        bars.append(
            f'<span class="c3-iter-bar" '
            f'style="height:{h}px;background:{bar_color};" '
            f'title="{_esc(title)}"></span>'
        )
    return (
        '<span class="c3-iter-cell"><span class="c3-iter-strip">'
        + "".join(bars)
        + "</span></span>"
    )


def _iter_drilldown_html(iterations: tuple[IterationRow, ...], loop_var_names: tuple[str, ...]) -> str:
    if not iterations:
        return ""
    max_t = max((it.time_s for it in iterations), default=0.001) or 0.001
    rows = []
    for it in iterations:
        kind = theme.kind_of(it.status.value)
        bar_color = theme.bar_color(kind)
        rail = theme.rail_color(it.status.value)
        pct = max(1.0, (it.time_s / max_t) * 100)
        bindings = ", ".join(f"<b>{_esc(repr(v))}</b>" for _, v in it.loop_bindings) or "—"
        var_label = ", ".join(loop_var_names) or "i"
        rows.append(
            f'<div class="c3-iter-row">'
            f'<span class="c3-iter-bullet" style="background:{rail};"></span>'
            f'<span class="c3-iter-key">{_esc(var_label)} = {bindings}</span>'
            f'<span class="c3-iter-bar-track"><span style="width:{pct:.1f}%;background:{bar_color};"></span></span>'
            f'<span class="c3-iter-time">{it.time_s:.3f}s</span>'
            f"</div>"
        )
    return f'<div class="c3-iter-table">{"".join(rows)}</div>'


# ---------------------------------------------------------------------------
# Loop / control / skipped renderers
# ---------------------------------------------------------------------------

def _loop_tip_html(
    *,
    title_code: str,
    total: int,
    cached: int,
    computed: int,
    total_time: float,
    total_saved: float,
    kind: str,
) -> str:
    """Hover tooltip for an aggregate loop row (head or body line)."""
    status_label = (
        "MIXED" if cached and computed
        else "CACHED" if cached and not computed
        else "COMPUTED"
    )
    pill = (
        f'<span class="c3-rt-status" '
        f'style="color:{theme.chip_fg(kind)};background:{theme.chip_bg(kind)};">'
        f"{status_label}</span>"
    )
    time_html = f"{total_time:.3f}s"
    if total_saved > theme.MIN_TIME_DISPLAY_S:
        time_html += f' <span class="c3-rt-saved">· saved {total_saved:.2f}s</span>'
    code_block = f'<pre class="c3-rt-code">{highlight_python(title_code)}</pre>'
    counts = []
    if total > 0:
        counts.append(f"<dt>Iterations</dt><dd>{total}</dd>")
    if cached:
        counts.append(f"<dt>Cached</dt><dd>{cached}</dd>")
    if computed:
        counts.append(f"<dt>Computed</dt><dd>{computed}</dd>")
    dl = f'<dl class="c3-rt-dl">{"".join(counts)}</dl>' if counts else ""
    return (
        '<div class="c3-rowtip">'
        f'<div class="c3-rt-h">{pill}<span class="c3-rt-time">{time_html}</span></div>'
        f"{code_block}"
        f"{dl}"
        "</div>"
    )


def _aggregate_kind(statuses: tuple[BadgeStatus, ...]) -> str:
    """Synthesise a kind across a group of iteration / row statuses."""
    cached = sum(1 for s in statuses if s in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED))
    computed = sum(1 for s in statuses if s is BadgeStatus.COMPUTED)
    if computed == 0 and cached > 0:
        return "cached"
    if cached == 0 and computed > 0:
        return "exec"
    return "exec"  # mixed -> exec coloring; rail picks blue via rail_color('mixed') below


def _for_loop_group_html(g: ForLoopGroup, max_time: float, *, is_upstream: bool) -> str:
    """Render one ForLoopGroup as **one** for-header row + N body-line rows.

    A loop with several body statements (the common case) used to emit
    one ``for ticker in [...]:`` header per statement, which read as
    duplicated noise. Per the v3 design (Badge.v3.jsx :: LoopBlock) the
    header is rendered once, aggregated across all stmts; each body
    statement gets a single body-line row underneath with its own
    inline iteration mini-histogram and a drill-down.
    """
    if not g.stmts:
        return ""
    rail_soft = " c3-rail-soft" if is_upstream else ""

    # ---- aggregate across every iteration of every body statement -----
    all_iters = [it for stmt in g.stmts for it in stmt.iterations]
    statuses = tuple(it.status for it in all_iters)
    cached = sum(1 for s in statuses if s in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED))
    total = len(all_iters)
    head_kind = _aggregate_kind(statuses)
    if cached > 0 and (total - cached) > 0:
        head_rail = theme.RAIL_MIXED
    elif head_kind == "cached":
        head_rail = theme.RAIL_CACHED
    else:
        head_rail = theme.RAIL_EXEC
    head_total_time = sum(it.time_s for it in all_iters)
    head_total_saved = sum(it.saved_time_s for it in all_iters)

    # Iterations-per-stmt — the loop's actual trip count.
    iters_per_stmt = len(g.stmts[0].iterations)

    # Sample 4 values from the first stmt's bindings for the header
    # preview: ``for ticker in ['TSLA', 'AAPL', 'MSFT', 'GOOG']:``. Use
    # ``repr()`` so strings keep their quotes, lists their brackets, etc. —
    # matching what the user actually typed.
    var_decl = ", ".join(g.loop_var_names) or "i"
    sample_values: list[str] = []
    for it in g.stmts[0].iterations[:4]:
        if not it.loop_bindings:
            continue
        sample_values.append(", ".join(_esc(repr(v)) for _, v in it.loop_bindings))
    if iters_per_stmt > len(sample_values):
        sample_values.append("…")
    values_preview = ", ".join(sample_values) if sample_values else "…"
    loop_header = f"for {var_decl} in [{values_preview}]:"

    head_meta = (
        f"{iters_per_stmt}× cached" if cached == total and total > 0
        else f"{cached}/{total} cached" if cached > 0
        else f"{iters_per_stmt} iters"
    )

    head_tip = _loop_tip_html(
        title_code=loop_header,
        total=total, cached=cached, computed=(total - cached),
        total_time=head_total_time, total_saved=head_total_saved, kind=head_kind,
    )
    head_rid = _uid("rx")
    head_row = (
        f'<div class="c3-rowx">'
        f'<input type="checkbox" class="c3-rxtog" id="{head_rid}">'
        f'<label class="c3-row c3-loop-head" for="{head_rid}" data-kind="{head_kind}">'
        f'<span class="c3-rail{rail_soft}" style="background:{head_rail};"></span>'
        f'<pre class="c3-code">{_code_html(loop_header)}</pre>'
        f'<span class="c3-loop-meta">{_esc(head_meta)}</span>'
        f"{_tbar(head_total_time, max_time, head_kind)}"
        f"{_time_chip(head_total_time, head_total_saved, head_kind)}"
        f"</label>"
        f"{head_tip}"
        f"</div>"
    )

    # ---- one body line per stmt, each with its own per-stmt aggregates --
    body_rows: list[str] = []
    for stmt in g.stmts:
        iters = stmt.iterations
        if not iters:
            continue
        stmt_statuses = tuple(it.status for it in iters)
        stmt_cached = sum(1 for s in stmt_statuses if s in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED))
        stmt_total = len(iters)
        stmt_time = sum(it.time_s for it in iters)
        stmt_saved = sum(it.saved_time_s for it in iters)
        stmt_kind = _aggregate_kind(stmt_statuses)
        if stmt_cached > 0 and (stmt_total - stmt_cached) > 0:
            stmt_rail = theme.RAIL_MIXED
        elif stmt_kind == "cached":
            stmt_rail = theme.RAIL_CACHED
        else:
            stmt_rail = theme.RAIL_EXEC
        body_drill = _iter_drilldown_html(iters, g.loop_var_names)
        body_tip = _loop_tip_html(
            title_code=stmt.base_code or "…",
            total=stmt_total, cached=stmt_cached, computed=(stmt_total - stmt_cached),
            total_time=stmt_time, total_saved=stmt_saved, kind=stmt_kind,
        )
        body_rid = _uid("rx")
        # Pick expansion content by iteration count: show the per-iteration
        # drill-down for small loops (you want to see every value), fall
        # back to the condensed counts panel when there'd be too many rows
        # to scan. Threshold picked to match what fits comfortably in a
        # notebook cell without scrolling.
        if stmt_total <= _ITER_INLINE_LIMIT:
            expansion = body_drill
        else:
            expansion = body_tip
        body_rows.append(
            f'<div class="c3-rowx c3-loop-body">'
            f'<input type="checkbox" class="c3-rxtog" id="{body_rid}">'
            f'<label class="c3-row" for="{body_rid}" data-kind="{stmt_kind}">'
            f'<span class="c3-rail c3-rail-soft" style="background:{stmt_rail};"></span>'
            f'<pre class="c3-code c3-code-body">    {_code_html(stmt.base_code or "…")}</pre>'
            f"{_iter_histogram_html(iters)}"
            f"{_tbar(stmt_time, max_time, stmt_kind)}"
            f"{_time_chip(stmt_time, stmt_saved, stmt_kind)}"
            f"</label>"
            f"{expansion}"
            f"</div>"
        )

    return head_row + "".join(body_rows)


def _control_group_html(cg: ControlGroup, max_time: float) -> str:
    rail = theme.rail_color(BadgeStatus.COMPUTED.value)
    statuses = tuple(r.status for r in cg.rows)
    kind = _aggregate_kind(statuses)
    head_code = f"{cg.branch_label}: {cg.header}" if cg.branch_label else cg.header
    total_time = sum(r.time_s for r in cg.rows)
    total_saved = sum(r.saved_time_s for r in cg.rows)

    cg_rid = _uid("rx")
    head = (
        f'<div class="c3-rowx">'
        f'<input type="checkbox" class="c3-rxtog" id="{cg_rid}">'
        f'<label class="c3-row" for="{cg_rid}" data-kind="{kind}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f'<pre class="c3-code">{_code_html(head_code)}</pre>'
        f'<span class="c3-loop-meta">{len(cg.rows)} stmt{"s" if len(cg.rows) != 1 else ""}</span>'
        f"{_tbar(total_time, max_time, kind)}"
        f"{_time_chip(total_time, total_saved, kind)}"
        f"</label>"
        f'<div class="c3-rowtip">'
        + "".join(_statement_row_html(r, max_time) for r in cg.rows)
        + "</div></div>"
    )
    return head


def _control_group_single_html(cgs: ControlGroupSingle, max_time: float) -> str:
    return _statement_row_html(cgs.row, max_time)


def _control_body_html(cb: ControlBody) -> str:
    if not cb.body_stmts:
        return ""
    inner = "\n".join(_esc(s) for s in cb.body_stmts)
    return (
        f'<details class="c3-control-body" style="padding:6px 12px 6px 30px;">'
        f'<summary style="cursor:pointer;font-size:10px;color:{theme.INK_4};list-style:none;">body</summary>'
        f'<pre style="font-family:{theme.FONT_MONO};font-size:11px;color:{theme.INK};margin:4px 0 0;">{inner}</pre>'
        f"</details>"
    )


def _skipped_bucket_html(sb: SkippedBucket, max_time: float) -> str:
    """Render the collapsible bucket of upstream statements that were *not* re-run.

    Semantically: each statement here produced data that some later upstream
    cell now restores from cache, so re-executing it would be pure waste —
    the downstream consumer already has the final value. See
    ``virtual_lineage._collect_skipped_statement_metrics`` for the
    dependency-walk that flags them.
    """
    if not sb.items:
        return ""
    n = len(sb.items)
    label = f"{n} upstream step{'s' if n != 1 else ''} not re-run"
    title = (
        "These upstream statements weren't re-executed because their outputs "
        "are only needed by downstream values that Cash restored from cache. "
        "Running them again would do work the cache already covered."
    )
    saved = (
        f"saved {sb.total_saved_time_s:.2f}s"
        if sb.total_saved_time_s > theme.MIN_TIME_DISPLAY_S else "—"
    )
    body = "".join(_render_section_item(i, max_time, is_upstream=True) for i in sb.items)
    return (
        f'<details class="c3-skipped">'
        f'<summary title="{_esc(title)}">'
        f'<span class="c3-upstream-caret"></span>'
        f'<span>{_esc(label)}</span>'
        f'<span class="c3-skipped-meta">{_esc(saved)}</span>'
        f"</summary>"
        f"{body}"
        f"</details>"
    )


# ---------------------------------------------------------------------------
# Decorator section
# ---------------------------------------------------------------------------

def _decorator_call_row_html(c: DecoratorCall, max_time: float) -> str:
    kind = theme.kind_of(c.status.value)
    rail = theme.rail_color(c.status.value)
    short_name = c.func_name.split(".")[-1] if "." in c.func_name else c.func_name
    status_text = "HIT" if c.status is BadgeStatus.RESTORED else "MISS"
    code = (
        f'<span class="c3-cache-tag">@cache</span> '
        f'<span class="c3-kw">{_esc(short_name)}</span>() '
        f'<span class="c3-com">{status_text}</span>'
    )
    return (
        f'<div class="c3-row" data-kind="{kind}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f'<pre class="c3-code">{code}</pre>'
        f'<span class="c3-dots-cell"></span>'
        f"{_tbar(c.time_s, max_time, kind)}"
        f"{_time_chip(c.time_s, 0.0, kind)}"
        f"</div>"
    )


def _decorator_group_html(g: DecoratorCallGroup, max_time: float) -> str:
    if not g.condensed:
        return "".join(_decorator_call_row_html(c, max_time) for c in g.calls)
    n = len(g.calls)
    hits = sum(1 for c in g.calls if c.status is BadgeStatus.RESTORED)
    misses = n - hits
    total_time = sum(c.time_s for c in g.calls)
    kind = "cached" if misses == 0 else "exec"
    rail = theme.RAIL_CACHED if misses == 0 else theme.RAIL_EXEC
    short = g.func_name.split(".")[-1] if "." in g.func_name else g.func_name
    summary_label = (
        f"all {n} cached" if misses == 0
        else f"{n} calls, all computed" if hits == 0
        else f"{hits}/{n} cached, {misses} computed"
    )
    # Per-call mini-strip for the breakdown
    max_call_t = max((c.time_s for c in g.calls), default=0.001) or 0.001
    strip = "".join(
        f'<span class="c3-deco-bar" style="height:{max(3, int((c.time_s / max_call_t) * 18))}px;'
        f'background:{theme.BAR_CACHED if c.status is BadgeStatus.RESTORED else theme.BAR_EXEC};" '
        f'title="call #{i + 1} · {"HIT" if c.status is BadgeStatus.RESTORED else "MISS"} · {c.time_s:.3f}s"></span>'
        for i, c in enumerate(g.calls)
    )
    dec_rid = _uid("rx")
    head = (
        f'<div class="c3-rowx">'
        f'<input type="checkbox" class="c3-rxtog" id="{dec_rid}">'
        f'<label class="c3-row" for="{dec_rid}" data-kind="{kind}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f'<pre class="c3-code"><span class="c3-cache-tag">@cache</span> '
        f'<span class="c3-kw">{_esc(short)}</span>() '
        f'<span class="c3-com">{_esc(summary_label)}</span></pre>'
        f'<span class="c3-dots-cell"></span>'
        f"{_tbar(total_time, max_time, kind)}"
        f"{_time_chip(total_time, 0.0, kind)}"
        f"</label>"
        f'<div class="c3-rowtip"><div class="c3-detail">'
        f'<div class="c3-detail-h"><span class="c3-cache-tag">@cache</span> '
        f"{hits} of {n} cached · {total_time:.2f}s total</div>"
        f'<div class="c3-deco-fn">'
        f'<div class="c3-deco-fn-name">{_esc(short)}()</div>'
        f'<div class="c3-deco-strip">{strip}</div>'
        f"</div></div></div></div>"
    )
    return head


# ---------------------------------------------------------------------------
# Overhead section
# ---------------------------------------------------------------------------

def _overhead_html(ob: OverheadBreakdown, max_time: float) -> str:
    """Render the whole overhead breakdown as a single dim row.

    Up to four sub-categories used to render as four near-zero rows,
    which dominated the badge visually for no information gain. Now:
    one row, with an inline ``upstream 0.05s · init 0.02s · other 0.92s``
    breakdown in the code cell, and the total in the time chip.
    """
    if not ob.entries:
        return ""
    parts = " · ".join(
        f'<span class="c3-ovh-part">{_esc(e.label)}'
        f'&nbsp;<span class="c3-ovh-time">{e.time_s:.3f}s</span></span>'
        for e in ob.entries
    )
    return (
        f'<div class="c3-row c3-ovh" data-kind="exec">'
        f'<span class="c3-rail c3-rail-soft" style="background:{theme.INK_4};"></span>'
        f'<div class="c3-ovh-cell">'
        f'<span class="c3-ovh-label">overhead</span>'
        f'<span class="c3-ovh-parts">{parts}</span>'
        f"</div>"
        f'<span class="c3-dots-cell"></span>'
        f"{_tbar(ob.total_s, max_time, 'exec')}"
        f"{_time_chip(ob.total_s, 0.0, 'exec')}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Section dispatch
# ---------------------------------------------------------------------------

def _render_section_item(item: SectionItem, max_time: float, *, is_upstream: bool) -> str:
    if isinstance(item, StatementRow):
        return _statement_row_html(item, max_time)
    if isinstance(item, ForLoopGroup):
        return _for_loop_group_html(item, max_time, is_upstream=is_upstream)
    if isinstance(item, ControlGroup):
        return _control_group_html(item, max_time)
    if isinstance(item, ControlGroupSingle):
        return _control_group_single_html(item, max_time)
    if isinstance(item, ControlBody):
        return _control_body_html(item)
    if isinstance(item, SkippedBucket):
        return _skipped_bucket_html(item, max_time)
    if isinstance(item, DecoratorCallGroup):
        return _decorator_group_html(item, max_time)
    if isinstance(item, OverheadBreakdown):
        return _overhead_html(item, max_time)
    raise TypeError(f"Unsupported BadgeView node: {type(item).__name__}")


def _section_label(kind: SectionKind, header: str) -> str:
    if kind is SectionKind.UPSTREAM:
        return ""  # rendered via <details class="c3-upstream"> wrapper instead
    if kind is SectionKind.CURRENT and not header:
        return ""
    return f'<div class="c3-section">{_esc(header)}</div>'


# ---------------------------------------------------------------------------
# Summary chip (header + sparkline + filter counts)
# ---------------------------------------------------------------------------

def _summary_meta(header: BadgeHeader) -> tuple[str, str, str]:
    """``(kind, label, sub)`` for the collapsed pill."""
    # RUNNING placeholder
    if header.current_step or header.total_steps or header.current_code:
        if header.total_steps:
            sub = f"({header.current_step}/{header.total_steps})"
        elif header.current_step:
            sub = f"(step {header.current_step})"
        else:
            sub = "…"
        return "exec", "PROCESSING", sub

    if header.computed_count == 0 and (header.restored_count > 0 or header.skipped_count > 0):
        if header.restored_count > 0:
            label = "CACHED"
            sub = (
                f"saved {header.total_saved_s:.2f}s"
                if header.total_saved_s > theme.MIN_TIME_DISPLAY_S else f"{header.total_exec_s:.2f}s"
            )
        else:
            label = "SKIPPED"
            sub = "already computed"
        return "cached", label, sub

    label = "EXECUTED"
    if header.total_saved_s > theme.MIN_TIME_DISPLAY_S:
        sub = f"{header.total_exec_s:.2f}s · saved {header.total_saved_s:.2f}s"
    else:
        sub = f"{header.total_exec_s:.2f}s"
    return "exec", label, sub


def _sparkline_html(badge: InteractiveBadge) -> str:
    """Per-current-statement aggregate, one bar each."""
    bars = []
    current = next((s for s in badge.sections if s.kind is SectionKind.CURRENT), None)
    if current is None:
        return ""
    times: list[tuple[float, str]] = []
    for item in current.items:
        t = _item_total_time(item)
        if isinstance(item, ForLoopGroup):
            statuses = tuple(it.status for ls in item.stmts for it in ls.iterations)
            kind = _aggregate_kind(statuses)
        elif isinstance(item, StatementRow):
            kind = theme.kind_of(item.status.value)
        elif isinstance(item, ControlGroup):
            kind = _aggregate_kind(tuple(r.status for r in item.rows))
        elif isinstance(item, ControlGroupSingle):
            kind = theme.kind_of(item.row.status.value)
        else:
            continue
        if t == 0 and kind == "warn":
            t = 0.01
        times.append((t, kind))
    if not times:
        return ""
    max_t = max(t for t, _ in times) or 0.001
    for t, kind in times:
        h = max(2, int((t / max_t) * 16))
        bars.append(
            f'<span class="c3-spark-bar" style="height:{h}px;background:{theme.bar_color(kind)};"></span>'
        )
    return (
        '<span class="c3-summary-spark"><span class="c3-spark">'
        + "".join(bars)
        + "</span></span>"
    )


def _filter_chips_html(header: BadgeHeader) -> str:
    """Static (non-interactive) state counters in the summary chip."""
    parts = []
    if header.computed_count:
        parts.append(
            f'<span class="c3-fchip c3-fchip-exec"><span class="c3-fchip-dot"></span>'
            f'exec<span class="c3-fchip-count">{header.computed_count}</span></span>'
        )
    if header.restored_count or header.skipped_count:
        cached = header.restored_count + header.skipped_count
        parts.append(
            f'<span class="c3-fchip c3-fchip-cached"><span class="c3-fchip-dot"></span>'
            f'cached<span class="c3-fchip-count">{cached}</span></span>'
        )
    if not parts:
        return ""
    return '<span class="c3-summary-chips">' + "".join(parts) + "</span>"


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _footer_html(footer: BugReportLink | None) -> str:
    bug = ""
    if footer is not None:
        bug = (
            f'<a class="c3-bug" href="{footer.url}" target="_blank" rel="noopener noreferrer" '
            f'title="Open a pre-filled GitHub issue to report incorrect caching behaviour">'
            f'Report incorrect caching<span class="c3-bug-arrow"> →</span></a>'
        )
    return (
        f'<div class="c3-footer">'
        f'<span class="c3-hint">click a row for detail</span>'
        f"{bug}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def render_html(badge: InteractiveBadge) -> str:
    """Render an :class:`InteractiveBadge` to v3-design HTML."""
    _reset_ids()
    kind, label, sub = _summary_meta(badge.header)
    max_time = _max_time(badge)

    body_html = ""
    upstream = next((s for s in badge.sections if s.kind is SectionKind.UPSTREAM), None)
    if upstream is not None and upstream.items:
        rows = "".join(_render_section_item(i, max_time, is_upstream=True) for i in upstream.items)
        n = sum(1 for i in upstream.items if not isinstance(i, SkippedBucket))
        saved_blob = (
            f"saved {badge.header.total_saved_s:.2f}s"
            if badge.header.total_saved_s > theme.MIN_TIME_DISPLAY_S else "—"
        )
        body_html += (
            f'<details class="c3-upstream" open>'
            f'<summary>'
            f'<span class="c3-upstream-caret"></span>'
            f'<span class="c3-upstream-label">upstream context</span>'
            f'<span class="c3-upstream-meta"><b>{n}</b> · {_esc(saved_blob)}</span>'
            f"</summary>"
            f'<div class="c3-upstream-body">{rows}</div>'
            f"</details>"
        )

    for section in badge.sections:
        if section.kind in (SectionKind.UPSTREAM,):
            continue
        body_html += _section_label(section.kind, section.header)
        for item in section.items:
            body_html += _render_section_item(item, max_time, is_upstream=False)

    sparkline = _sparkline_html(badge)
    chips = _filter_chips_html(badge.header)

    return (
        _STYLE_BLOCK
        + '<div class="c3-wrap">'
        + f'<details class="c3-card" data-kind="{kind}">'
        + f'<summary class="c3-summary">'
        + f'<span class="c3-summary-label">{_esc(label)}</span>'
        + f'<span class="c3-summary-sep">·</span>'
        + f'<span class="c3-summary-sub">{_esc(sub)}</span>'
        + sparkline
        + chips
        + f'<span class="c3-summary-caret"></span>'
        + "</summary>"
        + f'<div class="c3-panel">'
        + body_html
        + _footer_html(badge.footer)
        + "</div>"
        + "</details>"
        + "</div>"
    )


def render_status_badge_html(badge: StatusBadge) -> str:
    """Render the compact non-interactive status pill (unchanged from v3 design)."""
    color = (theme.BADGE_COLOR_RESTORED
             if badge.status is BadgeStatus.RESTORED
             else theme.BADGE_COLOR_DEFAULT)
    storage_str = "+".join(badge.storage_tiers) if badge.storage_tiers else ""
    saved_str = f" · saved {badge.time_saved_s:.2f}s" if badge.time_saved_s > 0 else ""
    source_str = f" · ← {badge.source}" if badge.source else ""
    storage_segment = f" · {storage_str}" if storage_str else ""
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'background:{color}1A;color:{color};font-family:{theme.FONT_MONO};'
        f'font-size:11px;font-weight:500;">'
        f"<b>{_esc(badge.status.value.upper())}</b>"
        f"{source_str}{storage_segment} {badge.execution_time_s:.3f}s{saved_str}"
        f"</span>"
    )
