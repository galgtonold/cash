"""HTML renderer for the BadgeView IR — v3 "code-first" design.

Walks an :class:`InteractiveBadge` tree and emits the v3 visual language
from ``design/notebook-badges/Badge.v3.jsx``:

* code-line foreground (syntax-highlighted, monospace)
* 5px state rail on the left of every row
* tier dots (RAM / DISK) instead of arrows + text
* per-row timing bar scaled to the cell-max
* state-tinted time chip on the right
* loop bodies render an inline mini-histogram of iteration times
* expansions use ``<details>`` and the CSS checkbox-hack so they survive
  Jupyter's HTML sanitiser and VS Code's renderer alike

Interactive layers that require JS (hover-revealed tooltips, keyboard
navigation, dynamic filter chips) are deliberately omitted — they would
silently break in VS Code's sanitised renderer. All "click to expand"
state is driven by hidden checkboxes + ``label[for]`` + CSS sibling
selectors. Filter chips are static visual indicators only.

Style is hoisted into a single per-badge ``<style>`` block so HTML output
stays compact and the CSS class names form a stable contract for tests.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Active tier list for the current render pass — set at the top of
# ``render_html`` from the badge's ``configured_tiers`` field. ``_dots()``
# reads it through ``.get()`` so every row, control body, iteration row,
# and tooltip drawer in one render shares the same tier rack without each
# helper taking a kwarg. Reset to ``()`` at the start of each render to
# prevent leakage between successive renders in the same thread.
_CONFIGURED_TIERS: ContextVar[tuple[str, ...]] = ContextVar(
    "_CONFIGURED_TIERS", default=(),
)

from .. import theme
from ..view import (
    BadgeHeader,
    BadgeStatus,
    BugReportLink,
    ControlGroup,
    ControlGroupSingle,
    DecoratorCall,
    DecoratorCallGroup,
    ForLoopGroup,
    InteractiveBadge,
    IterationRow,
    LoopStatement,
    OverheadBreakdown,
    OverheadEntry,
    SectionItem,
    SectionKind,
    SkippedBucket,
    StatementRow,
    SubUnitGroup,
    iter_iterations,
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
/* Neutralize host (Jupyter / VS Code) defaults — but only on the inline
   row code block (.c3-code). The expanded rowtip code block (.c3-rt-code)
   wants its own background, so we don't touch it. */
.c3-code {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
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
  /* Cap so a 100-statement cell can't push the filter chips off the
     edge — overflow clips the trailing bars rather than expanding. */
  max-width: 200px;
  overflow: hidden;
}}
.c3-spark {{
  display: inline-flex;
  align-items: flex-end;
  height: 16px;
  gap: 1px;
  flex-shrink: 0;
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

/* Panel — overflow is visible so click-to-expand tooltips can escape;
   we trade per-badge scroll for the badge growing with its content. */
.c3-panel {{
  background: {theme.BG_PANEL};
  padding: 0;
  min-width: 0;
  overflow: visible;
}}

/* Upstream subsection — reuses the EXACT same .c3-rowx + checkbox-hack
   pattern as expandable rows below, so it inherits everything that
   already works (grid alignment, VS Code rendering, hover state). The
   head is a real .c3-row label; the body is a sibling div that becomes
   visible when the checkbox is checked. No <details>/<summary> — that
   markup added a separate code path that VS Code's notebook renderer
   styled differently from real rows. */
.c3-upstream {{
  background: {theme.BG_UPSTREAM};
  border-bottom: 1px solid #ececec;
}}
/* Head-row content cell — sits in the row's "code" column.
   Plain block + inline children with explicit margin, so we don't
   rely on flex `gap` (which VS Code's older Chromium renderer
   distributes oddly — pushing the meta to the column centre). */
.c3-upstream-head-cell {{
  padding: 0 10px;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}}
.c3-upstream-label {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: {theme.INK_4};
}}
.c3-upstream-meta {{
  font-family: {theme.FONT_MONO};
  font-size: 10px;
  color: {theme.INK_5};
  /* padding-left rather than margin-left: some notebook hosts reset
     margins on bare <span> elements. The HTML also emits a literal
     &nbsp; before the meta as a belt-and-braces backup. */
  padding-left: 6px;
}}
.c3-upstream-body {{
  display: none;
  border-top: 1px solid #ececec;
  padding: 2px 0;
}}
/* Body becomes visible when the head row's hidden checkbox is checked
   (general-sibling combinator — same mechanism as .c3-rowtip). */
.c3-rxtog:checked ~ .c3-upstream-body {{ display: block; }}

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
  background: #fbfaf5;
  border-top: 1px solid #e6e1d2;
  border-bottom: 1px solid #e6e1d2;
  padding: 10px 14px 12px;
  font-family: {theme.FONT_SANS};
  font-size: 11px;
  color: {theme.INK};
  white-space: normal;
  box-shadow: inset 3px 0 0 #b69a4d;
}}
.c3-rxtog:checked ~ .c3-rowtip {{ display: block; }}
.c3-rxtog:checked ~ label.c3-row {{ background: {theme.BG_HOVER}; }}
/* Loop body expansion (per-iteration drill-down + tooltip) uses its
   own checkbox toggle but keeps the existing summary/drill-down
   markup inside the label. */
.c3-rxtog:checked ~ .c3-iter-table {{ display: flex; }}
.c3-iter-table {{ display: none; }}
.c3-rt-h {{
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 10px;
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
  font-variant-numeric: tabular-nums;
}}
.c3-rt-saved {{ color: {theme.RAIL_CACHED}; }}
.c3-rt-code {{
  margin: 0 0 12px;
  padding: 7px 10px;
  background: #fff;
  border-radius: 3px;
  font-family: {theme.FONT_MONO};
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 110px;
  overflow: hidden;
  color: {theme.INK};
  line-height: 1.45;
}}
.c3-rt-dl {{
  display: grid;
  grid-template-columns: 84px 1fr;
  gap: 3px 12px;
  margin: 0;
  font-size: 10px;
  align-items: baseline;
}}
.c3-rt-dl dt {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {theme.INK_4};
}}
.c3-rt-dl dd {{
  margin: 0;
  color: {theme.INK};
  font-family: {theme.FONT_MONO};
  font-size: 10px;
  word-break: break-word;
}}
.c3-rt-dl dd code {{
  font-family: {theme.FONT_MONO};
  background: #fff;
  border-radius: 2px;
  padding: 0 4px;
  font-size: 9.5px;
}}

.c3-rail {{ width: 5px; align-self: stretch; }}
.c3-rail-soft {{ opacity: 0.5; }}
/* Upstream rows are de-emphasised by softening their rail. This is a
   section-level fact (the row lives in the UPSTREAM section), so it lives
   here as a cascade off the section-body wrapper rather than as a per-row
   class the view layer has to stamp on every StatementRow. */
.c3-upstream-body .c3-rail {{ opacity: 0.5; }}

.c3-code {{
  /* !important throughout: Jupyter's .jp-RenderedHTMLCommon pre rules
     (padding: 0; line-height: 1.21429; menlo-first font stack, and
     overflow:auto which puts a horizontal scrollbar on every long row)
     would otherwise win on specificity and compress / scroll-bar
     every row. */
  margin: 0 !important;
  padding: 5px 10px !important;
  font-family: {theme.FONT_MONO} !important;
  font-size: 12px !important;
  color: {theme.INK} !important;
  white-space: pre !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  line-height: 1.4 !important;
}}
.c3-code-body  {{ color: {theme.INK_2}; }}
/* Faded suffix listing produced/restored variable names. Lives inline
   so the parent .c3-code's overflow:hidden + ellipsis hides it first
   when there's not enough room — never pushes the code itself off. */
.c3-row-vars {{
  margin-left: 10px;
  color: {theme.INK_5};
  font-size: 10.5px;
  font-style: italic;
}}
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

/* Randomness role pill — same vocabulary as the notification pill. Neutral
   (grey) for a seed or a seeded/reproducible draw; the warn-red family for an
   unseeded draw, whose cached value is a frozen replay. */
.c3-rng-pill {{
  display: inline-block;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: {theme.INK_4};
  padding: 1px 5px;
  margin: 0 4px;
  background: {theme.BG_PANEL};
  border: 1px solid {theme.RULE};
  border-radius: 3px;
  vertical-align: middle;
  white-space: nowrap;
}}
.c3-rng-pill.c3-rng-warn {{
  color: {theme.CHIP_FG_WARN};
  background: {theme.CHIP_BG_WARN};
  border-color: #f1c8c1;
}}
/* A row's RNG pill shares ONE grid cell with the code (this flex box) instead of
   being a SEPARATE grid item. The statement-row grid has exactly five columns
   (rail, code, dots, bar, chip); a sixth child overflowed to a new implicit row,
   wrapping the time chip onto a second line under the code. Here the code
   truncates (min-width:0) and the pill never shrinks, so the warning stays whole
   and the chip keeps its column. */
.c3-codepill {{
  display: flex;
  align-items: center;
  min-width: 0;
  overflow: hidden;
}}
.c3-codepill > .c3-code {{ min-width: 0; flex: 0 1 auto; }}
.c3-codepill > .c3-rng-pill {{ flex: 0 0 auto; }}

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
.c3-iter-more .c3-iter-key {{ color: {theme.INK_5}; font-style: italic; }}
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
.c3-deco-detail {{
  display: none;
  background: {theme.BG_DETAIL};
  border-top: 1px solid {theme.RULE_SOFT};
  padding: 8px 16px 10px 22px;
  font-size: 11px;
}}
.c3-rxtog:checked ~ .c3-deco-detail {{ display: block; }}
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

/* Control structure (if / elif / else / try / except).
   Head row is shown verbatim; body rows render inline beneath it (no
   click-to-expand) so the if's body is always visible — same way it
   reads in the source. */
.c3-ctrl {{ display: block; }}
.c3-ctrl-head {{ /* uses .c3-row grid via the markup */ }}
.c3-ctrl-body {{
  border-left: 2px solid #e8e1ce;
  /* Visual indent of one "tab" so nested loops/controls read at a
     similar shift as Python's 4-space source indent — the previous
     5px just aligned rails and looked unindented in deep nesting. */
  margin-left: 24px;
  background: #fbfaf3;
}}

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

import uuid as _uuid  # noqa: E402


def _reset_ids() -> None:
    # Retained for callers that used to clear the counter before rendering.
    # With per-checkbox UUIDs there is no shared mutable state to reset,
    # but we keep the function so the API stays stable.
    return None


def _uid(prefix: str = "id") -> str:
    """Return a globally-unique element id with the given prefix.

    Uses a fresh UUID per call rather than a shared counter + salt so two
    threads rendering badges concurrently cannot interleave salt
    rotations and end up with colliding ``#rx-<salt>-<n>`` IDs on the
    same badge.
    """
    return f"{prefix}-{_uuid.uuid4().hex[:10]}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_time(t: float) -> str:
    return f"{t:.2f}s" if t >= theme.MIN_TIME_DISPLAY_S else "0.00s"


def _snippet(code: str) -> str:
    """First-line snippet (raw, not yet escaped or highlighted).

    No character cap: the row's ``.c3-code`` cell uses
    ``overflow: hidden; text-overflow: ellipsis``, so the browser cuts
    at the column edge rather than hard-truncating mid-word.
    """
    if not code:
        return ""
    return code.splitlines()[0]


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
    configured_tiers: tuple[str, ...] | None = None,
) -> str:
    """N-dot tier indicator — one dot per configured backend tier.

    ``configured_tiers`` comes from the badge root and lists every tier
    the active backend exposes, in configured order (e.g. ``('RAM',
    'REDIS', 'DISK')``). Each dot's fill encodes whether that tier holds
    this entry; the row-level kind colour (cached/exec/warn) still
    derives from the row's status.

    Fallback chain when no configured tier list is available (e.g.
    a standalone renderer test that builds a row directly without
    going through ``render_html``): use this row's own
    ``storage_tiers``; otherwise render a single empty dot.
    """
    written = {t.upper() for t in storage_tiers}
    src_upper = (source or "").upper()

    if uncacheable_reasons:
        kind = "warn"
        def state_for(_tier: str) -> str:
            return "blocked"
        aggregate_title = "Uncacheable: " + ", ".join(uncacheable_reasons)
    elif status is BadgeStatus.RESTORED:
        kind = "cached"
        def state_for(tier: str) -> str:
            return "ring" if tier.upper() == src_upper else "empty"
        aggregate_title = f"Restored from {source or 'cache'}"
    elif status is BadgeStatus.SKIPPED:
        kind = "exec"
        def state_for(_tier: str) -> str:
            return "empty"
        aggregate_title = "Not re-run — downstream value was satisfied by cache"
    elif status in (BadgeStatus.WARNING, BadgeStatus.FUNCTION_CHANGED,
                    BadgeStatus.MODULE_RELOADED, BadgeStatus.ERROR):
        kind = "warn"
        def state_for(_tier: str) -> str:
            return "empty"
        aggregate_title = "—"
    elif storage_tiers:
        kind = "exec"
        def state_for(tier: str) -> str:
            return "solid" if tier.upper() in written else "empty"
        aggregate_title = "Cached to: " + "+".join(storage_tiers)
    else:
        kind = "exec"
        def state_for(_tier: str) -> str:
            return "empty"
        aggregate_title = "no storage info"

    # Tier list resolution:
    # 1. caller-supplied configured tiers (explicit kwarg wins)
    # 2. the render-pass ContextVar set by render_html
    # 3. this row's own storage_tiers (standalone-test fallback)
    # 4. a single empty placeholder so the cell always has at least one dot
    if configured_tiers is None:
        configured_tiers = _CONFIGURED_TIERS.get()
    tiers: tuple[str, ...] = configured_tiers or storage_tiers or ("",)

    def _dot_title(tier: str, state: str) -> str:
        if not tier:
            return aggregate_title
        if state == "solid":
            return f"{tier} — stored"
        if state == "ring":
            return f"{tier} — restored from here"
        if state == "blocked":
            return f"{tier} — uncacheable"
        return f"{tier} — not stored"

    dots_html = "".join(
        f'<span class="c3-dot c3-dot-{state_for(t)}" '
        f'title="{_esc(_dot_title(t, state_for(t)))}"></span>'
        for t in tiers
    )
    return (
        f'<span class="c3-dots-cell"><span class="c3-dots c3-dots-{kind}" '
        f'title="{_esc(aggregate_title)}">{dots_html}</span></span>'
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
        return sum(it.time_s for it in iter_iterations(item))
    if isinstance(item, ControlGroup):
        # Body rows can themselves be nested ControlGroups / ForLoopGroups
        # after the recursive view-builder grouping; sum recursively.
        return sum(_item_total_time(r) for r in item.rows)
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


def _rng_pill(row: StatementRow) -> str:
    """Inline text pill for a statement's RNG role (empty otherwise).

    Matches the notification-pill vocabulary — a terse uppercase tag, no emoji —
    so it reads as part of the badge rather than pasted on. Neutral grey for a
    seed or a seeded (reproducible) draw; the warn-red family for an unseeded
    draw, whose cached value is a frozen replay. A native ``title`` carries the
    detail on hover.
    """
    if row.random_effect == "seed":
        return '<span class="c3-rng-pill" title="Sets the RNG seed">seed</span>'
    if row.random_effect == "draw" and row.random_unseeded:
        return (
            '<span class="c3-rng-pill c3-rng-warn" title="Unseeded randomness — the '
            'cached value is a frozen replay, not a fresh draw. Seed the RNG or '
            'use @cash:no-cache to redraw.">unseeded</span>'
        )
    if row.random_effect == "draw":
        return '<span class="c3-rng-pill" title="Random draw (seeded, reproducible)">random</span>'
    return ""


# ---------------------------------------------------------------------------
# Statement row
# ---------------------------------------------------------------------------

def _rowtip_html(row: StatementRow) -> str:
    """Pure-CSS click-to-expand tooltip body for a :class:`StatementRow`."""
    kind = theme.kind_of(row.status.value)
    # Same word the text renderer uses for this row, and the same word the cell
    # header uses for the whole-cell case (CAS-272) -- this pill used to render
    # the raw enum value, so an HTML row read RESTORED under a CACHED header.
    label = (
        theme.LABEL_UNCACHEABLE
        if row.status is BadgeStatus.COMPUTED
        and (row.uncacheable_reasons or row.skipped_reason)
        else theme.label_of(row.status.value)
    )
    status_pill = (
        f'<span class="c3-rt-status" '
        f'style="color:{theme.chip_fg(kind)};background:{theme.chip_bg(kind)};">'
        f"{_esc(label)}</span>"
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
    if row.sub_units:
        # Per-call-SITE breakdown of the intercepted (on by default, CAS-243)
        # sub-calls this statement made -- see SubUnitGroup for why grouping
        # is by site rather than callee. One summary line, then one line per
        # site with its own hit ratio and cache-key prefix (the same-prefix-
        # across-runs signal the statement row already gives for itself).
        su_hits = sum(1 for g in row.sub_units for c in g.calls if c.status is BadgeStatus.RESTORED)
        su_n = sum(len(g.calls) for g in row.sub_units)
        sites = " · ".join(_esc(g.call_source) for g in row.sub_units[:3])
        dl_parts.append(f"<dt>Sub-calls</dt><dd>{su_hits}/{su_n} hit · {sites}</dd>")
        for g in row.sub_units:
            g_hits = sum(1 for c in g.calls if c.status is BadgeStatus.RESTORED)
            detail = f"{g_hits}/{len(g.calls)} hit"
            if g.key_prefix:
                detail += f" · <code>{_esc(g.key_prefix)}</code>"
            if g.condensed:
                detail += " · condensed"
            if g.miss_reason:
                detail += f" · {_esc(g.miss_reason)}"
            dl_parts.append(
                f"<dt class='cash-subunit'>{_esc(g.call_source)}</dt><dd>{detail}</dd>"
            )
    if row.changed_functions:
        dl_parts.append(f"<dt>Fn changed</dt><dd>{_esc(', '.join(row.changed_functions))}</dd>")
    if row.changed_modules:
        dl_parts.append(f"<dt>Modules reloaded</dt><dd>{_esc(', '.join(row.changed_modules))}</dd>")
    if row.cache_key_short:
        # Short prefix of the statement's cache key — same prefix across
        # re-runs means cash landed in the same slot; different prefix means
        # the inputs/code/file deps changed enough to force a new key.
        dl_parts.append(f'<dt>Key</dt><dd><code>{_esc(row.cache_key_short)}</code></dd>')
    if row.miss_reason and row.status is BadgeStatus.COMPUTED:
        # One-line attribution: why did this cell recompute instead of
        # restoring from cache? Populated by statement_processor on the
        # miss path; only shown on COMPUTED rows (a RESTORED row doesn't
        # have a "miss").
        dl_parts.append(f"<dt>Miss</dt><dd>{_esc(row.miss_reason)}</dd>")
    if row.random_effect == "seed":
        dl_parts.append("<dt>Random</dt><dd>sets the RNG seed</dd>")
    elif row.random_effect == "draw" and row.random_unseeded:
        dl_parts.append(
            "<dt>Random</dt><dd>unseeded — the cached value is a frozen replay, "
            "not a fresh draw. Seed the RNG for reproducibility, or use "
            "<code>@cash:no-cache</code> to redraw every run.</dd>"
        )
    elif row.random_effect == "draw":
        dl_parts.append("<dt>Random</dt><dd>random draw (seeded, reproducible)</dd>")

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

    # Notification rows (WARNING / FUNCTION_CHANGED / MODULE_RELOADED / ERROR)
    # get a text pill in the time chip and skip the timing bar. The row code
    # shows just the descriptor (function names / module names / error type)
    # — the pill conveys the kind, no need for a wordy prefix.
    if status is BadgeStatus.FUNCTION_CHANGED:
        descriptor = ", ".join(row.changed_functions) or _snippet(row.code) or "—"
        code_html = f'<pre class="c3-code">{_esc(descriptor)}</pre>'
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip("changed")
    elif status is BadgeStatus.MODULE_RELOADED:
        descriptor = ", ".join(row.changed_modules) or _snippet(row.code) or "—"
        code_html = f'<pre class="c3-code">{_esc(descriptor)}</pre>'
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip("reloaded")
    elif status is BadgeStatus.WARNING:
        descriptor = _snippet(row.code) or "—"
        code_html = f'<pre class="c3-code">{_code_html(descriptor)}</pre>'
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip("warn")
    elif status is BadgeStatus.ERROR:
        code_html = f'<pre class="c3-code">{_code_html(row.code)}</pre>'
        bar = '<span class="c3-tbar-cell"></span>'
        chip = _notif_chip("error")
    else:
        # Produced/restored variables get a faded ``← name1, name2`` suffix
        # so a user can see what the row hydrated or computed without
        # opening the click tooltip. Suffix overflows with the code if
        # there isn't room (CSS ellipsis on the parent <pre> takes over).
        names = row.restored_vars or row.output_vars
        suffix = (
            f'<span class="c3-row-vars">← {", ".join(_esc(n) for n in names)}</span>'
            if names else ""
        )
        code_html = f'<pre class="c3-code">{_code_html(row.code)}{suffix}</pre>'
        bar = _tbar(row.time_s, max_time, kind)
        chip = _time_chip(row.time_s, row.saved_time_s, kind)

    dots = _dots(
        status=status,
        storage_tiers=row.storage_tiers,
        source=row.source,
        uncacheable_reasons=row.uncacheable_reasons,
    )
    rng_pill = _rng_pill(row)
    drawer = _rowtip_html(row)

    # The pill and the code share ONE grid cell (a flex box) so the pill is not a
    # sixth child of the five-column row grid — a sixth child wraps the time chip
    # to a new line. Rows without a pill keep the code as the grid cell directly.
    code_cell = f'<div class="c3-codepill">{code_html}{rng_pill}</div>' if rng_pill else code_html

    rid = _uid("rx")
    return (
        f'<div class="c3-rowx">'
        f'<input type="checkbox" class="c3-rxtog" id="{rid}">'
        f'<label class="c3-row" for="{rid}" data-kind="{kind}" data-status="{status.value}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f"{code_cell}"
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


_ITER_VALUE_MAX = 40


def _fmt_iter_value(v: Any) -> str:
    """Compact repr for a per-iteration binding value.

    Inline dicts/lists with many items render as multi-line junk in the
    160px drill-down column. Collapse those to a count summary
    (``{3 items}``); truncate other long reprs in the middle so both the
    head and tail of the value stay visible.
    """
    if isinstance(v, dict):
        n = len(v)
        if n > 2:
            return f"{{{n} items}}"
    if isinstance(v, (list, tuple, set)) and len(v) > 4:
        opener, closer = ("[", "]") if isinstance(v, list) else ("(", ")") if isinstance(v, tuple) else ("{", "}")
        return f"{opener}{len(v)} items{closer}"
    r = repr(v)
    if len(r) <= _ITER_VALUE_MAX:
        return r
    keep = (_ITER_VALUE_MAX - 1) // 2
    return f"{r[:keep]}…{r[-keep:]}"


def _iter_drilldown_html(
    iterations: tuple[IterationRow, ...],
    loop_var_names: tuple[str, ...],
    *,
    max_rows: int = 0,
) -> str:
    if not iterations:
        return ""
    # When no iteration carries loop_bindings, the drill-down would be a
    # column of `var = —` rows that adds no information — skip it.
    if not any(it.loop_bindings for it in iterations):
        return ""
    max_t = max((it.time_s for it in iterations), default=0.001) or 0.001
    # Multi-var loops use Python tuple syntax: ``(cat, stats) = ('A', {...})``.
    # Reads like the loop's actual `for cat, stats in ...:` line.
    is_tuple = len(loop_var_names) > 1
    var_label = (
        f"({', '.join(loop_var_names)})" if is_tuple
        else (loop_var_names[0] if loop_var_names else "i")
    )
    bound_iters = [it for it in iterations if it.loop_bindings]
    total = len(bound_iters)
    visible = bound_iters[: max_rows] if max_rows and total > max_rows else bound_iters
    omitted = total - len(visible)
    rows = []
    for it in visible:
        kind = theme.kind_of(it.status.value)
        bar_color = theme.bar_color(kind)
        rail = theme.rail_color(it.status.value)
        pct = max(1.0, (it.time_s / max_t) * 100)
        vals = [f"<b>{_esc(_fmt_iter_value(v))}</b>" for _, v in it.loop_bindings]
        value_str = f"({', '.join(vals)})" if is_tuple else (vals[0] if vals else "—")
        # Why this iteration re-ran, as a hover title. The drill-down is a
        # fixed four-column grid (bullet / key / bar / time) and a fifth column
        # of prose would wreck it at any realistic reason length, so the
        # attribution rides on the row instead of taking space from it. Before
        # this the reason was computed and then discarded for anything inside a
        # loop -- see IterationRow.
        why = it.miss_reason or it.skipped_reason
        title = f' title="{_esc(why)}"' if why else ""
        rows.append(
            f'<div class="c3-iter-row"{title}>'
            f'<span class="c3-iter-bullet" style="background:{rail};"></span>'
            f'<span class="c3-iter-key">{_esc(var_label)} = {value_str}</span>'
            f'<span class="c3-iter-bar-track"><span style="width:{pct:.1f}%;background:{bar_color};"></span></span>'
            f'<span class="c3-iter-time">{it.time_s:.3f}s</span>'
            f"</div>"
        )
    if omitted > 0:
        rows.append(
            f'<div class="c3-iter-row c3-iter-more">'
            f'<span></span>'
            f'<span class="c3-iter-key">… +{omitted} more iteration{"s" if omitted != 1 else ""}</span>'
            f'<span></span><span></span>'
            f"</div>"
        )
    return f'<div class="c3-iter-table">{"".join(rows)}</div>'


def _loop_stmt_sub_units_html(sub_units: tuple[SubUnitGroup, ...]) -> str:
    """Sub-calls block for a loop-body statement's collapsed aggregate row.

    Sits INSIDE the same toggle-revealed area as ``_iter_drilldown_html``
    (nested under the loop body row, not a sibling of the loop) -- CAS-243
    task 9. *sub_units* here is already aggregated across every iteration
    (see ``view_builder``'s ``LoopStatement.sub_units``), so this shows one
    line per call SITE for the whole statement, not per iteration.
    """
    if not sub_units:
        return ""
    hits = sum(1 for g in sub_units for c in g.calls if c.status is BadgeStatus.RESTORED)
    n = sum(len(g.calls) for g in sub_units)
    rows = []
    for g in sub_units:
        g_hits = sum(1 for c in g.calls if c.status is BadgeStatus.RESTORED)
        detail = f"{g_hits}/{len(g.calls)} hit"
        if g.key_prefix:
            detail += f" · <code>{_esc(g.key_prefix)}</code>"
        if g.condensed:
            detail += " · condensed"
        if g.miss_reason:
            detail += f" · {_esc(g.miss_reason)}"
        rows.append(
            f'<div class="c3-iter-row cash-subunit">'
            f'<span class="c3-iter-key">{_esc(g.call_source)}</span>'
            f'<span class="c3-iter-key">{detail}</span>'
            f"</div>"
        )
    return (
        '<div class="c3-iter-table c3-subunit-table">'
        f'<div class="c3-iter-row c3-subunit-summary">Sub-calls: {hits}/{n} hit</div>'
        + "".join(rows)
        + "</div>"
    )


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
        else "EXECUTED"
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


def _item_saved_time(item) -> float:
    """Aggregate saved-time the same way _item_total_time aggregates time."""
    if isinstance(item, StatementRow):
        return item.saved_time_s
    if isinstance(item, ForLoopGroup):
        return sum(it.saved_time_s for it in iter_iterations(item))
    if isinstance(item, ControlGroup):
        return sum(_item_saved_time(r) for r in item.rows)
    if isinstance(item, ControlGroupSingle):
        return item.row.saved_time_s
    if isinstance(item, SkippedBucket):
        return item.total_saved_time_s
    return 0.0


def _walk_statuses(items) -> "list[BadgeStatus]":
    """Flatten the leaf statuses of any nested SectionItem tree."""
    out: list[BadgeStatus] = []
    for item in items:
        if isinstance(item, StatementRow):
            out.append(item.status)
        elif isinstance(item, ForLoopGroup):
            out.extend(it.status for it in iter_iterations(item))
        elif isinstance(item, ControlGroup):
            out.extend(_walk_statuses(item.rows))
        elif isinstance(item, ControlGroupSingle):
            out.append(item.row.status)
        # OverheadBreakdown / DecoratorCallGroup don't contribute statuses
        # at the cell-summary level — skip.
    return out


def _collect_iterations(items) -> "list[IterationRow]":
    """All IterationRows under a forest of SectionItems — list-input
    convenience over :func:`view.iter_iterations`. Used by the synthetic
    -outer for_loop_group head row to count iterations from nested data
    when it has no direct stmts."""
    return [it for item in items for it in iter_iterations(item)]


def _aggregate_kind(statuses: tuple[BadgeStatus, ...]) -> str:
    """Synthesise a kind across a group of iteration / row statuses."""
    cached = sum(1 for s in statuses if s in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED))
    computed = sum(1 for s in statuses if s is BadgeStatus.COMPUTED)
    if computed == 0 and cached > 0:
        return "cached"
    if cached == 0 and computed > 0:
        return "exec"
    return "exec"  # mixed -> exec coloring; rail picks blue via rail_color('mixed') below


def _for_loop_group_html(g: ForLoopGroup, max_time: float) -> str:
    """Render one ForLoopGroup as **one** for-header row + N body-line rows.

    A loop with several body statements (the common case) used to emit
    one ``for ticker in [...]:`` header per statement, which read as
    duplicated noise. Per the v3 design (Badge.v3.jsx :: LoopBlock) the
    header is rendered once, aggregated across all stmts; each body
    statement gets a single body-line row underneath with its own
    inline iteration mini-histogram and a drill-down.
    """
    if not g.stmts and not g.nested:
        return ""

    # ---- aggregate across every iteration of every body statement -----
    all_iters = [it for stmt in g.stmts for it in stmt.iterations]
    # Synthetic outer wrappers carry no direct stmts (the iteration data
    # lives under a head-suppressed inner inside a hoisted control). Walk
    # nested items to gather statuses for the head row.
    if not all_iters and g.nested:
        all_iters = list(_collect_iterations(g.nested))
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
    iters_per_stmt = len(g.stmts[0].iterations) if g.stmts else total

    # Sample 4 values from the first stmt's bindings for the header
    # preview: ``for ticker in ['TSLA', 'AAPL', 'MSFT', 'GOOG']:``. Use
    # ``repr()`` so strings keep their quotes, lists their brackets, etc. —
    # matching what the user actually typed.
    # Prefer the source-faithful header captured by the runtime
    # ("for cat in df['category'].unique():"). Fall back to synthesising
    # one from observed iteration values for older metrics that predate
    # the loop_header field.
    if g.loop_header:
        loop_header = g.loop_header
    elif not g.stmts:
        loop_header = "for …:"
    else:
        var_decl = ", ".join(g.loop_var_names) or "i"
        header_var_set = set(g.loop_var_names)
        sample_values: list[str] = []
        seen_values: set[tuple[Any, ...]] = set()
        for it in g.stmts[0].iterations:
            if not it.loop_bindings:
                continue
            kept = tuple(v for name, v in it.loop_bindings if name in header_var_set)
            if not kept:
                continue
            key = tuple(repr(v) for v in kept)
            if key in seen_values:
                continue
            seen_values.add(key)
            sample_values.append(", ".join(_esc(repr(v)) for v in kept))
            if len(sample_values) >= 4:
                break
        if len(seen_values) > len(sample_values) or iters_per_stmt > len(sample_values):
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
    if getattr(g, "suppress_head", False):
        # This for-loop is the head-suppressed inner copy under a hoisted
        # outer wrapper (see view_builder._hoist_outer_for_loop). The outer
        # already drew the loop header; here we only emit body stmt rows
        # so the iteration data renders as the control's body.
        head_row = ""
    else:
        head_row = (
            f'<div class="c3-rowx">'
            f'<input type="checkbox" class="c3-rxtog" id="{head_rid}">'
            f'<label class="c3-row c3-loop-head" for="{head_rid}" data-kind="{head_kind}">'
            f'<span class="c3-rail" style="background:{head_rail};"></span>'
            f'<pre class="c3-code">{_code_html(loop_header)}</pre>'
            f'<span class="c3-loop-meta">{_esc(head_meta)}</span>'
            f"{_tbar(head_total_time, max_time, head_kind)}"
            f"{_time_chip(head_total_time, head_total_saved, head_kind)}"
            f"</label>"
            f"{head_tip}"
            f"</div>"
        )

    # ---- body in source order: interleaves direct stmts (LoopStatement)
    # and nested for-loops / controls (SectionItem subtypes). View-builder
    # builds ``g.body`` from the runtime's body_index_chain so this loop's
    # body reads top-to-bottom matching the user's source. Falls back to
    # ``stmts + nested`` for older callers that don't set ``g.body``.
    ordered_body = g.body or (tuple(g.stmts) + tuple(g.nested))
    body_pieces: list[str] = []
    for item in ordered_body:
        if isinstance(item, LoopStatement):
            iters = item.iterations
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
            # Show every iteration when the count fits comfortably in a
            # cell; cap to the first N + a "… +M more" row beyond that.
            cap = 0 if stmt_total <= _ITER_INLINE_LIMIT else _ITER_INLINE_LIMIT
            expansion = (
                _iter_drilldown_html(iters, g.loop_var_names, max_rows=cap)
                + _loop_stmt_sub_units_html(item.sub_units)
            )
            body_rid = _uid("rx")
            body_pieces.append(
                f'<div class="c3-rowx c3-loop-body">'
                f'<input type="checkbox" class="c3-rxtog" id="{body_rid}">'
                f'<label class="c3-row" for="{body_rid}" data-kind="{stmt_kind}">'
                f'<span class="c3-rail c3-rail-soft" style="background:{stmt_rail};"></span>'
                f'<pre class="c3-code c3-code-body">{_code_html(item.base_code or "…")}</pre>'
                f"{_iter_histogram_html(iters)}"
                f"{_tbar(stmt_time, max_time, stmt_kind)}"
                f"{_time_chip(stmt_time, stmt_saved, stmt_kind)}"
                f"</label>"
                f"{expansion}"
                f"</div>"
            )
        else:
            body_pieces.append(_render_section_item(item, max_time))

    # Wrap the entire body in ONE shared .c3-ctrl-body so everything sits
    # at the same indent level — matching Python's 4-space convention
    # (head at L0, body at L1; deeper nesting falls to L2 via the inner
    # for-loop/control's own wrapper).
    inner = "".join(body_pieces)
    if inner and not getattr(g, "suppress_head", False):
        inner = f'<div class="c3-ctrl-body">{inner}</div>'
    return head_row + inner


def _control_group_html(cg: ControlGroup, max_time: float) -> str:
    # Walk nested items to gather every status — body rows may include
    # ForLoopGroups, ControlGroupSingles, etc. that don't carry .status
    # directly. We only need a kind for the aggregate visual treatment.
    statuses = tuple(_walk_statuses(cg.rows))
    kind = _aggregate_kind(statuses)
    # Rail tracks the body's aggregate kind: if every body row is cached,
    # the if-block reads as cached (green), not exec — otherwise the
    # control would look "computed" purely because of the if's bookkeeping
    # while its actual body did no work.
    if kind == "cached":
        rail = theme.RAIL_CACHED
    elif kind == "warn":
        rail = theme.RAIL_WARN
    else:
        rail = theme.RAIL_EXEC
    # The view-builder often puts the same string in both branch_label and
    # header (the metric's body_statements[0] is the if/for/while line
    # itself). Just show the header — the branch keyword is already in it.
    # Only prepend the branch label if it's a short keyword (else / except)
    # that doesn't already appear in the header.
    if cg.branch_label and cg.branch_label not in cg.header:
        head_code = f"{cg.branch_label}: {cg.header}"
    else:
        head_code = cg.header or cg.branch_label
    total_time = sum(_item_total_time(r) for r in cg.rows)
    total_saved = sum(_item_saved_time(r) for r in cg.rows)

    # Control body rows render INLINE as siblings of the head — same way
    # the source code reads (if line then body, both visible). No click-
    # to-expand on the control header; the body is the body. Body rows
    # are wrapped in a .c3-ctrl-body container that's lightly tinted so
    # the "which lines belong to this branch" relationship is obvious.
    body_parts = []
    for r in cg.rows:
        # body_statements metrics get wrapped as ControlGroupSingle by the
        # view-builder; unwrap to the inner StatementRow for the static path.
        if isinstance(r, ControlGroupSingle):
            body_parts.append(_static_statement_row_html(r.row, max_time, indented=True))
        elif isinstance(r, StatementRow):
            body_parts.append(_static_statement_row_html(r, max_time, indented=True))
        else:
            body_parts.append(_render_section_item(r, max_time))
    body_rows = "".join(body_parts)
    head = (
        f'<div class="c3-ctrl">'
        f'<div class="c3-row c3-ctrl-head" data-kind="{kind}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f'<pre class="c3-code">{_code_html(head_code)}</pre>'
        f'<span class="c3-loop-meta">{len(cg.rows)} stmt{"s" if len(cg.rows) != 1 else ""}</span>'
        f"{_tbar(total_time, max_time, kind)}"
        f"{_time_chip(total_time, total_saved, kind)}"
        f"</div>"
        f'<div class="c3-ctrl-body">{body_rows}</div>'
        f"</div>"
    )
    return head


def _static_statement_row_html(
    row: StatementRow, max_time: float, *, indented: bool = False,
) -> str:
    """A row with no click-to-expand wrapper — used inside control bodies
    where the body is always visible. ``indented`` is preserved as a
    flag for callers but no per-row prefix is added: the surrounding
    ``.c3-ctrl-body`` wrapper already supplies one indent level so the
    whole nested block reads at a single Python-like step inward."""
    del indented  # margin-left on the wrapper supplies the indent
    status = row.status
    kind = theme.kind_of(status.value)
    rail = theme.rail_color(status.value)
    code_html = f'<pre class="c3-code">{_code_html(row.code)}</pre>'
    dots = _dots(
        status=status,
        storage_tiers=row.storage_tiers,
        source=row.source,
        uncacheable_reasons=row.uncacheable_reasons,
    )
    bar = _tbar(row.time_s, max_time, kind)
    chip = _time_chip(row.time_s, row.saved_time_s, kind)
    return (
        f'<div class="c3-row" data-kind="{kind}" data-status="{status.value}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f"{code_html}"
        f"{dots}"
        f"{bar}"
        f"{chip}"
        f"</div>"
    )


def _control_group_single_html(cgs: ControlGroupSingle, max_time: float) -> str:
    row = cgs.row
    # While / with / try are processed as a single unit by the runtime
    # (no per-iteration metrics), but the runtime records the original
    # source lines on ``body_statements``. Render head + body lines like
    # a for-loop so the structure reads at a glance instead of one opaque
    # multi-line row.
    if row.body_statements and len(row.body_statements) > 1:
        return _multiline_control_html(row, max_time)
    return _statement_row_html(row, max_time)


def _multiline_control_html(row: StatementRow, max_time: float) -> str:
    """Render a single-unit control structure (``while``, ``with``, ``try``)
    as a for-loop-style block: the first body statement becomes the head
    (carrying the row's aggregate timing and status), subsequent lines
    render as static body rows inside a ``.c3-ctrl-body`` wrapper so the
    indent matches if-blocks and nested for-loops."""
    head_line = row.body_statements[0]
    body_lines = row.body_statements[1:]
    status = row.status
    kind = theme.kind_of(status.value)
    rail = theme.rail_color(status.value)
    head = (
        f'<div class="c3-row c3-loop-head" data-kind="{kind}">'
        f'<span class="c3-rail" style="background:{rail};"></span>'
        f'<pre class="c3-code">{_code_html(head_line)}</pre>'
        f'<span class="c3-loop-meta">{len(body_lines)} stmt{"s" if len(body_lines) != 1 else ""}</span>'
        f"{_tbar(row.time_s, max_time, kind)}"
        f"{_time_chip(row.time_s, row.saved_time_s, kind)}"
        f"</div>"
    )
    body_rows: list[str] = []
    for line in body_lines:
        body_rows.append(
            f'<div class="c3-row" data-kind="{kind}">'
            f'<span class="c3-rail c3-rail-soft" style="background:{theme.INK_5};"></span>'
            f'<pre class="c3-code c3-code-body">{_code_html(line)}</pre>'
            f'<span></span><span></span><span></span>'
            f"</div>"
        )
    return head + f'<div class="c3-ctrl-body">{"".join(body_rows)}</div>'


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
    body = "".join(_render_section_item(i, max_time) for i in sb.items)
    return (
        f'<details class="c3-skipped">'
        f'<summary title="{_esc(title)}">'
        f'<span>{_esc(label)}</span>'
        f'<span class="c3-skipped-meta">{_esc(saved)}</span>'
        f"</summary>"
        f"{body}"
        f"</details>"
    )


# ---------------------------------------------------------------------------
# Decorator section
# ---------------------------------------------------------------------------

#: Tag shown on a call cash wrapped itself via call interception (CAS-243,
#: on by default), as opposed to one the user decorated with ``@cash.cache``.
#: Same cache, but the reader needs to know which mechanism put it there.
#:
#: Deliberately NOT named after a directive: interception is unconditional
#: (no ``# @cash:cache-calls`` needed to make this label appear), and a tag
#: spelling ``cache-calls`` would also substring-match the opt-out
#: ``# @cash:no-cache-calls`` in any test or log scrape that greps for it.
_INTERCEPTED_TAG = "@intercepted"
_INTERCEPTED_TITLE = "cached automatically via call interception -- disable with @cash:no-cache-calls"


def _cache_tag_html(intercepted: bool) -> str:
    if intercepted:
        return (f'<span class="c3-cache-tag" title="{_INTERCEPTED_TITLE}">'
                f'{_INTERCEPTED_TAG}</span>')
    return '<span class="c3-cache-tag">@cache</span>'


def _decorator_call_row_html(
    c: DecoratorCall, max_time: float, *, intercepted: bool = False
) -> str:
    kind = theme.kind_of(c.status.value)
    rail = theme.rail_color(c.status.value)
    short_name = c.func_name.split(".")[-1] if "." in c.func_name else c.func_name
    status_text = "HIT" if c.status is BadgeStatus.RESTORED else "MISS"
    code = (
        f'{_cache_tag_html(intercepted)} '
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
        return "".join(
            _decorator_call_row_html(c, max_time, intercepted=g.intercepted)
            for c in g.calls
        )
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
        f'<pre class="c3-code">{_cache_tag_html(g.intercepted)} '
        f'<span class="c3-kw">{_esc(short)}</span>() '
        f'<span class="c3-com">{_esc(summary_label)}</span></pre>'
        f'<span class="c3-dots-cell"></span>'
        f"{_tbar(total_time, max_time, kind)}"
        f"{_time_chip(total_time, 0.0, kind)}"
        f"</label>"
        # Inline detail block (own class so we don't share rowtip's
        # row-properties styling) — revealed by the same checkbox-hack.
        f'<div class="c3-deco-detail">'
        f'<div class="c3-detail-h">{_cache_tag_html(g.intercepted)} '
        f"{hits} of {n} cached · {total_time:.2f}s total</div>"
        f'<div class="c3-deco-fn">'
        f'<div class="c3-deco-fn-name">{_esc(short)}()</div>'
        f'<div class="c3-deco-strip">{strip}</div>'
        f"</div></div></div>"
    )
    return head


# ---------------------------------------------------------------------------
# Overhead section
# ---------------------------------------------------------------------------

def _overhead_html(ob: OverheadBreakdown, max_time: float) -> str:
    """Render the whole overhead breakdown as a single dim row.

    The sub-categories used to render as separate near-zero rows, which
    dominated the badge visually for no information gain. Now: one row, with an
    inline ``upstream 0.05s · cache 0.06s · badge 0.03s · other 0.92s``
    breakdown in the code cell (each part carries a hover tooltip with the full
    description), and the total in the time chip.
    """
    if not ob.entries:
        return ""

    def _part(e: OverheadEntry) -> str:
        # Build the optional title in its own f-string: a backslash inside an
        # f-string *expression* is a SyntaxError before Python 3.12, and the
        # project still supports 3.10/3.11.
        title = f' title="{_esc(e.tooltip)}"' if e.tooltip else ""
        return (
            f'<span class="c3-ovh-part"{title}>{_esc(e.label)}'
            f'&nbsp;<span class="c3-ovh-time">{e.time_s:.3f}s</span></span>'
        )

    parts = " · ".join(_part(e) for e in ob.entries)
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

def _render_section_item(item: SectionItem, max_time: float) -> str:
    if isinstance(item, StatementRow):
        return _statement_row_html(item, max_time)
    if isinstance(item, ForLoopGroup):
        return _for_loop_group_html(item, max_time)
    if isinstance(item, ControlGroup):
        return _control_group_html(item, max_time)
    if isinstance(item, ControlGroupSingle):
        return _control_group_single_html(item, max_time)
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
            kind = _aggregate_kind(tuple(_walk_statuses(item.rows)))
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
    """Static (non-interactive) state counters in the summary chip.

    Four families: exec, uncacheable, cached, warn. A notification-only cell
    (e.g. only FUNCTION_CHANGED rows) gets only the warn chip; without it the
    summary would carry no counter at all.
    """
    parts = []
    if header.computed_count:
        parts.append(
            f'<span class="c3-fchip c3-fchip-exec"><span class="c3-fchip-dot"></span>'
            f'exec<span class="c3-fchip-count">{header.computed_count}</span></span>'
        )
    # Warn-coloured but its own chip: this is not a transient notification, it
    # is work this cell will repeat on every run until the user changes
    # something. Merged into `warn` it would be lost among RNG and
    # function-changed notices; merged into `exec` (where its rows are also
    # counted) it was invisible altogether.
    if header.uncacheable_count:
        parts.append(
            f'<span class="c3-fchip c3-fchip-warn"><span class="c3-fchip-dot"></span>'
            f'not cached<span class="c3-fchip-count">{header.uncacheable_count}'
            f'</span></span>'
        )
    if header.restored_count or header.skipped_count:
        cached = header.restored_count + header.skipped_count
        parts.append(
            f'<span class="c3-fchip c3-fchip-cached"><span class="c3-fchip-dot"></span>'
            f'cached<span class="c3-fchip-count">{cached}</span></span>'
        )
    if header.warn_count:
        parts.append(
            f'<span class="c3-fchip c3-fchip-warn"><span class="c3-fchip-dot"></span>'
            f'warn<span class="c3-fchip-count">{header.warn_count}</span></span>'
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
    # Publish the configured tier list for every nested ``_dots()`` call
    # in this render pass. Reset on exit so a subsequent render doesn't
    # inherit stale tiers from a previous one.
    _tier_token = _CONFIGURED_TIERS.set(badge.configured_tiers)
    try:
        return _render_html_impl(badge)
    finally:
        _CONFIGURED_TIERS.reset(_tier_token)


def _render_html_impl(badge: InteractiveBadge) -> str:
    kind, label, sub = _summary_meta(badge.header)
    max_time = _max_time(badge)

    body_html = ""
    upstream = next((s for s in badge.sections if s.kind is SectionKind.UPSTREAM), None)
    if upstream is not None and upstream.items:
        rows = "".join(_render_section_item(i, max_time) for i in upstream.items)
        n = sum(1 for i in upstream.items if not isinstance(i, SkippedBucket))
        up_saved = sum(_item_saved_time(i) for i in upstream.items)
        up_exec = sum(_item_total_time(i) for i in upstream.items)
        # Derive the aggregate kind from the actual item statuses so the
        # head row's rail / bar / chip match what's underneath — all
        # restored → cached (green), all computed → exec (ocker), mixed
        # → exec for bar but the rail picks blue via 'mixed' separately.
        statuses = tuple(_walk_statuses(upstream.items))
        up_kind = _aggregate_kind(statuses) if statuses else "cached"
        # Rail color: mirrors the per-row rail logic — if everything was
        # restored we use the cached rail, if anything was computed we
        # surface that on the rail too.
        if statuses and all(s in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED) for s in statuses):
            rail_col = theme.RAIL_CACHED
        elif statuses and all(s is BadgeStatus.COMPUTED for s in statuses):
            rail_col = theme.RAIL_EXEC
        else:
            rail_col = theme.RAIL_MIXED if statuses else theme.RAIL_CACHED
        # Scale the bar against the larger of saved or any badge-wide exec
        # time, so a big saving reads as a near-full bar even when the
        # restore overhead itself is invisibly small.
        bar_scale = max(max_time, up_saved, up_exec, theme.MIN_TIME_DISPLAY_S)
        bar_value = max(up_saved, up_exec)
        bar_html = _tbar(bar_value, bar_scale, up_kind)
        # Chip content: prefer the saved time when it's meaningful (the
        # whole point of the upstream section is to surface what the
        # cache saved). If nothing was saved but the steps still cost
        # real time, show that instead so the chip isn't a lonely "—".
        if up_saved > theme.MIN_TIME_DISPLAY_S:
            chip_display = f"↑{up_saved:.2f}s"
            chip_kind = "cached"
        elif up_exec > theme.MIN_TIME_DISPLAY_S:
            chip_display = _fmt_time(up_exec)
            chip_kind = up_kind
        else:
            chip_display = "—"
            chip_kind = up_kind
        chip_html = (
            f'<span class="c3-time-chip c3-time-chip-{chip_kind}">'
            f'<span>{chip_display}</span></span>'
        )
        plural = "s" if n != 1 else ""
        up_rid = _uid("rx-up")
        body_html += (
            f'<div class="c3-upstream c3-rowx">'
            f'<input type="checkbox" class="c3-rxtog" id="{up_rid}">'
            f'<label class="c3-row" for="{up_rid}" data-kind="{up_kind}">'
            f'<span class="c3-rail c3-rail-soft" style="background:{rail_col};"></span>'
            f'<div class="c3-upstream-head-cell">'
            f'<span class="c3-upstream-label">upstream context</span>'
            # &nbsp; hard-codes the separator so notebook hosts that strip
            # span margins (VS Code) still render visible spacing.
            f'<span class="c3-upstream-meta">&nbsp;{n} step{plural}</span>'
            f'</div>'
            f'<span class="c3-dots-cell"></span>'
            f'{bar_html}'
            f'{chip_html}'
            f"</label>"
            f'<div class="c3-upstream-body">{rows}</div>'
            f"</div>"
        )

    for section in badge.sections:
        if section.kind in (SectionKind.UPSTREAM,):
            continue
        body_html += _section_label(section.kind, section.header)
        for item in section.items:
            body_html += _render_section_item(item, max_time)

    sparkline = _sparkline_html(badge)
    chips = _filter_chips_html(badge.header)

    return (
        _STYLE_BLOCK
        + '<div class="c3-wrap">'
        + f'<details class="c3-card" data-kind="{kind}">'
        + '<summary class="c3-summary">'
        + f'<span class="c3-summary-label">{_esc(label)}</span>'
        + '<span class="c3-summary-sep">·</span>'
        + f'<span class="c3-summary-sub">{_esc(sub)}</span>'
        + sparkline
        + chips
        + '<span class="c3-summary-caret"></span>'
        + "</summary>"
        + '<div class="c3-panel">'
        + body_html
        + _footer_html(badge.footer)
        + "</div>"
        + "</details>"
        + "</div>"
    )
