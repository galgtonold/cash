"""HTML renderer for the BadgeView IR.

One ``isinstance``-dispatched function per node type. All style decisions
go through :mod:`..theme`; no hex codes inline (except for the few one-off
backgrounds that haven't been promoted to named tokens yet).

This renderer is the structural successor to the f-string assembly in
``_badge.py`` and ``_components.py``. It is intentionally not byte-equal
to the legacy output — that's the swap's job to reconcile via the
existing badge_display test suite.
"""

from __future__ import annotations

import threading
from typing import Any

from .. import theme
from ..view import (
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_id_counter = 0
_id_lock = threading.Lock()


def _reset_ids() -> None:
    global _id_counter
    with _id_lock:
        _id_counter = 0


def _uid(prefix: str = "id") -> str:
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return f"{prefix}_{_id_counter}"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _snippet(code: str, max_len: int = theme.CODE_SNIPPET_MAX_LEN) -> str:
    s = code.splitlines()[0][:max_len] if code else ""
    if len(code) > max_len:
        s += "…"
    return _esc(s)


def _fmt_time_s(t: float) -> str:
    return f"{t:.2f}s" if t > theme.MIN_TIME_DISPLAY_S else "-"


def _storage_display(tiers: tuple[str, ...]) -> str:
    if not tiers:
        return "-"
    return f"→ {'+'.join(tiers)}"


def _storage_with_reasons(row: StatementRow) -> str:
    """Storage cell that also surfaces uncacheable / skipped reasons."""
    if row.uncacheable_reasons:
        reason = _esc(", ".join(row.uncacheable_reasons))
        return f"<span title='Not cached: {reason}' style='cursor:help; color: #d9534f;'>🚫 No Cache</span>"
    if row.skipped_reason:
        return (
            f"<span title='{_esc(row.skipped_reason)}' "
            f"style='cursor:help; color: #e68a00;'>⚠️ Not Cached</span>"
        )
    if row.storage_tiers:
        return _storage_display(row.storage_tiers)
    return "-"


def _source_label(row: StatementRow) -> str:
    if row.status is BadgeStatus.RESTORED and row.source:
        return f"← {row.source}"
    return ""


# ---------------------------------------------------------------------------
# Row renderers
# ---------------------------------------------------------------------------

def _row_html(row: StatementRow) -> str:
    visual = theme.upstream_row_visual if row.is_upstream else theme.row_visual
    icon_label = visual(row.status.value)
    type_cell, color = _icon_cell(icon_label, row)
    code_cell = _code_cell(row)
    storage_cell = _source_or_storage_cell(row)
    time_cell = _time_cell(row)
    return (
        f"<tr style=\"border-bottom: 1px solid #eee;\">"
        f"<td style=\"padding: 4px; text-align: left; color: {color};\">{type_cell}</td>"
        f"<td style=\"padding: 4px; text-align: left; font-family: {theme.FONT_MONO};\">{code_cell}</td>"
        f"<td style=\"padding: 4px; text-align: left; font-size: 10px; color: #555;\">{storage_cell}</td>"
        f"<td style=\"padding: 4px; text-align: left;\">{time_cell}</td>"
        f"</tr>"
    )


def _icon_cell(visual: tuple[str, str, str], row: StatementRow) -> tuple[str, str]:
    icon_or_label, color, label = visual
    # upstream_row_visual returns the full "⬆️ Restored" text in slot 0;
    # row_visual returns just the icon. Detect and combine appropriately.
    if " " in icon_or_label:
        return icon_or_label, color
    return f"{icon_or_label} {label}", color


def _code_cell(row: StatementRow) -> str:
    code = _snippet(row.code)
    code_span = f'<span style="color: #666; font-size: 10px;">{code}</span>'
    if row.output_vars:
        vars_str = ", ".join(_esc(v) for v in row.output_vars)
        return f"{vars_str}<br>{code_span}"
    return code_span


def _source_or_storage_cell(row: StatementRow) -> str:
    if row.status is BadgeStatus.RESTORED and row.source:
        return _source_label(row)
    return _storage_with_reasons(row)


def _time_cell(row: StatementRow) -> str:
    if row.status is BadgeStatus.RESTORED and row.time_s > 0:
        return f"Saved {row.time_s:.2f}s"
    if row.status is BadgeStatus.SKIPPED:
        return f"Saved {row.time_s:.2f}s" if row.time_s > theme.MIN_TIME_DISPLAY_S else "-"
    return _fmt_time_s(row.time_s)


# ---------------------------------------------------------------------------
# Loop / control / skipped renderers
# ---------------------------------------------------------------------------

def _iteration_row_html(it: IterationRow) -> str:
    icon, color, _label = theme.row_visual(it.status.value)
    binding_str = ", ".join(f"{_esc(name)}={_esc(repr(val))}" for name, val in it.loop_bindings)
    return (
        f"<tr style=\"border-bottom: 1px solid #f0f0f0;\">"
        f"<td style=\"padding: 2px 4px 2px 24px; color: {color}; border-left: 2px solid #ddd;\">{icon}</td>"
        f"<td style=\"padding: 2px 4px; font-family: {theme.FONT_MONO}; font-size: 11px;\">"
        f"<span style=\"color: #666; font-size: 10px;\">{binding_str}</span></td>"
        f"<td style=\"padding: 2px 4px; font-size: 10px; color: #888;\">{_storage_display(it.storage_tiers)}</td>"
        f"<td style=\"padding: 2px 4px;\">{_fmt_time_s(it.time_s)}</td>"
        f"</tr>"
    )


def _loop_statement_html(ls: LoopStatement) -> str:
    gid = _uid("loop_stmt")
    n = len(ls.iterations)
    total = sum(it.time_s for it in ls.iterations)
    summary = (
        f"<tr style=\"border-bottom: 1px solid #eee;\">"
        f"<td style=\"padding: 4px; color: {theme._COLOR_COMPUTED};\">\U0001f501</td>"
        f"<td style=\"padding: 4px; font-family: {theme.FONT_MONO}; font-size: 11px;\">"
        f"<span onclick=\"document.querySelectorAll('.{gid}').forEach(e=>e.style.display=e.style.display==='none'?'table-row':'none')\" "
        f"style=\"cursor:pointer; color: #1a73e8;\">{_snippet(ls.base_code)} "
        f"<span style='color:#888;'>({n} iterations)</span></span></td>"
        f"<td style=\"padding: 4px; font-size: 10px; color: #888;\">-</td>"
        f"<td style=\"padding: 4px;\">{_fmt_time_s(total)}</td>"
        f"</tr>"
    )
    iter_rows = "".join(
        _wrap_hidden(_iteration_row_html(it), gid) for it in ls.iterations
    )
    return summary + iter_rows


def _wrap_hidden(row_html: str, class_name: str) -> str:
    """Inject a class onto an existing <tr> so the toggle JS can find it."""
    return row_html.replace(
        "<tr ", f"<tr class=\"{class_name}\" ", 1,
    ).replace(
        "border-bottom: 1px solid #f0f0f0;\">",
        "border-bottom: 1px solid #f0f0f0; display: none;\">",
        1,
    )


def _for_loop_group_html(g: ForLoopGroup) -> str:
    return "".join(_loop_statement_html(ls) for ls in g.stmts)


def _control_group_html(cg: ControlGroup) -> str:
    gid = _uid("ctrl")
    header_html = (
        f"<tr style=\"border-bottom: 1px solid #eee;\">"
        f"<td style=\"padding: 4px; color: {theme._COLOR_COMPUTED};\">\U0001f500</td>"
        f"<td style=\"padding: 4px; font-family: {theme.FONT_MONO}; font-size: 11px;\">"
        f"<span onclick=\"document.querySelectorAll('.{gid}').forEach(e=>e.style.display=e.style.display==='none'?'table-row':'none')\" "
        f"style=\"cursor:pointer; color: #1a73e8;\">{_esc(cg.branch_label)} {_esc(cg.header[:theme.HEADER_MAX_LEN])}</span></td>"
        f"<td style=\"padding: 4px; font-size: 10px; color: #888;\">-</td>"
        f"<td style=\"padding: 4px;\">-</td>"
        f"</tr>"
    )
    rows = "".join(
        _wrap_hidden(_row_html(r), gid) for r in cg.rows
    )
    return header_html + rows


def _control_group_single_html(cgs: ControlGroupSingle) -> str:
    return _row_html(cgs.row)


def _control_body_html(cb: ControlBody) -> str:
    if not cb.body_stmts:
        return ""
    inner = "\n".join(_esc(s) for s in cb.body_stmts)
    return (
        f"<tr><td colspan='4' style='padding:4px;'>"
        f"<details><summary style='cursor:pointer; color:#666; font-size:10px;'>body</summary>"
        f"<pre style='font-family:{theme.FONT_MONO}; font-size:11px; color:#333;'>{inner}</pre>"
        f"</details></td></tr>"
    )


def _skipped_bucket_html(sb: SkippedBucket) -> str:
    if not sb.items:
        return ""
    gid = _uid("skip")
    saved = f"Saved {sb.total_saved_time_s:.2f}s" if sb.total_saved_time_s > theme.MIN_TIME_DISPLAY_S else "—"
    header = (
        f"<tr style=\"border-bottom: 1px solid #eee;\">"
        f"<td style=\"padding: 4px; color: {theme._COLOR_SKIPPED};\">⏩</td>"
        f"<td style=\"padding: 4px; font-family: {theme.FONT_MONO}; font-size: 11px;\">"
        f"<span onclick=\"document.querySelectorAll('.{gid}').forEach(e=>e.style.display=e.style.display==='none'?'table-row':'none')\" "
        f"style=\"cursor:pointer; color: #666; font-style: italic;\">"
        f"{len(sb.items)} intermediate dependency step{'s' if len(sb.items) != 1 else ''}</span></td>"
        f"<td style=\"padding: 4px;\">-</td>"
        f"<td style=\"padding: 4px;\">{saved}</td>"
        f"</tr>"
    )
    body = "".join(_wrap_hidden(_render_section_item(i), gid) for i in sb.items)
    return header + body


# ---------------------------------------------------------------------------
# Decorator / overhead
# ---------------------------------------------------------------------------

def _decorator_call_html(c: DecoratorCall) -> str:
    icon, color, label = theme.row_visual(c.status.value)
    status_text = "HIT" if c.status is BadgeStatus.RESTORED else "MISS"
    return (
        f"<tr style=\"border-bottom: 1px solid #eee;\">"
        f"<td style=\"padding: 4px; color: {color};\">{icon} @cache</td>"
        f"<td style=\"padding: 4px; font-family: {theme.FONT_MONO};\">"
        f"<span style=\"color: {theme._COLOR_DECORATOR};\">{_esc(c.func_name)}()</span> "
        f"<span style=\"color: #888; font-size: 10px;\">{status_text}</span></td>"
        f"<td style=\"padding: 4px; font-size: 10px; color: #555;\">-</td>"
        f"<td style=\"padding: 4px;\">{c.time_s:.3f}s</td>"
        f"</tr>"
    )


def _decorator_group_html(g: DecoratorCallGroup) -> str:
    if not g.condensed:
        return "".join(_decorator_call_html(c) for c in g.calls)
    n = len(g.calls)
    hits = sum(1 for c in g.calls if c.status is BadgeStatus.RESTORED)
    misses = n - hits
    total_time = sum(c.time_s for c in g.calls)
    gid = _uid("dec")
    summary = (
        f"<tr style=\"border-bottom: 1px solid #eee;\">"
        f"<td style=\"padding: 4px; color: {theme._COLOR_DECORATOR};\">⚡ @cache</td>"
        f"<td style=\"padding: 4px; font-family: {theme.FONT_MONO};\">"
        f"<span onclick=\"document.querySelectorAll('.{gid}').forEach(e=>e.style.display=e.style.display==='none'?'table-row':'none')\" "
        f"style=\"cursor:pointer; color: {theme._COLOR_DECORATOR};\">{_esc(g.func_name)}() "
        f"<span style='color:#888;'>×{n} ({hits} hit, {misses} miss)</span></span></td>"
        f"<td style=\"padding: 4px; font-size: 10px; color: #555;\">-</td>"
        f"<td style=\"padding: 4px;\">{total_time:.3f}s</td>"
        f"</tr>"
    )
    body = "".join(_wrap_hidden(_decorator_call_html(c), gid) for c in g.calls)
    return summary + body


def _overhead_html(ob: OverheadBreakdown) -> str:
    rows = []
    for e in ob.entries:
        rows.append(
            f"<tr style=\"border-bottom: 1px solid #eee;\">"
            f"<td style=\"padding: 4px; color:#888;\">{e.label}</td>"
            f"<td style=\"padding: 4px; font-family: {theme.FONT_MONO}; color: #999; font-size: 10px;\">overhead</td>"
            f"<td style=\"padding: 4px; font-size: 10px; color: #666;\">-</td>"
            f"<td style=\"padding: 4px;\">{e.time_s:.3f}s</td>"
            f"</tr>"
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Section / dispatch
# ---------------------------------------------------------------------------

def _section_header_html(section: Section) -> str:
    if not section.header:
        return ""
    return (
        f"<tr><td colspan='4' style='background:#f5f5f5; font-weight:bold; "
        f"font-size:10px; padding:4px; color:#666; border-bottom:1px solid #eee;'>"
        f"{_esc(section.header)}</td></tr>"
    )


def _render_section_item(item: SectionItem) -> str:
    if isinstance(item, StatementRow):
        return _row_html(item)
    if isinstance(item, ForLoopGroup):
        return _for_loop_group_html(item)
    if isinstance(item, ControlGroup):
        return _control_group_html(item)
    if isinstance(item, ControlGroupSingle):
        return _control_group_single_html(item)
    if isinstance(item, ControlBody):
        return _control_body_html(item)
    if isinstance(item, SkippedBucket):
        return _skipped_bucket_html(item)
    if isinstance(item, DecoratorCallGroup):
        return _decorator_group_html(item)
    if isinstance(item, OverheadBreakdown):
        return _overhead_html(item)
    raise TypeError(f"Unsupported BadgeView node: {type(item).__name__}")


def _section_html(section: Section) -> str:
    return _section_header_html(section) + "".join(
        _render_section_item(i) for i in section.items
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _summary_parts(badge: InteractiveBadge) -> tuple[str, str, str, str, str, str]:
    """``(bg, color, border, icon, label, subtext)`` for the collapsed pill."""
    h = badge.header
    if h.status is BadgeStatus.WARNING and h.total_steps == 0 and h.current_step == 0 and h.current_code is None:
        # Fall through to default; otherwise WARNING is also the RUNNING placeholder.
        pass
    if h.current_step or h.total_steps or h.current_code:
        bg = theme.SUMMARY_BG_RUNNING
        color = theme.SUMMARY_COLOR_RUNNING
        border = theme.SUMMARY_BORDER_RUNNING
        icon = "⏳"
        if h.total_steps:
            label = f"Processing ({h.current_step}/{h.total_steps})"
        elif h.current_step:
            label = f"Processing (step {h.current_step})"
        else:
            label = "Processing..."
        subtext = ""
        if h.current_code:
            preview = h.current_code.splitlines()[0][:60]
            if len(h.current_code) > 60:
                preview += "..."
            subtext = f"- {_esc(preview)}"
        return bg, color, border, icon, label, subtext

    if h.computed_count == 0 and (h.restored_count > 0 or h.skipped_count > 0):
        bg = theme.SUMMARY_BG_CACHED
        color = theme.BADGE_COLOR_RESTORED
        border = theme.BADGE_COLOR_RESTORED
        if h.restored_count > 0:
            icon, label = "⚡", "CACHED"
            subtext = f"(Saved {h.total_saved_s:.2f}s)" if h.total_saved_s > theme.MIN_TIME_DISPLAY_S else ""
        else:
            icon, label = "⏩", "SKIPPED"
            subtext = "(Already computed)"
    else:
        bg = theme.SUMMARY_BG_EXECUTED
        color = theme.BADGE_COLOR_DEFAULT
        border = theme.BADGE_COLOR_DEFAULT
        icon, label = "⚙️", "EXECUTED"
        if h.total_saved_s > 0:
            subtext = f"({h.total_exec_s:.2f}s, Saved {h.total_saved_s:.2f}s)"
        else:
            subtext = f"({h.total_exec_s:.2f}s)"
    return bg, color, border, icon, label, subtext


def _footer_html(footer: BugReportLink) -> str:
    return (
        f"<div style='margin-top:8px; padding-top:6px; border-top:1px solid #f0f0f0; text-align:right;'>"
        f"<a href=\"{footer.url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
        f"style=\"display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; "
        f"background: #fff5f5; border: 1px solid #f5c6c6; border-radius: 12px; "
        f"color: #c0392b; font-size: 11px; font-weight: 500; text-decoration: none;\" "
        f"title=\"Open a pre-filled GitHub issue to report incorrect caching behaviour\">"
        f"🐛 Report incorrect caching</a></div>"
    )


def render_html(badge: InteractiveBadge) -> str:
    """Render an :class:`InteractiveBadge` to HTML."""
    _reset_ids()
    bg, color, border, icon, label, subtext = _summary_parts(badge)
    body_rows = "".join(_section_html(s) for s in badge.sections)
    footer = _footer_html(badge.footer) if badge.footer is not None else ""
    return f"""
    <details style="display: inline-block; border: 1px solid {border}; border-radius: 4px;
        background-color: {bg}; padding: 0; margin-top: 5px;
        font-family: {theme.FONT_SANS}; font-size: 12px; color: {color}; max-width: 100%;">
        <summary style="cursor: pointer; padding: 4px 8px; font-weight: bold;
            list-style: none; outline: none; display: inline-flex;
            align-items: baseline; gap: 6px;">
            <span>{icon} {label}</span>
            <span style="font-weight: normal; opacity: 0.8; font-family: {theme.FONT_MONO};">{subtext}</span>
        </summary>
        <div style="padding: 5px 10px; background-color: #ffffff;
            border-top: 1px solid {border}30; color: #333;
            max-height: 500px; overflow-y: auto;">
            <table style="width: 100%; min-width: 500px; border-collapse: collapse;
                font-size: 11px; table-layout: auto; color: #333; background-color: #ffffff;">
                <thead><tr style="border-bottom: 2px solid #ddd; color: #666;">
                    <th style="text-align: left; padding: 4px;">TYPE</th>
                    <th style="text-align: left; padding: 4px;">CONTENT</th>
                    <th style="text-align: left; padding: 4px;">STORAGE</th>
                    <th style="text-align: left; padding: 4px;">TIME</th>
                </tr></thead>
                <tbody>{body_rows}</tbody>
            </table>
            {footer}
        </div>
    </details>
    """


def render_status_badge_html(badge: StatusBadge) -> str:
    """Render the compact, non-interactive status pill."""
    color = (theme.BADGE_COLOR_RESTORED
             if badge.status is BadgeStatus.RESTORED
             else theme.BADGE_COLOR_DEFAULT)
    icon, _row_color, label = theme.row_visual(badge.status.value)
    storage_str = "+".join(badge.storage_tiers) if badge.storage_tiers else ""
    saved_str = f" (Saved {badge.time_saved_s:.2f}s)" if badge.time_saved_s > 0 else ""
    source_str = f" ← {badge.source}" if badge.source else ""
    return (
        f"<span style=\"display: inline-block; padding: 2px 8px; border-radius: 4px; "
        f"background-color: {color}1A; color: {color}; font-family: {theme.FONT_MONO}; "
        f"font-size: 11px; font-weight: 500;\">"
        f"{icon} {label} {badge.execution_time_s:.3f}s{saved_str}{source_str} "
        f"{storage_str}</span>"
    )
