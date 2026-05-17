"""Loop, control structure, and decorator HTML rendering components."""

from __future__ import annotations

from typing import Any

from cash.notebook.cache_status import CacheStatus

from ._grouping import (
    _html_escape,
    _unique_id,
    _unique_ordered,
    format_loop_var,
)
from ._types import (
    ControlGroup,
    ControlGroupSingle,
    ForLoopGroup,
    LoopGroup,
)
from .theme import (
    CODE_SNIPPET_MAX_LEN as _CODE_SNIPPET_MAX_LEN,
    FONT_MONO as _FONT_MONO,
    HEADER_MAX_LEN as _HEADER_MAX_LEN,
    MIN_TIME_DISPLAY_S as _MIN_TIME_DISPLAY_S,
)

# ---------------------------------------------------------------------------
# Control body HTML helper
# ---------------------------------------------------------------------------

def render_control_body_html(body_stmts: list[str]) -> str:
    """Render control structure body statements as an expandable HTML snippet.

    Returns an HTML ``<details>`` element showing the control structure header
    (first line) as the summary and individual body statements inside.
    """
    if not body_stmts:
        return ""

    header = body_stmts[0][:_HEADER_MAX_LEN]
    if len(body_stmts[0]) > _HEADER_MAX_LEN:
        header += "\u2026"
    # Escape HTML in the header
    header = header.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    inner_rows = ""
    for stmt in body_stmts[1:]:
        # Escape basic HTML characters for safe display
        safe = stmt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        inner_rows += f'<div style="padding: 1px 0; color: #555; white-space: pre;">{safe}</div>'

    return (
        f'<details style="cursor: pointer;">'
        f'<summary style="outline: none; color: #333;">{header}</summary>'
        f'<div style="margin-top: 2px; font-size: 10px; border-left: 2px solid #ddd; padding-left: 6px;">'
        f'{inner_rows}'
        f'</div></details>'
    )

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_toggle_js(gid: str, closed_arrow: str, open_arrow: str) -> str:
    """Build the JS onclick handler for expand/collapse."""
    return (
        f"event.stopPropagation();"
        f"(function(){{"
        f"var arrow=document.getElementById('{gid}_arrow');"
        f"var expanding=arrow&&arrow.textContent==='{closed_arrow}';"
        f"if(expanding){{"
        f"document.querySelectorAll('.{gid}_d').forEach(function(r){{r.style.display='table-row'}});"
        f"if(arrow)arrow.textContent='{open_arrow}'"
        f"}}else{{"
        f"document.querySelectorAll('.{gid}_a').forEach(function(r){{r.style.display='none'}});"
        f"if(arrow)arrow.textContent='{closed_arrow}'"
        f"}}"
        f"}})()"
    )

def _build_loop_label(
    group: dict[str, Any],
    loop_var_name: Any,
    loop_var_values: list,
) -> str:
    """Build the display label for a for-loop summary row."""
    all_var_names = group.get('all_loop_var_names', [loop_var_name] if loop_var_name else [])
    all_var_values = group.get('all_loop_var_values', {})

    if all_var_names and all_var_values:
        if len(all_var_names) == 1 and loop_var_name:
            vals = all_var_values.get(loop_var_name, loop_var_values)
            unique_vals = _unique_ordered([
                format_loop_var(v) if not isinstance(v, str) else v for v in vals
            ])
            if len(unique_vals) <= 5:
                vals_str = ", ".join(unique_vals)
                return f"🔁 <b>{loop_var_name}</b> ∈ [{vals_str}]"
            first_3 = ", ".join(unique_vals[:3])
            last_1 = unique_vals[-1]
            return f"🔁 <b>{loop_var_name}</b> ∈ [{first_3}, …, {last_1}]"
        parts = []
        for vn in all_var_names:
            vals = all_var_values.get(vn, [])
            unique_v = _unique_ordered([
                format_loop_var(v) if not isinstance(v, str) else v for v in vals
            ])
            if len(unique_v) <= 4:
                parts.append(f"<b>{vn}</b>∈[{', '.join(unique_v)}]")
            else:
                parts.append(f"<b>{vn}</b>∈[{unique_v[0]}, …, {unique_v[-1]}]")
        return f"🔁 {', '.join(parts)}"
    if loop_var_name and loop_var_values:
        unique_vals = _unique_ordered([
            format_loop_var(v) if not isinstance(v, str) else v for v in loop_var_values
        ])
        if len(unique_vals) <= 5:
            vals_str = ", ".join(unique_vals)
            return f"🔁 <b>{loop_var_name}</b> ∈ [{vals_str}]"
        first_3 = ", ".join(unique_vals[:3])
        last_1 = unique_vals[-1]
        return f"🔁 <b>{loop_var_name}</b> ∈ [{first_3}, …, {last_1}]"
    total_iters = max((len(sg['metrics']) for sg in group['stmt_groups']), default=0)
    return f"🔁 {total_iters} iterations"

def _aggregate_storage(
    all_metrics: list,
    cached_iters: int,
    computed_iters: int,
) -> str:
    """Compute aggregate storage display string for loop groups."""
    storage_parts: set = set()
    for m in all_metrics:
        for s in (m.get('storage', []) or []):
            storage_parts.add(s)
        if m.get('source'):
            storage_parts.add(m['source'])
    if cached_iters > 0 and computed_iters > 0:
        storage_display = "RAM" if 'RAM' in storage_parts else ("DISK" if 'DISK' in storage_parts else "-")
        return f"↔ {storage_display}"
    if cached_iters > 0:
        src = 'RAM' if 'RAM' in storage_parts else ('DISK' if 'DISK' in storage_parts else 'Cache')
        return f"← {src}"
    if storage_parts:
        ss = "+".join(sorted(storage_parts))
        return f"→ {ss}"
    return "-"

def _split_outer_inner_sgs(
    stmt_groups: list,
    loop_var_name: Any,
) -> tuple:
    """Split stmt_groups into outer-loop and inner-loop groups."""
    if not loop_var_name:
        return list(stmt_groups), []

    var_counts = []
    for sg in stmt_groups:
        sample_m = sg['metrics'][0] if sg['metrics'] else {}
        var_counts.append(len(sample_m.get('loop_vars', {})))
    outer_var_count = min(var_counts) if var_counts else 1

    outer_sgs = []
    inner_sgs = []
    for sg in stmt_groups:
        sample_m = sg['metrics'][0] if sg['metrics'] else {}
        n_vars = len(sample_m.get('loop_vars', {}))
        if n_vars > outer_var_count:
            inner_sgs.append(sg)
        else:
            outer_sgs.append(sg)
    return outer_sgs, inner_sgs

def _build_inner_loop_group(inner_sgs: list, outer_var_name: Any) -> dict[str, Any]:
    """Build a for_loop_group dict from inner-loop statement groups."""
    inner_var_name = None
    inner_all_var_values: dict[str, list] = {}
    inner_all_var_names: list = []
    for sg in inner_sgs:
        for m in sg['metrics']:
            lvars = m.get('loop_vars', {})
            for k, v in lvars.items():
                if k != outer_var_name:
                    if k not in inner_all_var_values:
                        inner_all_var_names.append(k)
                        inner_all_var_values[k] = []
                    inner_all_var_values[k].append(v)
                    if inner_var_name is None:
                        inner_var_name = k
    inner_var_values = inner_all_var_values.get(inner_var_name, []) if inner_var_name else []
    return {
        'type': 'for_loop_group',
        'stmt_groups': inner_sgs,
        'loop_var_name': inner_var_name,
        'loop_var_values': inner_var_values,
        'all_loop_var_names': inner_all_var_names,
        'all_loop_var_values': inner_all_var_values,
    }

def _stmt_storage(metrics: list, computed: int, cached: int) -> str:
    """Compute storage display for a single statement in a loop."""
    st_parts: set = set()
    for m in metrics:
        for s in (m.get('storage', []) or []):
            st_parts.add(s)
        if m.get('source'):
            st_parts.add(m['source'])
    if computed == 0 and cached > 0:
        src = 'RAM' if 'RAM' in st_parts else ('DISK' if 'DISK' in st_parts else 'Cache')
        return f"← {src}"
    if computed > 0 and st_parts:
        return f"→ {'+'.join(sorted(st_parts))}"
    if st_parts:
        return f"← {'+'.join(sorted(st_parts))}"
    return "-"

def _render_iteration_row(
    m: dict[str, Any],
    idx: int,
    iter_direct_class: str,
    iter_pad_left: int,
    loop_var_name: Any,
    loop_var_values: list,
) -> str:
    """Render a single iteration detail row."""
    ms = m.get('status', '?')
    mt = m.get('total_time', 0.0)
    if ms == CacheStatus.RESTORED:
        mi, mc, ml = "⚡", "#006644", "cached"
    elif ms == CacheStatus.SKIPPED:
        mi, mc, ml = "⏩", "#006644", "skipped"
    elif ms == CacheStatus.COMPUTED:
        mi, mc, ml = "⚙️", "#996300", "computed"
    else:
        mi, mc, ml = "❓", "#999", ms

    loop_vars = m.get('loop_vars', {})
    if loop_vars:
        var_strs = [f"{k}={format_loop_var(v)}" for k, v in loop_vars.items()]
        iter_label = ", ".join(var_strs)
    elif loop_var_name and idx < len(loop_var_values):
        iter_label = f"{loop_var_name}={loop_var_values[idx]}"
    else:
        iter_label = f"iter {idx + 1}"

    ms_time = f"{mt:.3f}s"
    ms_saved = m.get('saved_time', 0.0)
    if ms_saved > _MIN_TIME_DISPLAY_S:
        ms_time += f" (↑{ms_saved:.1f}s)"

    iter_storage_val = m.get('storage', [])
    iter_source = m.get('source', '')
    if ms == CacheStatus.RESTORED:
        iter_stor = f"← {iter_source or 'Cache'}"
    elif ms == CacheStatus.SKIPPED:
        iter_stor = "✓ RAM"
    elif iter_storage_val:
        ss = "+".join(iter_storage_val) if isinstance(iter_storage_val, list) else str(iter_storage_val)
        iter_stor = f"→ {ss}"
    else:
        iter_stor = "-"

    return f"""
    <tr class="{iter_direct_class}" style="display: none; border-bottom: 1px solid #f5f5f5;">
        <td style="padding: 2px 4px 2px {iter_pad_left}px; text-align: left; color: {mc}; white-space: nowrap; border-left: 2px solid #eee;">{mi}</td>
        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #555;">{ml} — {iter_label}</td>
        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #999;">{iter_stor}</td>
        <td style="padding: 2px 4px; text-align: left; white-space: nowrap; font-size: 10px; color: #555;">{ms_time}</td>
    </tr>"""


# ---------------------------------------------------------------------------
# For-loop rendering
# ---------------------------------------------------------------------------

def render_for_loop_group(
    group: ForLoopGroup,
    is_upstream: bool = False,
    indent_level: int = 0,
    ancestor_classes: str = "",
) -> str:
    """Render a for-loop group as flat rows in the main table.

    Returns a summary row (with JS-toggled expand/collapse) followed by
    hidden detail rows.  All rows use the same 4 columns as the main
    badge table (TYPE / CONTENT / STORAGE / TIME).
    """
    stmt_groups = group['stmt_groups']
    loop_var_name = group.get('loop_var_name')
    loop_var_values = group.get('loop_var_values', [])

    all_metrics = [m for sg in stmt_groups for m in sg['metrics']]
    total_iters = max((len(sg['metrics']) for sg in stmt_groups), default=0)

    # Per-iteration cached/computed
    iter_status: dict[int, bool] = {}
    for sg in stmt_groups:
        for idx, m in enumerate(sg['metrics']):
            if idx not in iter_status:
                iter_status[idx] = True
            if m.get('status') == CacheStatus.COMPUTED:
                iter_status[idx] = False
    cached_iters = sum(1 for v in iter_status.values() if v)
    computed_iters = sum(1 for v in iter_status.values() if not v)

    total_time = sum(m.get('total_time', 0.0) for m in all_metrics)
    saved_time = sum(m.get('saved_time', 0.0) for m in all_metrics)

    if computed_iters == 0:
        icon, color = "⚡", "#006644"
        status_text = f"All {total_iters} cached"
    elif cached_iters == 0:
        icon, color = "⚙️", "#996300"
        status_text = f"All {total_iters} computed"
    else:
        icon, color = "🔄", "#336699"
        status_text = f"{cached_iters} cached, {computed_iters} computed"

    time_str = f"{total_time:.2f}s"
    if saved_time > _MIN_TIME_DISPLAY_S:
        time_str += f" (Saved {saved_time:.2f}s)"

    loop_label = _build_loop_label(group, loop_var_name, loop_var_values)

    if is_upstream:
        type_prefix = "⬆️"
        if computed_iters == 0:
            type_label = "Restored"
        else:
            type_label = "Auto-exec"
    else:
        type_prefix = icon
        type_label = "Loop"

    storage_display = _aggregate_storage(all_metrics, cached_iters, computed_iters)

    gid = _unique_id("flg")

    # Separate outer vs inner loop statement groups
    outer_sgs, inner_sgs = _split_outer_inner_sgs(stmt_groups, loop_var_name)

    display_n_stmts = len(outer_sgs) + (1 if inner_sgs else 0)
    content_summary = (
        f"{loop_label} <span style='color:#888;'>"
        f"({status_text}, {display_n_stmts} stmt{'s' if display_n_stmts != 1 else ''})"
        f"</span>"
    )

    pad_left = 4 + indent_level * 20
    border_style = "border-left: 2px solid #ddd;" if indent_level > 0 else ""
    bg_color = "#f8f8f8" if indent_level > 0 else ""
    arrow_open = "▼" if indent_level == 0 else "▾"
    arrow_closed = "▶" if indent_level == 0 else "▸"
    arrow_size = "10px" if indent_level == 0 else "9px"
    arrow_color = "#999" if indent_level == 0 else "#bbb"
    row_display = "display: none;" if indent_level > 0 else ""

    row_class = ancestor_classes
    toggle_js = _build_toggle_js(gid, arrow_closed, arrow_open)
    bg_style = f"background-color: {bg_color};" if bg_color else ""

    html = f"""
    <tr class="{row_class}" style="{row_display} border-bottom: 1px solid #eee; cursor: pointer; {bg_style}" onclick="{toggle_js}">
        <td style="padding: 4px 4px 4px {pad_left}px; text-align: left; color: {color}; white-space: nowrap; {border_style}">{type_prefix} {type_label}</td>
        <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px; color: #333;"><span id="{gid}_arrow" style="color:{arrow_color}; font-size:{arrow_size};">{arrow_closed}</span> {content_summary}</td>
        <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">{storage_display}</td>
        <td style="padding: 4px; text-align: left; white-space: nowrap; color: #333;">{time_str}</td>
    </tr>
    """

    ancestor_a_only = " ".join(c for c in ancestor_classes.split() if c.endswith('_a'))
    child_ancestor_a = f"{ancestor_a_only} {gid}_a".strip()
    direct_child_class = f"{gid}_d {child_ancestor_a}"

    for sg in outer_sgs:
        html += render_loop_stmt_row(
            sg,
            parent_direct_class=direct_child_class,
            ancestor_classes=child_ancestor_a,
            indent_level=indent_level + 1,
        )

    if inner_sgs:
        inner_group = _build_inner_loop_group(inner_sgs, loop_var_name)
        html += render_for_loop_group(
            inner_group,
            is_upstream=is_upstream,
            indent_level=indent_level + 1,
            ancestor_classes=direct_child_class,
        )

    return html

def render_loop_stmt_row(
    sg: LoopGroup,
    parent_direct_class: str = "",
    ancestor_classes: str = "",
    indent_level: int = 1,
) -> str:
    """Render one statement inside a for-loop group as flat rows.

    Produces a statement summary row and per-iteration detail rows,
    all hidden by default.
    """
    metrics = sg['metrics']
    base_code = sg['base_code'].strip()
    code_preview = base_code.split('\n')[0][:_CODE_SNIPPET_MAX_LEN]
    if len(base_code) > _CODE_SNIPPET_MAX_LEN:
        code_preview += "…"
    code_preview = _html_escape(code_preview)

    total_iters = len(metrics)
    cached = sum(1 for m in metrics if m.get('status') in (CacheStatus.RESTORED, CacheStatus.SKIPPED))
    computed = sum(1 for m in metrics if m.get('status') == CacheStatus.COMPUTED)
    total_time = sum(m.get('total_time', 0.0) for m in metrics)
    saved_time = sum(m.get('saved_time', 0.0) for m in metrics)

    if computed == 0:
        s_icon, s_color = "⚡", "#006644"
        s_text = "all cached"
    elif cached == 0:
        s_icon, s_color = "⚙️", "#996300"
        s_text = "all computed"
    else:
        s_icon, s_color = "🔄", "#336699"
        s_text = f"{cached} cached, {computed} computed"

    time_cell = f"{total_time:.2f}s"
    if saved_time > _MIN_TIME_DISPLAY_S:
        time_cell += f" (↑{saved_time:.1f}s)"

    st_storage = _stmt_storage(metrics, computed, cached)

    sid = _unique_id("ssg")
    pad_left = 4 + indent_level * 20

    ancestor_a_only = " ".join(c for c in ancestor_classes.split() if c.endswith('_a'))
    child_ancestor_a = f"{ancestor_a_only} {sid}_a".strip()
    iter_direct_class = f"{sid}_d {child_ancestor_a}"

    toggle_js = _build_toggle_js(sid, "▸", "▾")

    html = f"""
    <tr class="{parent_direct_class}" style="display: none; border-bottom: 1px solid #f0f0f0; cursor: pointer;" onclick="{toggle_js}">
        <td style="padding: 4px 4px 4px {pad_left}px; text-align: left; color: {s_color}; white-space: nowrap; border-left: 2px solid #ddd;">{s_icon}</td>
        <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span id="{sid}_arrow" style="color:#bbb; font-size:9px;">▸</span> <span style="color: #666; font-size: 10px;">{code_preview}</span> <span style="color:#999; font-style: italic;">({s_text})</span></td>
        <td style="padding: 4px; text-align: left; font-size: 10px; color: #888;">{st_storage}</td>
        <td style="padding: 4px; text-align: left; white-space: nowrap;">{time_cell}</td>
    </tr>
    """

    iter_pad_left = pad_left + 20
    max_visible = 5
    show_all = total_iters <= max_visible * 2
    loop_var_name = sg.get('loop_var_name')
    loop_var_values = sg.get('loop_var_values', [])

    for idx, m in enumerate(metrics):
        if not show_all and max_visible <= idx < total_iters - max_visible:
            if idx == max_visible:
                hidden = total_iters - max_visible * 2
                html += f"""
                <tr class="{iter_direct_class}" style="display: none;">
                    <td style="padding: 2px 4px 2px {iter_pad_left}px; border-left: 2px solid #eee;"></td>
                    <td colspan="3" style="padding: 2px 4px; text-align: center; color: #999; font-size: 10px;">… {hidden} more iterations …</td>
                </tr>"""
            continue

        html += _render_iteration_row(
            m, idx, iter_direct_class, iter_pad_left,
            loop_var_name, loop_var_values,
        )

    return html


# ---------------------------------------------------------------------------
# Control structure rendering
# ---------------------------------------------------------------------------

def render_control_group(item: ControlGroup, is_upstream: bool = False) -> str:
    """Render a control structure (if/elif/else) as flat rows in the main table.

    Returns a summary row followed by hidden detail rows (one per body
    statement).  All rows use the same 4 columns as the main badge table.
    """
    metrics = item['metrics']
    branch_label = item.get('branch_label', '')
    header = item.get('header', branch_label)

    total_time = sum(m.get('total_time', 0.0) for m in metrics)
    saved_time = sum(m.get('saved_time', 0.0) for m in metrics)
    cached = sum(1 for m in metrics if m.get('status') in (CacheStatus.RESTORED, CacheStatus.SKIPPED))
    computed = sum(1 for m in metrics if m.get('status') == CacheStatus.COMPUTED)

    if computed == 0 and cached > 0:
        icon, color, label = ("⚡", "#006644", "Cached") if not is_upstream else ("⬆️", "#006644", "Restored")
    elif cached == 0:
        icon, color, label = ("⚙️", "#996300", "Executed") if not is_upstream else ("⬆️", "#996300", "Upstream")
    else:
        icon, color, label = ("🔄", "#336699", "Mixed") if not is_upstream else ("⬆️", "#336699", "Upstream")

    time_str = f"{total_time:.2f}s"
    if saved_time > _MIN_TIME_DISPLAY_S:
        time_str += f" (Saved {saved_time:.2f}s)"

    ctrl_storage_display = _control_storage(metrics, computed, cached)

    all_vars: list = []
    for m in metrics:
        for v in (
            m.get('restored_vars', [])
            or m.get('evaluated_vars', [])
            or [o for o in (m.get('output_vars', []) or []) if isinstance(o, str)]
        ):
            if v not in all_vars:
                all_vars.append(v)
    vars_display = ", ".join(all_vars) if all_vars else ""

    header_display = header[:_CODE_SNIPPET_MAX_LEN]
    if len(header) > _CODE_SNIPPET_MAX_LEN:
        header_display += "…"
    header_display = _html_escape(header_display)

    n_stmts = len(metrics)
    summary_label = f"{n_stmts} stmt{'s' if n_stmts != 1 else ''}"
    safe_branch = _html_escape(branch_label)

    gid = _unique_id("cg")

    content_summary = ""
    if vars_display:
        content_summary += f"{vars_display} — "
    content_summary += f"<span style='color:#666; font-size: 10px;'>{header_display} ({summary_label})</span>"

    toggle_js = _build_toggle_js(gid, "▶", "▼")

    html = f"""
    <tr style="border-bottom: 1px solid #eee; cursor: pointer;" onclick="{toggle_js}">
        <td style="padding: 4px; text-align: left; color: {color}; white-space: nowrap;">{icon} {label}</td>
        <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span id="{gid}_arrow" style="color:#999; font-size:10px;">▶</span> {content_summary}</td>
        <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">{ctrl_storage_display}</td>
        <td style="padding: 4px; text-align: left; white-space: nowrap;">{time_str}</td>
    </tr>
    """

    child_class = f"{gid}_d {gid}_a"

    html += f"""
    <tr class="{child_class}" style="display: none;">
        <td style="padding: 2px 4px 2px 24px; border-left: 2px solid #ddd;"></td>
        <td colspan="3" style="padding: 2px 4px; font-weight: bold; font-size: 11px;"><span style="color: #666; font-size: 10px;">{safe_branch}:</span></td>
    </tr>"""

    for m in metrics:
        html += _render_control_stmt_row(m, child_class)

    return html

def _ctrl_single_icon_color_label(status: str, is_upstream: bool) -> tuple[str, str, str]:
    """Return ``(icon, color, label)`` for a single control structure row."""
    if is_upstream:
        if status == CacheStatus.RESTORED:
            return "⬆️", "#006644", "Restored"
        return "⬆️", "#555", "Upstream"
    if status == CacheStatus.RESTORED:
        return "⚡", "#006644", "Restored"
    if status == CacheStatus.COMPUTED:
        return "⚙️", "#996300", "Executed"
    if status == CacheStatus.SKIPPED:
        return "⏩", "#006644", "Skipped"
    return "❓", "#999", status


def _ctrl_single_body_rows_html(
    body_stmts: list[str],
    child_class: str,
    status: str,
    storage_display: str,
) -> str:
    """Return HTML for the body rows of a single control structure."""
    html = ""
    for stmt in body_stmts:
        safe = _html_escape(stmt)
        stripped = stmt.strip()
        is_header = any(
            stripped.startswith(kw)
            for kw in ('if ', 'elif ', 'else:', 'while ', 'for ',
                       'try:', 'except', 'finally:', 'with ')
        )
        if is_header:
            html += f"""
            <tr class="{child_class}" style="display: none;">
                <td style="padding: 2px 4px 2px 24px; border-left: 2px solid #ddd;"></td>
                <td colspan="3" style="padding: 2px 4px; font-weight: bold; font-size: 11px;"><span style="color: #666; font-size: 10px;">{safe}</span></td>
            </tr>"""
        else:
            if status == CacheStatus.RESTORED:
                si, sc = "⚡", "#006644"
            elif status == CacheStatus.SKIPPED:
                si, sc = "⏩", "#006644"
            elif status == CacheStatus.COMPUTED:
                si, sc = "⚙️", "#996300"
            else:
                si, sc = "❓", "#999"

            html += f"""
            <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                <td style="padding: 2px 4px 2px 24px; text-align: left; color: {sc}; white-space: nowrap; border-left: 2px solid #ddd;">{si}</td>
                <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span style="color: #666; font-size: 10px;">{safe}</span></td>
                <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">{storage_display}</td>
                <td style="padding: 2px 4px; text-align: left; white-space: nowrap;"></td>
            </tr>"""
    return html


def render_control_group_single(item: ControlGroupSingle, is_upstream: bool = False) -> str:
    """Render a single-unit control structure (while/with/try) as flat rows.

    Returns a summary row followed by hidden detail rows for the body
    statements, all using the same 4-column layout.
    """
    metric = item['metric']
    status = metric.get('status', 'UNKNOWN')
    body_stmts = metric.get('body_statements', [])
    control_type = metric.get('control_type', 'block')
    exec_time = metric.get('total_time', 0.0)
    saved_time = metric.get('saved_time', 0.0)
    storage_val = metric.get('storage', [])

    icon, color, label = _ctrl_single_icon_color_label(status, is_upstream)

    restored_vars = metric.get('restored_vars', [])
    eval_vars = metric.get('evaluated_vars', [])
    out_vars = metric.get('output_vars', [])
    vars_list = restored_vars or eval_vars or [o for o in (out_vars or []) if isinstance(o, str)]
    vars_display = ", ".join(vars_list) if vars_list else ""

    time_str = f"{exec_time:.2f}s"
    if saved_time > _MIN_TIME_DISPLAY_S:
        time_str = f"Saved {saved_time:.2f}s" if status == CacheStatus.RESTORED else time_str

    storage_display = _single_control_storage(metric, status, is_upstream, storage_val)

    header_line = body_stmts[0] if body_stmts else control_type
    header_display = header_line[:_CODE_SNIPPET_MAX_LEN]
    if len(header_line) > _CODE_SNIPPET_MAX_LEN:
        header_display += "…"
    header_display = _html_escape(header_display)

    n_stmts = sum(
        1 for s in body_stmts
        if not any(
            s.strip().startswith(kw)
            for kw in ('if ', 'elif ', 'else:', 'while ', 'for ',
                       'try:', 'except', 'finally:', 'with ')
        )
    )
    summary_label = f"{n_stmts} stmt{'s' if n_stmts != 1 else ''}"

    gid = _unique_id("cgs")

    content_summary = ""
    if vars_display:
        content_summary += f"{vars_display} — "
    content_summary += f"<span style='color:#666; font-size: 10px;'>{header_display} ({summary_label})</span>"

    toggle_js = _build_toggle_js(gid, "▶", "▼")

    html = f"""
    <tr style="border-bottom: 1px solid #eee; cursor: pointer;" onclick="{toggle_js}">
        <td style="padding: 4px; text-align: left; color: {color}; white-space: nowrap;">{icon} {label}</td>
        <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span id="{gid}_arrow" style="color:#999; font-size:10px;">▶</span> {content_summary}</td>
        <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">{storage_display}</td>
        <td style="padding: 4px; text-align: left; white-space: nowrap;">{time_str}</td>
    </tr>
    """

    child_class = f"{gid}_d {gid}_a"

    html += _ctrl_single_body_rows_html(body_stmts, child_class, status, storage_display)

    return html


# ---------------------------------------------------------------------------
# Decorator rendering
# ---------------------------------------------------------------------------

def render_decorator_calls(calls: list[dict[str, Any]]) -> str:
    """Render decorator cache call events as badge rows.

    Groups calls by function name.  If a single function is called
    more than 3 times, the individual calls are collapsed into a
    condensed summary row with expand/collapse.
    """
    if not calls:
        return ""

    from collections import defaultdict
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        groups[c.get('func_name', '?')].append(c)

    html = ""

    for func_name, func_calls in groups.items():
        total_calls = len(func_calls)
        cached_calls = sum(1 for c in func_calls if c.get('cache_hit'))
        computed_calls = total_calls - cached_calls
        total_time = sum(c.get('execution_time', 0.0) for c in func_calls)

        short_name = func_name.split('.')[-1] if '.' in func_name else func_name
        safe_name = _html_escape(short_name)

        if total_calls <= 3:
            for c in func_calls:
                hit = c.get('cache_hit', False)
                ct = c.get('execution_time', 0.0)
                if hit:
                    c_icon, c_color, status_text = "⚡", "#006644", "HIT"
                else:
                    c_icon, c_color, status_text = "⚙️", "#996300", "MISS"
                html += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color: {c_color};">{c_icon} @cache</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #1a73e8;">{safe_name}()</span> <span style="color: #888; font-size: 10px;">{status_text}</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;">-</td>
                    <td style="padding: 4px; text-align: left;">{ct:.3f}s</td>
                </tr>
                """
        else:
            html += _render_condensed_decorator_group(
                safe_name, func_calls, cached_calls, computed_calls, total_calls, total_time,
            )

    return html


# ---------------------------------------------------------------------------
# Private control/storage helpers
# ---------------------------------------------------------------------------

def _ctrl_stmt_icon_color(s: str) -> tuple[str, str]:
    """Return ``(icon, color)`` for a control statement status."""
    if s == CacheStatus.RESTORED:
        return "⚡", "#006644"
    if s == CacheStatus.SKIPPED:
        return "⏩", "#006644"
    if s == CacheStatus.COMPUTED:
        return "⚙️", "#996300"
    return "❓", "#999"


def _ctrl_stmt_storage_display(m: dict[str, Any], s: str) -> str:
    """Return storage display string for a control statement."""
    storage_val = m.get('storage', [])
    source = m.get('source', '')
    reasons = m.get('uncacheable_reasons', [])
    if s == CacheStatus.RESTORED:
        return f"← {source or 'Cache'}"
    if s == CacheStatus.SKIPPED:
        return "✓ RAM"
    if reasons:
        return "🚫"
    if storage_val:
        ss = "+".join(storage_val) if isinstance(storage_val, list) else str(storage_val)
        return f"→ {ss}"
    return "-"


def _render_control_stmt_row(m: dict[str, Any], child_class: str) -> str:
    """Render a single statement inside a control group."""
    s = m.get('status', 'UNKNOWN')
    t = m.get('total_time', 0.0)
    raw_code = m.get('code', '')
    if '# control_context:' in raw_code:
        raw_code = '\n'.join(
            line for line in raw_code.split('\n')
            if not line.startswith('# control_context:')
        )
    stmt_preview = raw_code.splitlines()[0][:_CODE_SNIPPET_MAX_LEN] if raw_code else ''
    if len(raw_code) > _CODE_SNIPPET_MAX_LEN:
        stmt_preview += "…"
    stmt_preview = _html_escape(stmt_preview)

    si, sc = _ctrl_stmt_icon_color(s)
    stor = _ctrl_stmt_storage_display(m, s)

    time_cell = f"{t:.3f}s"
    ms_saved = m.get('saved_time', 0.0)
    if ms_saved > _MIN_TIME_DISPLAY_S:
        time_cell += f" (↑{ms_saved:.1f}s)"

    stmt_vars = (
        m.get('restored_vars', [])
        or m.get('evaluated_vars', [])
        or [o for o in (m.get('output_vars', []) or []) if isinstance(o, str)]
    )
    stmt_vars_str = ", ".join(stmt_vars)
    vars_part = f"{stmt_vars_str}<br>" if stmt_vars_str else ""

    return f"""
    <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
        <td style="padding: 2px 4px 2px 24px; text-align: left; color: {sc}; white-space: nowrap; border-left: 2px solid #ddd;">{si}</td>
        <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;">{vars_part}<span style="color: #666; font-size: 10px;">{stmt_preview}</span></td>
        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">{stor}</td>
        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{time_cell}</td>
    </tr>"""

def _control_storage(metrics: list, computed: int, cached: int) -> str:
    """Compute control group aggregate storage display."""
    storage_parts: set = set()
    for m in metrics:
        for s in (m.get('storage', []) or []):
            storage_parts.add(s)
        if m.get('source'):
            storage_parts.add(m['source'])
    reasons_any = any(m.get('uncacheable_reasons') for m in metrics)
    if computed == 0 and cached > 0:
        src = 'RAM' if 'RAM' in storage_parts else ('DISK' if 'DISK' in storage_parts else 'Cache')
        return f"← {src}"
    if reasons_any:
        return "🚫"
    if storage_parts:
        ss = "+".join(sorted(storage_parts))
        return f"→ {ss}"
    return "-"

def _single_control_storage(
    m: dict[str, Any],
    status: str,
    is_upstream: bool,
    storage_val: Any,
) -> str:
    """Compute storage display for a single-unit control structure."""
    source = m.get('source', '')
    reasons = m.get('uncacheable_reasons', [])
    skipped_reason = m.get('skipped_reason', '')
    if is_upstream:
        if status == CacheStatus.RESTORED:
            src = source or 'DISK'
            return f"← {src}"
        if storage_val:
            ss = "+".join(storage_val) if isinstance(storage_val, list) else str(storage_val)
            return f"→ {ss}"
        return "-"
    if status == CacheStatus.RESTORED:
        src = source or 'Cache'
        return f"← {src}"
    if status == CacheStatus.SKIPPED:
        return "✓ In RAM"
    if reasons:
        return "🚫 No Cache"
    if skipped_reason:
        return "⚠️ Not Cached"
    if storage_val:
        ss = "+".join(storage_val) if isinstance(storage_val, list) else str(storage_val)
        return f"→ {ss}"
    return "-"

def _render_condensed_decorator_group(
    safe_name: str,
    func_calls: list,
    cached_calls: int,
    computed_calls: int,
    total_calls: int,
    total_time: float,
) -> str:
    """Render a condensed decorator group with expand/collapse."""
    gid = _unique_id("dec")

    if cached_calls == total_calls:
        summary_icon, summary_color = "⚡", "#006644"
        summary_text = f"ALL {total_calls} CACHED"
    elif cached_calls > 0:
        summary_icon, summary_color = "⚡", "#1a73e8"
        summary_text = f"{cached_calls}/{total_calls} cached, {computed_calls} computed"
    else:
        summary_icon, summary_color = "⚙️", "#996300"
        summary_text = f"{total_calls} calls, all computed"

    toggle_js = _build_toggle_js(gid, "▶", "▼")

    html = f"""
    <tr style="border-bottom: 1px solid #eee; cursor: pointer;" onclick="{toggle_js}">
        <td style="padding: 4px; text-align: left; color: {summary_color};">{summary_icon} @cache</td>
        <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span id="{gid}_arrow" style="color:#999; font-size:10px;">▶</span> <span style="color: #1a73e8;">{safe_name}()</span> <span style="color: #888; font-size: 10px;">{summary_text}</span></td>
        <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;">-</td>
        <td style="padding: 4px; text-align: left;">{total_time:.3f}s</td>
    </tr>
    """

    child_class = f"{gid}_d {gid}_a"

    show_calls = func_calls
    omitted = 0
    if total_calls > 10:
        show_calls = func_calls[:5] + func_calls[-5:]
        omitted = total_calls - 10

    for i, c in enumerate(show_calls):
        hit = c.get('cache_hit', False)
        ct = c.get('execution_time', 0.0)
        c_icon = "⚡" if hit else "⚙️"
        c_color = "#006644" if hit else "#996300"

        html += f"""
        <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
            <td style="padding: 2px 4px 2px 24px; text-align: left; color: {c_color}; border-left: 2px solid #c8d8ee;">{c_icon}</td>
            <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span style="color: #888;">call #{func_calls.index(c) + 1}</span></td>
            <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">{'HIT' if hit else 'MISS'}</td>
            <td style="padding: 2px 4px; text-align: left;">{ct:.3f}s</td>
        </tr>
        """

        if omitted > 0 and i == 4:
            html += f"""
            <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                <td style="padding: 2px 4px 2px 24px; border-left: 2px solid #c8d8ee;"></td>
                <td colspan="3" style="padding: 2px 4px; text-align: center; font-size: 10px; color: #999; font-style: italic;">... {omitted} more calls ...</td>
            </tr>
            """

    return html
