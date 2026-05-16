"""Badge row rendering, section builders, and main interactive badge."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from cash.notebook.cache_status import CacheStatus

from ._components import (
    render_control_group,
    render_control_group_single,
    render_decorator_calls,
    render_for_loop_group,
)
from ._grouping import (
    _reset_unique_ids,
    _unique_id,
    group_loop_iterations,
)
from ._types import (
    _BADGE_COLOR_DEFAULT,
    _BADGE_COLOR_RESTORED,
    _CODE_SNIPPET_MAX_LEN,
    _FONT_MONO,
    _FONT_SANS,
    _HEADER_MAX_LEN,
    _MIN_TIME_DISPLAY_MS,
    _MIN_TIME_DISPLAY_S,
)

# ---------------------------------------------------------------------------
# Storage / tooltip helpers
# ---------------------------------------------------------------------------

def _build_storage_tooltip(storage_val: Any) -> str:
    """Build an explanatory tooltip for → RAM, → RAM+DISK, etc."""
    tiers = [s.upper() for s in storage_val] if isinstance(storage_val, list) else [str(storage_val).upper()]

    if 'RAM' in tiers and 'DISK' in tiers:
        return 'Cached in memory (fast access) and persisted to disk (survives kernel restart).'
    if 'RAM' in tiers:
        return ('Cached in memory only. Fast access but lost on kernel restart. '
                'Execution was too fast for disk persistence to be worthwhile.')
    if 'DISK' in tiers:
        return 'Cached to disk. Survives kernel restart.'
    return f"Cached to: {', '.join(tiers)}"


def _source_restore_tooltip(source: str) -> str:
    """Return tooltip text for a given cache restore source label."""
    if source == 'RAM':
        return 'Restored from in-memory cache (fastest tier). May also be persisted to disk.'
    if source == 'DISK':
        return 'Restored from disk cache. Result was loaded from persistent storage.'
    return f'Restored from {source} cache.'


# ---------------------------------------------------------------------------
# Badge row renderers
# ---------------------------------------------------------------------------

def _render_upstream_badge_row(m: dict[str, Any], code_snippet: str) -> str:
    """Render an upstream metric as an HTML badge row."""
    status = m.get('status', CacheStatus.COMPUTED)
    exec_time = m.get('total_time', 0.0)
    source = m.get('source', '')

    if status == 'FUNCTION_CHANGED':
        # Function source change notification
        changed_names = ', '.join(m.get('changed_functions', []))
        return f"""
                <tr style="border-bottom: 1px solid #eee; background-color: #fff3e0;">
                    <td style="padding: 4px; text-align: left; color: #e65100;">🔄 Changed</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #bf360c;">{changed_names}</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #e65100;">Re-executing dependents</td>
                    <td style="padding: 4px; text-align: left;">-</td>
                </tr>
                """
    if status == 'MODULE_RELOADED':
        # Module source change notification
        mod_names = ', '.join(sorted(m.get('changed_modules', {}).keys()))
        return f"""
                <tr style="border-bottom: 1px solid #eee; background-color: #fff3e0;">
                    <td style="padding: 4px; text-align: left; color: #e65100;">🔄 Reloaded</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #bf360c;">{mod_names or code_snippet}</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #e65100;">Re-executing dependents</td>
                    <td style="padding: 4px; text-align: left;">-</td>
                </tr>
                """
    if status == 'WARNING':
        # Opaque call pattern or other warning
        return f"""
                <tr style="border-bottom: 1px solid #eee; background-color: #fffde7;">
                    <td style="padding: 4px; text-align: left; color: #f57f17;">⚠️ Warning</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #795548;">{code_snippet}</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #f57f17;">Cache may be unreliable</td>
                    <td style="padding: 4px; text-align: left;">-</td>
                </tr>
                """
    if status == CacheStatus.RESTORED:
        # Virtually restored from cache
        source_display = f"← {source}" if source else "← DISK"
        source_tooltip = _source_restore_tooltip(source)
        restore_time = exec_time  # exec_time already captures total_time
        saved_time = m.get('saved_time', 0.0)
        time_display = f"{restore_time:.2f}s"
        if saved_time > 0:
            time_display += f" (Saved {saved_time:.2f}s)"
        return f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color: #006644;">⬆️ Restored</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #666;">{code_snippet}</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;"><span title='{source_tooltip}' style='cursor:help;'>{source_display}</span></td>
                    <td style="padding: 4px; text-align: left;">{time_display}</td>
                </tr>
                """
    # Upstream COMPUTED (auto-executed) or other status
    eval_vars = ", ".join(m.get('evaluated_vars', []) or m.get('output_vars', []) or [])
    code_part = f'<span style="color: #666; font-size: 10px;">{code_snippet}</span>'
    content_html = f"{eval_vars}<br>{code_part}" if eval_vars else code_part

    up_storage_val = m.get('storage', [])
    up_reasons = m.get('uncacheable_reasons', [])
    up_skipped_reason = m.get('skipped_reason', '')
    if up_reasons:
        up_storage_display = "🚫"
    elif up_skipped_reason:
        up_storage_display = "⚠️"
    elif up_storage_val:
        ss = "+".join(up_storage_val) if isinstance(up_storage_val, list) else str(up_storage_val)
        up_storage_display = f"→ {ss}"
    else:
        up_storage_display = "-"

    return f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color: #996300;">⬆️ Executed</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};">{content_html}</td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">{up_storage_display}</td>
                    <td style="padding: 4px; text-align: left;">{exec_time:.2f}s</td>
                </tr>
                """


def _computed_storage_display_html(m: dict[str, Any]) -> str:
    """Return the storage cell HTML for a COMPUTED metric."""
    storage_val = m.get('storage', [])
    reasons = m.get('uncacheable_reasons', [])
    skipped_reason = m.get('skipped_reason', '')
    if reasons:
        reason_text = ", ".join(reasons)
        return f"<span title='Not cached: {reason_text}' style='cursor:help; color: #d9534f;'>🚫 No Cache</span>"
    if skipped_reason:
        return f"<span title='{skipped_reason}' style='cursor:help; color: #e68a00;'>⚠️ Not Cached</span>"
    if storage_val:
        storage_str = "+".join(storage_val) if isinstance(storage_val, list) else str(storage_val)
        storage_tooltip = _build_storage_tooltip(storage_val)
        return f"<span title='{storage_tooltip}' style='cursor:help;'>→ {storage_str}</span>"
    # Provide a reason for why no storage info is available
    exec_time = m.get('total_time', 0.0) or m.get('execution_time', 0.0)
    outputs = m.get('evaluated_vars', []) or m.get('outputs', [])
    if not outputs:
        tooltip = "No variable outputs to cache (e.g. print-only statement). Stdout is still cached."
        return f"<span title='{tooltip}' style='cursor:help; color: #999;'>- no outputs</span>"
    if exec_time < 0.01:
        tooltip = "Execution was too fast for disk promotion. Stored in RAM only on next run."
        return f"<span title='{tooltip}' style='cursor:help; color: #999;'>- trivial</span>"
    tooltip = "No storage info available. This may indicate a backend configuration issue."
    return f"<span title='{tooltip}' style='cursor:help; color: #999;'>-</span>"


def _render_current_badge_row(m: dict[str, Any], status: str, code_snippet: str) -> str:
    """Render a non-upstream metric as an HTML badge row."""
    if status == CacheStatus.RESTORED:
        saved = m.get('saved_time', 0.0)
        restored_vars = ", ".join(m.get('restored_vars', [])) or "No outputs"
        source = m.get('source', 'Cache')
        source_display = f"← {source}"
        source_tooltip = _source_restore_tooltip(source)
        code_part = f'<span style="color: #666; font-size: 10px;">{code_snippet}</span>'
        return f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 4px; text-align: left; color: #006644;">⚡ Restored</td>
                <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};">{restored_vars}<br>{code_part}</td>
                <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;"><span title='{source_tooltip}' style='cursor:help;'>{source_display}</span></td>
                <td style="padding: 4px; text-align: left;">Saved {saved:.2f}s</td>
            </tr>
            """
    if status == CacheStatus.COMPUTED:
        exec_time = m.get('total_time', 0.0)
        eval_vars = ", ".join(m.get('evaluated_vars', []))
        code_part = f'<span style="color: #666; font-size: 10px;">{code_snippet}</span>'
        content_html = f"{eval_vars}<br>{code_part}" if eval_vars else code_part
        storage_display = _computed_storage_display_html(m)
        return f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 4px; text-align: left; color: #996300;">⚙️ Executed</td>
                <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};">{content_html}</td>
                <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">{storage_display}</td>
                <td style="padding: 4px; text-align: left;">{exec_time:.2f}s</td>
            </tr>
            """
    if status == CacheStatus.SKIPPED:
        exec_time = m.get('total_time', 0.0)
        outputs = m.get('output_vars', []) or m.get('outputs', [])
        outputs = [o for o in (outputs or []) if isinstance(o, str)]
        outputs_str = ", ".join(outputs) if outputs else ""
        code_part = f'<span style="color: #666; font-size: 10px;">{code_snippet}</span>'
        content_html = f"{outputs_str}<br>{code_part}" if outputs_str else code_part
        return f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 4px; text-align: left; color: #006644;">⏩ Skipped</td>
                <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};">{content_html}</td>
                <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;" title="Already computed by upstream">✓ In RAM</td>
                <td style="padding: 4px; text-align: left;">{exec_time:.2f}s</td>
            </tr>
            """
    return ""


def _render_badge_row(m: dict[str, Any], is_upstream: bool = False) -> str:
    """Render a single metric as an HTML table row for the interactive badge."""
    raw_code = m.get('code', '')
    # Strip __iteration_context__ / control_context comments from display
    if '# __iteration_context__:' in raw_code or '# control_context:' in raw_code:
        raw_code = '\n'.join(line for line in raw_code.split('\n') if not line.startswith('# __iteration_context__:') and not line.startswith('# control_context:'))
    code_snippet = raw_code.splitlines()[0][:_CODE_SNIPPET_MAX_LEN] if raw_code else ''
    if len(raw_code) > _CODE_SNIPPET_MAX_LEN:
        code_snippet += "…"

    if is_upstream:
        return _render_upstream_badge_row(m, code_snippet)

    status = m.get('status', 'UNKNOWN')
    return _render_current_badge_row(m, status, code_snippet)


def _compute_badge_stats(
    metrics_list: list[dict[str, Any]],
) -> tuple[float, float, int, int, int]:
    """Compute summary statistics from a metrics list.

    Returns ``(total_saved, total_exec, restored_count, computed_count, skipped_count_stat)``.
    """
    total_saved = 0.0
    total_exec = 0.0
    restored_count = 0
    computed_count = 0
    skipped_count_stat = 0
    notification_statuses = {'FUNCTION_CHANGED', 'MODULE_RELOADED', 'WARNING'}
    for m in metrics_list:
        status_val = m.get('status', '')

        # Skip pure notification items from statistics
        if status_val in notification_statuses:
            continue

        # Handle RESTORED status first (regardless of is_upstream)
        if status_val == CacheStatus.RESTORED:
            restored_count += 1
            total_saved += m.get('saved_time', 0.0)
            # Add the restore overhead to total execution time
            total_exec += m.get('total_time', 0.0)

        elif status_val == CacheStatus.SKIPPED:
            total_saved += m.get('saved_time', 0.0)
            skipped_count_stat += 1

        elif m.get('is_upstream', False) or status_val == CacheStatus.COMPUTED:
            computed_count += 1
            total_exec += m.get('total_time', 0.0)

    return total_saved, total_exec, restored_count, computed_count, skipped_count_stat


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _render_grouped_item_rows(item: dict[str, Any], child_class: str) -> str:
    """Render one grouped item (single/loop/control) as child rows inside a collapsible section."""
    if item['type'] == 'single':
        em = item['metric']
        e_code = em.get('code', '')
        if '# __iteration_context__:' in e_code or '# control_context:' in e_code:
            e_code = '\n'.join(line for line in e_code.split('\n') if not line.startswith('# __iteration_context__:') and not line.startswith('# control_context:'))
        e_snippet = e_code.splitlines()[0][:_CODE_SNIPPET_MAX_LEN] if e_code else ''
        if len(e_code) > _CODE_SNIPPET_MAX_LEN:
            e_snippet += "\u2026"
        e_snippet = e_snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        e_eval_vars = ", ".join(em.get('evaluated_vars', []) or em.get('output_vars', []) or [])
        e_content = f"{e_eval_vars}<br><span style='color:#666;font-size:10px;'>{e_snippet}</span>" if e_eval_vars else f"<span style='color:#666;font-size:10px;'>{e_snippet}</span>"
        e_time = em.get('total_time', 0.0)
        e_storage_val = em.get('storage', [])
        if e_storage_val:
            ss = "+".join(e_storage_val) if isinstance(e_storage_val, list) else str(e_storage_val)
            e_storage_display = f"\u2192 {ss}"
        else:
            e_storage_display = "-"
        return f"""
                    <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #996300; white-space: nowrap; border-left: 2px solid #ddd;">\u2b06\ufe0f</td>
                        <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;">{e_content}</td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">{e_storage_display}</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{e_time:.2f}s</td>
                    </tr>
                    """
    if item['type'] == 'for_loop_group':
        all_e_metrics = [m for sg in item['stmt_groups'] for m in sg['metrics']]
        e_total_time = sum(m.get('total_time', 0.0) for m in all_e_metrics)
        e_time_str = f"{e_total_time:.2f}s" if e_total_time > _MIN_TIME_DISPLAY_S else "-"
        return f"""
                    <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #996300; white-space: nowrap; border-left: 2px solid #ddd;">\u2b06\ufe0f</td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 11px; color: #666;">\U0001f501 for loop ({len(all_e_metrics)} iterations)</td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">-</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{e_time_str}</td>
                    </tr>
                    """
    if item['type'] in ('control_group', 'control_group_single'):
        if item['type'] == 'control_group':
            e_cg_metrics = item['metrics']
            ce_header = item.get('header', '\u2026')[:_HEADER_MAX_LEN]
        else:
            em = item['metric']
            e_cg_metrics = [em]
            cs_stmts = em.get('body_statements', [])
            ce_header = cs_stmts[0][:_HEADER_MAX_LEN] if cs_stmts else '\u2026'
        if len(ce_header) > _HEADER_MAX_LEN:
            ce_header = ce_header[:_HEADER_MAX_LEN] + '\u2026'
        ce_header = ce_header.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        e_time_total = sum(m.get('total_time', 0.0) for m in e_cg_metrics)
        return f"""
                    <tr class="{child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #996300; white-space: nowrap; border-left: 2px solid #ddd;">\u2b06\ufe0f</td>
                        <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span style='color:#666;font-size:10px;'>{ce_header}</span></td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">-</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{e_time_total:.2f}s</td>
                    </tr>
                    """
    return ""


def _render_skipped_item_row(s_item: dict[str, Any], skip_child_class: str) -> str:
    """Render one skipped grouped item as a child row in the skipped section."""
    if s_item['type'] == 'single':
        sm = s_item['metric']
        s_code = sm.get('code', '')
        # Strip iteration/control context
        if '# __iteration_context__:' in s_code or '# control_context:' in s_code:
            s_code = '\n'.join(line for line in s_code.split('\n') if not line.startswith('# __iteration_context__:') and not line.startswith('# control_context:'))
        s_snippet = s_code.splitlines()[0][:_CODE_SNIPPET_MAX_LEN] if s_code else ''
        if len(s_code) > _CODE_SNIPPET_MAX_LEN:
            s_snippet += "\u2026"
        s_snippet = s_snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s_saved = sm.get('saved_time', 0.0)
        s_time_str = f"{s_saved:.2f}s" if s_saved > _MIN_TIME_DISPLAY_S else "-"
        return f"""
                    <tr class="{skip_child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #006644; white-space: nowrap; border-left: 2px solid #ddd;">\u23e9</td>
                        <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span style="color: #666; font-size: 10px;">{s_snippet}</span></td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">-</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{s_time_str}</td>
                    </tr>
                    """
    if s_item['type'] == 'for_loop_group':
        all_s_metrics = [m for sg in s_item['stmt_groups'] for m in sg['metrics']]
        s_total_saved = sum(m.get('saved_time', 0.0) for m in all_s_metrics)
        s_time_str = f"{s_total_saved:.2f}s" if s_total_saved > _MIN_TIME_DISPLAY_S else "-"
        return f"""
                    <tr class="{skip_child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #006644; white-space: nowrap; border-left: 2px solid #ddd;">\u23e9</td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 11px; color: #666;">\U0001f501 for loop ({len(all_s_metrics)} iterations)</td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">-</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{s_time_str}</td>
                    </tr>
                    """
    if s_item['type'] == 'control_group':
        s_cg_metrics = s_item['metrics']
        cs_header = s_item.get('header', '\u2026')[:_HEADER_MAX_LEN]
        if len(s_item.get('header', '')) > _HEADER_MAX_LEN:
            cs_header += '\u2026'
        cs_header = cs_header.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s_saved = sum(m.get('saved_time', 0.0) for m in s_cg_metrics)
        s_time_str = f"{s_saved:.2f}s" if s_saved > _MIN_TIME_DISPLAY_S else "-"
        return f"""
                    <tr class="{skip_child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #006644; white-space: nowrap; border-left: 2px solid #ddd;">\u23e9</td>
                        <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span style="color: #666; font-size: 10px;">{cs_header}</span></td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">-</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{s_time_str}</td>
                    </tr>
                    """
    if s_item['type'] == 'control_group_single':
        sm = s_item['metric']
        cs_stmts = sm.get('body_statements', [])
        cs_header = cs_stmts[0][:_HEADER_MAX_LEN] if cs_stmts else '\u2026'
        if cs_stmts and len(cs_stmts[0]) > _HEADER_MAX_LEN:
            cs_header += '\u2026'
        cs_header = cs_header.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s_saved = sm.get('saved_time', 0.0)
        s_time_str = f"{s_saved:.2f}s" if s_saved > _MIN_TIME_DISPLAY_S else "-"
        return f"""
                    <tr class="{skip_child_class}" style="display: none; border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 2px 4px 2px 24px; text-align: left; color: #006644; white-space: nowrap; border-left: 2px solid #ddd;">\u23e9</td>
                        <td style="padding: 2px 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span style="color: #666; font-size: 10px;">{cs_header}</span></td>
                        <td style="padding: 2px 4px; text-align: left; font-size: 10px; color: #888;">-</td>
                        <td style="padding: 2px 4px; text-align: left; white-space: nowrap;">{s_time_str}</td>
                    </tr>
                    """
    return ""


def _build_executed_subsection_html(upstream_executed: list[dict[str, Any]]) -> str:
    """Return HTML rows for the auto-executed upstream subsection."""
    if not upstream_executed:
        return ""

    exec_count = len(upstream_executed)
    exec_time_total = sum(m.get('total_time', 0.0) for m in upstream_executed)
    exec_gid = _unique_id("uex")
    exec_time_display = f"{exec_time_total:.2f}s" if exec_time_total > _MIN_TIME_DISPLAY_S else "—"

    # Aggregate storage tiers from child items for summary display
    exec_storage_tiers: set[str] = set()
    for em in upstream_executed:
        em_storage = em.get('storage', [])
        if isinstance(em_storage, list):
            exec_storage_tiers.update(s.upper() for s in em_storage)
        elif em_storage:
            exec_storage_tiers.add(str(em_storage).upper())
    if exec_storage_tiers:
        # Consistent order: RAM before DISK
        tier_order = ['RAM', 'DISK']
        sorted_tiers = [t for t in tier_order if t in exec_storage_tiers]
        sorted_tiers.extend(sorted(exec_storage_tiers - set(tier_order)))
        exec_storage_summary = f"\u2192 {'+'.join(sorted_tiers)}"
    else:
        exec_storage_summary = "—"

    exec_toggle_js = (
        f"(function(){{"
        f"var arrow=document.getElementById('{exec_gid}_arrow');"
        f"var expanding=arrow&&arrow.textContent==='\u25b6';"
        f"if(expanding){{"
        f"document.querySelectorAll('.{exec_gid}_d').forEach(function(r){{r.style.display='table-row'}});"
        f"arrow.textContent='\u25bc'"
        f"}}else{{"
        f"document.querySelectorAll('.{exec_gid}_a').forEach(function(r){{r.style.display='none'}});"
        f"arrow.textContent='\u25b6'"
        f"}}"
        f"}})()"
    )

    rows = f"""
            <tr style="border-bottom: 1px solid #eee; cursor: pointer;" onclick="{exec_toggle_js}">
                <td style="padding: 4px; text-align: left; color: #996300;">\u2b06\ufe0f Auto-exec</td>
                <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span id="{exec_gid}_arrow" style="color:#999; font-size:10px;">\u25b6</span> <span style="color: #666; font-style: italic;">{exec_count} upstream statement{'s' if exec_count != 1 else ''} auto-executed</span></td>
                <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;">{exec_storage_summary}</td>
                <td style="padding: 4px; text-align: left;">{exec_time_display}</td>
            </tr>
            """

    exec_child_class = f"{exec_gid}_d {exec_gid}_a"
    exec_grouped = group_loop_iterations(upstream_executed)
    for e_item in exec_grouped:
        if e_item['type'] == 'for_loop_group':
            rows += render_for_loop_group(
                e_item,
                is_upstream=True,
                indent_level=1,
                ancestor_classes=exec_child_class,
            )
        else:
            rows += _render_grouped_item_rows(e_item, exec_child_class)

    return rows


def _build_skipped_subsection_html(skipped_list: list[dict[str, Any]]) -> str:
    """Return HTML rows for the skipped upstream steps subsection."""
    if not skipped_list:
        return ""

    skipped_count = len(skipped_list)
    skipped_time = sum(m.get('saved_time', 0.0) for m in skipped_list)
    skipped_grouped = group_loop_iterations(skipped_list)
    skip_gid = _unique_id("skip")
    time_display = f"Saved {skipped_time:.2f}s" if skipped_time > _MIN_TIME_DISPLAY_S else "—"

    # Summary row with expand/collapse toggle
    skip_toggle_js = (
        f"(function(){{"
        f"var arrow=document.getElementById('{skip_gid}_arrow');"
        f"var expanding=arrow&&arrow.textContent==='\u25b6';"
        f"if(expanding){{"
        f"document.querySelectorAll('.{skip_gid}_d').forEach(function(r){{r.style.display='table-row'}});"
        f"arrow.textContent='\u25bc'"
        f"}}else{{"
        f"document.querySelectorAll('.{skip_gid}_a').forEach(function(r){{r.style.display='none'}});"
        f"arrow.textContent='\u25b6'"
        f"}}"
        f"}})()"
    )

    rows = f"""
            <tr style="border-bottom: 1px solid #eee; cursor: pointer;" onclick="{skip_toggle_js}">
                <td style="padding: 4px; text-align: left; color: #006644;">\u23e9 Skipped</td>
                <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO}; font-size: 11px;"><span id="{skip_gid}_arrow" style="color:#999; font-size:10px;">\u25b6</span> <span style="color: #666; font-style: italic;">{skipped_count} intermediate dependency step{'s' if skipped_count != 1 else ''}</span></td>
                <td style="padding: 4px; text-align: left; font-size: 10px; color: #555;">-</td>
                <td style="padding: 4px; text-align: left;">{time_display}</td>
            </tr>
            """

    skip_child_class = f"{skip_gid}_d {skip_gid}_a"
    for s_item in skipped_grouped:
        rows += _render_skipped_item_row(s_item, skip_child_class)

    return rows


def _build_current_cell_section_html(
    current_list: list[dict[str, Any]],
    has_upstream: bool,
) -> str:
    """Return HTML rows for the current-cell section of the badge table."""
    rows = ""
    if not current_list:
        return rows

    if has_upstream:
        rows += "<tr><td colspan='4' style='background:#f5f5f5; font-weight:bold; font-size:10px; padding:4px; color:#666; border-bottom:1px solid #eee; border-top:1px solid #eee;'>CURRENT CELL</td></tr>"

    grouped_rows = group_loop_iterations(current_list)
    for item in grouped_rows:
        if item['type'] == 'single':
            rows += _render_badge_row(item['metric'])
        elif item['type'] == 'for_loop_group':
            rows += render_for_loop_group(item)
        elif item['type'] == 'control_group':
            rows += render_control_group(item)
        elif item['type'] == 'control_group_single':
            rows += render_control_group_single(item)

    # Decorator Call Summary (condensed display for @cash.cache calls)
    all_decorator_calls: list[dict[str, Any]] = []
    for m in current_list:
        all_decorator_calls.extend(m.get('decorator_calls', []))
    if all_decorator_calls:
        rows += "<tr><td colspan='4' style='background:#e8f0fe; font-weight:bold; font-size:10px; padding:4px; color:#1a73e8; border-bottom:1px solid #c8d8ee; border-top:1px solid #c8d8ee;'>DECORATOR CACHE (@cash.cache)</td></tr>"
        rows += render_decorator_calls(all_decorator_calls)

    return rows


def _build_upstream_section_html(
    upstream_restored: list[dict[str, Any]],
    upstream_executed: list[dict[str, Any]],
    skipped_list: list[dict[str, Any]],
) -> str:
    """Return the HTML rows for the upstream history section of the badge."""
    details_rows = ""

    # Show restored/warning items directly (these represent cache savings)
    upstream_restored_grouped = group_loop_iterations(upstream_restored)
    for item in upstream_restored_grouped:
        if item['type'] == 'single':
            details_rows += _render_badge_row(item['metric'], is_upstream=True)
        elif item['type'] == 'for_loop_group':
            details_rows += render_for_loop_group(item, is_upstream=True)
        elif item['type'] == 'control_group':
            details_rows += render_control_group(item, is_upstream=True)
        elif item['type'] == 'control_group_single':
            details_rows += render_control_group_single(item, is_upstream=True)

    details_rows += _build_executed_subsection_html(upstream_executed)
    details_rows += _build_skipped_subsection_html(skipped_list)

    return details_rows


def _build_overhead_section_html(
    timing_breakdown: dict[str, float],
    cell_total_time: float,
    metrics_list: list[dict[str, Any]],
) -> str:
    """Return HTML rows for the overhead breakdown section of the badge."""
    # Calculate overhead: total time - sum of all statement times
    statements_time = sum(m.get('total_time', 0.0) for m in metrics_list)
    overhead = cell_total_time - statements_time

    if overhead <= 0.001:  # Only show if overhead is more than 1ms
        return ""

    rows = "<tr><td colspan='4' style='background:#f5f5f5; font-weight:bold; font-size:10px; padding:4px; color:#666; border-bottom:1px solid #eee; border-top:1px solid #eee;'>OVERHEAD</td></tr>"

    # Add breakdown rows
    badge_init = timing_breakdown.get('badge_init', 0)
    # Use upstream_check directly - it already has restore time excluded (calculated at line 650)
    upstream_check = timing_breakdown.get('upstream_check', 0)
    badge_progress = timing_breakdown.get('badge_progress', 0)

    other_overhead = overhead - (badge_init + upstream_check + badge_progress)

    if upstream_check > _MIN_TIME_DISPLAY_MS:
        rows += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color:#888;">↻ Upstream check</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #999; font-size: 10px;">Lineage simulation (excl. restore)</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">-</td>
                    <td style="padding: 4px; text-align: left;">{upstream_check:.3f}s</td>
                </tr>
                """

    if badge_init > _MIN_TIME_DISPLAY_MS:
        rows += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color:#888;">🏷️ Badge init</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #999; font-size: 10px;">Initial badge render</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">-</td>
                    <td style="padding: 4px; text-align: left;">{badge_init:.3f}s</td>
                </tr>
                """

    if badge_progress > _MIN_TIME_DISPLAY_MS:
        rows += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color:#888;">📊 Progress updates</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #999; font-size: 10px;">Badge progress renders</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">-</td>
                    <td style="padding: 4px; text-align: left;">{badge_progress:.3f}s</td>
                </tr>
                """

    if other_overhead > _MIN_TIME_DISPLAY_MS:
        rows += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 4px; text-align: left; color:#888;">⚙️ Other</td>
                    <td style="padding: 4px; text-align: left; font-family: {_FONT_MONO};"><span style="color: #999; font-size: 10px;">AST parsing, output handling</span></td>
                    <td style="padding: 4px; text-align: left; font-size: 10px; color: #666;">-</td>
                    <td style="padding: 4px; text-align: left;">{other_overhead:.3f}s</td>
                </tr>
                """

    return rows


def _badge_style_parts(
    status: str,
    computed_count: int,
    restored_count: int,
    skipped_count_stat: int,
    total_saved: float,
    summary_time: float,
    current_code: str | None,
    current_step: int,
    total_steps: int,
) -> tuple[str, str, str, str, str, str]:
    """Return ``(summary_bg, summary_color, summary_border, icon, label, subtext)``."""
    if status == "RUNNING":
        # RUNNING STATE
        summary_bg = "#f0f0f0"
        summary_color = "#666"
        summary_border = "#ccc"
        icon = "⏳"

        # Format: Processing (X/Y) or Processing (step X) for upstream
        if total_steps and total_steps > 0:
            progress_str = f"({current_step}/{total_steps})"
        elif current_step > 0:
            progress_str = f"(step {current_step})"
        else:
            progress_str = "..."

        label = f"Processing {progress_str}"

        # Subtext is the current code being executed
        if current_code:
            code_preview = current_code.splitlines()[0][:60]
            if len(current_code) > 60:
                code_preview += "..."
            subtext = f"- {code_preview}"
        else:
            subtext = ""

    else:
        # DONE STATE
        if computed_count == 0 and (restored_count > 0 or skipped_count_stat > 0):
            # FULLY CACHED (restored from cache and/or skipped)
            summary_bg = "#e6fffa"
            summary_color = "#006644"
            summary_border = "#006644"

            if restored_count > 0:
                icon = "⚡"
                label = "CACHED"
                subtext = f"(Saved {total_saved:.2f}s)" if total_saved > _MIN_TIME_DISPLAY_S else ""
            else:
                # All statements were skipped (already computed)
                icon = "⏩"
                label = "SKIPPED"
                subtext = "(Already computed)"
        else:
            # EXECUTED (Partial or Full)
            summary_bg = "#fffbe6"
            summary_color = "#996300"
            summary_border = "#996300"
            icon = "⚙️"
            label = "EXECUTED"
            if total_saved > 0:
                subtext = f"({summary_time:.2f}s, Saved {total_saved:.2f}s)"
            else:
                subtext = f"({summary_time:.2f}s)"

    return summary_bg, summary_color, summary_border, icon, label, subtext


# ---------------------------------------------------------------------------
# Bug report URL builder
# ---------------------------------------------------------------------------

_ISSUES_BASE = "https://github.com/galgtonold/cash/issues/new"

def _build_bug_report_url(metrics_list: list[dict[str, Any]], context: dict | None = None) -> str:
    """Build a pre-filled GitHub issue URL for incorrect caching reports.

    The URL must stay under ~8000 chars (GitHub rejects longer ones).
    Sections are added in priority order; lower-priority sections are
    dropped when the budget is exhausted.
    """
    _MAX_URL = 7800  # leave margin below GitHub's ~8192 limit
    ctx = context or {}
    version = ctx.get('version', 'unknown')
    python_version = ctx.get('python_version', '(unknown)')
    backend = ctx.get('backend', '(unknown)')
    notebook_source: list[str] = ctx.get('notebook_source', [])

    # --- Badge content (text version — always included) ---
    badge_lines: list[str] = []
    for m in metrics_list:
        if m.get('is_upstream'):
            continue
        code = (m.get('code') or '').strip()
        # Strip __iteration_context__ / control_context comments from display
        if '# __iteration_context__:' in code or '# control_context:' in code:
            code = '\n'.join(line for line in code.split('\n') if not line.startswith('# __iteration_context__:') and not line.startswith('# control_context:')).strip()
        if len(code) > 100:
            code = code[:97] + '...'
        st = str(m.get('status', '')).replace('CacheStatus.', '')
        t = m.get('total_time') or m.get('execution_time') or 0.0
        saved = m.get('saved_time') or 0.0
        outs_raw = m.get('output_vars', []) or m.get('outputs', [])
        outs = ', '.join(o for o in (outs_raw or []) if isinstance(o, str))
        outs_str = f' | {outs}' if outs else ''
        badge_lines.append(f"  {st:>8} | {t:>6.3f}s | saved {saved:>6.3f}s | {code}{outs_str}")
    badge_text = "\n".join(badge_lines) if badge_lines else "(no metrics)"

    # --- Notebook source (from .ipynb — variable length, may be trimmed) ---
    def _make_nb_source(max_chars_per_cell: int) -> str:
        if not notebook_source:
            return ""
        parts = []
        for i, cell in enumerate(notebook_source, 1):
            snippet = cell.strip()
            if len(snippet) > max_chars_per_cell:
                snippet = snippet[:max_chars_per_cell - 3] + "..."
            parts.append(f"# --- Cell {i} ---\n{snippet}")
        return (
            "<details><summary>Notebook source</summary>\n\n"
            "```python\n"
            + "\n\n".join(parts)
            + "\n```\n\n</details>\n\n"
        )

    # --- Skeleton (always present) ---
    skeleton = (
        "**Describe the incorrect behavior:**\n"
        "<!-- What did cash do wrong? What did you expect instead? -->\n\n"
        "**Cash badge output:**\n"
        "```\n"
        "{badge}\n"
        "```\n\n"
        "**Expected behavior:**\n"
        "<!-- e.g. 'Should have re-executed but was RESTORED from cache' -->\n\n"
        "{nb_source}"
        "**Environment:**\n"
        f"- Cash version: {version}\n"
        f"- Python: {python_version}\n"
        f"- Backend: {backend}\n\n"
        "**Additional context:**\n"
        "<!-- Paste any relevant `%cash_debug on` output here -->"
    )

    url_prefix = f"{_ISSUES_BASE}?title=Incorrect+caching+behavior&labels=bug%2Ccaching-behavior&body="

    def _url_len(body: str) -> int:
        return len(url_prefix) + len(quote(body))

    # Try with full notebook source (300 chars/cell)
    body = skeleton.format(badge=badge_text, nb_source=_make_nb_source(300))
    if _url_len(body) <= _MAX_URL:
        return url_prefix + quote(body)

    # Try with shorter notebook source (150 chars/cell)
    body = skeleton.format(badge=badge_text, nb_source=_make_nb_source(150))
    if _url_len(body) <= _MAX_URL:
        return url_prefix + quote(body)

    # Try without notebook source
    truncated_note = (
        "> **Note:** Notebook source was too large to include. "
        "Please paste the relevant cells below.\n\n"
    )
    body = skeleton.format(badge=badge_text, nb_source=truncated_note)
    if _url_len(body) <= _MAX_URL:
        return url_prefix + quote(body)

    # Last resort: trim badge text too
    if len(badge_text) > 500:
        badge_text = badge_text[:500] + "\n  ... (truncated)"
    body = skeleton.format(badge=badge_text, nb_source=truncated_note)
    return url_prefix + quote(body)


# ---------------------------------------------------------------------------
# Main badge builder
# ---------------------------------------------------------------------------

def render_interactive_badge(
    metrics_list: list[dict[str, Any]],
    badge_mode: str,
    debug: bool = False,
    display_id: str | None = None,
    status: str = "DONE",
    current_step: int = 0,
    total_steps: int = 0,
    current_code: str | None = None,
    cell_total_time: float | None = None,
    timing_breakdown: dict[str, float] | None = None,
    bug_report_context: dict | None = None,
) -> str:
    """Build the interactive HTML badge string for cell execution results.

    This is a pure function — it computes and returns the HTML string.
    The caller is responsible for display / publish.

    Returns:
        The rendered HTML string, or an empty string if *badge_mode* is not
        ``'html'``.
    """
    # In non-html badge mode, this function is a no-op
    # (print mode uses _print_text_badge instead, off mode shows nothing)
    if badge_mode != 'html':
        return ""

    # Reset the deterministic ID counter so the same badge structure
    # produces the same element IDs across re-renders (needed for
    # preserving which sections the user has expanded/collapsed).
    _reset_unique_ids()

    metrics_list = metrics_list or []

    # Partition metrics into groups
    upstream_list = [m for m in metrics_list if m.get('is_upstream', False) and m.get('status') != CacheStatus.SKIPPED]
    # Sub-categorize upstream: restored (shown directly) vs re-executed (collapsible)
    upstream_restored = [m for m in upstream_list if m.get('status') in (CacheStatus.RESTORED, 'FUNCTION_CHANGED', 'MODULE_RELOADED', 'WARNING')]
    upstream_executed = [m for m in upstream_list if m.get('status') not in (CacheStatus.RESTORED, 'FUNCTION_CHANGED', 'MODULE_RELOADED', 'WARNING')]
    # Skipped list: only upstream skipped (intermediate dependencies), not current cell skipped
    skipped_list = [m for m in metrics_list if m.get('status') == CacheStatus.SKIPPED and m.get('is_upstream', False)]
    # Current list: all non-upstream statements
    current_list = [m for m in metrics_list if not m.get('is_upstream', False)]

    # Compute summary statistics
    total_saved, total_exec, restored_count, computed_count, skipped_count_stat = _compute_badge_stats(metrics_list)

    # Use the actual cell total time if provided (more accurate than sum of metrics)
    summary_time = cell_total_time if cell_total_time is not None else total_exec

    # --- Build Details Table Rows ---
    details_rows = ""

    # 1. Upstream Rows (with loop grouping)
    if upstream_list or skipped_list:
        details_rows += "<tr><td colspan='4' style='background:#f5f5f5; font-weight:bold; font-size:10px; padding:4px; color:#666; border-bottom:1px solid #eee;'>UPSTREAM HISTORY</td></tr>"
        details_rows += _build_upstream_section_html(upstream_restored, upstream_executed, skipped_list)

    # 2. Completed Current Rows (with loop grouping) and decorator calls
    details_rows += _build_current_cell_section_html(current_list, bool(upstream_list or skipped_list))

    # 3. Overhead Breakdown (if available)
    if timing_breakdown and cell_total_time is not None:
        details_rows += _build_overhead_section_html(timing_breakdown, cell_total_time, metrics_list)

    # --- Badge Style Logic ---
    summary_bg, summary_color, summary_border, icon, label, subtext = _badge_style_parts(
        status, computed_count, restored_count, skipped_count_stat,
        total_saved, summary_time, current_code, current_step, total_steps,
    )

    return f"""
    <details style="
        display: inline-block;
        border: 1px solid {summary_border};
        border-radius: 4px;
        background-color: {summary_bg};
        padding: 0;
        margin-top: 5px;
        font-family: {_FONT_SANS};
        font-size: 12px;
        color: {summary_color};
        max-width: 100%;
    ">
        <summary style="
            cursor: pointer;
            padding: 4px 8px;
            font-weight: bold;
            list-style: none;
            outline: none;
            display: inline-flex;
            align-items: baseline;
            gap: 6px;
        ">
            <span>{icon} {label}</span><span style="font-weight: normal; opacity: 0.8; font-family: {_FONT_MONO};">{subtext}</span>
        </summary>

        <div style="
            padding: 5px 10px;
            background-color: #ffffff;
            border-top: 1px solid {summary_border}30;
            color: #333;
            max-height: 500px;
            overflow-y: auto;
        ">
            <table style="width: 100%; min-width: 500px; border-collapse: collapse; font-size: 11px; table-layout: auto; color: #333; background-color: #ffffff;">
                <thead>
                     <tr style="border-bottom: 2px solid #ddd; color: #666; background-color: #ffffff;">
                        <th style="text-align: left; padding: 4px; white-space: nowrap;">TYPE</th>
                        <th style="text-align: left; padding: 4px;">CONTENT</th>
                        <th style="text-align: left; padding: 4px; white-space: nowrap;">STORAGE</th>
                        <th style="text-align: left; padding: 4px; white-space: nowrap;">TIME</th>
                    </tr>
                </thead>
                <tbody>
                    {details_rows}
                </tbody>
            </table>
            <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid #f0f0f0; text-align: right;">
                <a href="{_build_bug_report_url(metrics_list, bug_report_context)}" target="_blank" rel="noopener noreferrer"
                   style="display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px;
                          background: #fff5f5; border: 1px solid #f5c6c6; border-radius: 12px;
                          color: #c0392b; font-size: 11px; font-weight: 500; text-decoration: none;"
                   title="Open a pre-filled GitHub issue to report incorrect caching behaviour"
                >🐛 Report incorrect caching</a>
            </div>
        </div>
    </details>
    """


# ---------------------------------------------------------------------------
# Simple status badge
# ---------------------------------------------------------------------------

def render_status_badge(
    status: str,
    execution_time: float = 0.0,
    time_saved: float = 0.0,
    source: str | None = None,
    storage: Any = None,
) -> str:
    """Render a simple colored status badge as an HTML string.

    Parameters
    ----------
    status : str
        Cache status value (e.g. 'COMPUTED', 'RESTORED').
    execution_time : float
        Wall-clock execution time in seconds.
    time_saved : float
        Time saved by cache restore in seconds.
    source : str, optional
        Cache backend source label (e.g. 'memory', 'disk').
    storage : str or list, optional
        Cache backend storage label(s).

    Returns
    -------
    str
        An HTML ``<div>`` badge string.
    """
    color = _BADGE_COLOR_RESTORED if status == CacheStatus.RESTORED else _BADGE_COLOR_DEFAULT

    parts = [f"<b>{status}</b>"]

    if source:
        parts.append(f"FROM: {source}")

    if storage:
        storage_str = "+".join(storage) if isinstance(storage, list) else str(storage)
        parts.append(f"IN: {storage_str}")

    if status == CacheStatus.RESTORED and time_saved > 0:
        parts.append(f"Saved {time_saved:.2f}s")
    elif execution_time > 0:
        parts.append(f"{execution_time:.2f}s")

    text = " | ".join(parts)

    return f"""
    <div style="
        display: inline-block;
        padding: 4px 8px;
        background-color: {color};
        color: white;
        border-radius: 4px;
        font-family: {_FONT_MONO};
        font-size: 11px;
        margin-bottom: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    ">{text}</div>
    """
