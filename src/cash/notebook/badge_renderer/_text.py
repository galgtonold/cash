"""Text-mode badge (plain print output)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cash.notebook.cache_status import CacheStatus
from cash.utils import safe_text

from .theme import HEADER_MAX_LEN as _HEADER_MAX_LEN


def _text_badge_metric_line(m: dict[str, Any]) -> tuple[str, bool, str, float, float]:
    """Return ``(line, is_upstream, kind, saved, exec_time)`` for one metric.

    ``kind`` is one of ``'restored'``, ``'computed'``, ``'skipped'``, ``'other'``.
    """
    status = m.get('status', 'UNKNOWN')
    raw_code = m.get('code', '')
    # Strip __iteration_context__ / control_context comments from display
    if '# __iteration_context__:' in raw_code or '# control_context:' in raw_code:
        raw_code = '\n'.join(
            line for line in raw_code.split('\n')
            if not line.startswith('# __iteration_context__:')
            and not line.startswith('# control_context:')
        )
    code_snippet = raw_code.splitlines()[0][:_HEADER_MAX_LEN] if raw_code else ''
    is_upstream = m.get('is_upstream', False)
    exec_time = m.get('total_time', 0.0)
    saved_time = m.get('saved_time', 0.0)

    if status == CacheStatus.RESTORED:
        tag = '⚡ RESTORED' if not is_upstream else '⬆️ RESTORED'
        line = f"  {tag}: {code_snippet}  (saved {saved_time:.2f}s)"
        return line, is_upstream, 'restored', saved_time, exec_time
    if status == CacheStatus.SKIPPED:
        tag = '⏩ SKIPPED' if not is_upstream else '⬆️ SKIPPED'
        line = f"  {tag}: {code_snippet}"
        return line, is_upstream, 'skipped', saved_time, exec_time
    if status == CacheStatus.COMPUTED:
        skipped_reason = m.get('skipped_reason', '')
        uncacheable_reasons = m.get('uncacheable_reasons', [])
        storage_val = m.get('storage', [])
        if uncacheable_reasons:
            tag = '🚫 NOT CACHED' if not is_upstream else '⬆️ NOT CACHED'
            reason_text = ", ".join(uncacheable_reasons)
            line = f"  {tag}: {code_snippet}  ({exec_time:.2f}s) — {reason_text}"
        elif skipped_reason:
            tag = '⚠️ NOT CACHED' if not is_upstream else '⬆️ NOT CACHED'
            line = f"  {tag}: {code_snippet}  ({exec_time:.2f}s) — {skipped_reason}"
        elif storage_val:
            storage_str = "+".join(storage_val) if isinstance(storage_val, list) else str(storage_val)
            tag = '⚙️ COMPUTED' if not is_upstream else '⬆️ COMPUTED'
            line = f"  {tag}: {code_snippet}  ({exec_time:.2f}s) → {storage_str}"
        else:
            tag = '⚙️ COMPUTED' if not is_upstream else '⬆️ COMPUTED'
            line = f"  {tag}: {code_snippet}  ({exec_time:.2f}s)"
        return line, is_upstream, 'computed', saved_time, exec_time
    if status == 'FUNCTION_CHANGED':
        line = f"  🔄 FUNC CHANGED: {code_snippet}"
        return line, is_upstream, 'other', saved_time, exec_time
    if status == 'MODULE_RELOADED':
        mod_names = ', '.join(sorted(m.get('changed_modules', {}).keys()))
        line = f"  🔄 MODULE RELOADED: {mod_names or code_snippet}"
        return line, is_upstream, 'other', saved_time, exec_time
    if status == 'WARNING':
        line = f"  ⚠️ WARNING: {code_snippet}"
        return line, is_upstream, 'other', saved_time, exec_time
    line = f"  {status}: {code_snippet}  ({exec_time:.2f}s)"
    return line, is_upstream, 'other', saved_time, exec_time


def _text_badge_header(
    computed_count: int,
    restored_count: int,
    skipped_count: int,
    total_saved: float,
    cell_total_time: float | None,
) -> str:
    """Return the summary header line for a text badge."""
    if computed_count == 0 and restored_count > 0:
        return f"⚡ CACHED (saved {total_saved:.2f}s)"
    if computed_count == 0 and skipped_count > 0:
        return "⏩ SKIPPED (already computed)"
    if total_saved > 0:
        return (
            f"⚙️ EXECUTED ({cell_total_time:.2f}s, saved {total_saved:.2f}s)"
            if cell_total_time
            else f"⚙️ EXECUTED (saved {total_saved:.2f}s)"
        )
    return (
        f"⚙️ EXECUTED ({cell_total_time:.2f}s)"
        if cell_total_time
        else "⚙️ EXECUTED"
    )


def _text_badge_decorator_lines(metrics_list: list[dict[str, Any]]) -> list[str]:
    """Return text lines summarising @cash.cache decorator calls."""
    all_dec_calls: list[dict[str, Any]] = []
    for m in metrics_list:
        all_dec_calls.extend(m.get('decorator_calls', []))
    if not all_dec_calls:
        return []
    dec_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in all_dec_calls:
        dec_groups[c.get('func_name', '?')].append(c)
    lines = ["  @cash.cache:"]
    for fn, fc in dec_groups.items():
        cached = sum(1 for c in fc if c.get('cache_hit'))
        total = len(fc)
        tt = sum(c.get('execution_time', 0.0) for c in fc)
        short_fn = fn.split('.')[-1] if '.' in fn else fn
        lines.append(f"    {short_fn}(): {cached}/{total} cached ({tt:.3f}s)")
    return lines


def print_text_badge(
    metrics_list: list[dict[str, Any]],
    cell_total_time: float | None = None,
) -> None:
    """Print a plain-text summary of cell execution (for ``print`` badge mode).

    This is the text-mode counterpart of :func:`render_interactive_badge`.
    It writes directly to stdout.
    """
    metrics_list = metrics_list or []
    if not metrics_list:
        return

    total_saved = 0.0
    restored_count = 0
    computed_count = 0
    skipped_count = 0

    upstream_lines: list[str] = []
    current_lines: list[str] = []

    for m in metrics_list:
        line, is_upstream, kind, saved_time, _exec_time = _text_badge_metric_line(m)
        if kind == 'restored':
            restored_count += 1
            total_saved += saved_time
        elif kind == 'skipped':
            skipped_count += 1
            total_saved += saved_time
        elif kind == 'computed':
            computed_count += 1

        if is_upstream:
            upstream_lines.append(line)
        else:
            current_lines.append(line)

    header = _text_badge_header(computed_count, restored_count, skipped_count, total_saved, cell_total_time)

    output = [f"[Cash] {header}"]
    if upstream_lines:
        output.append("  Upstream:")
        output.extend(f"  {line}" for line in upstream_lines)
    if current_lines:
        output.extend(current_lines)

    output.extend(_text_badge_decorator_lines(metrics_list))

    print(safe_text("\n".join(output)))
