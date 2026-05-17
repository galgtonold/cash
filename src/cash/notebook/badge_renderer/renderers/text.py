"""Text renderer for the BadgeView IR.

Walks an :class:`InteractiveBadge` tree and produces a plain-text summary
suitable for ``print()`` output. The text format is intentionally flat
(no collapsible groups, no HTML chrome) — readers are humans scanning a
notebook stdout stream.
"""

from __future__ import annotations

from .. import theme
from ..view import (
    BadgeHeader,
    BadgeStatus,
    ControlGroup,
    ControlGroupSingle,
    DecoratorCallGroup,
    ForLoopGroup,
    InteractiveBadge,
    IterationRow,
    OverheadBreakdown,
    Section,
    SectionItem,
    SectionKind,
    SkippedBucket,
    StatementRow,
)


def _header_line(h: BadgeHeader) -> str:
    if h.computed_count == 0 and h.restored_count > 0:
        return f"⚡ CACHED (saved {h.total_saved_s:.2f}s)"
    if h.computed_count == 0 and h.skipped_count > 0:
        return "⏩ SKIPPED (already computed)"
    if h.total_saved_s > 0:
        return (f"⚙️ EXECUTED ({h.total_exec_s:.2f}s, saved {h.total_saved_s:.2f}s)"
                if h.total_exec_s else f"⚙️ EXECUTED (saved {h.total_saved_s:.2f}s)")
    return (f"⚙️ EXECUTED ({h.total_exec_s:.2f}s)"
            if h.total_exec_s else "⚙️ EXECUTED")


def _row_tag(row: StatementRow) -> str:
    prefix = "⬆️" if row.is_upstream else _status_icon(row.status)
    label = _status_label(row.status, row)
    return f"{prefix} {label}"


def _status_icon(status: BadgeStatus) -> str:
    if status is BadgeStatus.RESTORED:
        return "⚡"
    if status is BadgeStatus.SKIPPED:
        return "⏩"
    if status is BadgeStatus.COMPUTED:
        return "⚙️"
    if status in (BadgeStatus.FUNCTION_CHANGED, BadgeStatus.MODULE_RELOADED):
        return "🔄"
    if status is BadgeStatus.WARNING:
        return "⚠️"
    return "•"


def _status_label(status: BadgeStatus, row: StatementRow) -> str:
    if status is BadgeStatus.RESTORED:
        return "RESTORED"
    if status is BadgeStatus.SKIPPED:
        return "SKIPPED"
    if status is BadgeStatus.COMPUTED:
        if row.uncacheable_reasons:
            return "NOT CACHED"  # 🚫 in HTML
        if row.skipped_reason:
            return "NOT CACHED"  # ⚠️ in HTML
        return "COMPUTED"
    if status is BadgeStatus.FUNCTION_CHANGED:
        return "FUNC CHANGED"
    if status is BadgeStatus.MODULE_RELOADED:
        return "MODULE RELOADED"
    if status is BadgeStatus.WARNING:
        return "WARNING"
    return status.value.upper()


def _row_line(row: StatementRow) -> str:
    code = row.code.splitlines()[0][:theme.HEADER_MAX_LEN] if row.code else ""
    tag = _row_tag(row)
    if row.status is BadgeStatus.RESTORED:
        return f"  {tag}: {code}  (saved {row.time_s:.2f}s)"
    if row.status is BadgeStatus.SKIPPED:
        return f"  {tag}: {code}"
    if row.status is BadgeStatus.COMPUTED:
        if row.uncacheable_reasons:
            reasons = ", ".join(row.uncacheable_reasons)
            return f"  {tag}: {code}  ({row.time_s:.2f}s) — {reasons}"
        if row.skipped_reason:
            return f"  {tag}: {code}  ({row.time_s:.2f}s) — {row.skipped_reason}"
        if row.storage_tiers:
            return f"  {tag}: {code}  ({row.time_s:.2f}s) → {'+'.join(row.storage_tiers)}"
        return f"  {tag}: {code}  ({row.time_s:.2f}s)"
    return f"  {tag}: {code}  ({row.time_s:.2f}s)"


def _iteration_line(it: IterationRow, *, is_upstream: bool) -> str:
    pseudo = StatementRow(
        status=it.status,
        code=it.code,
        time_s=it.time_s,
        storage_tiers=it.storage_tiers,
        is_upstream=is_upstream,
    )
    return _row_line(pseudo)


def _item_lines(item: SectionItem, *, is_upstream: bool) -> list[str]:
    if isinstance(item, StatementRow):
        return [_row_line(item)]
    if isinstance(item, ForLoopGroup):
        out: list[str] = []
        for ls in item.stmts:
            for it in ls.iterations:
                out.append(_iteration_line(it, is_upstream=is_upstream))
        return out
    if isinstance(item, ControlGroup):
        return [_row_line(r) for r in item.rows]
    if isinstance(item, ControlGroupSingle):
        return [_row_line(item.row)]
    if isinstance(item, SkippedBucket):
        out = []
        for sub in item.items:
            out.extend(_item_lines(sub, is_upstream=is_upstream))
        return out
    if isinstance(item, OverheadBreakdown | DecoratorCallGroup):
        return []  # rendered separately
    return []


def _decorator_lines(sections: tuple[Section, ...]) -> list[str]:
    dec_section = next(
        (s for s in sections if s.kind is SectionKind.DECORATORS),
        None,
    )
    if dec_section is None or not dec_section.items:
        return []
    lines = ["  @cash.cache:"]
    for g in dec_section.items:
        if not isinstance(g, DecoratorCallGroup):
            continue
        total = len(g.calls)
        cached = sum(1 for c in g.calls if c.status is BadgeStatus.RESTORED)
        time_s = sum(c.time_s for c in g.calls)
        short = g.func_name.split(".")[-1] if "." in g.func_name else g.func_name
        lines.append(f"    {short}(): {cached}/{total} cached ({time_s:.3f}s)")
    return lines


def render_text(badge: InteractiveBadge) -> str:
    """Render an :class:`InteractiveBadge` to a flat plain-text summary."""
    lines = [f"[Cash] {_header_line(badge.header)}"]

    upstream_section = next(
        (s for s in badge.sections if s.kind is SectionKind.UPSTREAM),
        None,
    )
    if upstream_section is not None:
        up_lines: list[str] = []
        for item in upstream_section.items:
            up_lines.extend(_item_lines(item, is_upstream=True))
        if up_lines:
            lines.append("  Upstream:")
            lines.extend(f"  {line}" for line in up_lines)

    current_section = next(
        (s for s in badge.sections if s.kind is SectionKind.CURRENT),
        None,
    )
    if current_section is not None:
        for item in current_section.items:
            lines.extend(_item_lines(item, is_upstream=False))

    lines.extend(_decorator_lines(badge.sections))
    return "\n".join(lines)
