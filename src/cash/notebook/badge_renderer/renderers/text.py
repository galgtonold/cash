"""Text renderer for the BadgeView IR.

Walks an :class:`InteractiveBadge` tree and produces a plain-text summary
suitable for ``print()`` output. The text format carries no chrome — no
collapsible groups, no per-row drawers — because readers are humans
scanning a notebook stdout stream. Nesting is conveyed by indentation
alone: a loop or control nested inside another control body renders its
rows one step further in.
"""

from __future__ import annotations

from .. import theme
from .._reasons import guard_summary_line, is_guard_reason, shorten_skipped_reason
from ..view import (
    BadgeHeader,
    BadgeStatus,
    ControlGroup,
    ControlGroupSingle,
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


def _row_tag(row: StatementRow, *, is_upstream: bool) -> str:
    prefix = "⬆️" if is_upstream else _status_icon(row.status)
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


def _row_line(row: StatementRow, *, is_upstream: bool) -> str:
    code = row.code.splitlines()[0][:theme.HEADER_MAX_LEN] if row.code else ""
    tag = _row_tag(row, is_upstream=is_upstream)
    if row.status is BadgeStatus.RESTORED:
        return f"  {tag}: {code}  (saved {row.time_s:.2f}s)"
    if row.status is BadgeStatus.SKIPPED:
        return f"  {tag}: {code}"
    if row.status is BadgeStatus.COMPUTED:
        if row.uncacheable_reasons:
            reasons = ", ".join(row.uncacheable_reasons)
            return f"  {tag}: {code}  ({row.time_s:.2f}s) — {reasons}"
        if row.skipped_reason:
            # Shortened, not dropped: the row still says it wasn't cached and
            # why. The guard's full paragraph is emitted once per cell by
            # ``_guard_summary_lines`` instead of once per statement (CAS-182).
            return (f"  {tag}: {code}  ({row.time_s:.2f}s) — "
                    f"{shorten_skipped_reason(row.skipped_reason)}")
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
    )
    return _row_line(pseudo, is_upstream=is_upstream)


#: Leaf item types that render as exactly one line at their parent's level.
#: Anything else is a *group* and gets one extra indent step when it appears
#: inside a control body or a loop body (CAS-195).
_LEAF_ITEMS = (StatementRow, ControlGroupSingle)

#: One indent step, matching the two-space lead-in ``_row_line`` already emits.
_INDENT = "  "


def _loop_body(item: ForLoopGroup) -> tuple:
    """Body of *item* in source order.

    ``ForLoopGroup.body`` interleaves direct :class:`LoopStatement`s with
    loops/controls nested inside the body, ordered by the runtime's
    ``body_index_chain``. Older metric sources leave it empty, so fall back
    to ``stmts + nested`` the same way the HTML renderer does.
    """
    return item.body or (tuple(item.stmts) + tuple(item.nested))


def _item_lines(item: SectionItem, *, is_upstream: bool, indent: int = 0) -> list[str]:
    """Lines for *item*, recursing into nested groups.

    A control or loop body may hold further ``ControlGroup`` /
    ``ControlGroupSingle`` / ``ForLoopGroup`` nodes rather than only flat
    ``StatementRow``s (``view_builder`` builds ``ControlGroup.rows`` by the
    same recursive dispatch used at the top level). Dispatch on type at every
    level; feeding a group to ``_row_line`` raised ``AttributeError``, which
    escaped into the kernel's message handler and hung the client (CAS-195).
    """
    pad = _INDENT * indent
    if isinstance(item, StatementRow):
        return [pad + _row_line(item, is_upstream=is_upstream)]
    if isinstance(item, ForLoopGroup):
        out: list[str] = []
        for sub in _loop_body(item):
            if isinstance(sub, LoopStatement):
                out.extend(
                    pad + _iteration_line(it, is_upstream=is_upstream)
                    for it in sub.iterations
                )
            else:
                out.extend(
                    _item_lines(sub, is_upstream=is_upstream, indent=indent + 1)
                )
        return out
    if isinstance(item, ControlGroup):
        out = []
        for r in item.rows:
            step = 0 if isinstance(r, _LEAF_ITEMS) else 1
            out.extend(
                _item_lines(r, is_upstream=is_upstream, indent=indent + step)
            )
        return out
    if isinstance(item, ControlGroupSingle):
        return [pad + _row_line(item.row, is_upstream=is_upstream)]
    if isinstance(item, SkippedBucket):
        out = []
        for sub in item.items:
            out.extend(_item_lines(sub, is_upstream=is_upstream, indent=indent))
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


def _iter_rows(item: SectionItem):
    """Yield every StatementRow reachable under *item*.

    Recurses through nested groups: ``ControlGroup.rows`` may hold further
    groups rather than bare ``StatementRow``s, and the callers below read
    ``.skipped_reason`` off whatever this yields (CAS-195).
    """
    if isinstance(item, StatementRow):
        yield item
    elif isinstance(item, ControlGroup):
        for r in item.rows:
            yield from _iter_rows(r)
    elif isinstance(item, ControlGroupSingle):
        yield item.row
    elif isinstance(item, SkippedBucket):
        for sub in item.items:
            yield from _iter_rows(sub)
    elif isinstance(item, ForLoopGroup):
        # A loop's own body statements are IterationRows, which carry no
        # skipped_reason — but a control nested in the body does hold real
        # StatementRows, so descend into the non-LoopStatement children only.
        for sub in _loop_body(item):
            if not isinstance(sub, LoopStatement):
                yield from _iter_rows(sub)


def _guard_summary_lines(badge: InteractiveBadge) -> list[str]:
    """The guard's explanation, once per cell rather than once per statement."""
    count = sum(
        1
        for section in badge.sections
        for item in section.items
        for row in _iter_rows(item)
        if is_guard_reason(row.skipped_reason)
    )
    line = guard_summary_line(count)
    return [line] if line else []


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
    lines.extend(_guard_summary_lines(badge))
    return "\n".join(lines)
