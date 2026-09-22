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
from .._headline import mixed_headline
from .._reasons import guard_summary_line, is_guard_reason, shorten_skipped_reason, stale_export_text
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


# This renderer is ASCII-ONLY, deliberately. ``%cash_badge print`` exists for
# headless / agent runs, so its output is read by a DIFFERENT process than the
# one that wrote it: nbconvert, a log scraper, an agent parsing the .ipynb. The
# writing kernel always has a UTF-8 stdout, so an emoji is encoded happily into
# the notebook and then crashes the reader with UnicodeEncodeError on a cp1252
# console -- a traceback instead of the badge, for exactly the audience the mode
# was built for. `safe_text` cannot help: it degrades on the WRITER's encoding,
# which is never the one that fails.
#
# ``_rng_suffix`` below already stated this rule for the RNG marker; the status
# icons simply predated it. The label ("CACHED", "EXECUTED", ...) already
# carries the meaning, so dropping the glyph loses no information -- and matches
# the plain-text format the docs advertise.


def _header_line(h: BadgeHeader) -> str:
    if h.status is BadgeStatus.ERROR:
        return (f"ERROR ({h.total_exec_s:.2f}s)" if h.total_exec_s else "ERROR") + _uncacheable_suffix(h)
    if h.computed_count == 0 and h.restored_count > 0:
        return f"CACHED (saved {h.total_saved_s:.2f}s)"
    if h.computed_count == 0 and h.skipped_count > 0:
        return "SKIPPED (already computed)"
    if h.restored_count and h.computed_count:
        # Counts, and CACHED when restoring saved more than running cost: a
        # sweep cell that restored 12 fits and ran `sweep_rows = []` read
        # "EXECUTED · saved 257s" (round 29, r29s4).
        label, counts = mixed_headline(h)
        line = f"{label} ({counts}; {h.total_exec_s:.2f}s, saved {h.total_saved_s:.2f}s)"
    elif h.total_saved_s > 0:
        line = (f"EXECUTED ({h.total_exec_s:.2f}s, saved {h.total_saved_s:.2f}s)"
                if h.total_exec_s else f"EXECUTED (saved {h.total_saved_s:.2f}s)")
    else:
        line = (f"EXECUTED ({h.total_exec_s:.2f}s)"
                if h.total_exec_s else "EXECUTED")
    return line + _uncacheable_suffix(h)


def _uncacheable_suffix(h: BadgeHeader) -> str:
    """Name the work that will be paid for again on every future run.

    Only ever appended to an EXECUTED header: an uncacheable row is a row that
    ran, so a cell with one can't reach the CACHED or SKIPPED branches above.
    """
    if not h.uncacheable_count:
        return ""
    return f" - {h.uncacheable_count} not cached"


def _row_tag(row: StatementRow, *, is_upstream: bool) -> str:
    prefix = _status_icon(row.status)
    label = _status_label(row.status, row)
    # "^" marks a statement pulled in from an upstream cell -- real information,
    # so it survives the de-emoji as an ASCII marker rather than being dropped.
    tag = f"{prefix} {label}" if prefix else label
    return f"^{tag}" if is_upstream else tag


def _status_icon(status: BadgeStatus) -> str:
    """ASCII marker, or "" where the label alone already says it."""
    if status in (BadgeStatus.FUNCTION_CHANGED, BadgeStatus.MODULE_RELOADED):
        return "~"
    if status is BadgeStatus.WARNING:
        return "!"
    if status in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED, BadgeStatus.COMPUTED):
        return ""
    return "-"


def _status_label(status: BadgeStatus, row: StatementRow) -> str:
    if status is BadgeStatus.COMPUTED and (row.uncacheable_reasons or row.skipped_reason):
        return theme.LABEL_UNCACHEABLE
    return theme.label_of(status.value)


def _rng_suffix(row: StatementRow) -> str:
    """ASCII RNG marker for the text renderer (empty when no random effect).

    ASCII only: this lands in a kernel's stderr, where a Windows console
    codepage would mangle an emoji.
    """
    if row.random_effect == "seed":
        return "  [seed]"
    if row.random_effect == "draw" and row.random_unseeded:
        return "  [random: unseeded]"
    if row.random_effect == "draw":
        return "  [random]"
    return ""


def _row_line(row: StatementRow, *, is_upstream: bool) -> str:
    code = row.code.splitlines()[0][:theme.HEADER_MAX_LEN] if row.code else ""
    code = code + _rng_suffix(row)
    tag = _row_tag(row, is_upstream=is_upstream)
    if row.status is BadgeStatus.RESTORED:
        # "saved" == the compute we avoided (``saved_time_s``), NOT ``time_s``
        # (the tiny deserialise/restore wall-clock). Showing time_s here labelled
        # a ~0.01s restore as "saved 0.01s" for a statement whose compute — and
        # true saving — was e.g. 0.44s.
        return f"  {tag}: {code}  (saved {row.saved_time_s:.2f}s)"
    if row.status is BadgeStatus.SKIPPED:
        return f"  {tag}: {code}"
    if row.status is BadgeStatus.COMPUTED:
        return _row_line_computed(row, tag, code)
    return f"  {tag}: {code}  ({row.time_s:.2f}s)"


def _row_line_computed(row: StatementRow, tag: str, code: str) -> str:
    # The statement ran, but what it exists for may have come from the cache:
    # "EXECUTED (0.03s)" read like a suspiciously fast re-fit, with only the
    # footer saying otherwise (round 25, r25s1).
    saved = sum(c.time_s for g in (row.sub_units or ()) for c in g.calls
                if c.status is BadgeStatus.RESTORED)
    timing = f"({row.time_s:.2f}s, saved {saved:.2f}s by cached calls)" if saved else f"({row.time_s:.2f}s)"
    if row.uncacheable_reasons:
        return f"  {tag}: {code}  {timing} - {', '.join(row.uncacheable_reasons)}"
    if row.skipped_reason:
        # Shortened, not dropped: the row still says it wasn't cached and
        # why. The guard's full paragraph is emitted once per cell by
        # ``_guard_summary_lines`` instead of once per statement.
        cause = f" ({row.guard_cause})" if row.guard_cause else ""
        return f"  {tag}: {code}  {timing} - {shorten_skipped_reason(row.skipped_reason)}{cause}"
    if row.storage_tiers:
        return f"  {tag}: {code}  {timing} -> {'+'.join(row.storage_tiers)}"
    return f"  {tag}: {code}  {timing}"


def _iteration_pseudo_row(it: IterationRow) -> StatementRow:
    """A ``StatementRow`` view of *it*, for reuse of the row-rendering helpers.

    Carries ``sub_units`` through (CAS-243 task 9) -- an iteration's
    intercepted sub-calls otherwise never reach ``_sub_unit_lines``, since a
    loop-body statement is an ``IterationRow``, not a ``StatementRow``.
    """
    return StatementRow(
        status=it.status,
        code=it.code,
        time_s=it.time_s,
        saved_time_s=it.saved_time_s,
        storage_tiers=it.storage_tiers,
        # Carry the reasons across the bridge. Omitting them is what made a
        # loop-body row silent: every helper below reads them off the
        # StatementRow, so a field missing here is a field that cannot render.
        miss_reason=it.miss_reason,
        skipped_reason=it.skipped_reason,
        sub_units=it.sub_units,
    )


def _iteration_lines(it: IterationRow, pad: str, *, is_upstream: bool) -> list[str]:
    """The iteration's own line plus one ``sub-call ...`` line per call site,
    nested at the SAME indent the loop's other body lines use -- not as
    siblings of the loop (CAS-243 task 9)."""
    pseudo = _iteration_pseudo_row(it)
    return [pad + _row_line(pseudo, is_upstream=is_upstream), *_sub_unit_lines(pseudo, pad)]


def _sub_unit_lines(row: StatementRow, pad: str) -> list[str]:
    """One line per call SITE inside *row* (CAS-243 intercepted sub-calls).

    Mirrors the cell-level ``[intercepted]`` line's job at
    statement granularity: grouped by ``(call_source, occurrence_index)``,
    not by callee -- see ``SubUnitGroup`` for why. Nothing when the
    statement made no intercepted calls, and none for a site where cash did
    nothing: every call missed and none was stored -- too cheap to cache.
    ``sub-call roc_auc_score(...): 0/1 hit`` on every run of a report cell
    said only that (round 25, r25s1).
    """
    return [
        f"{pad}    sub-call {g.call_source}: "
        f"{sum(1 for c in g.calls if c.status is BadgeStatus.RESTORED)}/{len(g.calls)} hit"
        + (f", {g.ran_plain} run plain (too cheap to cache)" if getattr(g, "ran_plain", 0) else "")
        for g in row.sub_units
        if not (g.unstored and g.unstored == len(g.calls) and not g.miss_reason)
    ]


#: Leaf item types that render as exactly one line at their parent's level.
#: Anything else is a *group* and gets one extra indent step when it appears
#: inside a control body or a loop body.
_LEAF_ITEMS = (StatementRow, ControlGroupSingle)

#: One indent step, matching the two-space lead-in ``_row_line`` already emits.
_INDENT = "  "


#: Past this many passes a loop-body statement is summarised on one line.
_LOOP_ROWS_MAX = 5


def _loop_statement_summary(stmt: LoopStatement, pad: str, *, is_upstream: bool) -> list[str]:
    """One line for every pass of *stmt*, and its sub-calls across all of them.

    A row per pass put 126+ lines on the badge for 63 machines, each not-cached
    row repeating the same reason (round 25, r25s3). The counts and the
    distinct reasons are what a reader looks for; a short loop keeps its rows.
    """
    its = stmt.iterations
    restored = [it for it in its if it.status is BadgeStatus.RESTORED]
    ran = [it for it in its if it.status is not BadgeStatus.RESTORED]
    code = (stmt.base_code or (its[0].code if its else "")).splitlines()
    code = (code[0] if code else "")[:theme.HEADER_MAX_LEN]
    parts = []
    if restored:
        parts.append(f"{len(restored)} cached (saved {sum(it.saved_time_s for it in restored):.2f}s)")
    if ran:
        parts.append(f"{len(ran)} ran ({sum(it.time_s for it in ran):.2f}s)")
    reasons = list(dict.fromkeys(
        shorten_skipped_reason(it.skipped_reason) for it in ran if it.skipped_reason))
    lead = "^" if is_upstream else ""
    line = f"{pad}  {lead}LOOP x{len(its)}: {code}  - {', '.join(parts)}"
    if reasons:
        line += " - not cached: " + "; ".join(reasons[:2])
    return [line, *_sub_unit_lines(StatementRow(status=BadgeStatus.COMPUTED, code=code, time_s=0.0,
                                                sub_units=stmt.sub_units), pad)]


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
    escaped into the kernel's message handler and hung the client.
    """
    pad = _INDENT * indent
    if isinstance(item, StatementRow):
        return [pad + _row_line(item, is_upstream=is_upstream), *_sub_unit_lines(item, pad)]
    if isinstance(item, ForLoopGroup):
        out: list[str] = []
        for sub in _loop_body(item):
            if isinstance(sub, LoopStatement):
                if len(sub.iterations) > _LOOP_ROWS_MAX:
                    out.extend(_loop_statement_summary(sub, pad, is_upstream=is_upstream))
                    continue
                for it in sub.iterations:
                    out.extend(_iteration_lines(it, pad, is_upstream=is_upstream))
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
        return [pad + _row_line(item.row, is_upstream=is_upstream), *_sub_unit_lines(item.row, pad)]
    if isinstance(item, SkippedBucket):
        # One line, as the HTML badge folds it: each of these is a step the
        # repair did not need, and a row apiece put 18 ``^SKIPPED: import os``
        # style rows into a report cell's badge (round 25, r25s1).
        n = len(item.items)
        lead = "^" if is_upstream else ""
        out = [f"{pad}  {lead}{n} upstream step{'s' if n != 1 else ''} not re-run "
               f"(what they built is already current)"] if n else []
        # A write the repair left alone although its data changed: the file
        # is out of date, and "already current" would say otherwise (r28s3).
        for code, paths in item.stale_exports:
            out.append(f"{pad}  {lead}{stale_export_text(code, paths)}")
        return out
    if isinstance(item, OverheadBreakdown | DecoratorCallGroup):
        return []  # rendered separately
    return []


#: A statement that cannot be cached and ran in less than this is folded with
#: its neighbours; a slower one keeps its row and its reason.
_FOLD_BELOW_S = 0.1
#: Fewer consecutive cheap not-cached rows than this keep their rows.
_FOLD_MIN_ROWS = 3


def _foldable(item: SectionItem) -> bool:
    return (isinstance(item, StatementRow) and item.status is BadgeStatus.COMPUTED
            and bool(item.uncacheable_reasons or item.skipped_reason)
            and not is_guard_reason(item.skipped_reason)
            and not item.sub_units and item.time_s < _FOLD_BELOW_S)


def _items_lines(items, *, is_upstream: bool) -> list[str]:
    """Lines for a section's items, folding runs of cheap not-cached rows.

    ``ax.axhline(...)``, ``ax.legend()``, ``fig.savefig(...)``, ``to_csv(...)``:
    a row apiece, each with its reason, put 10 of 12 lines of a chart cell's
    badge on steps that re-run in milliseconds and must (round 25, r25s1 and
    r25s2). They cost nothing to re-run; one line names them.
    """
    out: list[str] = []
    items = list(items)
    i = 0
    while i < len(items):
        j = i
        while j < len(items) and _foldable(items[j]):
            j += 1
        if j - i >= _FOLD_MIN_ROWS:
            run = items[i:j]
            names = [(r.code or "").splitlines()[0].split("(")[0].strip()[:40] for r in run]
            shown = ", ".join(names[:6]) + (", ..." if len(names) > 6 else "")
            lead = "^" if is_upstream else ""
            out.append(f"  {lead}re-ran {len(run)} quick steps that are never cached "
                       f"({sum(r.time_s for r in run):.2f}s): {shown}")
            i = j
            continue
        out.extend(_item_lines(items[i], is_upstream=is_upstream))
        i += 1
    return out


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
        # Name the mechanism for a call cash wrapped itself: the user decorated
        # nothing, so an unlabelled entry reads as someone else's doing.
        # Deliberately not named after a directive -- interception is
        # unconditional (CAS-243 default-on), and spelling this
        # ``cache-calls`` would substring-collide with the opt-out
        # ``no-cache-calls`` in any grep/assertion over badge text.
        via = " [intercepted]" if g.intercepted else ""
        lines.append(f"    {short}(){via}: {cached}/{total} cached ({time_s:.3f}s)")
    return lines


def _iter_rows(item: SectionItem):
    """Yield every StatementRow reachable under *item*.

    Recurses through nested groups: ``ControlGroup.rows`` may hold further
    groups rather than bare ``StatementRow``s, and the callers below read
    ``.skipped_reason`` off whatever this yields.
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
    codes = [
        row.code or ""
        for section in badge.sections
        for item in section.items
        for row in _iter_rows(item)
        if is_guard_reason(row.skipped_reason)
    ]
    line = guard_summary_line(len(codes), codes)
    return [line] if line else []


def render_text(badge: InteractiveBadge) -> str:
    """Render an :class:`InteractiveBadge` to a flat plain-text summary."""
    lines = [f"[Cash] {_header_line(badge.header)}"]

    upstream_section = next(
        (s for s in badge.sections if s.kind is SectionKind.UPSTREAM),
        None,
    )
    if upstream_section is not None:
        up_lines = _items_lines(upstream_section.items, is_upstream=True)
        if up_lines:
            lines.append("  Upstream:")
            lines.extend(f"  {line}" for line in up_lines)

    current_section = next(
        (s for s in badge.sections if s.kind is SectionKind.CURRENT),
        None,
    )
    if current_section is not None:
        lines.extend(_items_lines(current_section.items, is_upstream=False))

    lines.extend(_decorator_lines(badge.sections))
    lines.extend(_guard_summary_lines(badge))
    return "\n".join(lines)
