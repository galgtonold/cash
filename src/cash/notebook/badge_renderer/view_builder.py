"""Build a BadgeView IR tree from ``ProcessResult``-shaped metric dicts.

This module is the single seam between the runtime (which produces raw
metric dicts) and the renderers (which consume :class:`BadgeView` nodes).
All status-string parsing, loop/control grouping, partitioning, and
bug-report URL construction live here — renderers never see a raw dict.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from cash.notebook.cache_status import CacheStatus

from .theme import MIN_TIME_DISPLAY_MS
from .view import (
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
    OverheadEntry,
    Section,
    SectionItem,
    SectionKind,
    SkippedBucket,
    StatementRow,
    StatusBadge,
)

_NOTIFICATION_STATUSES = frozenset({"FUNCTION_CHANGED", "MODULE_RELOADED", "WARNING"})
_RESTORED_LIKE_STATUSES = frozenset({"FUNCTION_CHANGED", "MODULE_RELOADED", "WARNING"})


# ---------------------------------------------------------------------------
# Loop / control grouping (inlined from the deprecated _grouping.py)
# ---------------------------------------------------------------------------

def _make_loop_group(base_code: str, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    all_var_names: list[str] = []
    var_values: dict[str, list[Any]] = {}
    for m in metrics:
        for k, v in (m.get("loop_vars") or {}).items():
            if k not in var_values:
                all_var_names.append(k)
                var_values[k] = []
            var_values[k].append(v)
    return {
        "type": "loop_group",
        "base_code": base_code,
        "metrics": metrics,
        "all_loop_var_names": all_var_names,
        "all_loop_var_values": var_values,
    }


def _group_loop_iterations(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two-pass grouping: per-statement loop bodies, then wrap consecutive loops.

    Returns intermediate dict items (``single``, ``for_loop_group``,
    ``control_group``, ``control_group_single``) that
    :func:`_section_item_from_grouped` translates into BadgeView nodes.
    """
    pass1: list[dict[str, Any]] = []
    loop_stmt_groups: dict[str, list[dict[str, Any]]] = {}
    control_groups: dict[str, list[dict[str, Any]]] = {}

    def _flush_loops() -> None:
        for base_code, mlist in loop_stmt_groups.items():
            pass1.append(_make_loop_group(base_code, mlist))
        loop_stmt_groups.clear()

    def _flush_controls() -> None:
        for _ctx_hash, mlist in control_groups.items():
            branch_label = mlist[0].get("branch_label", "")
            body_stmts = mlist[0].get("body_statements", [])
            header = body_stmts[0] if body_stmts else branch_label
            pass1.append({
                "type": "control_group",
                "metrics": mlist,
                "branch_label": branch_label,
                "header": header,
            })
        control_groups.clear()

    for m in metrics:
        code = m.get("code", "")
        if "# __iteration_context__:" in code:
            _flush_controls()
            actual = "\n".join(
                line for line in code.split("\n")
                if not line.startswith("# __iteration_context__:")
                and not line.startswith("# control_context:")
            )
            loop_stmt_groups.setdefault(actual, []).append(m)
        elif m.get("control_context"):
            _flush_loops()
            control_groups.setdefault(m["control_context"], []).append(m)
        else:
            _flush_loops()
            _flush_controls()
            if m.get("body_statements") and not m.get("control_context"):
                pass1.append({"type": "control_group_single", "metric": m})
            else:
                pass1.append({"type": "single", "metric": m})
    _flush_loops()
    _flush_controls()

    # Pass 2: wrap consecutive loop_group items into a single for_loop_group.
    result: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def _flush_pending() -> None:
        if pending:
            result.append({"type": "for_loop_group", "stmt_groups": list(pending)})
            pending.clear()

    for item in pass1:
        if item["type"] == "loop_group":
            pending.append(item)
        else:
            _flush_pending()
            result.append(item)
    _flush_pending()
    return result


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

def map_status(raw: Any) -> BadgeStatus:
    """Map a runtime status (``CacheStatus`` or raw string) to a :class:`BadgeStatus`.

    Total function: any unknown input collapses to :attr:`BadgeStatus.WARNING`
    so renderers never crash on a malformed metric.
    """
    if isinstance(raw, CacheStatus):
        raw = raw.value
    key = str(raw or "").upper()
    if key == "RESTORED":
        return BadgeStatus.RESTORED
    if key == "COMPUTED":
        return BadgeStatus.COMPUTED
    if key == "SKIPPED":
        return BadgeStatus.SKIPPED
    if key == "ERROR":
        return BadgeStatus.ERROR
    if key == "FUNCTION_CHANGED":
        return BadgeStatus.FUNCTION_CHANGED
    if key == "MODULE_RELOADED":
        return BadgeStatus.MODULE_RELOADED
    return BadgeStatus.WARNING


def _summary_status(restored: int, computed: int, skipped: int) -> BadgeStatus:
    """Decide the top-level :attr:`BadgeHeader.status` from per-row counts."""
    if computed == 0 and (restored > 0 or skipped > 0):
        return BadgeStatus.RESTORED if restored > 0 else BadgeStatus.SKIPPED
    if restored > 0 and computed > 0:
        return BadgeStatus.MIXED
    return BadgeStatus.COMPUTED


# ---------------------------------------------------------------------------
# Cell-level statistics
# ---------------------------------------------------------------------------

def _compute_stats(metrics: list[dict[str, Any]]) -> tuple[float, float, int, int, int]:
    """``(total_saved, total_exec, restored_count, computed_count, skipped_count)``."""
    total_saved = 0.0
    total_exec = 0.0
    restored = computed = skipped = 0
    for m in metrics:
        status = str(m.get("status", ""))
        if status in _NOTIFICATION_STATUSES:
            continue
        if status == str(CacheStatus.RESTORED):
            restored += 1
            total_saved += m.get("saved_time", 0.0)
            total_exec += m.get("total_time", 0.0)
        elif status == str(CacheStatus.SKIPPED):
            skipped += 1
            total_saved += m.get("saved_time", 0.0)
        elif m.get("is_upstream", False) or status == str(CacheStatus.COMPUTED):
            computed += 1
            total_exec += m.get("total_time", 0.0)
    return total_saved, total_exec, restored, computed, skipped


# ---------------------------------------------------------------------------
# Helpers — pull tuples out of dict metrics safely
# ---------------------------------------------------------------------------

def _tup_str(seq: Any) -> tuple[str, ...]:
    if isinstance(seq, list | tuple):
        return tuple(str(x) for x in seq)
    if seq is None:
        return ()
    return (str(seq),)


def _strip_context_comments(code: str) -> str:
    if "# __iteration_context__:" not in code and "# control_context:" not in code:
        return code
    return "\n".join(
        line for line in code.split("\n")
        if not line.startswith("# __iteration_context__:")
        and not line.startswith("# control_context:")
    )


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _statement_row_from_metric(m: dict[str, Any], *, is_upstream: bool = False) -> StatementRow:
    """Translate one metric dict into a :class:`StatementRow`."""
    status = map_status(m.get("status"))

    time_s = float(m.get("total_time", 0.0) or m.get("execution_time", 0.0))
    saved_time_s = float(m.get("saved_time", 0.0) or 0.0)

    output_vars = _tup_str(m.get("evaluated_vars") or m.get("output_vars") or m.get("outputs"))
    restored_vars = _tup_str(m.get("restored_vars"))

    raw_decorator = m.get("decorator_calls", []) or []
    dec_calls = tuple(
        DecoratorCall(
            func_name=str(c.get("func_name", "?")),
            status=BadgeStatus.RESTORED if c.get("cache_hit") else BadgeStatus.COMPUTED,
            time_s=float(c.get("execution_time", 0.0)),
        )
        for c in raw_decorator
    )

    changed_modules = m.get("changed_modules") or {}
    if isinstance(changed_modules, dict):
        changed_modules_tup = tuple(sorted(str(k) for k in changed_modules.keys()))
    else:
        changed_modules_tup = _tup_str(changed_modules)

    return StatementRow(
        status=status,
        code=_strip_context_comments(str(m.get("code", ""))),
        time_s=time_s,
        saved_time_s=saved_time_s,
        storage_tiers=_tup_str(m.get("storage")),
        source=m.get("source") or None,
        output_vars=output_vars,
        restored_vars=restored_vars,
        uncacheable_reasons=_tup_str(m.get("uncacheable_reasons")),
        skipped_reason=m.get("skipped_reason") or None,
        is_upstream=is_upstream,
        changed_functions=_tup_str(m.get("changed_functions")),
        changed_modules=changed_modules_tup,
        decorator_calls=dec_calls,
    )


def _iteration_row(m: dict[str, Any]) -> IterationRow:
    loop_vars = m.get("loop_vars") or {}
    bindings = tuple((str(k), v) for k, v in loop_vars.items()) if isinstance(loop_vars, dict) else ()
    return IterationRow(
        status=map_status(m.get("status")),
        code=_strip_context_comments(str(m.get("code", ""))),
        time_s=float(m.get("total_time", 0.0)),
        saved_time_s=float(m.get("saved_time", 0.0) or 0.0),
        storage_tiers=_tup_str(m.get("storage")),
        loop_bindings=bindings,
    )


# ---------------------------------------------------------------------------
# Grouped-item translation
# ---------------------------------------------------------------------------

def _section_item_from_grouped(item: dict[str, Any], *, is_upstream: bool) -> SectionItem:
    """Translate one ``group_loop_iterations`` intermediate dict into a node."""
    kind = item["type"]
    if kind == "single":
        return _statement_row_from_metric(item["metric"], is_upstream=is_upstream)

    if kind == "for_loop_group":
        stmt_groups = item.get("stmt_groups", [])
        loop_var_names = tuple(stmt_groups[0].get("all_loop_var_names", [])) if stmt_groups else ()
        stmts = tuple(
            LoopStatement(
                base_code=str(sg.get("base_code", "")),
                iterations=tuple(_iteration_row(m) for m in sg.get("metrics", [])),
            )
            for sg in stmt_groups
        )
        return ForLoopGroup(loop_var_names=loop_var_names, stmts=stmts)

    if kind == "control_group":
        return ControlGroup(
            branch_label=str(item.get("branch_label", "")),
            header=str(item.get("header", "")),
            rows=tuple(_statement_row_from_metric(m, is_upstream=is_upstream)
                       for m in item.get("metrics", [])),
        )

    if kind == "control_group_single":
        return ControlGroupSingle(
            row=_statement_row_from_metric(item["metric"], is_upstream=is_upstream),
        )

    raise ValueError(f"Unknown grouped-item type: {kind!r}")


def _skipped_bucket(skipped_metrics: list[dict[str, Any]]) -> SkippedBucket | None:
    if not skipped_metrics:
        return None
    total_saved = sum(float(m.get("saved_time", 0.0)) for m in skipped_metrics)
    grouped = _group_loop_iterations(skipped_metrics)
    items: list[StatementRow | ForLoopGroup] = []
    for g in grouped:
        node = _section_item_from_grouped(g, is_upstream=True)
        # SkippedBucket only carries StatementRow | ForLoopGroup per the IR.
        # Control-group skipped items are rendered as their single row for the bucket.
        if isinstance(node, StatementRow | ForLoopGroup):
            items.append(node)
        elif isinstance(node, ControlGroupSingle):
            items.append(node.row)
        elif isinstance(node, ControlGroup):
            items.extend(node.rows)
    return SkippedBucket(items=tuple(items), total_saved_time_s=total_saved)


# ---------------------------------------------------------------------------
# Decorator translation
# ---------------------------------------------------------------------------

_CONDENSE_THRESHOLD = 3

def _decorator_groups(metrics: list[dict[str, Any]]) -> tuple[DecoratorCallGroup, ...]:
    raw_calls: list[dict[str, Any]] = []
    for m in metrics:
        raw_calls.extend(m.get("decorator_calls", []) or [])
    if not raw_calls:
        return ()
    by_func: dict[str, list[dict[str, Any]]] = {}
    for c in raw_calls:
        by_func.setdefault(c.get("func_name", "?"), []).append(c)
    groups: list[DecoratorCallGroup] = []
    for func_name, calls in by_func.items():
        dc_calls = tuple(
            DecoratorCall(
                func_name=str(func_name),
                status=BadgeStatus.RESTORED if c.get("cache_hit") else BadgeStatus.COMPUTED,
                time_s=float(c.get("execution_time", 0.0)),
            )
            for c in calls
        )
        groups.append(DecoratorCallGroup(
            func_name=str(func_name),
            calls=dc_calls,
            condensed=len(dc_calls) > _CONDENSE_THRESHOLD,
        ))
    return tuple(groups)


# ---------------------------------------------------------------------------
# Overhead translation
# ---------------------------------------------------------------------------

_OVERHEAD_LABELS = {
    "upstream_check": ("↻ Upstream check", "Lineage simulation (excl. restore)"),
    "badge_init": ("🏷️ Badge init", "Initial badge render"),
    "badge_progress": ("📊 Progress updates", "Badge progress renders"),
}

def _overhead_section(
    timing_breakdown: dict[str, float] | None,
    cell_total_time: float | None,
    metrics: list[dict[str, Any]],
) -> Section | None:
    if not timing_breakdown or cell_total_time is None:
        return None
    statements_time = sum(float(m.get("total_time", 0.0)) for m in metrics)
    overhead = cell_total_time - statements_time
    if overhead <= MIN_TIME_DISPLAY_MS:
        return None

    upstream_check = float(timing_breakdown.get("upstream_check", 0.0))
    badge_init = float(timing_breakdown.get("badge_init", 0.0))
    badge_progress = float(timing_breakdown.get("badge_progress", 0.0))
    other = overhead - (badge_init + upstream_check + badge_progress)

    entries: list[OverheadEntry] = []
    for key, value in (
        ("upstream_check", upstream_check),
        ("badge_init", badge_init),
        ("badge_progress", badge_progress),
    ):
        if value > MIN_TIME_DISPLAY_MS:
            label, _detail = _OVERHEAD_LABELS[key]
            entries.append(OverheadEntry(label=label, time_s=value))
    if other > MIN_TIME_DISPLAY_MS:
        entries.append(OverheadEntry(label="⚙️ Other", time_s=other))
    if not entries:
        return None
    return Section(
        kind=SectionKind.OVERHEAD,
        header="OVERHEAD",
        items=(OverheadBreakdown(entries=tuple(entries), total_s=overhead),),
    )


# ---------------------------------------------------------------------------
# Bug-report URL
# ---------------------------------------------------------------------------

_ISSUES_BASE = "https://github.com/galgtonold/cash/issues/new"
_BUG_URL_MAX = 7800

def build_bug_report_url(metrics: list[dict[str, Any]], context: dict | None = None) -> str:
    """Build a pre-filled GitHub issue URL — moved verbatim from ``_badge.py``."""
    ctx = context or {}
    version = ctx.get("version", "unknown")
    python_version = ctx.get("python_version", "(unknown)")
    backend = ctx.get("backend", "(unknown)")
    notebook_source: list[str] = ctx.get("notebook_source", [])

    badge_lines: list[str] = []
    for m in metrics:
        if m.get("is_upstream"):
            continue
        code = _strip_context_comments(str(m.get("code") or "")).strip()
        if len(code) > 100:
            code = code[:97] + "..."
        st = str(m.get("status", "")).replace("CacheStatus.", "")
        t = m.get("total_time") or m.get("execution_time") or 0.0
        saved = m.get("saved_time") or 0.0
        outs_raw = m.get("output_vars", []) or m.get("outputs", [])
        outs = ", ".join(o for o in (outs_raw or []) if isinstance(o, str))
        outs_str = f" | {outs}" if outs else ""
        badge_lines.append(f"  {st:>8} | {t:>6.3f}s | saved {saved:>6.3f}s | {code}{outs_str}")
    badge_text = "\n".join(badge_lines) if badge_lines else "(no metrics)"

    def _nb_source(max_chars: int) -> str:
        if not notebook_source:
            return ""
        parts = []
        for i, cell in enumerate(notebook_source, 1):
            snippet = cell.strip()
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars - 3] + "..."
            parts.append(f"# --- Cell {i} ---\n{snippet}")
        return (
            "<details><summary>Notebook source</summary>\n\n```python\n"
            + "\n\n".join(parts)
            + "\n```\n\n</details>\n\n"
        )

    skeleton = (
        "**Describe the incorrect behavior:**\n"
        "<!-- What did cash do wrong? What did you expect instead? -->\n\n"
        "**Cash badge output:**\n```\n{badge}\n```\n\n"
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
    prefix = f"{_ISSUES_BASE}?title=Incorrect+caching+behavior&labels=bug%2Ccaching-behavior&body="

    def _url_len(body: str) -> int:
        return len(prefix) + len(quote(body))

    for nb_chars in (300, 150):
        body = skeleton.format(badge=badge_text, nb_source=_nb_source(nb_chars))
        if _url_len(body) <= _BUG_URL_MAX:
            return prefix + quote(body)

    truncated_note = (
        "> **Note:** Notebook source was too large to include. "
        "Please paste the relevant cells below.\n\n"
    )
    body = skeleton.format(badge=badge_text, nb_source=truncated_note)
    if _url_len(body) <= _BUG_URL_MAX:
        return prefix + quote(body)

    if len(badge_text) > 500:
        badge_text = badge_text[:500] + "\n  ... (truncated)"
    body = skeleton.format(badge=badge_text, nb_source=truncated_note)
    return prefix + quote(body)


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------

def build_interactive_badge(
    metrics_list: list[dict[str, Any]],
    *,
    status: str = "DONE",
    current_step: int = 0,
    total_steps: int = 0,
    current_code: str | None = None,
    cell_total_time: float | None = None,
    timing_breakdown: dict[str, float] | None = None,
    bug_report_context: dict | None = None,
) -> InteractiveBadge:
    """Build an :class:`InteractiveBadge` view from a metrics list."""
    metrics = metrics_list or []

    upstream_all = [m for m in metrics if m.get("is_upstream", False)
                    and str(m.get("status")) != str(CacheStatus.SKIPPED)]
    upstream_restored_like = [
        m for m in upstream_all
        if str(m.get("status")) in {str(CacheStatus.RESTORED), *_RESTORED_LIKE_STATUSES}
    ]
    upstream_executed = [m for m in upstream_all if m not in upstream_restored_like]
    upstream_skipped = [m for m in metrics if str(m.get("status")) == str(CacheStatus.SKIPPED)
                        and m.get("is_upstream", False)]
    current = [m for m in metrics if not m.get("is_upstream", False)]

    total_saved, total_exec, restored, computed, skipped_count = _compute_stats(metrics)
    summary_time = cell_total_time if cell_total_time is not None else total_exec
    header = BadgeHeader(
        status=BadgeStatus.WARNING if status == "RUNNING" else _summary_status(restored, computed, skipped_count),
        restored_count=restored,
        computed_count=computed,
        skipped_count=skipped_count,
        total_saved_s=total_saved,
        total_exec_s=summary_time,
        current_step=current_step,
        total_steps=total_steps,
        current_code=current_code,
    )

    sections: list[Section] = []

    if upstream_all or upstream_skipped:
        items: list[SectionItem] = []
        for g in _group_loop_iterations(upstream_restored_like):
            items.append(_section_item_from_grouped(g, is_upstream=True))
        for g in _group_loop_iterations(upstream_executed):
            items.append(_section_item_from_grouped(g, is_upstream=True))
        bucket = _skipped_bucket(upstream_skipped)
        if bucket is not None:
            items.append(bucket)
        sections.append(Section(
            kind=SectionKind.UPSTREAM,
            header="UPSTREAM HISTORY",
            items=tuple(items),
        ))

    current_items: list[SectionItem] = []
    for g in _group_loop_iterations(current):
        current_items.append(_section_item_from_grouped(g, is_upstream=False))
    sections.append(Section(
        kind=SectionKind.CURRENT,
        header="CURRENT CELL" if (upstream_all or upstream_skipped) else "",
        items=tuple(current_items),
    ))

    dec_groups = _decorator_groups(current)
    if dec_groups:
        sections.append(Section(
            kind=SectionKind.DECORATORS,
            header="DECORATOR CACHE (@cash.cache)",
            items=dec_groups,
        ))

    overhead = _overhead_section(timing_breakdown, cell_total_time, metrics)
    if overhead is not None:
        sections.append(overhead)

    footer = BugReportLink(url=build_bug_report_url(metrics, bug_report_context))
    return InteractiveBadge(header=header, sections=tuple(sections), footer=footer)


def build_status_badge(
    *,
    status: Any,
    execution_time: float,
    time_saved: float = 0.0,
    source: str | None = None,
    storage: Any = (),
) -> StatusBadge:
    """Build a :class:`StatusBadge` (the compact non-interactive pill)."""
    return StatusBadge(
        status=map_status(status),
        execution_time_s=float(execution_time),
        time_saved_s=float(time_saved),
        source=source,
        storage_tiers=_tup_str(storage),
    )


__all__ = [
    "build_interactive_badge",
    "build_status_badge",
    "build_bug_report_url",
    "map_status",
]
