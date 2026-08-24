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

from .theme import MIN_TIME_DISPLAY_MS, label_of
from .view import (
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
    Section,
    SectionItem,
    SectionKind,
    SkippedBucket,
    StatementRow,
    build_sub_unit_groups,
)

# Statuses that don't represent a "ran" or "restored" data operation. They
# are surfaced as warning rows in the badge but excluded from restored /
# computed / skipped counts and partitioned alongside restored rows for
# the upstream-context split (they don't carry their own work, they
# annotate other work).
_NOTIFICATION_STATUSES = frozenset({"FUNCTION_CHANGED", "MODULE_RELOADED", "WARNING"})


# ---------------------------------------------------------------------------
# Loop / control grouping
# ---------------------------------------------------------------------------

def _make_loop_group(base_code: str, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    # Collect ALL loop vars present in the metrics, in order. The runtime
    # records ``loop_vars = {outer: ..., inner: ...}`` on the innermost
    # statement's metrics, and these are exactly the variables the
    # per-iteration drill-down should display. The synthesised loop
    # header (when no source ``loop_header`` is available) uses only the
    # outermost; the iter drill-down can show all of them at once.
    all_var_names: list[str] = []
    var_values: dict[str, list[Any]] = {}
    for m in metrics:
        lv = m.get("loop_vars") or {}
        if not isinstance(lv, dict):
            continue
        for k, v in lv.items():
            if k not in var_values:
                all_var_names.append(k)
                var_values[k] = []
            var_values[k].append(v)
    # Source-faithful header from the runtime (e.g. "for cat in df['category'].unique():").
    # Picked from the first metric that carries it — the runtime stamps every
    # body metric of the same loop with the same header.
    loop_header = next(
        (str(m["loop_header"]) for m in metrics if m.get("loop_header")),
        "",
    )
    # Full enclosing chain (outermost → innermost). Same across all metrics
    # in a loop_group since they share base_code and thus position.
    chain = next(
        (tuple(m["loop_header_chain"]) for m in metrics if m.get("loop_header_chain")),
        (loop_header,) if loop_header else (),
    )
    # Source-order position inside the enclosing for-loop's body, recorded
    # by the runtime as ``body_index``. Used by the view-builder to sort
    # body stmts and nested controls in source order.
    body_index = min(
        (int(m["body_index"]) for m in metrics if "body_index" in m),
        default=10_000,
    )
    return {
        "type": "loop_group",
        "base_code": base_code,
        "metrics": metrics,
        "all_loop_var_names": all_var_names,
        "all_loop_var_values": var_values,
        "loop_header": loop_header,
        "loop_header_chain": chain,
        "body_index": body_index,
    }


def _group_loop_iterations(
    metrics: list[dict[str, Any]],
    *,
    synthesize_intermediate_loops: bool = True,
) -> list[dict[str, Any]]:
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
        # Pre-merge: if the same inner control fires N times across outer
        # loop iterations, each firing arrives with a distinct ``ctx`` hash
        # but the same ``branch_label`` (the if/elif/else/for line). Combine
        # those into one bucket so the renderer shows a single control with
        # an inner for-loop body, not N sibling control_groups. Buckets
        # without a branch_label (rare: synthetic metrics) stay keyed by ctx.
        merged: dict[str, list[dict[str, Any]]] = {}
        for ctx_hash, mlist in control_groups.items():
            branch_label = next(
                (str(m.get("branch_label")) for m in mlist if m.get("branch_label")),
                "",
            )
            key = branch_label or ctx_hash
            merged.setdefault(key, []).extend(mlist)
        for _key, mlist in merged.items():
            branch_label = next(
                (str(m.get("branch_label")) for m in mlist if m.get("branch_label")),
                "",
            )
            body_stmts = next(
                (m.get("body_statements") for m in mlist if m.get("body_statements")),
                [],
            )
            header = body_stmts[0] if body_stmts else branch_label
            # Recursively group the body so nested loops or nested
            # controls within this branch get their own grouping (not
            # flattened into a row-per-iteration spam). Strip the
            # control_context off the inner metrics so the recursion
            # doesn't re-collect them at this same level.
            inner_metrics = [
                {k: v for k, v in m.items() if k != "control_context"}
                for m in mlist
            ]
            # Synthesis inside a control's body is enabled iff the
            # control's body actually CONTAINS for-loops (T10:
            # ``if cond: for i: for j: body`` — the for-loops are in the
            # if-body, so synthesising for-i around the for-j inside is
            # correct). For T6 (``for c: if cond: body``) the chain's
            # for-loops are OUTSIDE the control; synthesising them inside
            # would duplicate the outer loops, so we disable.
            body_haystack = "\n".join(
                str(s)
                for m in mlist
                for s in (m.get("body_statements") or [])
            )
            chain_headers = []
            for m in mlist:
                ch = m.get("loop_header_chain")
                if ch:
                    chain_headers = list(ch)
                    break
            synth_inside = bool(chain_headers and any(
                h in body_haystack for h in chain_headers
            ))
            sub_items = _group_loop_iterations(
                inner_metrics, synthesize_intermediate_loops=synth_inside,
            )
            body_idx = min(
                (int(m["body_index"]) for m in mlist if "body_index" in m),
                default=10_000,
            )
            pass1.append({
                "type": "control_group",
                "metrics": mlist,
                "sub_items": sub_items,
                "branch_label": branch_label,
                "header": header,
                "body_index": body_idx,
            })
        control_groups.clear()

    for m in metrics:
        code = m.get("code", "")
        ctx = m.get("control_context")
        has_iter = "# __iteration_context__:" in code
        if ctx:
            # Control context wins over iteration context — a loop *inside*
            # an if/else gets bucketed under the control, then recursively
            # sub-grouped at flush-time so the loop still looks like a loop.
            # BUT: if this control is itself nested inside a loop (has_iter),
            # don't flush the outer loop's stmt_groups — otherwise the outer
            # for-loop gets split into two halves around the control, one
            # for iterations where the control's true-branch ran and one
            # for iterations where it didn't.
            if not has_iter:
                _flush_loops()
            control_groups.setdefault(ctx, []).append(m)
        elif has_iter:
            _flush_controls()
            actual = "\n".join(
                line for line in code.split("\n")
                if not line.startswith("# __iteration_context__:")
                and not line.startswith("# control_context:")
            )
            loop_stmt_groups.setdefault(actual, []).append(m)
        else:
            _flush_loops()
            _flush_controls()
            if m.get("body_statements"):
                pass1.append({"type": "control_group_single", "metric": m})
            else:
                pass1.append({"type": "single", "metric": m})
    _flush_loops()
    _flush_controls()

    # Pass 1.5: merge control_groups across the whole pass1 list that share
    # the same branch_label. They represent the same inner control
    # structure fired across different outer-loop iterations — each firing
    # got flushed separately (often interleaved with sibling body
    # statements that triggered _flush_controls), so without this pass T6
    # ends up with N sibling if-blocks instead of one.
    merged_pass1: list[dict[str, Any]] = []
    by_label: dict[str, int] = {}
    for item in pass1:
        if item.get("type") == "control_group" and item.get("branch_label"):
            label = str(item["branch_label"])
            if label in by_label:
                target = merged_pass1[by_label[label]]
                target["metrics"].extend(item["metrics"])
                inner = [
                    {k: v for k, v in m.items() if k != "control_context"}
                    for m in target["metrics"]
                ]
                # Synthesis is disabled for control-body re-grouping —
                # same rationale as _flush_controls above (T6 duplication).
                target["sub_items"] = _group_loop_iterations(
                    inner, synthesize_intermediate_loops=False,
                )
                continue
            by_label[label] = len(merged_pass1)
        merged_pass1.append(item)
    pass1 = merged_pass1

    # Pass 2: wrap consecutive loop_group items into a single for_loop_group.
    # When loop_groups have DIFFERENT loop_headers (e.g. nested
    # `for cat:` body stmts vs `for win:` body stmts in T2) they belong to
    # different for-loops and must be flushed into separate for_loop_groups
    # so each gets its own source-faithful header.
    pass2: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def _flush_pending() -> None:
        if pending:
            chain = next(
                (p.get("loop_header_chain", ()) for p in pending if p.get("loop_header_chain")),
                (),
            )
            pass2.append({
                "type": "for_loop_group",
                "stmt_groups": list(pending),
                "loop_header_chain": tuple(chain),
            })
            pending.clear()

    def _pending_header() -> str:
        return next((p.get("loop_header", "") for p in pending if p.get("loop_header")), "")

    for item in pass1:
        if item["type"] == "loop_group":
            item_header = item.get("loop_header", "")
            existing = _pending_header()
            if pending and existing and item_header and existing != item_header:
                _flush_pending()
            pending.append(item)
        else:
            _flush_pending()
            pass2.append(item)
    _flush_pending()

    # Pass 2.5: hoist for-loops that surround a control_group out of the
    # control's sub_items. When a control_group has chain (A, B, C) and
    # its sub_items contain a for_loop_group with the same chain, that
    # for_loop_group is the for-C loop that SURROUNDS the if (in source,
    # `for c in range(2): if cond: body`). Without hoisting, the renderer
    # would draw the if first and the for-c inside it (inverted). After
    # hoisting, the for-c is the outer wrapper with the control as its
    # nested child and the inner stmt rows marked head-suppressed so
    # they render as the if-body without a duplicate "for c" header.
    #
    # SKIP hoisting if a real for_loop_group with the same chain already
    # exists at the top level — in T13 (`for x: stmt; if cond: stmt; stmt`)
    # the real for-x carries the body stmts and pass 3 will nest the
    # control under it. Synthesising a duplicate outer would split the
    # for-x in two.
    existing_chains = {
        _item_chain(s) for s in pass2
        if s.get("type") == "for_loop_group" and _item_chain(s)
    }
    new_pass2: list[dict[str, Any]] = []
    for item in pass2:
        if (
            item.get("type") == "control_group"
            and _item_chain(item) in existing_chains
        ):
            # A real for_loop_group with this chain already exists as a
            # sibling — pass 3 will nest the control under it. Mark any
            # for_loop_group inside the control's sub_items with the same
            # chain as head-suppressed so we don't draw a duplicate
            # "for x in ...:" header inside the control body (T13).
            ctrl_chain = _item_chain(item)
            for sub in item.get("sub_items", []):
                if (sub.get("type") == "for_loop_group"
                        and _item_chain(sub) == ctrl_chain):
                    sub["_suppress_head"] = True
            new_pass2.append(item)
        else:
            new_pass2.append(_hoist_outer_for_loop(item))
    pass2 = new_pass2

    # Pass 3: nest by loop_header_chain so deep loops actually sit inside
    # their parents instead of rendering as siblings. A for_loop_group
    # with chain (A, B, C) belongs inside the for_loop_group with chain
    # (A, B); a control_group with chain (A, B, C) belongs inside the
    # for_loop_group at chain (A, B, C) (its hoisted outer-for parent).
    return _nest_by_chain(pass2, synthesize=synthesize_intermediate_loops)


def _hoist_outer_for_loop(item: dict[str, Any]) -> dict[str, Any]:
    """If ``item`` is a control_group whose sub_items contain a for_loop_group
    with the same chain (= the loop that SURROUNDS this control in source),
    swap them: the for-loop becomes the outer wrapper, this control becomes
    its nested child, and the original inner for-loop is marked
    ``_suppress_head`` so its iteration rows render as the control's body
    without a duplicate for-loop header.

    Heuristic to tell T6 (``for c: if cond: body``) from T10
    (``if cond: for i: for j: body``): if the wrapper's loop_header appears
    as a substring in the control's recorded ``body_statements``, the
    for-loop is INSIDE the control (T10) — don't hoist. Otherwise the
    for-loop wraps the control (T6) — hoist.
    """
    if item.get("type") != "control_group":
        return item
    ctrl_chain = _item_chain(item)
    if not ctrl_chain:
        return item
    sub_items = item.get("sub_items", [])
    wrapper_idx = next(
        (i for i, s in enumerate(sub_items)
         if s.get("type") == "for_loop_group" and _item_chain(s) == ctrl_chain),
        None,
    )
    if wrapper_idx is None:
        return item
    wrapper = sub_items[wrapper_idx]
    wrapper_header = wrapper.get("loop_header") or (
        wrapper["stmt_groups"][0].get("loop_header", "") if wrapper.get("stmt_groups") else ""
    )
    # If the wrapper's for-line appears in the control's recorded body
    # source, the for-loop is INSIDE the control (e.g. T10) — keep it
    # nested inside and skip hoisting.
    if wrapper_header:
        body_stmts = []
        for m in item.get("metrics", []):
            bs = m.get("body_statements") or []
            body_stmts.extend(str(s) for s in bs)
        haystack = "\n".join(body_stmts)
        if wrapper_header in haystack:
            return item
    others = [s for i, s in enumerate(sub_items) if i != wrapper_idx]
    inner = dict(wrapper)
    inner["_suppress_head"] = True
    item["sub_items"] = others + [inner]
    # Outer wrapper carries the loop_header and chain but has no direct
    # stmt_groups — the iteration data lives in the inner, head-suppressed
    # copy now nested under the control.
    return {
        "type": "for_loop_group",
        "stmt_groups": [],
        "loop_header": wrapper.get("loop_header") or (
            wrapper["stmt_groups"][0].get("loop_header", "") if wrapper.get("stmt_groups") else ""
        ),
        "loop_header_chain": list(ctrl_chain),
        "nested": [item],
        "_synthetic_outer": True,
    }


def _item_chain(item: dict[str, Any]) -> tuple[str, ...]:
    if item.get("type") == "for_loop_group":
        return tuple(item.get("loop_header_chain", ()))
    if item.get("type") == "control_group":
        metrics = item.get("metrics", [])
        ch: tuple[str, ...] = ()
        for m in metrics:
            mch = m.get("loop_header_chain")
            if mch:
                ch = tuple(mch)
                break
        if not ch:
            return ()
        # If the chain's for-loops appear in the control's recorded
        # body_statements, those for-loops are INSIDE the control's body,
        # not the loops surrounding the control. The control then has no
        # enclosing for-loop at the cell level — return empty chain so
        # outer nesting/synthesis doesn't wrap it in those loops.
        body_stmts: list[str] = []
        for m in metrics:
            for s in (m.get("body_statements") or []):
                body_stmts.append(str(s))
        haystack = "\n".join(body_stmts)
        if haystack and any(header in haystack for header in ch):
            return ()
        return ch
    return ()


def _nest_by_chain(items: list[dict[str, Any]], *, synthesize: bool = True) -> list[dict[str, Any]]:
    """Convert a flat list of for_loop_groups + controls into a nested tree
    using each item's enclosing-loop chain. Items with empty chain stay at
    the top level. When the chain skips levels (e.g. T10's
    ``if cond: for i: for j: body`` has metrics with chain (for-i, for-j)
    but no for_loop_group for ``for-i`` because its body is only a control
    structure), synthesize the missing intermediate for_loop_groups so
    every level of the chain has its own header in the rendered tree.

    Synthetics are inserted at the position of their first child so the
    rendered order matches source order (the missing outer wrap doesn't
    leap-frog earlier siblings)."""
    # Index existing real for_loop_groups by their chain.
    by_chain: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in items:
        if item.get("type") == "for_loop_group":
            ch = _item_chain(item)
            if ch:
                by_chain[ch] = item
                item.setdefault("nested", [])

    result: list[dict[str, Any]] = []
    for item in items:
        chain = _item_chain(item)
        if not chain:
            result.append(item)
            continue
        if item.get("type") == "for_loop_group":
            parent_chain = chain[:-1]
        else:
            # control_group, etc. live INSIDE their innermost enclosing
            # for-loop — same chain.
            parent_chain = chain

        # Walk up the parent chain; for each prefix without an existing
        # for_loop_group, synthesize one (when allowed). Build the
        # synthetic chain outermost → innermost so we can nest them
        # correctly.
        synth_chain: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        pc = parent_chain
        while pc:
            if pc in by_chain and by_chain[pc] is not item:
                break
            if pc in by_chain and by_chain[pc] is item:
                pc = pc[:-1]
                continue
            if not synthesize:
                break
            synth = {
                "type": "for_loop_group",
                "stmt_groups": [],
                "loop_header": pc[-1],
                "loop_header_chain": list(pc),
                "nested": [],
                "_synthetic_outer": True,
            }
            by_chain[pc] = synth
            synth_chain.append((pc, synth))
            pc = pc[:-1]

        # synth_chain is innermost-first; reverse to outermost-first.
        synth_chain.reverse()

        # The outermost element to attach: either the first synthetic
        # (if any were created) or the item itself.
        outermost = synth_chain[0][1] if synth_chain else item

        # Where does the outermost attach? Walk up from outermost's
        # parent_chain to find an existing real for-loop ancestor.
        outermost_parent_chain = (
            synth_chain[0][0][:-1] if synth_chain
            else parent_chain
        )
        outer_parent = None
        ppc = outermost_parent_chain
        while ppc:
            if ppc in by_chain and by_chain[ppc] is not outermost:
                outer_parent = by_chain[ppc]
                break
            ppc = ppc[:-1]

        # Chain the synthetics: outermost.nested = [next] = ... = [innermost]
        for i in range(len(synth_chain) - 1):
            synth_chain[i][1]["nested"].append(synth_chain[i + 1][1])
        # Finally attach the item to the innermost synthetic (if any).
        if synth_chain:
            synth_chain[-1][1]["nested"].append(item)

        # Place outermost in the right spot.
        if outer_parent is not None:
            outer_parent["nested"].append(outermost)
        else:
            # If we created synthetics, the OUTERMOST takes the slot;
            # the item itself moves into the synthetic tree.
            result.append(outermost)
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

def _compute_stats(
    metrics: list[dict[str, Any]],
) -> tuple[float, float, int, int, int, int]:
    """``(total_saved, total_exec, restored, computed, skipped, uncacheable)``.

    ERROR rows are counted as computed (they consumed time and ran the
    operation, just unsuccessfully) so the cell summary reflects that
    something happened instead of silently dropping the failure.

    ``uncacheable`` is a SUBSET of ``computed``, not a fourth bucket: those
    rows really did run, so they belong in the exec tally. It is counted
    separately only so the header can say that some of that work will be paid
    for again on every future run — see ``BadgeHeader.uncacheable_count``.
    """
    total_saved = 0.0
    total_exec = 0.0
    restored = computed = skipped = uncacheable = 0
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
        elif (m.get("is_upstream", False)
                or status == str(CacheStatus.COMPUTED)
                or status == str(CacheStatus.ERROR)):
            computed += 1
            total_exec += m.get("total_time", 0.0)
            # Same test the renderers use to label a row NOT CACHED, so the
            # header count can never disagree with the rows below it.
            if m.get("uncacheable_reasons") or m.get("skipped_reason"):
                uncacheable += 1
    return total_saved, total_exec, restored, computed, skipped, uncacheable


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

def _statement_display_time(m: dict[str, Any]) -> float:
    """The wall-time to show for a statement row.

    For a COMPUTED statement this is the pure compute (``execution_time``), NOT
    ``total_time``: total_time also includes cash's OWN per-statement overhead
    (hashing + serialising the result into the cache), which can dwarf the
    compute for a large object — a DataFrame that computes in 0.01s but takes
    0.08s to serialise showed "0.09s" here while crediting only "saved 0.01s" on
    restore (saved == the compute you avoid, by design). Showing the compute
    keeps the executed time consistent with the saved time; the serialisation
    cost is attributed to the overhead section instead so the breakdown still
    sums to the cell's wall-clock total. For a RESTORED statement execution_time
    is 0, so this falls back to total_time — the restore/deserialise time.
    """
    return float(m.get("execution_time", 0.0) or m.get("total_time", 0.0))


def _statement_row_from_metric(m: dict[str, Any]) -> StatementRow:
    """Translate one metric dict into a :class:`StatementRow`."""
    status = map_status(m.get("status"))

    time_s = _statement_display_time(m)
    saved_time_s = float(m.get("saved_time", 0.0) or 0.0)

    # ``evaluated_vars`` is the AST-derived list of variable names this
    # statement writes. Do NOT fall through to ``metrics['outputs']`` — that
    # key holds rich/display objects (IPython.utils.capture.RichOutput), not
    # variable names, and ``str()``-ing them leaks ``<RichOutput at 0x..>``
    # into the badge's "Produced" field.
    output_vars = tuple(x for x in _tup_str(m.get("evaluated_vars")) if not x.startswith("<"))
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
    # Per-call-site grouping of the same raw events (CAS-243), for the
    # "Sub-calls" drawer section — see SubUnitGroup for why site rather
    # than callee. Only events with intercepted=True contribute; a
    # hand-decorated call has no call site and stays out of this list.
    sub_units = tuple(build_sub_unit_groups(raw_decorator))

    changed_modules = m.get("changed_modules") or {}
    if isinstance(changed_modules, dict):
        changed_modules_tup = tuple(sorted(str(k) for k in changed_modules.keys()))
    else:
        changed_modules_tup = _tup_str(changed_modules)

    # Short prefix of the statement cache key — first 8 hex chars after the
    # ``stmt:`` prefix the runtime stamps onto cache keys. Empty for callers
    # that don't expose the key (e.g. mock metrics in tests).
    raw_key = str(m.get("cache_key") or "")
    if raw_key.startswith("stmt:"):
        raw_key = raw_key[5:]
    cache_key_short = raw_key[:8]

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
        changed_functions=_tup_str(m.get("changed_functions")),
        changed_modules=changed_modules_tup,
        decorator_calls=dec_calls,
        sub_units=sub_units,
        body_statements=_tup_str(m.get("body_statements")),
        cache_key_short=cache_key_short,
        miss_reason=m.get("miss_reason") or None,
        random_effect=m.get("random_effect") or None,
        random_unseeded=bool(m.get("random_unseeded", False)),
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
        miss_reason=m.get("miss_reason") or None,
        skipped_reason=m.get("skipped_reason") or None,
        loop_bindings=bindings,
        # Same per-call-site grouping as ``_statement_row_from_metric`` (CAS-243
        # task 9) — a loop-body statement's intercepted sub-calls otherwise
        # never reach the badge at all, since it renders as an IterationRow,
        # not a StatementRow.
        sub_units=tuple(build_sub_unit_groups(m.get("decorator_calls") or [])),
    )


# ---------------------------------------------------------------------------
# Grouped-item translation
# ---------------------------------------------------------------------------

def _section_item_from_grouped(item: dict[str, Any]) -> SectionItem:
    """Translate one ``group_loop_iterations`` intermediate dict into a node."""
    kind = item["type"]
    if kind == "single":
        return _statement_row_from_metric(item["metric"])

    if kind == "for_loop_group":
        stmt_groups = item.get("stmt_groups", [])
        loop_var_names = tuple(stmt_groups[0].get("all_loop_var_names", [])) if stmt_groups else ()
        loop_header = str(item.get("loop_header", "")) or next(
            (str(sg.get("loop_header", "")) for sg in stmt_groups if sg.get("loop_header")),
            "",
        )
        # Sort body stmts (loop_groups) and nested children (control_groups,
        # nested for_loop_groups) into source order. Each metric carries a
        # ``body_index_chain`` set by the runtime; this for-loop sits at
        # depth ``len(self_chain) - 1``, so its body items are sorted by
        # ``chain[depth]``. Falls back to insertion order when an item has
        # no chain (synthetic outer wrappers).
        self_chain = tuple(item.get("loop_header_chain", []))
        depth = max(0, len(self_chain) - 1)

        def _body_idx(sub: dict[str, Any], pos: int) -> tuple[int, int]:
            metrics = sub.get("metrics", []) or [
                m for sg in sub.get("stmt_groups", []) for m in sg.get("metrics", [])
            ]
            for m in metrics:
                chain = m.get("body_index_chain")
                if chain and depth < len(chain):
                    return (int(chain[depth]), pos)
            if "body_index" in sub:
                return (int(sub["body_index"]), pos)
            return (10_000, pos)

        ordered: list[tuple[tuple[int, int], str, dict[str, Any]]] = []
        for pos, sg in enumerate(stmt_groups):
            ordered.append((_body_idx(sg, pos), "stmt", sg))
        for pos, child in enumerate(item.get("nested", []), start=len(stmt_groups)):
            ordered.append((_body_idx(child, pos), "child", child))
        ordered.sort(key=lambda t: t[0])

        stmts_list: list[LoopStatement] = []
        nested_list: list[Any] = []
        body_list: list[Any] = []
        for _, item_kind, sub in ordered:
            if item_kind == "stmt":
                stmt_metrics = sub.get("metrics", [])
                # Aggregate sub-calls across ALL iterations of this body
                # statement (CAS-243 task 9) -- the HTML renderer shows this
                # statement as one collapsed row, so it needs one combined
                # view rather than per-iteration groups.
                raw_calls_all_iters = [
                    c for m in stmt_metrics for c in (m.get("decorator_calls") or [])
                ]
                ls = LoopStatement(
                    base_code=str(sub.get("base_code", "")),
                    iterations=tuple(_iteration_row(m) for m in stmt_metrics),
                    sub_units=tuple(build_sub_unit_groups(raw_calls_all_iters)),
                )
                stmts_list.append(ls)
                body_list.append(ls)
            else:
                child_item = _section_item_from_grouped(sub)
                nested_list.append(child_item)
                body_list.append(child_item)
        return ForLoopGroup(
            loop_var_names=loop_var_names,
            stmts=tuple(stmts_list),
            loop_header=loop_header,
            nested=tuple(nested_list),
            suppress_head=bool(item.get("_suppress_head", False)),
            body=tuple(body_list),
        )

    if kind == "control_group":
        sub_items = item.get("sub_items")
        if sub_items:
            # Body was recursively grouped — translate each sub-item with
            # the same dispatch we use at the top level. Loops nested in
            # this control body therefore render as ForLoopGroups, not as
            # a flat list of N per-iteration StatementRows.
            rows: tuple[Any, ...] = tuple(
                _section_item_from_grouped(g)
                for g in sub_items
            )
        else:
            rows = tuple(
                _statement_row_from_metric(m)
                for m in item.get("metrics", [])
            )
        return ControlGroup(
            branch_label=str(item.get("branch_label", "")),
            header=str(item.get("header", "")),
            rows=rows,
        )

    if kind == "control_group_single":
        return ControlGroupSingle(
            row=_statement_row_from_metric(item["metric"]),
        )

    raise ValueError(f"Unknown grouped-item type: {kind!r}")


def _skipped_bucket(skipped_metrics: list[dict[str, Any]]) -> SkippedBucket | None:
    if not skipped_metrics:
        return None
    total_saved = sum(float(m.get("saved_time", 0.0)) for m in skipped_metrics)
    grouped = _group_loop_iterations(skipped_metrics)
    items: list[StatementRow | ForLoopGroup] = []
    for g in grouped:
        node = _section_item_from_grouped(g)
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
            # A group is intercepted only if every call in it was — a name that
            # is sometimes hand-decorated and sometimes not should not be
            # labelled as cash's doing.
            intercepted=all(c.get("intercepted") for c in calls),
        ))
    return tuple(groups)


# ---------------------------------------------------------------------------
# Overhead translation
# ---------------------------------------------------------------------------

# Labels stay short (one word) so up to four parts fit on the single collapsed
# overhead row without the cell's ellipsis clipping the rightmost one; the full
# meaning rides along as a hover tooltip. ``badge`` merges the display's setup
# and progress-render costs — one number for "what the badge itself cost".
_OVERHEAD_LABELS = {
    "upstream_check": "upstream",
    "cache_write": "cache",
    "remote_validate": "remote",
    "badge": "badge",
    "other": "other",
}
_OVERHEAD_TOOLTIPS = {
    "upstream_check": "re-checking and re-restoring upstream cells",
    "cache_write": "hashing inputs and serialising results into the cache",
    "remote_validate": "asking object storage whether tracked remote data changed",
    "badge": "building and updating Cash's badge display",
    "other": "cell time not attributed to a category above",
}

def _overhead_section(
    timing_breakdown: dict[str, float] | None,
    cell_total_time: float | None,
    metrics: list[dict[str, Any]],
) -> Section | None:
    if not timing_breakdown or cell_total_time is None:
        return None
    # Match what the statement rows DISPLAY (compute for executed, restore time
    # for restored), so overhead = wall-clock minus shown-times absorbs cash's
    # per-statement serialisation cost and the breakdown still sums to the total.
    statements_time = sum(_statement_display_time(m) for m in metrics)
    overhead = cell_total_time - statements_time
    if overhead <= MIN_TIME_DISPLAY_MS:
        return None

    upstream_check = float(timing_breakdown.get("upstream_check", 0.0))
    badge_init = float(timing_breakdown.get("badge_init", 0.0))
    badge_progress = float(timing_breakdown.get("badge_progress", 0.0))
    # cash's per-statement cost that ISN'T the compute/restore the row shows:
    # hashing the inputs + serialising the result into the cache (the dominant
    # overhead for a large object). It equals sum(total_time) - sum(displayed),
    # i.e. exactly the extra carried by COMPUTED rows now that they show only the
    # compute — so surfacing it here keeps that cost visible instead of hiding it
    # inside "other". (RESTORED rows already display their full total_time, so
    # they contribute nothing.)
    cache_write = max(
        0.0,
        sum(float(m.get("total_time", 0.0)) for m in metrics) - statements_time,
    )
    # Network round trips asking object storage whether tracked remote data
    # moved. It gets its own line because it is the one overhead a user cannot
    # otherwise see or act on: it happens on the HIT path, where the badge
    # reports a saving and nothing reports what establishing it cost.
    remote_validate = float(timing_breakdown.get("remote_validate", 0.0))
    remote_count = int(timing_breakdown.get("remote_validate_count", 0))
    # The badge's own setup + progress-render costs read as one "badge" number.
    badge = badge_init + badge_progress
    other = overhead - (badge + upstream_check + cache_write + remote_validate)

    entries: list[OverheadEntry] = []
    for key, value in (
        ("upstream_check", upstream_check),
        ("cache_write", cache_write),
        ("remote_validate", remote_validate),
        ("badge", badge),
        ("other", other),
    ):
        if value > MIN_TIME_DISPLAY_MS:
            tooltip = _OVERHEAD_TOOLTIPS[key]
            if key == "remote_validate" and remote_count:
                # The count is the actionable half - it is what tells you when
                # to trade N metadata requests for one prefix listing.
                tooltip = (
                    f"{tooltip} ({remote_count} "
                    f"{'source' if remote_count == 1 else 'sources'} checked)"
                )
            entries.append(OverheadEntry(
                label=_OVERHEAD_LABELS[key],
                time_s=value,
                tooltip=tooltip,
            ))
    if not entries:
        return None
    return Section(
        kind=SectionKind.OVERHEAD,
        header="",  # the single-row renderer carries its own label
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
        # The words the reporter actually SAW on their badge. This dump used
        # the internal status names, so an issue said RESTORED about a row the
        # user watched render as CACHED -- and the reporter is the one who has
        # to recognise their own cell in the preview.
        st = label_of(str(m.get("status", "")).replace("CacheStatus.", "").lower())
        t = m.get("total_time") or m.get("execution_time") or 0.0
        saved = m.get("saved_time") or 0.0
        outs_raw = m.get("outputs", [])
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
        "<!-- e.g. 'Should have re-executed but the badge read CACHED' -->\n\n"
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
    configured_tiers: tuple[str, ...] = (),
) -> InteractiveBadge:
    """Build an :class:`InteractiveBadge` view from a metrics list.

    ``configured_tiers`` is the ordered tier-label list reported by the
    active backend (``cash.backend.tier_labels()``). It drives the dot
    indicator's slot count; when empty the renderer falls back to the
    per-row ``storage_tiers`` data.
    """
    metrics = metrics_list or []

    upstream_all = [m for m in metrics if m.get("is_upstream", False)
                    and str(m.get("status")) != str(CacheStatus.SKIPPED)]
    upstream_restored_like = [
        m for m in upstream_all
        if str(m.get("status")) in {str(CacheStatus.RESTORED), *_NOTIFICATION_STATUSES}
    ]
    upstream_executed = [m for m in upstream_all if m not in upstream_restored_like]
    upstream_skipped = [m for m in metrics if str(m.get("status")) == str(CacheStatus.SKIPPED)
                        and m.get("is_upstream", False)]
    current = [m for m in metrics if not m.get("is_upstream", False)]

    (total_saved, total_exec, restored, computed, skipped_count,
     uncacheable_count) = _compute_stats(metrics)
    # Notification statuses (WARN / FUNCTION_CHANGED / MODULE_RELOADED / ERROR)
    # don't count as restored/computed/skipped but should appear in the
    # summary chip strip — otherwise a notification-only cell shows no chip.
    warn_count = sum(
        1 for m in metrics
        if str(m.get("status")) in _NOTIFICATION_STATUSES
        or str(m.get("status")) == str(CacheStatus.ERROR)
        # Unseeded randomness is advisory but user-visible: its cached value is a
        # frozen replay, so it belongs in the header's warning tally.
        or bool(m.get("random_unseeded", False))
    )
    summary_time = cell_total_time if cell_total_time is not None else total_exec
    # Honest header saving: never advertise more than the cell's NET
    # win. When we know the cell's wall time we subtract cash's own overhead
    # (wall time minus the user compute that ran) from the gross saving, so the
    # collapsed badge can't claim a 7s saving when the cell's own overhead ate
    # into it. Floor at 0 here — the per-cell badge stays non-alarming and the
    # negative-net story lives in the aggregate %cash_stats. With no wall time
    # (unit renders that pass no cell_total_time) fall back to the gross value.
    header_saved = total_saved
    if cell_total_time is not None:
        cell_compute = sum(
            float(m.get("execution_time", 0.0) or 0.0)
            for m in metrics
            if str(m.get("status")) == str(CacheStatus.COMPUTED)
        )
        cell_overhead = max(0.0, cell_total_time - cell_compute)
        header_saved = max(0.0, total_saved - cell_overhead)
    header = BadgeHeader(
        status=BadgeStatus.WARNING if status == "RUNNING" else _summary_status(restored, computed, skipped_count),
        restored_count=restored,
        computed_count=computed,
        skipped_count=skipped_count,
        uncacheable_count=uncacheable_count,
        warn_count=warn_count,
        total_saved_s=header_saved,
        total_exec_s=summary_time,
        current_step=current_step,
        total_steps=total_steps,
        current_code=current_code,
    )

    sections: list[Section] = []

    if upstream_all or upstream_skipped:
        items: list[SectionItem] = []
        for g in _group_loop_iterations(upstream_restored_like):
            items.append(_section_item_from_grouped(g))
        for g in _group_loop_iterations(upstream_executed):
            items.append(_section_item_from_grouped(g))
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
        current_items.append(_section_item_from_grouped(g))
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
    return InteractiveBadge(
        header=header,
        sections=tuple(sections),
        footer=footer,
        configured_tiers=tuple(configured_tiers),
    )


__all__ = [
    "build_interactive_badge",
    "build_bug_report_url",
    "map_status",
]
