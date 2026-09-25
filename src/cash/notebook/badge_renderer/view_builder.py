"""Build a BadgeView IR tree from ``ProcessResult``-shaped metric dicts.

This module is the single seam between the runtime (which produces raw
metric dicts) and the renderers (which consume :class:`BadgeView` nodes).
Each metric dict is read exactly once, into a typed :class:`_Metric`; loop
and control grouping then works on typed drafts, which are translated into
the frozen view nodes at the end. Renderers never see a raw dict.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Union
from urllib.parse import quote

from cash.control_markers import iteration_digest, strip_markers
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
    SubUnitGroup,
)

# Rows that annotate other work rather than doing any: shown as warning rows,
# left out of the restored / computed / skipped counts.
_NOTIFICATIONS = frozenset({CacheStatus.FUNCTION_CHANGED, CacheStatus.MODULE_RELOADED, CacheStatus.WARNING})

# The one mapping from what the runtime reports to what the badge shows.
_BADGE_STATUS = {
    CacheStatus.RESTORED: BadgeStatus.RESTORED,
    CacheStatus.COMPUTED: BadgeStatus.COMPUTED,
    CacheStatus.SKIPPED: BadgeStatus.SKIPPED,
    CacheStatus.ERROR: BadgeStatus.ERROR,
    CacheStatus.FUNCTION_CHANGED: BadgeStatus.FUNCTION_CHANGED,
    CacheStatus.MODULE_RELOADED: BadgeStatus.MODULE_RELOADED,
}

# A body position for items that carry none: after everything that does.
_NO_INDEX = 10_000


def map_status(raw: Any) -> BadgeStatus:
    """The :class:`BadgeStatus` for a runtime status (member or string).

    Total: anything unrecognised is a WARNING row, so a malformed metric
    renders instead of raising.
    """
    return _BADGE_STATUS.get(CacheStatus.parse(raw), BadgeStatus.WARNING)


# ---------------------------------------------------------------------------
# Reading a metric dict, once
# ---------------------------------------------------------------------------


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _tup_str(seq: Any) -> tuple[str, ...]:
    if isinstance(seq, list | tuple):
        return tuple(str(x) for x in seq)
    if seq is None:
        return ()
    return (str(seq),)


def _opt_str(value: Any) -> str | None:
    return str(value) if value else None


def _ints(seq: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in seq or ())
    except (TypeError, ValueError):
        return ()


@dataclass(frozen=True)
class _Call:
    """One ``decorator_calls`` event: a decorated or intercepted call."""

    func_name: str
    cache_hit: bool
    execution_time: float
    time_saved: float
    intercepted: bool
    call_source: str
    occurrence_index: int
    cache_key: str
    miss_reason: str | None
    ran_plain: bool
    #: ``False`` only when the runtime said the result was not stored.
    not_stored: bool

    @classmethod
    def parse(cls, e: dict[str, Any]) -> _Call:
        try:
            occ = int(e.get("occurrence_index", 0) or 0)
        except (TypeError, ValueError):
            occ = 0
        return cls(
            func_name=str(e.get("func_name", "?")),
            cache_hit=bool(e.get("cache_hit")),
            execution_time=_float(e.get("execution_time")),
            time_saved=_float(e.get("time_saved")),
            intercepted=bool(e.get("intercepted")),
            call_source=str(e.get("call_source", "?")),
            occurrence_index=occ,
            cache_key=str(e.get("cache_key") or ""),
            miss_reason=_opt_str(e.get("miss_reason")),
            ran_plain=bool(e.get("ran_plain")),
            not_stored=e.get("stored") is False,
        )

    @property
    def unstored_miss(self) -> bool:
        """Missed, ran through the cache, and its result was not kept."""
        return not self.cache_hit and not self.ran_plain and self.not_stored

    @property
    def status(self) -> BadgeStatus:
        return BadgeStatus.RESTORED if self.cache_hit else BadgeStatus.COMPUTED


@dataclass(frozen=True)
class _Metric:
    """The fields of one metric dict the badge reads."""

    status: CacheStatus
    #: With cash's iteration / control markers still in: grouping keys on them.
    code: str
    display_code: str | None
    #: The statement's own compute: its wall time less ``cash_tax``, cash's
    #: time inside it (see ``_own_compute``).
    execution_time: float
    total_time: float
    saved_time: float
    is_upstream: bool
    storage: tuple[str, ...]
    source: str | None
    evaluated_vars: tuple[str, ...]
    restored_vars: tuple[str, ...]
    uncacheable_reasons: tuple[str, ...]
    skipped_reason: str | None
    guard_cause: str | None
    printed: str
    changed_functions: tuple[str, ...]
    changed_modules: tuple[str, ...]
    calls: tuple[_Call, ...]
    body_statements: tuple[str, ...]
    cache_key: str
    miss_reason: str | None
    random_effect: str | None
    random_unseeded: bool
    loop_vars: tuple[tuple[str, Any], ...]
    loop_header: str
    loop_header_chain: tuple[str, ...]
    body_index: int | None
    body_index_chain: tuple[int, ...]
    control_context: str | None
    branch_label: str
    stale_export: bool
    written_paths: tuple[str, ...]
    output_names: tuple[str, ...]

    @classmethod
    def parse(cls, m: dict[str, Any]) -> _Metric:
        loop_vars = m.get("loop_vars")
        changed_modules = m.get("changed_modules") or {}
        try:
            body_index = int(m["body_index"]) if "body_index" in m else None
        except (TypeError, ValueError):
            body_index = None
        return cls(
            status=CacheStatus.parse(m.get("status")),
            code=str(m.get("code") or ""),
            display_code=strip_markers(str(m["display_code"])) if m.get("display_code") else None,
            execution_time=_own_compute(m),
            total_time=_float(m.get("total_time")),
            saved_time=_float(m.get("saved_time")),
            is_upstream=bool(m.get("is_upstream", False)),
            storage=_tup_str(m.get("storage")),
            source=_opt_str(m.get("source")),
            # Variable names the statement writes. Not ``outputs``: that holds
            # display objects, whose str() is ``<RichOutput at 0x..>``.
            evaluated_vars=tuple(x for x in _tup_str(m.get("evaluated_vars")) if not x.startswith("<")),
            restored_vars=_tup_str(m.get("restored_vars")),
            uncacheable_reasons=_tup_str(m.get("uncacheable_reasons")),
            skipped_reason=_opt_str(m.get("skipped_reason")),
            guard_cause=_opt_str(m.get("guard_cause")),
            printed=_printed(m),
            changed_functions=_tup_str(m.get("changed_functions")),
            changed_modules=(
                tuple(sorted(str(k) for k in changed_modules))
                if isinstance(changed_modules, dict)
                else _tup_str(changed_modules)
            ),
            calls=tuple(_Call.parse(e) for e in (m.get("decorator_calls") or ()) if isinstance(e, dict)),
            body_statements=_tup_str(m.get("body_statements") or None),
            cache_key=str(m.get("cache_key") or ""),
            miss_reason=_opt_str(m.get("miss_reason")),
            random_effect=_opt_str(m.get("random_effect")),
            random_unseeded=bool(m.get("random_unseeded", False)),
            loop_vars=tuple((str(k), v) for k, v in loop_vars.items()) if isinstance(loop_vars, dict) else (),
            loop_header=str(m.get("loop_header") or ""),
            loop_header_chain=tuple(str(h) for h in m.get("loop_header_chain") or ()),
            body_index=body_index,
            body_index_chain=_ints(m.get("body_index_chain")),
            control_context=_opt_str(m.get("control_context")),
            branch_label=str(m.get("branch_label") or ""),
            stale_export=bool(m.get("stale_export")),
            written_paths=_tup_str(m.get("written_paths") or None),
            output_names=tuple(o for o in (m.get("outputs") or ()) if isinstance(o, str)),
        )

    @property
    def display_time(self) -> float:
        """The time a statement row shows.

        For a COMPUTED statement the compute alone (``execution_time``), not
        ``total_time``, which adds cash's hashing and serialising: showing that
        made a 0.01 s compute read 0.09 s beside "saved 0.01s". The difference
        is attributed to the overhead section instead. A RESTORED statement has
        no execution time, so it shows its restore time.
        """
        return self.execution_time or self.total_time


def _own_compute(m: dict[str, Any]) -> float:
    """What the statement's own code took, without cash's time inside it.

    A statement's wall time includes what cash does while it runs: recording
    the files it reads, and keying, hashing and storing the calls it routes.
    For a call that returns a large frame that is a noticeable share, and the
    row showed it while the same statement restored showed "saved" net of it
    (``CallRouting.price``): "0.92s" beside "saved 0.82s" for one
    computation. ``%cash_stats`` counts compute the same way. The difference
    lands in the overhead row's ``cache`` part, with the rest of cash's cost.
    """
    return max(0.0, _float(m.get("execution_time")) - _float(m.get("cash_tax")))


def _printed(m: dict[str, Any]) -> str:
    """What a statement printed, stdout then stderr."""
    parts = [str(m.get(k) or "") for k in ("stdout", "stderr")]
    return "".join(p if p.endswith("\n") or not p else p + "\n" for p in parts).rstrip("\n")


# ---------------------------------------------------------------------------
# Grouping drafts
# ---------------------------------------------------------------------------


@dataclass
class _LoopStmt:
    """Every iteration of one loop-body statement."""

    base_code: str
    metrics: list[_Metric]
    #: Every loop variable the metrics bind, in first-seen order.
    var_names: tuple[str, ...]
    header: str
    chain: tuple[str, ...]
    body_index: int

    @classmethod
    def of(cls, base_code: str, metrics: list[_Metric]) -> _LoopStmt:
        names: dict[str, None] = {}
        for m in metrics:
            for name, _ in m.loop_vars:
                names.setdefault(name, None)
        header = next((m.loop_header for m in metrics if m.loop_header), "")
        chain = next((m.loop_header_chain for m in metrics if m.loop_header_chain), (header,) if header else ())
        return cls(
            base_code=base_code,
            metrics=metrics,
            var_names=tuple(names),
            header=header,
            chain=chain,
            body_index=_min_body_index(metrics),
        )


@dataclass
class _ForDraft:
    """A ``for`` loop: its body statements and the loops/controls nested in it."""

    stmts: list[_LoopStmt]
    chain: tuple[str, ...]
    header: str = ""
    nested: list[_Draft] = field(default_factory=list)
    #: The inner copy under a hoisted outer wrapper, which draws the header.
    suppress_head: bool = False

    @property
    def loop_header(self) -> str:
        return self.header or next((s.header for s in self.stmts if s.header), "")


@dataclass
class _ControlDraft:
    """One taken branch of an ``if`` / ``try``, its body grouped recursively."""

    metrics: list[_Metric]
    sub_items: list[_Draft]
    branch_label: str
    header: str
    body_index: int


@dataclass
class _SingleDraft:
    metric: _Metric


@dataclass
class _ControlSingleDraft:
    """A control structure run as one unit (``while``, ``with``, ...)."""

    metric: _Metric


_Draft = Union[_ForDraft, _ControlDraft, _SingleDraft, _ControlSingleDraft]


def _min_body_index(metrics: Iterable[_Metric]) -> int:
    return min((m.body_index for m in metrics if m.body_index is not None), default=_NO_INDEX)


def _body_haystack(metrics: Iterable[_Metric]) -> str:
    return "\n".join(s for m in metrics for s in m.body_statements)


def _without_control(metrics: Iterable[_Metric]) -> list[_Metric]:
    """*metrics* with their control context dropped, for regrouping a body."""
    return [dataclasses.replace(m, control_context=None) for m in metrics]


def _chain_of(item: _Draft | _LoopStmt) -> tuple[str, ...]:
    """The loops enclosing *item*, outermost first; empty when none do."""
    if isinstance(item, _ForDraft):
        return item.chain
    if isinstance(item, _ControlDraft):
        chain = next((m.loop_header_chain for m in item.metrics if m.loop_header_chain), ())
        # Loops named in the control's own body are inside it, not around it.
        haystack = _body_haystack(item.metrics)
        if chain and haystack and any(header in haystack for header in chain):
            return ()
        return chain
    return ()


def _group(metrics: list[_Metric], *, synthesize_intermediate_loops: bool = True) -> list[_Draft]:
    """Group a flat metric list into loops and controls, nested by source."""
    pass1: list[_Draft | _LoopStmt] = []
    loop_stmts: dict[str, list[_Metric]] = {}
    controls: dict[str, list[_Metric]] = {}

    def flush_loops() -> None:
        pass1.extend(_LoopStmt.of(base, ms) for base, ms in loop_stmts.items())
        loop_stmts.clear()

    def flush_controls() -> None:
        # The same inner control fired across outer iterations arrives under a
        # distinct context each time but one branch label: one bucket.
        merged: dict[str, list[_Metric]] = {}
        for ctx, ms in controls.items():
            label = next((m.branch_label for m in ms if m.branch_label), "")
            merged.setdefault(label or ctx, []).extend(ms)
        for ms in merged.values():
            label = next((m.branch_label for m in ms if m.branch_label), "")
            body = next((m.body_statements for m in ms if m.body_statements), ())
            # Loops that surround the control must not be synthesized inside
            # it; only loops its own body contains may be.
            chain = next((m.loop_header_chain for m in ms if m.loop_header_chain), ())
            haystack = _body_haystack(ms)
            synth_inside = bool(chain and any(h in haystack for h in chain))
            pass1.append(
                _ControlDraft(
                    metrics=ms,
                    sub_items=_group(_without_control(ms), synthesize_intermediate_loops=synth_inside),
                    branch_label=label,
                    header=body[0] if body else label,
                    body_index=_min_body_index(ms),
                )
            )
        controls.clear()

    for m in metrics:
        has_iter = iteration_digest(m.code) is not None
        if m.control_context:
            # A control nested in a loop must not split the loop around it.
            if not has_iter:
                flush_loops()
            controls.setdefault(m.control_context, []).append(m)
        elif has_iter:
            flush_controls()
            loop_stmts.setdefault(strip_markers(m.code), []).append(m)
        else:
            flush_loops()
            flush_controls()
            pass1.append(_ControlSingleDraft(m) if m.body_statements else _SingleDraft(m))
    flush_loops()
    flush_controls()

    # Controls with one branch label, flushed apart by sibling statements, are
    # one control fired in several outer iterations.
    merged_pass1: list[_Draft | _LoopStmt] = []
    by_label: dict[str, _ControlDraft] = {}
    for item in pass1:
        if isinstance(item, _ControlDraft) and item.branch_label:
            target = by_label.get(item.branch_label)
            if target is not None:
                target.metrics.extend(item.metrics)
                target.sub_items = _group(_without_control(target.metrics), synthesize_intermediate_loops=False)
                continue
            by_label[item.branch_label] = item
        merged_pass1.append(item)

    # Consecutive loop statements form one loop, split where the header changes.
    pass2: list[_Draft] = []
    pending: list[_LoopStmt] = []

    def flush_pending() -> None:
        if pending:
            chain = next((p.chain for p in pending if p.chain), ())
            pass2.append(_ForDraft(stmts=list(pending), chain=chain))
            pending.clear()

    for item in merged_pass1:
        if isinstance(item, _LoopStmt):
            existing = next((p.header for p in pending if p.header), "")
            if pending and existing and item.header and existing != item.header:
                flush_pending()
            pending.append(item)
        else:
            flush_pending()
            pass2.append(item)
    flush_pending()

    # A loop that surrounds a control is drawn around it, not inside it --
    # unless a real loop with that chain is already a sibling, which the
    # nesting pass puts the control under.
    existing_chains = {c for s in pass2 if isinstance(s, _ForDraft) and (c := _chain_of(s))}
    hoisted: list[_Draft] = []
    for item in pass2:
        if isinstance(item, _ControlDraft) and _chain_of(item) in existing_chains:
            chain = _chain_of(item)
            for sub in item.sub_items:
                if isinstance(sub, _ForDraft) and sub.chain == chain:
                    sub.suppress_head = True
            hoisted.append(item)
        else:
            hoisted.append(_hoist_outer_for_loop(item))

    return _nest_by_chain(hoisted, synthesize=synthesize_intermediate_loops)


def _hoist_outer_for_loop(item: _Draft) -> _Draft:
    """Swap a control and the loop that surrounds it in source.

    The loop becomes the outer wrapper with the control nested in it, and its
    iteration rows stay under the control as a head-suppressed copy. A loop
    whose header appears in the control's own body is inside the control
    (``if c: for i: ...``) and is left alone.
    """
    if not isinstance(item, _ControlDraft):
        return item
    chain = _chain_of(item)
    if not chain:
        return item
    idx = next(
        (i for i, s in enumerate(item.sub_items) if isinstance(s, _ForDraft) and s.chain == chain),
        None,
    )
    if idx is None:
        return item
    wrapper = item.sub_items[idx]
    assert isinstance(wrapper, _ForDraft)
    header = wrapper.loop_header
    if header and header in _body_haystack(item.metrics):
        return item
    others = [s for i, s in enumerate(item.sub_items) if i != idx]
    item.sub_items = [*others, dataclasses.replace(wrapper, suppress_head=True)]
    return _ForDraft(stmts=[], chain=chain, header=header, nested=[item])


def _nest_by_chain(items: list[_Draft], *, synthesize: bool = True) -> list[_Draft]:
    """Nest loops and controls under the loops that enclose them.

    A loop with chain (A, B, C) goes inside the loop with chain (A, B); a
    control with chain (A, B, C) inside the loop at (A, B, C). A level of the
    chain with no loop of its own (``if c: for i: for j: ...`` records no
    metric for ``for i``) gets a synthesized one when *synthesize* allows,
    placed where its first child was so source order holds.
    """
    by_chain: dict[tuple[str, ...], _ForDraft] = {
        item.chain: item for item in items if isinstance(item, _ForDraft) and item.chain
    }

    result: list[_Draft] = []
    for item in items:
        chain = _chain_of(item)
        if not chain:
            result.append(item)
            continue
        # A loop's parent encloses it; a control sits in its innermost loop.
        parent_chain = chain[:-1] if isinstance(item, _ForDraft) else chain

        synth: list[tuple[tuple[str, ...], _ForDraft]] = []
        pc = parent_chain
        while pc:
            found = by_chain.get(pc)
            if found is item:
                pc = pc[:-1]
                continue
            if found is not None or not synthesize:
                break
            made = _ForDraft(stmts=[], chain=pc, header=pc[-1])
            by_chain[pc] = made
            synth.append((pc, made))
            pc = pc[:-1]
        synth.reverse()  # outermost first

        outermost: _Draft = synth[0][1] if synth else item
        ppc = synth[0][0][:-1] if synth else parent_chain
        outer_parent = None
        while ppc:
            found = by_chain.get(ppc)
            if found is not None and found is not outermost:
                outer_parent = found
                break
            ppc = ppc[:-1]

        for (_, outer), (_, inner) in zip(synth, synth[1:], strict=False):
            outer.nested.append(inner)
        if synth:
            synth[-1][1].nested.append(item)

        if outer_parent is not None:
            outer_parent.nested.append(outermost)
        else:
            result.append(outermost)
    return result


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------


def _sub_unit_groups(calls: Iterable[_Call]) -> tuple[SubUnitGroup, ...]:
    """Intercepted calls grouped by call SITE (``call_source``, occurrence).

    A hand-decorated call has no site and stays in the flat decorator list.
    """
    buckets: dict[tuple[str, int], list[_Call]] = {}
    for c in calls:
        if c.intercepted:
            buckets.setdefault((c.call_source, c.occurrence_index), []).append(c)
    return tuple(
        SubUnitGroup(
            call_source=source,
            occurrence_index=occ,
            calls=tuple(
                DecoratorCall(
                    func_name=c.func_name,
                    status=c.status,
                    time_s=c.time_saved if c.cache_hit else c.execution_time,
                )
                for c in cs
            ),
            condensed=len(cs) > _CONDENSE_THRESHOLD,
            key_prefix=cs[0].cache_key[:13],
            miss_reason=next((c.miss_reason for c in cs if c.miss_reason), None),
            ran_plain=sum(1 for c in cs if c.ran_plain),
            unstored=sum(1 for c in cs if c.unstored_miss),
        )
        for (source, occ), cs in buckets.items()
    )


def build_sub_unit_groups(events: Iterable[Any]) -> tuple[SubUnitGroup, ...]:
    """Per-site groups for raw ``decorator_calls`` events; non-dicts are skipped."""
    return _sub_unit_groups(_Call.parse(e) for e in events or () if isinstance(e, dict))


def _statement_row(m: _Metric) -> StatementRow:
    status = map_status(m.status)
    # The statement key's hash after its ``stmt:`` namespace.
    key = m.cache_key.removeprefix("stmt:")
    return StatementRow(
        status=status,
        code=strip_markers(m.code),
        time_s=m.display_time,
        display_code=m.display_code,
        saved_time_s=m.saved_time,
        storage_tiers=m.storage,
        source=m.source,
        output_vars=m.evaluated_vars,
        restored_vars=m.restored_vars,
        uncacheable_reasons=m.uncacheable_reasons,
        skipped_reason=m.skipped_reason,
        guard_cause=m.guard_cause,
        output_text=m.printed if m.is_upstream and status is BadgeStatus.COMPUTED else "",
        changed_functions=m.changed_functions,
        changed_modules=m.changed_modules,
        decorator_calls=tuple(
            DecoratorCall(func_name=c.func_name, status=c.status, time_s=c.execution_time) for c in m.calls
        ),
        sub_units=_sub_unit_groups(m.calls),
        body_statements=m.body_statements,
        cache_key_short=key[:8],
        miss_reason=m.miss_reason,
        random_effect=m.random_effect,
        random_unseeded=m.random_unseeded,
    )


def _iteration_row(m: _Metric) -> IterationRow:
    return IterationRow(
        status=map_status(m.status),
        code=strip_markers(m.code),
        time_s=m.total_time,
        saved_time_s=m.saved_time,
        storage_tiers=m.storage,
        miss_reason=m.miss_reason,
        skipped_reason=m.skipped_reason,
        loop_bindings=m.loop_vars,
        sub_units=_sub_unit_groups(m.calls),
    )


def _loop_statement(s: _LoopStmt) -> LoopStatement:
    return LoopStatement(
        base_code=s.base_code,
        iterations=tuple(_iteration_row(m) for m in s.metrics),
        # Across all iterations: the HTML badge shows the statement as one row.
        sub_units=_sub_unit_groups(c for m in s.metrics for c in m.calls),
    )


def _node(item: _Draft) -> SectionItem:
    """Translate one draft into its view node."""
    if isinstance(item, _SingleDraft):
        return _statement_row(item.metric)
    if isinstance(item, _ControlSingleDraft):
        return ControlGroupSingle(row=_statement_row(item.metric))
    if isinstance(item, _ControlDraft):
        rows = (
            tuple(_node(g) for g in item.sub_items)
            if item.sub_items
            else tuple(_statement_row(m) for m in item.metrics)
        )
        return ControlGroup(branch_label=item.branch_label, header=item.header, rows=rows)
    return _for_loop_group(item)


def _for_loop_group(item: _ForDraft) -> ForLoopGroup:
    # This loop sits at depth len(chain) - 1; its body items sort by the
    # runtime's body_index_chain at that depth, in source order.
    depth = max(0, len(item.chain) - 1)

    def position(sub: _LoopStmt | _Draft, pos: int) -> tuple[int, int]:
        if isinstance(sub, _ForDraft):
            metrics: list[_Metric] = [m for s in sub.stmts for m in s.metrics]
            fallback = _NO_INDEX
        elif isinstance(sub, _LoopStmt | _ControlDraft):
            metrics, fallback = sub.metrics, sub.body_index
        else:
            metrics, fallback = [sub.metric], _NO_INDEX
        for m in metrics:
            if depth < len(m.body_index_chain):
                return (m.body_index_chain[depth], pos)
        return (fallback, pos)

    ordered: list[tuple[tuple[int, int], _LoopStmt | _Draft]] = [
        (position(s, pos), s) for pos, s in enumerate(item.stmts)
    ]
    ordered += [(position(c, pos), c) for pos, c in enumerate(item.nested, start=len(item.stmts))]
    ordered.sort(key=lambda t: t[0])

    stmts: list[LoopStatement] = []
    nested: list[SectionItem] = []
    body: list[Any] = []
    for _, sub in ordered:
        if isinstance(sub, _LoopStmt):
            node: Any = _loop_statement(sub)
            stmts.append(node)
        else:
            node = _node(sub)
            nested.append(node)
        body.append(node)
    return ForLoopGroup(
        loop_var_names=item.stmts[0].var_names if item.stmts else (),
        stmts=tuple(stmts),
        loop_header=item.loop_header,
        nested=tuple(nested),
        suppress_head=item.suppress_head,
        body=tuple(body),
    )


def _nodes(metrics: list[_Metric]) -> list[SectionItem]:
    return [_node(g) for g in _group(metrics)]


def _skipped_bucket(skipped: list[_Metric]) -> SkippedBucket | None:
    if not skipped:
        return None
    stale = tuple((m.code, m.written_paths) for m in skipped if m.stale_export)
    skipped = [m for m in skipped if not m.stale_export]
    items: list[StatementRow | ForLoopGroup] = []
    for node in _nodes(skipped):
        # The bucket holds rows and loops; a control contributes its rows.
        if isinstance(node, StatementRow | ForLoopGroup):
            items.append(node)
        elif isinstance(node, ControlGroupSingle):
            items.append(node.row)
        elif isinstance(node, ControlGroup):
            items.extend(node.rows)
    return SkippedBucket(
        items=tuple(items),
        total_saved_time_s=sum(m.saved_time for m in skipped),
        stale_exports=stale,
    )


# ---------------------------------------------------------------------------
# Cell-level statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Stats:
    total_saved: float = 0.0
    total_exec: float = 0.0
    restored: int = 0
    computed: int = 0
    skipped: int = 0
    #: A subset of ``computed``: rows that ran and were not stored.
    uncacheable: int = 0


def _compute_stats(metrics: list[_Metric]) -> _Stats:
    """Counts and times for the header.

    ERROR rows count as computed: they ran, unsuccessfully. ``uncacheable``
    uses the test the renderers use to label a row NOT CACHED, so the header
    can never disagree with the rows under it.
    """
    total_saved = total_exec = 0.0
    restored = computed = skipped = uncacheable = 0
    for m in metrics:
        if m.status in _NOTIFICATIONS:
            continue
        if m.status is CacheStatus.RESTORED:
            restored += 1
            total_saved += m.saved_time
            total_exec += m.total_time
        elif m.status is CacheStatus.SKIPPED:
            skipped += 1
            total_saved += m.saved_time
        elif m.is_upstream or m.status in (CacheStatus.COMPUTED, CacheStatus.ERROR):
            computed += 1
            total_exec += m.total_time
            if m.uncacheable_reasons or m.skipped_reason:
                uncacheable += 1
    return _Stats(total_saved, total_exec, restored, computed, skipped, uncacheable)


def _header_status(status: str, stats: _Stats, errored: bool) -> BadgeStatus:
    """The header's status.

    ``RUNNING`` and ``BYPASSED`` describe the cell, not its rows, and arrive
    with no rows at all, so the caller says so. A statement that raised makes
    the header say ERROR rather than EXECUTED.
    """
    if status == "RUNNING":
        return BadgeStatus.RUNNING
    if status == "BYPASSED":
        return BadgeStatus.BYPASSED
    if errored:
        return BadgeStatus.ERROR
    if stats.computed == 0 and (stats.restored > 0 or stats.skipped > 0):
        return BadgeStatus.RESTORED if stats.restored > 0 else BadgeStatus.SKIPPED
    if stats.restored > 0 and stats.computed > 0:
        return BadgeStatus.MIXED
    return BadgeStatus.COMPUTED


# ---------------------------------------------------------------------------
# Decorator translation
# ---------------------------------------------------------------------------

_CONDENSE_THRESHOLD = 3


def _decorator_groups(metrics: list[_Metric]) -> tuple[DecoratorCallGroup, ...]:
    by_func: dict[str, list[_Call]] = {}
    for m in metrics:
        for c in m.calls:
            # An intercepted call found under the cost floor and not kept is
            # not a cache that failed; the sub-call lines count it instead.
            if c.intercepted and c.unstored_miss:
                continue
            by_func.setdefault(c.func_name, []).append(c)
    return tuple(
        DecoratorCallGroup(
            func_name=func_name,
            calls=tuple(DecoratorCall(func_name=func_name, status=c.status, time_s=c.execution_time) for c in calls),
            condensed=len(calls) > _CONDENSE_THRESHOLD,
            # Only when every call was: a name sometimes decorated by hand is
            # not labelled as cash's doing.
            intercepted=all(c.intercepted for c in calls),
        )
        for func_name, calls in by_func.items()
    )


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
    "cache_write": "hashing inputs, recording file reads and serialising results into the cache",
    "remote_validate": "asking object storage whether tracked remote data changed",
    "badge": "building and updating Cash's badge display",
    "other": "cell time not attributed to a category above",
}


def _overhead_section(
    timing_breakdown: dict[str, float] | None,
    cell_total_time: float | None,
    metrics: list[_Metric],
) -> Section | None:
    if not timing_breakdown or cell_total_time is None:
        return None
    # Match what the statement rows DISPLAY (compute for executed, restore time
    # for restored), so overhead = wall-clock minus shown-times absorbs cash's
    # per-statement serialisation cost and the breakdown still sums to the total.
    statements_time = sum(m.display_time for m in metrics)
    overhead = cell_total_time - statements_time
    if overhead <= MIN_TIME_DISPLAY_MS:
        return None

    upstream_check = float(timing_breakdown.get("upstream_check", 0.0))
    badge = float(timing_breakdown.get("badge_init", 0.0)) + float(timing_breakdown.get("badge_progress", 0.0))
    # Hashing inputs and serialising results, and cash's time inside the
    # statement (``cash_tax``): what COMPUTED rows carry beyond the compute
    # they show. RESTORED rows show their full total_time.
    cache_write = max(0.0, sum(m.total_time for m in metrics) - statements_time)
    # Round trips asking object storage whether tracked remote data moved: the
    # one overhead on the HIT path, where nothing else reports what it cost.
    remote_validate = float(timing_breakdown.get("remote_validate", 0.0))
    remote_count = int(timing_breakdown.get("remote_validate_count", 0))
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
                # The count tells you when to trade N metadata requests for
                # one prefix listing.
                tooltip = f"{tooltip} ({remote_count} {'source' if remote_count == 1 else 'sources'} checked)"
            entries.append(OverheadEntry(label=_OVERHEAD_LABELS[key], time_s=value, tooltip=tooltip))
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
    """A pre-filled GitHub issue URL describing this cell's badge."""
    return _bug_report_url([_Metric.parse(m) for m in metrics], context)


def _bug_report_url(metrics: list[_Metric], context: dict | None) -> str:
    ctx = context or {}
    version = ctx.get("version", "unknown")
    python_version = ctx.get("python_version", "(unknown)")
    backend = ctx.get("backend", "(unknown)")
    notebook_source: list[str] = ctx.get("notebook_source", [])

    badge_lines: list[str] = []
    for m in metrics:
        if m.is_upstream:
            continue
        code = strip_markers(m.code).strip()
        if len(code) > 100:
            code = code[:97] + "..."
        # The word the reporter SAW on their badge, so they recognise the row.
        st = label_of(map_status(m.status))
        t = m.total_time or m.execution_time
        outs = ", ".join(m.output_names)
        outs_str = f" | {outs}" if outs else ""
        badge_lines.append(f"  {st:>8} | {t:>6.3f}s | saved {m.saved_time:>6.3f}s | {code}{outs_str}")
    badge_text = "\n".join(badge_lines) if badge_lines else "(no metrics)"

    def _nb_source(max_chars: int) -> str:
        if not notebook_source:
            return ""
        parts = []
        for i, cell in enumerate(notebook_source, 1):
            snippet = cell.strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3] + "..."
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

    truncated_note = "> **Note:** Notebook source was too large to include. Please paste the relevant cells below.\n\n"
    body = skeleton.format(badge=badge_text, nb_source=truncated_note)
    if _url_len(body) <= _BUG_URL_MAX:
        return prefix + quote(body)

    if len(badge_text) > 500:
        badge_text = badge_text[:500] + "\n  ... (truncated)"
    body = skeleton.format(badge=badge_text, nb_source=truncated_note)
    return prefix + quote(body)


# ---------------------------------------------------------------------------
# Top-level builder
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
    metrics = [_Metric.parse(m) for m in metrics_list or []]

    upstream = [m for m in metrics if m.is_upstream and m.status is not CacheStatus.SKIPPED]
    upstream_skipped = [m for m in metrics if m.is_upstream and m.status is CacheStatus.SKIPPED]
    current = [m for m in metrics if not m.is_upstream]

    stats = _compute_stats(metrics)
    # Notification and error rows are not counted above but get the warn chip,
    # as does unseeded randomness: its cached value is a frozen replay.
    warn_count = sum(
        1 for m in metrics if m.status in _NOTIFICATIONS or m.status is CacheStatus.ERROR or m.random_unseeded
    )
    # Never advertise more than the cell's NET saving: with its wall time
    # known, cash's own overhead (wall time minus the compute that ran) comes
    # off the gross saving, floored at 0. Without one, the gross value.
    header_saved = stats.total_saved
    if cell_total_time is not None:
        cell_compute = sum(m.execution_time for m in metrics if m.status is CacheStatus.COMPUTED)
        header_saved = max(0.0, stats.total_saved - max(0.0, cell_total_time - cell_compute))
    errored = any(m.status is CacheStatus.ERROR for m in metrics)
    header = BadgeHeader(
        status=_header_status(status, stats, errored),
        restored_count=stats.restored,
        computed_count=stats.computed,
        skipped_count=stats.skipped,
        uncacheable_count=stats.uncacheable,
        warn_count=warn_count,
        total_saved_s=header_saved,
        total_exec_s=cell_total_time if cell_total_time is not None else stats.total_exec,
        current_step=current_step,
        total_steps=total_steps,
        current_code=current_code,
    )

    sections: list[Section] = []
    if upstream or upstream_skipped:
        # In the notebook's order, not restores first.
        items: list[SectionItem] = _nodes(upstream)
        bucket = _skipped_bucket(upstream_skipped)
        if bucket is not None:
            items.append(bucket)
        sections.append(Section(kind=SectionKind.UPSTREAM, header="UPSTREAM HISTORY", items=tuple(items)))

    sections.append(
        Section(
            kind=SectionKind.CURRENT,
            header="CURRENT CELL" if (upstream or upstream_skipped) else "",
            items=tuple(_nodes(current)),
        )
    )

    dec_groups = _decorator_groups(current)
    if dec_groups:
        sections.append(Section(kind=SectionKind.DECORATORS, header="DECORATOR CACHE (@cash.cache)", items=dec_groups))

    overhead = _overhead_section(timing_breakdown, cell_total_time, metrics)
    if overhead is not None:
        sections.append(overhead)

    return InteractiveBadge(
        header=header,
        sections=tuple(sections),
        footer=BugReportLink(url=_bug_report_url(metrics, bug_report_context)),
        configured_tiers=tuple(configured_tiers),
    )


__all__ = [
    "build_interactive_badge",
    "build_bug_report_url",
    "build_sub_unit_groups",
    "map_status",
]
