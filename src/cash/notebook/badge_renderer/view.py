"""BadgeView — the component-tree IR for badge rendering.

This module defines the *semantic* node types that describe what a badge
should show, independent of output format. Renderers (HTML today; Text
and React future) consume the tree and own all presentation decisions
(colors, fonts, layout) via :mod:`.theme`.

Design rules:

* Nodes are **frozen dataclasses** — once a view is built, it is immutable.
* Props are **semantic**, never presentational. ``status=BadgeStatus.RESTORED``
  is correct; ``color='#006644'`` is not — color lives in the theme module.
* Lists are stored as :class:`tuple` so the whole tree is hashable.
* The node set is closed: every visible row in the badge must be
  representable here. New badge variants add a node, never a string
  discriminator.
* Nodes carry only data the renderer needs. Raw metric dicts and bug-report
  context are absorbed by the view-builder and never leak into the tree.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

# ---------------------------------------------------------------------------
# Status enum — display-layer concept (broader than ``CacheStatus``).
# ---------------------------------------------------------------------------


class BadgeStatus(str, Enum):
    """Display status for a row, header, or badge.

    Broader than :class:`cash.notebook.cache_status.CacheStatus` because it
    includes purely-presentational values (``MIXED``) and upstream-change
    notifications (``FUNCTION_CHANGED``, ``MODULE_RELOADED``, ``WARNING``)
    that the cache subsystem does not model.

    The view-builder maps from the runtime status string (or
    ``CacheStatus`` instance) to one of these values; renderers map each
    value to (icon, color, label) via :mod:`.theme`.
    """

    RESTORED = "restored"
    COMPUTED = "computed"
    SKIPPED = "skipped"
    MIXED = "mixed"  # summary header when a cell has both restored + computed
    # A header-only state: the cell is still running. Previously "RUNNING" was
    # squeezed into WARNING, which the summary renderer does not read, so a
    # progress badge published without step information fell through to
    # EXECUTED and claimed the cell had finished in 0.00s.
    RUNNING = "running"
    # Also header-only: cash aborted its own pipeline and handed the cell to
    # IPython uncached. Explicit rather than inferred from "the header has no
    # rows", because a cell of imports or magics has no rows either and has
    # not been bypassed -- inferring it put "cash stepped aside" on the first
    # cell of the demo tour.
    BYPASSED = "bypassed"
    ERROR = "error"
    FUNCTION_CHANGED = "function_changed"
    MODULE_RELOADED = "module_reloaded"
    WARNING = "warning"


class SectionKind(str, Enum):
    """Identifies the role of a :class:`Section` within an interactive badge."""

    UPSTREAM = "upstream"
    CURRENT = "current"
    DECORATORS = "decorators"
    OVERHEAD = "overhead"


@dataclass(frozen=True)
class Rollup:
    """Totals over every leaf row under a node, computed once as it is built.

    A leaf is a :class:`StatementRow` or an :class:`IterationRow`. Each
    container sums its children's rollups, so a loop head, a control header,
    the upstream head, the sparkline and the bar scale read the same numbers.
    """

    time_s: float = 0.0
    saved_s: float = 0.0
    leaves: int = 0
    cached: int = 0  # RESTORED or SKIPPED
    computed: int = 0

    @classmethod
    def of_leaf(cls, status: BadgeStatus, time_s: float, saved_s: float) -> Rollup:
        return cls(
            time_s=time_s,
            saved_s=saved_s,
            leaves=1,
            cached=int(status in (BadgeStatus.RESTORED, BadgeStatus.SKIPPED)),
            computed=int(status is BadgeStatus.COMPUTED),
        )

    @classmethod
    def of(cls, items: Iterable[Any]) -> Rollup:
        """The sum over *items*; a node without leaf rows contributes nothing."""
        time_s = saved_s = 0.0
        leaves = cached = computed = 0
        for item in items:
            r = getattr(item, "rollup", None)
            if r is None:
                continue
            time_s += r.time_s
            saved_s += r.saved_s
            leaves += r.leaves
            cached += r.cached
            computed += r.computed
        return cls(time_s, saved_s, leaves, cached, computed)

    @property
    def kind(self) -> str:
        """``cached`` when something was served and nothing computed, else ``exec``."""
        return "cached" if self.cached and not self.computed else "exec"

    @property
    def mixed(self) -> bool:
        """Some leaves came from the cache and some did not."""
        return 0 < self.cached < self.leaves


def _rollup_field() -> Any:
    return field(init=False, repr=False, compare=False)


def _set_rollup(node: Any, rollup: Rollup) -> None:
    object.__setattr__(node, "rollup", rollup)


# ---------------------------------------------------------------------------
# Row nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatementRow:
    """A single non-loop, non-control statement row.

    ``time_s`` is always the *actual* elapsed time of the operation (cache
    restore overhead for RESTORED rows, real compute time for COMPUTED).
    ``saved_time_s`` is the wall-clock saving the cache produced — zero
    for COMPUTED rows that paid the full price.
    """

    status: BadgeStatus
    code: str
    time_s: float
    #: The statement as the user WROTE it, for display only. ``None`` when the
    #: original source could not be recovered (a control body, a loop-split
    #: iteration, a statement cash rewrote) -- renderers fall back to ``code``.
    display_code: str | None = None
    saved_time_s: float = 0.0
    storage_tiers: tuple[str, ...] = ()  # e.g. ('RAM',), ('RAM', 'DISK')
    source: str | None = None  # restore source, e.g. 'RAM', 'DISK'
    output_vars: tuple[str, ...] = ()  # what the row produced this time
    restored_vars: tuple[str, ...] = ()  # vars hydrated from cache (RESTORED rows)
    uncacheable_reasons: tuple[str, ...] = ()  # populated when status==COMPUTED but not cached
    skipped_reason: str | None = None  # populated when the row was skipped at cache time
    guard_cause: str | None = None  # for "unstable key": what kept changing the key
    #: What an upstream statement the repair re-ran printed (stdout, stderr).
    #: Kept out of the cell's own output -- it belongs to another cell -- and
    #: shown under the step instead (round 29, r29s3).
    output_text: str = ""
    # Notification-row metadata (FUNCTION_CHANGED / MODULE_RELOADED rows).
    changed_functions: tuple[str, ...] = ()
    changed_modules: tuple[str, ...] = ()
    # Inline detail surfaced by the renderer (tooltip / drill-down).
    decorator_calls: tuple["DecoratorCall", ...] = ()
    # For single-unit control structures (``while``, ``with``, ``try``) the
    # runtime records the original source lines (header + body) here so the
    # renderer can show them expanded — header line as the row code, body
    # lines as nested static rows — instead of one opaque multi-line row.
    body_statements: tuple[str, ...] = ()
    # Short prefix of the statement's cache key, shown in the row's
    # expanded detail. Lets a user see whether two re-runs of the same
    # statement landed in the same cache slot.
    cache_key_short: str = ""
    # One-line attribution for COMPUTED rows, so the badge can answer "why did
    # this re-run?". Only the reasons the runtime works out anyway are here --
    # the earlier ``_diagnose_miss`` fallback that probed the backend to
    # explain the rest was O(N^2) in cache size and was deleted in the
    # 2026-05-18 overhead pass. Nothing may reintroduce a cache walk to fill a
    # gap below.
    #
    # Populated, and by whom:
    #   "cache TTL expired (...)"      statement/freshness.py
    #   "file changed: ..."            statement/freshness.py
    #   "input changed: x, y"          upstream/checker.py, from the backward
    #                                  scan's pre-execution lineage comparison
    #   "re-run to restore the random stream"   upstream/checker.py
    #
    # Deliberately NOT populated: the first run of a statement. Proving the
    # absence of an entry is what made the old fallback expensive, and "first
    # time" is self-evident to someone running a cell for the first time.
    # ``tests/test_notebook/test_miss_reason.py`` pins that absence on purpose.
    miss_reason: str | None = None
    # RNG role, surfaced as a chip: 'seed' sets a global seed, 'draw' consumes
    # randomness, None for an ordinary statement. ``random_unseeded`` marks a
    # draw/fit with no frozen seed — its cached value is a frozen replay.
    random_effect: str | None = None
    random_unseeded: bool = False
    # Per-call-SITE groups of intercepted (on by default, CAS-243) sub-calls
    # made from inside this statement (CAS-243 call-unit caching). Distinct
    # from ``decorator_calls`` above: that field is flat and keyed on nothing
    # in particular, which is fine for the "@cache: N/M hits" summary but
    # wrong for debugging — see :class:`SubUnitGroup` for why site matters.
    sub_units: tuple["SubUnitGroup", ...] = ()
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of_leaf(self.status, self.time_s, self.saved_time_s))


@dataclass(frozen=True)
class IterationRow:
    """One iteration of a loop-body statement."""

    status: BadgeStatus
    code: str
    time_s: float
    saved_time_s: float = 0.0
    storage_tiers: tuple[str, ...] = ()
    # Why this iteration re-ran, or why it was not cached -- the same two
    # fields ``StatementRow`` carries, for the same reason. They were missing
    # here, so every reason the runtime worked out was computed and then
    # dropped on the floor for anything inside a loop: the row rendered with
    # no attribution at all. Loops are where the expensive work is, which
    # makes this the worst place to go quiet.
    miss_reason: str | None = None
    skipped_reason: str | None = None
    loop_bindings: tuple[tuple[str, Any], ...] = ()
    """Resolved loop variables for this iteration (denormalised onto the row).

    Stored as a tuple of ``(name, value)`` pairs (not a dict) so the whole
    node remains hashable. Renderers iterate in order.
    """
    # Per-call-SITE groups of this iteration's intercepted sub-calls (CAS-243
    # task 9) — same field, same purpose as ``StatementRow.sub_units``. A
    # loop-body statement renders as an ``IterationRow``, not a
    # ``StatementRow`` (see ``view_builder._iteration_row``), so it needs
    # its own copy of this field or a sub-call made inside a loop body is
    # silently dropped rather than merely misplaced.
    sub_units: tuple["SubUnitGroup", ...] = ()
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of_leaf(self.status, self.time_s, self.saved_time_s))


@dataclass(frozen=True)
class LoopStatement:
    """All iterations of one statement within a ``for`` loop body."""

    base_code: str
    iterations: tuple[IterationRow, ...]
    # Per-call-site sub-call groups AGGREGATED across every iteration (CAS-243
    # task 9) -- built from the union of all iterations' raw ``decorator_calls``
    # events, re-grouped by ``(call_source, occurrence_index)``. Renderers that
    # show this statement as one collapsed aggregate row (the HTML renderer's
    # loop-body row) show this instead of per-iteration ``IterationRow.sub_units``,
    # which stay available for renderers (text) that keep iterations separate.
    sub_units: tuple["SubUnitGroup", ...] = ()
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of(children(self)))


@dataclass(frozen=True)
class ForLoopGroup:
    """A ``for`` loop and its per-statement iteration groups."""

    loop_var_names: tuple[str, ...]
    stmts: tuple[LoopStatement, ...]
    # Source-faithful header line ("for cat in df['category'].unique():"). The
    # runtime captures this from the AST and stamps every body metric with
    # it. Empty when the metric source predates the loop_header field;
    # renderers fall back to constructing a header from loop_var_names +
    # iteration values in that case.
    loop_header: str = ""
    # Loops or controls that live INSIDE this for-loop's body. The runtime
    # records each metric's full enclosing-loop header chain; the view
    # builder uses that to nest a child for-loop's items under its parent
    # instead of rendering them as siblings. Renderers walk this after
    # ``stmts`` so the body reads stmts-first then nested-loops/controls.
    nested: tuple[Any, ...] = ()
    # Set when the for-loop is rendered as the inner head-less wrapper
    # under a hoisted outer (e.g. `for c in range(2): if cond: body` where
    # the renderer draws the for-c header at the outer position and this
    # node provides the iteration data without re-drawing the head).
    suppress_head: bool = False
    # Body of the for-loop in source order: a mixed tuple of
    # ``LoopStatement`` (direct body stmts) and ``SectionItem`` (nested
    # for-loops, control-groups). Renderers iterate this when they need
    # source order; the type-segregated ``stmts`` / ``nested`` views
    # above stay because aggregation passes (timing sums, status roll-up)
    # walk only one type at a time.
    body: tuple[Any, ...] = ()
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of(children(self)))


@dataclass(frozen=True)
class ControlGroup:
    """Grouped metrics for an ``if``/``elif``/``else`` branch.

    ``rows`` can contain any :data:`SectionItem` — a control body may
    itself contain loops or further control groups. Renderers dispatch
    on item type when walking the body. Type is intentionally loose
    (``tuple[Any, ...]``) because the closed union :data:`SectionItem`
    is defined below and would create a forward-reference cycle.
    """

    branch_label: str
    header: str
    rows: tuple[Any, ...]
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of(children(self)))


@dataclass(frozen=True)
class ControlGroupSingle:
    """A single-statement control structure (e.g. standalone ``if`` with no body grouping)."""

    row: StatementRow
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of(children(self)))


@dataclass(frozen=True)
class SkippedBucket:
    """The collapsed-by-default group of intermediate-dependency skipped steps.

    Sits inside an :class:`Section` with ``kind=SectionKind.UPSTREAM``.
    """

    items: tuple[Union["StatementRow", "ForLoopGroup"], ...]
    total_saved_time_s: float
    #: ``(code, paths)`` of file writers left out of the repair although what
    #: they write changed: their file on disk is out of date (round 28).
    stale_exports: tuple[tuple[str, tuple[str, ...]], ...] = ()
    rollup: Rollup = _rollup_field()

    def __post_init__(self) -> None:
        _set_rollup(self, Rollup.of(children(self)))


@dataclass(frozen=True)
class DecoratorCall:
    """A single ``@cash.cache`` call event."""

    func_name: str
    status: BadgeStatus  # RESTORED for hit, COMPUTED for miss
    time_s: float


@dataclass(frozen=True)
class DecoratorCallGroup:
    """All calls to one decorated function, with a condense-when->3 hint."""

    func_name: str
    calls: tuple[DecoratorCall, ...]
    condensed: bool
    # True when cash wrapped this callee itself via call interception
    # (on by default, CAS-243) rather than the user decorating it. Same
    # cache, same section — but the reader needs to know where it came from.
    # Defaults False so pre-existing metrics keep reading as decorated.
    intercepted: bool = False


@dataclass(frozen=True)
class SubUnitGroup:
    """All calls at ONE intercepted call site.

    Grouped by call SITE (``call_source`` + ``occurrence_index``) rather
    than by callee, unlike :class:`DecoratorCallGroup`: the same function
    called from two different places is two independent cache lines —
    different arguments, different hit/miss history, quite possibly a
    different cache key. Merging them by ``func_name`` alone hides exactly
    what a reader is trying to debug when they open the drawer.

    ``key_prefix`` mirrors ``StatementRow.cache_key_short``: the same
    prefix across re-runs means cash landed in the same slot for this call,
    which is the single most useful field for confirming a sub-call cached
    the way you expect. ``miss_reason`` carries per-site miss attribution
    when the runtime supplies one; it is ``None`` when not (yet) available
    rather than a fabricated guess.
    """

    call_source: str
    occurrence_index: int
    calls: tuple[DecoratorCall, ...]
    condensed: bool
    key_prefix: str
    miss_reason: str | None = None
    #: Calls the many-cheap-calls guard ran without the cache.
    ran_plain: int = 0
    #: Misses whose result was not stored: below the cost floor, or refused.
    unstored: int = 0


@dataclass(frozen=True)
class OverheadEntry:
    """One line of the overhead breakdown subsection.

    ``label`` is kept short so several parts fit on the single collapsed
    overhead row without ellipsis-clipping; ``tooltip`` carries the full
    description for a hover ``title`` in renderers that support it.
    """

    label: str
    time_s: float
    tooltip: str = ""


@dataclass(frozen=True)
class OverheadBreakdown:
    """Per-cell breakdown of cash-internal overhead times."""

    entries: tuple[OverheadEntry, ...]
    total_s: float


# ---------------------------------------------------------------------------
# Container nodes
# ---------------------------------------------------------------------------

# The closed union of node types that can appear inside ``Section.items``.
# Kept as a type alias purely for documentation; renderers dispatch by
# ``isinstance`` against the concrete classes.
SectionItem = Union[
    StatementRow,
    ForLoopGroup,
    ControlGroup,
    ControlGroupSingle,
    SkippedBucket,
    DecoratorCallGroup,
    OverheadBreakdown,
]


def children(item: Any) -> tuple[Any, ...]:
    """The child edges of the badge tree, defined once.

    Every rollup is the sum over these edges, and :func:`iter_leaves` walks
    them, so no two passes can disagree about what a node contains.
    """
    if isinstance(item, LoopStatement):
        return item.iterations
    if isinstance(item, ForLoopGroup):
        return (*item.stmts, *item.nested)
    if isinstance(item, ControlGroup):
        return item.rows
    if isinstance(item, ControlGroupSingle):
        return (item.row,)
    if isinstance(item, SkippedBucket):
        return item.items
    return ()


def iter_leaves(item: Any) -> Iterator[StatementRow | IterationRow]:
    """Every leaf row under *item*, in tree order."""
    if isinstance(item, StatementRow | IterationRow):
        yield item
        return
    for child in children(item):
        yield from iter_leaves(child)


def rollup_of(item: Any) -> Rollup:
    """*item*'s rollup; a node with no leaf rows under it has an empty one."""
    return getattr(item, "rollup", None) or Rollup()


@dataclass(frozen=True)
class Section:
    """A labelled group of items inside an :class:`InteractiveBadge`."""

    kind: SectionKind
    header: str
    items: tuple[SectionItem, ...]


@dataclass(frozen=True)
class BadgeHeader:
    """The summary line shown when the interactive badge is collapsed."""

    status: BadgeStatus
    restored_count: int = 0
    computed_count: int = 0
    skipped_count: int = 0
    # Notification rows: WARNING + FUNCTION_CHANGED + MODULE_RELOADED + ERROR.
    # Surfaced as a separate warn chip in the summary so e.g. a
    # notification-only cell still gets a count badge instead of nothing.
    warn_count: int = 0
    total_saved_s: float = 0.0
    total_exec_s: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_code: str | None = None
    # Rows that ran and were NOT stored (too cheap, a side effect, an
    # unpicklable value, ``# @cash:no-cache``). Counted separately from
    # ``warn_count`` on purpose: a notification is transient and usually
    # self-explanatory, whereas an uncacheable statement is a STANDING
    # property that recurs on every single run and is normally the user's to
    # fix. Folded into ``computed_count`` it was invisible -- a permanently
    # uncacheable cell looked exactly like one the user had just edited.
    # (Appended rather than grouped with the other counts so positional
    # construction of this dataclass keeps working.)
    uncacheable_count: int = 0


@dataclass(frozen=True)
class BugReportLink:
    """The footer 'Report incorrect caching' link. URL is pre-computed by the view-builder."""

    url: str


@dataclass(frozen=True)
class InteractiveBadge:
    """Root node of the expandable, multi-section badge.

    Produced by :func:`cash.notebook.badge_renderer.view_builder.build_interactive_badge`
    from a ``metrics_list``.
    """

    header: BadgeHeader
    sections: tuple[Section, ...]
    footer: BugReportLink | None = None
    # Ordered labels of all tiers the active backend exposes — drives the
    # dot indicator (one dot per configured tier). Empty when the runtime
    # cannot describe the tiers (legacy callers, mocked metric in a
    # standalone test); the renderer falls back to per-row storage_tiers.
    configured_tiers: tuple[str, ...] = ()


__all__ = [
    "BadgeStatus",
    "SectionKind",
    "StatementRow",
    "IterationRow",
    "LoopStatement",
    "ForLoopGroup",
    "ControlGroup",
    "ControlGroupSingle",
    "SkippedBucket",
    "DecoratorCall",
    "DecoratorCallGroup",
    "SubUnitGroup",
    "OverheadEntry",
    "OverheadBreakdown",
    "SectionItem",
    "Rollup",
    "children",
    "iter_leaves",
    "rollup_of",
    "Section",
    "BadgeHeader",
    "BugReportLink",
    "InteractiveBadge",
]
