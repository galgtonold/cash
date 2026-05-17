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
* The node set is closed: every visible row in the legacy renderer must be
  representable here. New badge variants add a node, never a string
  discriminator.
* Nodes carry only data the renderer needs. Raw metric dicts and bug-report
  context are absorbed by the view-builder and never leak into the tree.

See CONTEXT.md → ``BadgeView`` for the architectural framing and the list
of consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    MIXED = "mixed"                    # summary header when a cell has both restored + computed
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


# ---------------------------------------------------------------------------
# Row nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatementRow:
    """A single non-loop, non-control statement row."""

    status: BadgeStatus
    code: str
    time_s: float                                # for RESTORED rows this is the saved time
    storage_tiers: tuple[str, ...] = ()          # e.g. ('RAM',), ('RAM', 'DISK')
    source: str | None = None                    # restore source, e.g. 'RAM', 'DISK'
    output_vars: tuple[str, ...] = ()            # named outputs (vars assigned)
    uncacheable_reasons: tuple[str, ...] = ()    # populated when status==COMPUTED but not cached
    skipped_reason: str | None = None            # populated when the row was skipped at cache time
    is_upstream: bool = False                    # affects icon (⬆️ vs ⚡/⚙️)


@dataclass(frozen=True)
class IterationRow:
    """One iteration of a loop-body statement."""

    status: BadgeStatus
    code: str
    time_s: float
    storage_tiers: tuple[str, ...] = ()
    loop_bindings: tuple[tuple[str, Any], ...] = ()
    """Resolved loop variables for this iteration (denormalised onto the row).

    Stored as a tuple of ``(name, value)`` pairs (not a dict) so the whole
    node remains hashable. Renderers iterate in order.
    """


@dataclass(frozen=True)
class LoopStatement:
    """All iterations of one statement within a ``for`` loop body."""

    base_code: str
    iterations: tuple[IterationRow, ...]


@dataclass(frozen=True)
class ForLoopGroup:
    """A ``for`` loop and its per-statement iteration groups."""

    loop_var_names: tuple[str, ...]
    stmts: tuple[LoopStatement, ...]


@dataclass(frozen=True)
class ControlGroup:
    """Grouped metrics for an ``if``/``elif``/``else`` branch."""

    branch_label: str
    header: str
    rows: tuple[StatementRow, ...]


@dataclass(frozen=True)
class ControlGroupSingle:
    """A single-statement control structure (e.g. standalone ``if`` with no body grouping)."""

    row: StatementRow


@dataclass(frozen=True)
class ControlBody:
    """The expandable ``<details>`` body listing the source statements inside a control block."""

    body_stmts: tuple[str, ...]


@dataclass(frozen=True)
class SkippedBucket:
    """The collapsed-by-default group of intermediate-dependency skipped steps.

    Sits inside an :class:`Section` with ``kind=SectionKind.UPSTREAM``.
    """

    items: tuple[Union["StatementRow", "ForLoopGroup"], ...]
    total_saved_time_s: float


@dataclass(frozen=True)
class DecoratorCall:
    """A single ``@cash.cache`` call event."""

    func_name: str
    status: BadgeStatus                          # RESTORED for hit, COMPUTED for miss
    time_s: float


@dataclass(frozen=True)
class DecoratorCallGroup:
    """All calls to one decorated function, with a condense-when->3 hint."""

    func_name: str
    calls: tuple[DecoratorCall, ...]
    condensed: bool


@dataclass(frozen=True)
class OverheadEntry:
    """One line of the overhead breakdown subsection."""

    label: str
    time_s: float


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
    ControlBody,
    SkippedBucket,
    DecoratorCallGroup,
    OverheadBreakdown,
]


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
    total_saved_s: float = 0.0
    total_exec_s: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_code: str | None = None


@dataclass(frozen=True)
class BugReportLink:
    """The footer 'Report incorrect caching' link. URL is pre-computed by the view-builder."""

    url: str


@dataclass(frozen=True)
class InteractiveBadge:
    """Root node of the expandable, multi-section badge.

    Produced by :func:`cash.notebook.badge_renderer.view_builder.build_view`
    from a ``metrics_list``.
    """

    header: BadgeHeader
    sections: tuple[Section, ...]
    footer: BugReportLink | None = None


@dataclass(frozen=True)
class StatusBadge:
    """Compact non-interactive badge (the binary restored/computed pill).

    Produced from a single ``ProcessResult``-like dict by the view-builder.
    Today the legacy ``render_status_badge`` uses only a two-colour palette;
    the IR keeps the same prop set so the renderer can decide whether to
    expand the palette in the future without changing the data model.
    """

    status: BadgeStatus
    execution_time_s: float
    time_saved_s: float
    source: str | None = None
    storage_tiers: tuple[str, ...] = ()


__all__ = [
    "BadgeStatus",
    "SectionKind",
    "StatementRow",
    "IterationRow",
    "LoopStatement",
    "ForLoopGroup",
    "ControlGroup",
    "ControlGroupSingle",
    "ControlBody",
    "SkippedBucket",
    "DecoratorCall",
    "DecoratorCallGroup",
    "OverheadEntry",
    "OverheadBreakdown",
    "SectionItem",
    "Section",
    "BadgeHeader",
    "BugReportLink",
    "InteractiveBadge",
    "StatusBadge",
]
