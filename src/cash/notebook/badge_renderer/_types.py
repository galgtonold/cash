"""Badge type definitions and display constants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from ..statement_processor import ProcessResult

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
_CODE_SNIPPET_MAX_LEN = 120  # Characters to show before truncating code snippets
_HEADER_MAX_LEN = 80         # Characters to show before truncating header lines
_MIN_TIME_DISPLAY_S = 0.01   # Minimum time (seconds) worth displaying in badges
_MIN_TIME_DISPLAY_MS = 0.001 # Minimum time (milliseconds) worth displaying

# ---------------------------------------------------------------------------
# Badge TypedDicts
# ---------------------------------------------------------------------------

class LoopGroup(TypedDict):
    """A group of iteration metrics for a single loop body statement."""

    type: str  # Literal['loop_group']
    base_code: str
    metrics: list[ProcessResult]
    loop_var_name: str | None
    loop_var_values: list[Any]
    all_loop_var_names: list[str]
    all_loop_var_values: dict[str, list[Any]]

class ForLoopGroup(TypedDict):
    """Wrapper around consecutive loop groups for a single ``for`` loop."""

    type: str  # Literal['for_loop_group']
    stmt_groups: list[LoopGroup]
    loop_var_name: str | None
    loop_var_values: list[Any]

class ControlGroup(TypedDict):
    """Grouped metrics for an if/elif/else control structure."""

    type: str  # Literal['control_group']
    metrics: list[ProcessResult]
    branch_label: str
    header: str

class ControlGroupSingle(TypedDict):
    """A single control structure metric (standalone if/else)."""

    type: str  # Literal['control_group_single']
    metric: ProcessResult

class SingleMetric(TypedDict):
    """A non-grouped single statement metric."""

    type: str  # Literal['single']
    metric: ProcessResult

GroupedBadgeItem = SingleMetric | ForLoopGroup | ControlGroup | ControlGroupSingle

# ---------------------------------------------------------------------------
# Simple status badge colors
# ---------------------------------------------------------------------------
_BADGE_COLOR_RESTORED = "#006644"
_BADGE_COLOR_DEFAULT = "#996300"
