"""Data grouping pipeline for badge metrics.

Groups raw per-statement metrics into structured items (loop groups,
control groups, single statements) for hierarchical badge display.
"""

from __future__ import annotations

import random as _rnd
import threading as _threading
from typing import Any

from ._types import (
    ForLoopGroup,
    GroupedBadgeItem,
    LoopGroup,
)

# ---------------------------------------------------------------------------
# Utility helpers (shared across rendering submodules)
# ---------------------------------------------------------------------------

# Counter-based unique ID generator — produces stable, deterministic IDs
# within a single render pass.  Call ``_reset_unique_ids()`` at the start
# of each ``render_interactive_badge()`` so the same badge structure always
# generates the same element IDs (needed to preserve expanded/collapsed
# state across progress-badge re-renders).
_id_counter = 0
_id_lock = _threading.Lock()

def format_loop_var(value: Any) -> str:
    """Format a loop variable value for display in the badge.

    Truncates long strings and uses repr for non-string types.
    """
    text = repr(value) if not isinstance(value, str) else value
    if len(text) > 30:
        text = text[:27] + "..."
    return text


def _reset_unique_ids() -> None:
    """Reset the counter so the next render pass starts from 1."""
    global _id_counter
    with _id_lock:
        _id_counter = 0


def _unique_id(prefix: str = "id") -> str:
    """Generate a short deterministic CSS-safe identifier.

    IDs are stable across re-renders as long as the badge structure
    (number and order of grouped items) stays the same.
    """
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return f"{prefix}_{_id_counter}"


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for badge content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _unique_ordered(items: list) -> list:
    """Deduplicate while preserving order."""
    seen: set = set()
    result: list = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Data grouping
# ---------------------------------------------------------------------------

def _flush_loop_groups_into(
    loop_stmt_groups: dict[str, list[dict[str, Any]]],
    intermediate: list[dict[str, Any]],
) -> None:
    """Flush pending loop groups into intermediate, clearing the dict."""
    if loop_stmt_groups:
        for base_code, metrics in loop_stmt_groups.items():
            intermediate.append(create_loop_group(base_code, metrics))
        loop_stmt_groups.clear()


def _flush_control_groups_into(
    control_groups: dict[str, list[dict[str, Any]]],
    intermediate: list[dict[str, Any]],
) -> None:
    """Flush pending control groups into intermediate, clearing the dict."""
    if control_groups:
        for _ctx_hash, metrics in control_groups.items():
            branch_label = metrics[0].get('branch_label', '')
            body_stmts = metrics[0].get('body_statements', [])
            header = body_stmts[0] if body_stmts else branch_label
            intermediate.append({
                'type': 'control_group',
                'metrics': metrics,
                'branch_label': branch_label,
                'header': header,
            })
        control_groups.clear()


def _group_pass1_intermediate(metrics_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pass 1: detect loop/control contexts and group into intermediate items."""
    intermediate: list[dict[str, Any]] = []
    loop_stmt_groups: dict[str, list[dict[str, Any]]] = {}
    control_groups: dict[str, list[dict[str, Any]]] = {}

    for m in metrics_list:
        code = m.get('code', '')

        if '# __iteration_context__:' in code:
            _flush_control_groups_into(control_groups, intermediate)
            lines = code.split('\n')
            actual_code = '\n'.join(
                line for line in lines
                if not line.startswith('# __iteration_context__:')
                and not line.startswith('# control_context:')
            )
            if actual_code not in loop_stmt_groups:
                loop_stmt_groups[actual_code] = []
            loop_stmt_groups[actual_code].append(m)

        elif m.get('control_context'):
            _flush_loop_groups_into(loop_stmt_groups, intermediate)
            ctx_hash = m['control_context']
            if ctx_hash not in control_groups:
                control_groups[ctx_hash] = []
            control_groups[ctx_hash].append(m)

        else:
            _flush_loop_groups_into(loop_stmt_groups, intermediate)
            _flush_control_groups_into(control_groups, intermediate)
            if m.get('body_statements') and not m.get('control_context'):
                intermediate.append({'type': 'control_group_single', 'metric': m})
            else:
                intermediate.append({'type': 'single', 'metric': m})

    _flush_loop_groups_into(loop_stmt_groups, intermediate)
    _flush_control_groups_into(control_groups, intermediate)
    return intermediate


def _wrap_pass2_loop_groups(intermediate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pass 2: wrap consecutive loop_group items into for_loop_group."""
    result: list[dict[str, Any]] = []
    pending_loop_groups: list[dict[str, Any]] = []

    def _flush_pending_loops() -> None:
        if not pending_loop_groups:
            return
        first_grp = pending_loop_groups[0]
        result.append({
            'type': 'for_loop_group',
            'stmt_groups': list(pending_loop_groups),
            'loop_var_name': first_grp.get('loop_var_name'),
            'loop_var_values': first_grp.get('loop_var_values', []),
        })
        pending_loop_groups.clear()

    for item in intermediate:
        if item['type'] == 'loop_group':
            pending_loop_groups.append(item)
        else:
            _flush_pending_loops()
            result.append(item)

    _flush_pending_loops()
    return result


def group_loop_iterations(metrics_list: list[dict[str, Any]]) -> list[GroupedBadgeItem]:
    """Group metrics from loops and control structures for collapsed badge display.

    Processing pipeline:

    1. Detect ``# __iteration_context__:`` metrics → group by base code into
       ``loop_group`` items (one per statement, all iterations collapsed).
    2. Detect ``control_context`` metrics → group into ``control_group``
       items (one per if/elif/else block with per-statement detail).
    3. Wrap consecutive ``loop_group`` items into a single
       ``for_loop_group`` so the badge shows one expandable "for" row.

    Returns a list of items with types:

    - ``{'type': 'single', 'metric': m}``
    - ``{'type': 'for_loop_group', 'stmt_groups': [...], ...}``
    - ``{'type': 'control_group', 'metrics': [...], ...}``
    - ``{'type': 'control_group_single', 'metric': m}``
    """
    intermediate = _group_pass1_intermediate(metrics_list)
    return _wrap_pass2_loop_groups(intermediate)

def create_loop_group(base_code: str, metrics: list[dict[str, Any]]) -> LoopGroup:
    """Create a loop group from a list of iteration metrics for the same statement.

    Extracts loop variable information from ``loop_vars`` in metrics.
    Supports multiple loop vars (e.g., ``for a, b in zip(...)``).
    """
    all_var_names: list = []
    var_values: dict[str, list] = {}

    for m in metrics:
        loop_vars = m.get('loop_vars', {})
        for k, v in loop_vars.items():
            if k not in var_values:
                all_var_names.append(k)
                var_values[k] = []
            var_values[k].append(v)

    loop_var_name = all_var_names[0] if all_var_names else None
    loop_var_values = var_values.get(loop_var_name, []) if loop_var_name else []

    return {
        'type': 'loop_group',
        'base_code': base_code,
        'metrics': metrics,
        'loop_var_name': loop_var_name,
        'loop_var_values': loop_var_values,
        'all_loop_var_names': all_var_names,
        'all_loop_var_values': var_values,
    }
