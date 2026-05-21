from __future__ import annotations

"""Lineage graph tracking for variable computation history."""

import json
import time
from dataclasses import dataclass, field

__all__ = ["ProvenanceRecord", "ProvenanceTracker"]

@dataclass
class ProvenanceRecord:
    """A single computation event in the provenance graph."""
    variable: str
    code: str
    inputs: list[str]
    timestamp: float
    cell_index: int | None = None
    status: str = "computed"  # computed, restored, skipped
    duration_ms: float = 0.0
    lineage_hash: str = ""
    file_deps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "code": self.code,
            "inputs": self.inputs,
            "timestamp": self.timestamp,
            "cell_index": self.cell_index,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "lineage_hash": self.lineage_hash[:16] + "..." if self.lineage_hash else "",
            "file_deps": self.file_deps,
        }

class ProvenanceTracker:
    """Tracks the full provenance graph for all computed variables.

    Each variable gets a list of ProvenanceRecords showing how it was
    created or modified over time. This enables:

    - **Time-travel**: See the value of a variable at any point in history
    - **Dependency graph**: Visualize which variables depend on which
    - **Audit trail**: Know exactly how a result was produced
    - **Reproducibility**: Replay the exact sequence of computations
    """

    def __init__(self):
        # variable_name -> list of ProvenanceRecords (ordered by time)
        self._history: dict[str, list[ProvenanceRecord]] = {}
        # All records in chronological order
        self._timeline: list[ProvenanceRecord] = []
        # Maximum records per variable (to prevent unbounded growth)
        self.max_history_per_var = 50
        # Maximum total timeline entries
        self.max_timeline = 1000

    def record(self, variable: str, code: str, inputs: list[str],
               status: str = "computed", duration_ms: float = 0.0,
               lineage_hash: str = "", cell_index: int = None,
               file_deps: list[str] = None):
        """Record a computation event.

        Args:
            variable: The output variable name.
            code: The source code that produced it.
            inputs: List of input variable names.
            status: 'computed', 'restored', or 'skipped'.
            duration_ms: Time taken in milliseconds.
            lineage_hash: The lineage hash for this computation.
            cell_index: Which cell this was executed in.
            file_deps: File dependencies used.
        """
        record = ProvenanceRecord(
            variable=variable,
            code=code,
            inputs=inputs,
            timestamp=time.time(),
            cell_index=cell_index,
            status=status,
            duration_ms=duration_ms,
            lineage_hash=lineage_hash,
            file_deps=file_deps or [],
        )

        # Add to variable history
        if variable not in self._history:
            self._history[variable] = []
        self._history[variable].append(record)

        # Trim per-variable history
        if len(self._history[variable]) > self.max_history_per_var:
            self._history[variable] = self._history[variable][-self.max_history_per_var:]

        # Add to timeline
        self._timeline.append(record)
        if len(self._timeline) > self.max_timeline:
            self._timeline = self._timeline[-self.max_timeline:]

    def get_history(self, variable: str) -> list[ProvenanceRecord]:
        """Get the computation history for a variable."""
        return self._history.get(variable, [])

    def get_latest(self, variable: str) -> ProvenanceRecord | None:
        """Get the most recent computation record for a variable."""
        history = self._history.get(variable, [])
        return history[-1] if history else None

    def get_dependencies(self, variable: str) -> set[str]:
        """Get all transitive dependencies for a variable.

        Returns the set of all variables that directly or indirectly
        contribute to this variable's value.
        """
        deps = set()
        visited = set()

        def _walk(var_name):
            if var_name in visited:
                return
            visited.add(var_name)
            latest = self.get_latest(var_name)
            if latest:
                for inp in latest.inputs:
                    deps.add(inp)
                    _walk(inp)

        _walk(variable)
        return deps

    def get_dependents(self, variable: str) -> set[str]:
        """Get all variables that depend on this variable."""
        dependents = set()
        for var_name, history in self._history.items():
            if history:
                latest = history[-1]
                if variable in latest.inputs:
                    dependents.add(var_name)
        return dependents

    def get_timeline(self, limit: int = 20,
                     variable: str = None) -> list[ProvenanceRecord]:
        """Get recent timeline entries.

        Args:
            limit: Maximum entries to return.
            variable: Optional filter by variable name.
        """
        filtered = [r for r in self._timeline if r.variable == variable] if variable else self._timeline
        return filtered[-limit:]

    def _format_graph_section(self, variable: str, max_depth: int = 5) -> list:
        """Return lines for the dependency-graph block.

        Renders a real tree: each level is indented by depth, with box-drawing
        connectors that show parent/child relationships. Self-references and
        already-visited nodes are skipped to break cycles. Inputs without a
        provenance record (typically imported modules and built-ins picked up
        from AST analysis, e.g. ``np``, ``len``) render as ``(external)`` leaves
        so the tree doesn't try to expand them further.
        """
        lines = ["", "  Dependency Graph:"]
        visited = {variable}
        rendered_any = False

        def _walk(var: str, prefix: str) -> None:
            nonlocal rendered_any
            latest = self.get_latest(var)
            if latest is None or not latest.inputs:
                return
            # Filter out self-references and already-visited nodes.
            deps_here = [inp for inp in latest.inputs if inp != var and inp not in visited]
            if not deps_here:
                return
            # Render in a stable order (sorted), with the last sibling using
            # └─ and earlier siblings using ├─ so the tree reads cleanly.
            depth_now = (len(prefix) // 3) if prefix else 0
            for i, inp in enumerate(sorted(deps_here)):
                is_last = (i == len(deps_here) - 1)
                connector = "└─ " if is_last else "├─ "
                inp_latest = self.get_latest(inp)
                if inp_latest is None:
                    suffix = " (external)"
                else:
                    suffix = f" ← {inp_latest.code.strip()[:40]}"
                lines.append(f"    {prefix}{connector}{inp}{suffix}")
                rendered_any = True
                visited.add(inp)
                # Only recurse into nodes that have their own provenance and
                # haven't exceeded the depth cap (avoids deep / wide blowups
                # on chains of pure helpers).
                if inp_latest is not None and depth_now + 1 < max_depth:
                    child_prefix = prefix + ("   " if is_last else "│  ")
                    _walk(inp, child_prefix)

        _walk(variable, "")
        if not rendered_any:
            lines.append("    (no dependencies)")

        dependents = self.get_dependents(variable) - {variable}
        if dependents:
            lines.append(f"  Dependents: {', '.join(sorted(dependents))}")
        return lines

    def _format_timeline_section(self, history: list) -> list:
        """Return lines for the computation-timeline block."""
        lines = ["", "  Timeline:"]
        for record in history[-10:]:
            ts = time.strftime('%H:%M:%S', time.localtime(record.timestamp))
            icon = {"computed": "🔧", "restored": "📦", "skipped": "⏭️"}.get(record.status, "❓")
            lines.append(f"    {ts} {icon} {record.status} ({record.duration_ms:.1f}ms)")
        return lines

    def format_provenance(self, variable: str,
                          show_graph: bool = False,
                          show_timeline: bool = False) -> str:
        """Format provenance info as a readable string.

        Args:
            variable: The variable to show provenance for.
            show_graph: Include dependency graph visualization.
            show_timeline: Include computation timeline.
        """
        lines = []
        history = self.get_history(variable)

        if not history:
            return f"No provenance recorded for '{variable}'"

        latest = history[-1]
        lines.append(f"📋 Provenance for '{variable}':")
        lines.append(f"  Last computed: {time.strftime('%H:%M:%S', time.localtime(latest.timestamp))}")
        lines.append(f"  Status: {latest.status}")
        lines.append(f"  Code: {latest.code.strip()[:80]}")
        if latest.inputs:
            lines.append(f"  Inputs: {', '.join(latest.inputs)}")
        if latest.file_deps:
            lines.append(f"  File deps: {', '.join(latest.file_deps)}")
        if latest.duration_ms > 0:
            lines.append(f"  Duration: {latest.duration_ms:.1f}ms")
        lines.append(f"  History: {len(history)} records")

        if show_graph:
            lines.extend(self._format_graph_section(variable))
        if show_timeline:
            lines.extend(self._format_timeline_section(history))

        return "\n".join(lines)

    def to_json(self, variable: str = None) -> str:
        """Export provenance data as JSON.

        Args:
            variable: Export for specific variable, or all if None.
        """
        records = self.get_history(variable) if variable else self._timeline

        return json.dumps(
            [r.to_dict() for r in records],
            indent=2,
            default=str
        )

    def clear(self):
        self._history.clear()
        self._timeline.clear()

    @property
    def tracked_variables(self) -> set[str]:
        """Set of all variables with provenance data."""
        return set(self._history.keys())

