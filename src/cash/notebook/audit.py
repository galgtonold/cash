from __future__ import annotations

"""Persistent audit trail of cache operations for compliance and debugging."""

import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from typing import Any

__all__ = ["AuditEntry", "AuditLogger"]

logger = logging.getLogger(__name__)

@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: float
    operation: str         # 'cache_hit', 'cache_miss', 'cache_store', 'cache_delete',
                           # 'cache_restore', 'cache_skip', 'cache_invalidate'
    variable: str          # Variable name or cache key
    code: str = ""         # Code that triggered the operation
    status: str = ""       # 'success', 'error'
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['timestamp_str'] = self.timestamp_str
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

class AuditLogger:
    """Persistent audit logger for cache operations.

    Maintains an in-memory buffer and optionally writes to a file.
    """

    def __init__(self, file_path: str | None = None, max_entries: int = 5000):
        self._entries: list[AuditEntry] = []
        self._max_entries = max_entries
        self._file_path = file_path
        self._file_handle = None
        self._enabled = False

    def enable(self, file_path: str | None = None):
        """Enable audit logging, optionally writing to a file."""
        self._enabled = True
        if file_path:
            self._file_path = file_path
        if self._file_path:
            try:
                self._file_handle = open(self._file_path, 'a', encoding='utf-8')  # noqa: SIM115 - persistent handle, not a one-shot operation
            except OSError as e:
                logger.warning("Could not open audit log file: %s", e)
                self._file_handle = None

    def disable(self):
        """Disable audit logging and close file handle."""
        self._enabled = False
        if self._file_handle:
            with contextlib.suppress(OSError):
                self._file_handle.close()
            self._file_handle = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, operation: str, variable: str, code: str = "",
            status: str = "success", duration_ms: float = 0.0,
            **details: Any):
        """Record an audit entry."""
        if not self._enabled:
            return

        entry = AuditEntry(
            timestamp=time.time(),
            operation=operation,
            variable=variable,
            code=code[:200],  # Truncate long code
            status=status,
            duration_ms=round(duration_ms, 2),
            details=details,
        )

        self._entries.append(entry)

        # Evict old entries
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Write to file
        if self._file_handle:
            try:
                self._file_handle.write(entry.to_json() + '\n')
                self._file_handle.flush()
            except OSError as e:
                logger.warning("Audit log write failed: %s", e)

    def get_entries(self, operation: str | None = None,
                    variable: str | None = None,
                    limit: int = 50) -> list[AuditEntry]:
        entries = self._entries
        if operation:
            entries = [e for e in entries if e.operation == operation]
        if variable:
            entries = [e for e in entries if e.variable == variable]
        return entries[-limit:]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of audit entries."""
        if not self._entries:
            return {"total": 0}

        ops = {}
        for e in self._entries:
            ops[e.operation] = ops.get(e.operation, 0) + 1

        first = self._entries[0].timestamp_str
        last = self._entries[-1].timestamp_str

        return {
            "total": len(self._entries),
            "operations": ops,
            "time_range": f"{first} to {last}",
            "unique_variables": len({e.variable for e in self._entries}),
        }

    def clear(self):
        self._entries.clear()

    def format_entries(self, entries: list[AuditEntry] | None = None,
                       as_json: bool = False) -> str:
        if entries is None:
            entries = self._entries[-50:]

        if not entries:
            return "No audit entries."

        if as_json:
            return json.dumps([e.to_dict() for e in entries], indent=2, default=str)

        lines = []
        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime('%H:%M:%S')
            dur = f"{e.duration_ms:.1f}ms" if e.duration_ms > 0 else ""
            code_preview = e.code.split('\n')[0][:40] if e.code else ""
            line = f"[{ts}] {e.operation:16s} {e.variable:20s} {dur:>8s}  {code_preview}"
            lines.append(line)

        return '\n'.join(lines)

    def shutdown(self):
        """Cleanup resources."""
        self.disable()

