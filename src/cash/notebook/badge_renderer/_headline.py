"""The headline of a cell that both restored and ran statements."""
from __future__ import annotations

from .view import BadgeHeader


def mixed_headline(h: BadgeHeader) -> tuple[str, str]:
    """``(label, counts)``: CACHED when restoring saved more time than running
    took, else EXECUTED; the counts in the order the label reads them."""
    restored = f"{h.restored_count} restored"
    ran = f"{h.computed_count} ran"
    if h.total_saved_s > h.total_exec_s:
        return "CACHED", f"{restored}, {ran}"
    return "EXECUTED", f"{ran}, {restored}"
