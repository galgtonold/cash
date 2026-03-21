"""Cache status enum and execution result types for statement processing."""

from __future__ import annotations

import enum


class CacheStatus(enum.Enum):
    """Status of a processed statement."""

    COMPUTED = "COMPUTED"
    RESTORED = "RESTORED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # noqa: D105
        return self.value

class ExecutionResult:
    """Result of executing a single statement.

    Replaces the ad-hoc inline ``class Result`` definitions that were
    scattered across :mod:`~cash.notebook.statement_processor`.

    Parameters
    ----------
    success:
        Whether the statement executed without error.
    skipped:
        ``True`` when the statement was skipped (e.g. redundant import).
    error:
        The exception instance, if execution failed.
    tb_string:
        Formatted traceback string, if execution failed.
    """

    __slots__ = ("success", "skipped", "error", "tb_string")

    def __init__(
        self,
        *,
        success: bool = True,
        skipped: bool = False,
        error: BaseException | None = None,
        tb_string: str | None = None,
    ) -> None:
        self.success = success
        self.skipped = skipped
        self.error = error
        self.tb_string = tb_string
