"""Cache status enum and execution result types for statement processing."""

from __future__ import annotations

import enum


class CacheStatus(enum.Enum):
    """What happened to one statement, as its metric reports it.

    The last three are notification rows the cell executor adds to a cell's
    metrics (a function or module changed under the cell, an advisory); they
    are not the outcome of running a statement. A metric may carry a member
    or its value; :meth:`parse` reads either.
    """

    COMPUTED = "COMPUTED"
    RESTORED = "RESTORED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"
    FUNCTION_CHANGED = "FUNCTION_CHANGED"
    MODULE_RELOADED = "MODULE_RELOADED"
    WARNING = "WARNING"

    def __str__(self) -> str:  # noqa: D105
        return self.value

    @classmethod
    def parse(cls, raw: object) -> CacheStatus:
        """The member *raw* names, case-insensitively; ``UNKNOWN`` for anything else."""
        if isinstance(raw, cls):
            return raw
        try:
            return cls(str(raw).upper())
        except ValueError:
            return cls.UNKNOWN


class ExecutionResult:
    """Result of executing a single statement.

    Replaces the ad-hoc inline ``class Result`` definitions that were
    scattered across the statement processor.

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
