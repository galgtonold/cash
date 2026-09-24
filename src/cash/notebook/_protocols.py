"""Protocol types for the notebook subsystem.

These protocols define the minimal interfaces that the notebook subsystem
requires from external objects (the IPython shell, the ``Cash`` instance).
Using protocols instead of ``Any`` provides:

- Better IDE support (autocomplete, type checking)
- Documentation of expected interfaces
- Easier testing with mock objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cash.backends import CacheBackend


@runtime_checkable
class ShellProtocol(Protocol):
    """Minimal interface for an IPython-like shell.

    The notebook subsystem only requires ``user_ns`` (the user namespace
    dict) and ``run_cell`` for upstream re-execution.
    """

    user_ns: dict[str, Any]

    def run_cell(self, raw_cell: str, *, silent: bool = False) -> Any:
        """Execute a code cell in the shell."""
        ...


@runtime_checkable
class CashInstanceProtocol(Protocol):
    """Minimal interface for the ``Cash`` instance used by the notebook subsystem.

    ``StatementProcessor`` and ``UpstreamChecker`` access the ``Cash``
    object only through its ``.backend`` attribute. That is typed as the
    backend base class itself, which every backend subclasses and which
    declares, with a default where one makes sense, every method the
    notebook calls (``get_metadata``, ``set_metadata_only``, ``hold_notices``,
    ...).
    """

    backend: CacheBackend
