"""Where cash's log records go: the one place that configures the ``cash`` logger.

The entry points share one record of what cash itself installed:

* `enable` -- ``debug=True`` / ``verbose=True`` / ``CASH_DEBUG``. Lowers the
  ``cash`` level and, when nothing would print a record (a script's default),
  adds one stderr handler that stands down once the application logs.
* `enable_console` -- ``%cash_debug on``. Replaces cash's own handlers with
  one that writes to the current ``sys.stdout``, so records reach the cell.
* `setup_logging` -- ``%cash_debug json`` / ``%cash_debug file``. Replaces
  cash's own handlers with a console handler and optionally a JSON file.
* `disable` -- ``%cash_debug off``. Removes cash's own handlers and the level
  cash set.

Handlers the application put on the ``cash`` logger are never removed.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any

__all__ = ["JsonFormatter", "application_handlers", "disable", "enable", "enable_console", "setup_logging"]

#: Handlers cash added to the ``cash`` logger; everything else is the application's.
_OWN_HANDLERS: list[logging.Handler] = []

#: The level cash last set on the ``cash`` logger. A level anyone else set is
#: theirs, and `enable` does not lower it.
_LEVEL_SET: int | None = None


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra fields if present
        for key in ("event", "duration_ms", "cache_key", "status", "variable", "module", "cell_id", "backend"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry)


def application_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Handlers that would print a record from *logger*, other than cash's own.

    Walks up the hierarchy the way `logging` does. pytest's capture handlers
    and ``NullHandler`` print nothing and do not count, so the answer means
    "the application configured logging".
    """
    found = []
    current: logging.Logger | None = logger
    while current is not None:
        for handler in current.handlers:
            if handler in _OWN_HANDLERS or isinstance(handler, logging.NullHandler):
                continue
            if (type(handler).__module__ or "").startswith("_pytest"):
                continue
            found.append(handler)
        if not current.propagate:
            break
        current = current.parent
    return found


def _add_own(cash_logger: logging.Logger, handler: logging.Handler) -> None:
    cash_logger.addHandler(handler)
    _OWN_HANDLERS.append(handler)


def _remove_own(cash_logger: logging.Logger) -> None:
    for handler in _OWN_HANDLERS:
        cash_logger.removeHandler(handler)
        handler.close()  # a log file is released, not left open until exit
    _OWN_HANDLERS.clear()


def enable(level: int) -> None:
    """Make ``cash`` records at *level* reach the user.

    Sets the ``cash`` level unless someone else set one, and adds a stderr
    handler only when nothing (cash's or the application's) would print a
    record. An application that configured logging keeps its handlers,
    format and levels. stderr, because stdout is often the program's output.
    """
    global _LEVEL_SET
    cash_logger = logging.getLogger("cash")
    if cash_logger.level == logging.NOTSET or (cash_logger.level == _LEVEL_SET and cash_logger.level > level):
        cash_logger.setLevel(level)
        _LEVEL_SET = level
    if not _OWN_HANDLERS and not application_handlers(cash_logger):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        handler.addFilter(_StandDownWhenTheAppLogs())
        _add_own(cash_logger, handler)


class _CurrentStdoutHandler(logging.StreamHandler):
    """A ``StreamHandler`` that always writes to the *current* ``sys.stdout``.

    Under ipykernel, ``sys.stdout`` is swapped to a per-cell output proxy on
    each execution.  A vanilla ``StreamHandler(sys.stdout)`` captures the
    stream at construction time, so debug records emitted during later cells
    would be routed to whatever stdout was active when ``%cash_debug on`` ran.
    Resolving ``sys.stdout`` at emit time keeps debug output landing in the
    cell that produced it.
    """

    def __init__(self) -> None:
        super().__init__(stream=sys.stdout)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value: Any) -> None:
        # logging.StreamHandler.__init__ assigns self.stream; ignore the stored
        # value and always defer to the live sys.stdout via the getter.
        pass


def enable_console(level: int = logging.DEBUG) -> None:
    """Send ``cash`` records at *level* to the current ``sys.stdout``.

    For ``%cash_debug on``: raising the level alone relied on root-logger
    propagation to reach the cell, which recent ipykernel no longer does, so
    debug markers never appeared. Replaces the handlers cash added before, so
    switching from ``json`` to ``on`` does not print every record twice.
    """
    global _LEVEL_SET
    cash_logger = logging.getLogger("cash")
    cash_logger.setLevel(level)
    _LEVEL_SET = level
    _remove_own(cash_logger)
    handler = _CurrentStdoutHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _add_own(cash_logger, handler)


def disable() -> None:
    """Undo `enable`, `enable_console` and `setup_logging`: remove cash's own
    handlers, and put the ``cash`` level back to ``NOTSET`` if cash set it.
    The application's handlers and levels stay."""
    global _LEVEL_SET
    cash_logger = logging.getLogger("cash")
    _remove_own(cash_logger)
    if _LEVEL_SET is not None and cash_logger.level == _LEVEL_SET:
        cash_logger.setLevel(logging.NOTSET)
    _LEVEL_SET = None


def setup_logging(level: int = logging.INFO, json_output: bool = False, log_file: str | None = None) -> None:
    """Send ``cash`` records at *level* to the console, and optionally to a JSON file.

    Replaces the handlers cash added before (so calling it twice does not
    print every line twice); handlers the application added stay.

    Args:
        level: Logging level (DEBUG, INFO, etc.)
        json_output: If True, use JSON formatter for console output
        log_file: Optional path for JSON file logging
    """
    global _LEVEL_SET
    cash_logger = logging.getLogger("cash")
    cash_logger.setLevel(level)
    _LEVEL_SET = level
    _remove_own(cash_logger)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(JsonFormatter() if json_output else logging.Formatter("[%(name)s] %(message)s"))
    _add_own(cash_logger, console)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JsonFormatter())
        _add_own(cash_logger, fh)


def _app_would_emit(levelno: int) -> bool:
    """Would one of the application's handlers print a ``cash`` record at *levelno*?

    The ``cash`` logger's own level first: a record it drops reaches no handler.
    """
    cash_logger = logging.getLogger("cash")
    if not cash_logger.isEnabledFor(levelno):
        return False
    return any(h.level <= levelno for h in application_handlers(cash_logger))


class _StandDownWhenTheAppLogs(logging.Filter):
    """Keep cash's stderr handler quiet once the application logs.

    A program that configures logging after ``import cash`` (imports at the
    top, ``basicConfig`` in ``main``) would otherwise get every line twice.
    Checked per record, so it follows the application's setup as it changes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # The exit summary is only logged when the application has a handler
        # for it, and written to stderr directly otherwise.
        if record.name == "cash.summary":
            return False
        return not _app_would_emit(record.levelno)
