"""A write into a pandas table served from the cache that an array taken
from the table before the write does not see.

A served table shares its memory with the cache's copy, so pandas copies a
column before writing it (`frame_sharing`). A numpy array taken from the
column before the write (``v = y.to_numpy()``, ``y.values``) then keeps
the old values, where a plain run writes the column in place and the array
sees the write. After each statement, and at the end of each cell, a write
pandas made to such a table while an array still read its old memory
(`frame_sharing.views_left_behind`) warns NOTEBOOK-SERVED-VIEW, naming the
variables that hold such an array.

A table read from disk after a restart is not shared (`InMemoryBackend.promote`):
it behaves as in a plain run, and never warns.
"""

from __future__ import annotations

import logging
from typing import Any

from ...backends import frame_sharing
from ...diagnostics import log_diagnostic, warn_diagnostic
from ...exceptions import CashWarning

logger = logging.getLogger(__name__)

__all__ = ["ServedFrameWrites"]


class ServedFrameWrites:
    """Warns NOTEBOOK-SERVED-VIEW once per statement whose write left an
    array of a served table behind; at a cell's end, each time."""

    def __init__(self) -> None:
        self._warned: set[str] = set()

    @staticmethod
    def cell_starts() -> None:
        """Forget writes made before the cell: cash's own, between cells.
        From the first cell on, a write is counted after its statement
        rather than when pandas makes it (`frame_sharing.checked_by_statements`)."""
        frame_sharing.checked_by_statements()
        frame_sharing.forget_asked()

    def check(self, user_ns: dict[str, Any], code: str | None, location: tuple[str, int] | None) -> None:
        """Warn when a write since the last check left an array of a served
        table behind (*code*: the statement that ran, None at a cell's end)."""
        try:
            roots = frame_sharing.views_left_behind()
            if not roots:
                return
            names = sorted(
                name
                for name, value in list(user_ns.items())
                if not name.startswith("_") and frame_sharing.reads_memory_of(value, roots)
            )
            del roots
            self._warn(names, code, location)
        except Warning:
            raise  # an "error" warning filter asked for it
        except Exception:
            logger.debug("could not check writes into served tables", exc_info=True)

    def _warn(self, names: list[str], code: str | None, location: tuple[str, int] | None) -> None:
        if code is not None:
            if code in self._warned:
                return
            self._warned.add(code)
        held = (
            ("the array " if len(names) == 1 else "the arrays ") + ", ".join(f"'{name}'" for name in names)
            if names
            else "an array"
        )
        what = (
            f"a write into a pandas table served from the cache did not reach {held}, taken from the table "
            "before the write (with .to_numpy() or .values). The served table shares its memory with the "
            "cache's copy, so pandas copied the column before writing it, and the array kept the old "
            "values. In a plain run the write reaches the array whenever no other table shares the column."
        )
        fix = (
            "take the array after the write, or copy the table where it is served (df = df.copy()) and "
            "take the array from the copy, so that it sees later writes as in a plain run."
        )
        log_diagnostic(logger, "NOTEBOOK-SERVED-VIEW", what, fix)
        warn_diagnostic(CashWarning, "NOTEBOOK-SERVED-VIEW", what, fix, location=location)
