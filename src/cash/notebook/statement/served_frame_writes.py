"""A write into a pandas table served from the cache that an array taken
from the table before the write does not see.

A served table shares its memory with the cache's copy, so pandas copies a
column before writing it (`frame_sharing`). A numpy array taken from the
column before the write (``v = y.to_numpy()``, ``y.values``) then keeps
the old values, where a plain run writes the column in place and the array
sees the write. Each write is counted as pandas makes it; one made while
an array still read the table's old memory warns NOTEBOOK-SERVED-VIEW at
the end of its statement (`frame_sharing.views_left_behind`), naming the
variables that still hold such an array.

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
    """Warns NOTEBOOK-SERVED-VIEW once per statement of a cell run whose
    write left an array of a served table behind; at a cell's end, each
    time.

    While a cell runs, pandas' writes are counted as they are made (an
    array taken inside a function is gone by the end of the statement) and
    reported here, at the end of the statement, where the variables holding
    the array can be named (`frame_sharing.defer_reports`). Between cells,
    and while cash runs no cells (``%cash_off``), a write warns where it is
    made (CACHE-SERVED-VIEW)."""

    def __init__(self) -> None:
        self._warned: set[str] = set()
        self._deferring = False

    def cell_starts(self) -> None:
        """Forget writes made before the cell, and keep the cell's writes
        for its statements' checks (`cell_ends` undoes it)."""
        self._warned.clear()
        if not self._deferring:
            frame_sharing.defer_reports(True)
            self._deferring = True
        frame_sharing.forget_asked()

    def cell_ends(self) -> None:
        """Back to warning at each write, once the cell's last check ran."""
        if self._deferring:
            frame_sharing.defer_reports(False)
            self._deferring = False

    def check(self, user_ns: dict[str, Any], code: str | None, location: tuple[str, int] | None) -> None:
        """Warn when a write since the last check left an array of a served
        table behind (*code*: the statement that ran, None at a cell's end)."""
        try:
            writes, roots = frame_sharing.views_left_behind()
            if not writes:
                return
            names = sorted(
                name
                for name, value in list(user_ns.items())
                if roots and not name.startswith("_") and frame_sharing.reads_memory_of(value, roots)
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
            else "an array (one no variable holds any more, such as one taken inside a function)"
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
