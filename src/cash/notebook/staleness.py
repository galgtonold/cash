"""Did the saved .ipynb fall behind the editor? Sometimes cash can PROVE it.

cash reads the cells it did not execute from the saved `.ipynb`, so an unsaved
edit is invisible to it: edit an upstream cell, run a downstream one, and cash
reads the old value, concludes nothing changed and restores the previous answer
while the screen shows new code. Silent, and the silence is the problem.

There is one place the proof is free. IPython hands cash the code it is actually
running; cash separately reads that same cell out of the file. If those differ,
the file is out of date -- and therefore so is every OTHER cell cash read from
it. That is a fact, not a heuristic.

What this canNOT do, stated so nobody mistakes it for a fix: it fires only when
the cell being RUN is the edited one. Edit cell 3, run cell 7, and cell 7 still
matches the file. Detectors for that case were designed and rejected during
design -- in-kernel detection cannot reach it. This is a floor.
"""
from __future__ import annotations

import os


class StalenessTracker:
    """Remembers, for the session, that the notebook file was proven stale."""

    def __init__(self) -> None:
        self._stale = False
        self._saved_at: float | None = None
        self._hint: str | None = None

    def observe(self, *, running_code: str, file_code: str | None,
                notebook_path: str | None) -> bool:
        """Compare what is running against what the file says, and remember.

        Returns True only on the transition into "stale" so the caller can emit
        one notification rather than one per statement.
        """
        mtime = _mtime(notebook_path)
        # A save is the only thing that can clear the verdict: the file has
        # been rewritten, so whatever it now holds is current as of that write.
        if self._stale and mtime is not None and mtime != self._saved_at:
            self.reset()

        if file_code is None or notebook_path is None or mtime is None:
            return False            # no proof available; not the same as "fresh"
        if _normalise(running_code) == _normalise(file_code):
            return False
        if self._stale:
            return False            # already known; do not re-notify

        self._stale = True
        self._saved_at = mtime
        first_line = running_code.strip().splitlines()[0][:60] if running_code.strip() else None
        self._hint = _to_ascii(first_line) if first_line else None
        return True

    def is_stale(self) -> bool:
        return self._stale

    def saved_at(self) -> float | None:
        return self._saved_at

    def hint(self) -> str | None:
        return self._hint

    def reset(self) -> None:
        self._stale = False
        self._saved_at = None
        self._hint = None


def _normalise(code: str) -> str:
    """Ignore differences no human made.

    nbformat and editors disagree about trailing newlines and line endings
    routinely. Firing on those would make the warning noise, and a warning the
    user learns to ignore is worse than no warning.
    """
    return code.replace("\r\n", "\n").strip()


def _mtime(path: str | None) -> float | None:
    if not path:
        return None
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None                 # degrade, never raise


def _to_ascii(text: str) -> str:
    """Guarantee true ASCII by replacing non-encodable characters.

    Task 3 embeds hint() in a badge code field. The downstream consumer reads
    the badge via a different process on a console whose codepage cash cannot
    know at write time -- cp1252, cp437, cp850, or anything else. ASCII is the
    only encoding safely assumed to be a subset of all of them: sanitising to
    cp1252 (as this used to) still lets accented Latin-1 characters through
    unescaped, which then crashes a cp437/cp850 reader. Replace anything
    outside plain ASCII rather than lose the hint entirely -- a
    lossy-but-present diagnostic beats none.
    """
    return text.encode("ascii", errors="replace").decode("ascii")
