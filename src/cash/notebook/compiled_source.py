"""Stable, linecache-registered pseudo-filenames for compiled cell statements.

Cash compiles every statement under a synthetic filename. Reusing one literal
``<cash>`` for all of them meant a traceback frame inside a function *defined in
a cell* could never resolve its source: Python printed ``File "<cash>", line N``
with no source line (and ``inspect.getsource`` reported "Could not get source"),
so every debug loop lost the failing line.

Each statement now gets a name derived from its source hash — stable across
re-runs of identical source, so the linecache entry is reused rather than grown
without bound, yet distinct per statement so their line numbers cannot collide —
and its source is registered in :mod:`linecache` so tracebacks resolve normally.
"""

from __future__ import annotations

import hashlib
import linecache
import re

# Every cash-compiled unit's filename starts with this; frame filters match the
# prefix because each name carries a per-statement digest.
CASH_FILENAME_PREFIX = "<cash-"


#: ``call_interception.HELPER_NAME``, spelled here: that module imports this one's
#: neighbours, and a cycle would cost more than a constant.
_HELPER = "__cash_call__"
_USER_CALL = re.compile(r"__cash_call__\((.+?), \d+\)\(")


def register_cell_source(code: str) -> str:
    """Return a stable pseudo-filename for *code* and register it in linecache.

    Call this immediately before ``compile()`` and pass the result as the
    filename, so any frame raised from the compiled unit can show its source.
    """
    digest = hashlib.sha1(code.encode("utf-8", "replace")).hexdigest()[:12]
    name = f"{CASH_FILENAME_PREFIX}{digest}>"
    # linecache entry: (size, mtime, lines, fullname). ``mtime=None`` marks it a
    # synthetic in-memory file so ``linecache.checkcache()`` never evicts it by
    # comparing against a real stat() — there is no file on disk to compare to.
    #
    # The lines are for DISPLAY -- a traceback, a warning -- and a statement
    # whose calls go through the cache reads as ``__cash_call__(fit, 0)(g)``
    # there. A pandas warning quoted that as the user's line; shown as
    # the user wrote it. What runs is ``code``, compiled by
    # the caller.
    shown = _USER_CALL.sub(r"\1(", code) if _HELPER in code else code
    linecache.cache[name] = (len(shown), None, shown.splitlines(True), name)
    return name


def is_cash_filename(filename: str | None) -> bool:
    """True for any filename cash compiled a user statement under.

    Prefix-based so it covers every per-statement ``<cash-abc123…>`` name.
    """
    return bool(filename) and filename.startswith(CASH_FILENAME_PREFIX)
