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
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Every cash-compiled unit's filename starts with this; frame filters match the
# prefix because each name carries a per-statement digest.
CASH_FILENAME_PREFIX = "<cash-"


#: ``call_interception.HELPER_NAME``, spelled here: that module imports this one's
#: neighbours, and a cycle would cost more than a constant.
_HELPER = "__cash_call__"
_USER_CALL = re.compile(r"__cash_call__\((.+?), \d+\)\(")
#: A plain-name callee's spelling (``call_interception._routed``).
_USER_NAME_CALL = re.compile(
    r"\((\w+) if \1 is __cash_plain__\[\d+\] and __cash_count__\[\d+\]\(\) else __cash_call__\(\1, \d+\)\)\("
)


#: Where the top-level statement being run stands (`statement_place`):
#: ``(compiler, cell number, first line, cell text)``, or None.
_PLACE: list[tuple[Any, int, int, str] | None] = [None]


@contextmanager
def statement_place(compiler: Any, number: int, first_line: int, cell: str) -> Iterator[None]:
    """Compile the statements run inside as standing at *first_line* of cell
    *number* (*cell* is its text), as IPython compiles the whole cell: a
    traceback through a function defined there reads ``Cell In[2], line 6,
    in f(a)``, as it does without cash, rather than ``File <cash-...>:2``.
    *compiler* is the shell's ``compile`` (IPython's ``CachingCompiler``),
    which labels the frames of the filenames it was told about."""
    before = _PLACE[0]
    _PLACE[0] = (compiler, number, first_line, cell)
    try:
        yield
    finally:
        _PLACE[0] = before


def line_offset() -> int:
    """How many lines the statement being compiled stands below its cell's
    first (`statement_place`); `shifted` moves its code there."""
    place = _PLACE[0]
    return place[2] - 1 if place is not None and place[2] > 1 else 0


def shifted(code: types.CodeType, lines: int) -> types.CodeType:
    """*code*, and the code of every function and class it defines, *lines*
    lines further down. Line tables are relative to ``co_firstlineno``."""
    if not lines:
        return code
    consts = tuple(shifted(c, lines) if isinstance(c, types.CodeType) else c for c in code.co_consts)
    return code.replace(co_firstlineno=code.co_firstlineno + lines, co_consts=consts)


def register_cell_source(code: str) -> str:
    """Return a stable pseudo-filename for *code* and register it in linecache.

    Call this immediately before ``compile()`` and pass the result as the
    filename, so any frame raised from the compiled unit can show its source.
    Inside `statement_place` the unit is compiled at its line in the cell
    (`line_offset`, `shifted`): the name then also stands for that line and
    that cell, its lines are the cell's, and the shell's compiler labels it
    with the cell's number.
    """
    place = _PLACE[0]
    keyed = code if place is None else f"{code}\0{place[2]}\0{place[3]}"
    digest = hashlib.sha1(keyed.encode("utf-8", "replace")).hexdigest()[:12]
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
    shown = _USER_CALL.sub(r"\1(", _USER_NAME_CALL.sub(r"\1(", code)) if _HELPER in code else code
    lines = shown.splitlines(True)
    if place is not None:
        compiler, number, first, cell = place
        offset = first - 1 if first > 1 else 0
        cell_lines = cell.splitlines(True)
        mine = [line.rstrip("\r\n") for line in lines]
        if [line.rstrip("\r\n") for line in cell_lines[offset : offset + len(mine)]] == mine:
            lines = cell_lines  # the lines around it too, as IPython shows them
        else:
            lines = ["\n"] * offset + lines
        named = getattr(compiler, "_filename_map", None)
        if isinstance(named, dict):
            named[name] = number
    linecache.cache[name] = (sum(map(len, lines)), None, lines, name)
    return name


def is_cash_filename(filename: str | None) -> bool:
    """True for any filename cash compiled a user statement under.

    Prefix-based so it covers every per-statement ``<cash-abc123…>`` name.
    """
    return bool(filename) and filename.startswith(CASH_FILENAME_PREFIX)
