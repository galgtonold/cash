"""The execution count a cell's code sees while cash runs it.

IPython 9 raises ``shell.execution_count`` as a cell starts, before its code
runs: inside cell ``In[N]`` the count reads ``N + 1`` and the display hook
files the cell's value under ``N`` (``DisplayHook.prompt_count`` is
``execution_count - 1``). IPython 8 raises it after the cell, so its code
reads ``N``. cash runs a cell's statements itself and hands IPython a
stand-in afterwards, so under IPython 9 the count the code read was one
behind plain's.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

#: The shell attribute holding ``N`` while :func:`counting_as_ipython` runs.
_RUNNING = "_cash_running_cell_count"


def running_cell_count(shell: Any) -> int:
    """``N`` of the cell running now (``In[N]``, ``Cell In[N]``), whichever
    count its code sees."""
    count = getattr(shell, _RUNNING, None)
    if isinstance(count, int):
        return count
    count = getattr(shell, "execution_count", 0)
    return count if isinstance(count, int) else 0


def _lead(shell: Any) -> int:
    """How far ahead of the cell's own number the count runs while the cell
    runs: 1 where IPython raises it first, else 0. Read off the display hook,
    which files a value under ``execution_count - lead``."""
    count = getattr(shell, "execution_count", None)
    try:
        lead = count - shell.displayhook.prompt_count
    except (AttributeError, TypeError):
        return 0
    return lead if lead in (0, 1) else 0


@contextlib.contextmanager
def counting_as_ipython(shell: Any, store_history: bool) -> Iterator[None]:
    """While cash runs a cell's statements, give its code the count IPython
    would. Put back after, so the stand-in IPython runs next takes the
    cell's own number."""
    count = getattr(shell, "execution_count", None)
    lead = _lead(shell) if store_history and isinstance(count, int) else 0
    if not lead:
        yield
        return
    try:
        setattr(shell, _RUNNING, count)
        shell.execution_count = count + lead
    except (AttributeError, TypeError):
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(AttributeError, TypeError):
            shell.execution_count = count
            delattr(shell, _RUNNING)
