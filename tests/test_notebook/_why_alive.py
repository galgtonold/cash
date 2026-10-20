"""Why `other_thread_alive` said yes, for a read-count test that fails now and
then (``CASH_TEST_WHY_ALIVE=<dir>``).

Every yes is kept in memory with the test that was running and the thread
that made it -- its name, class, the code it runs, or for one ``threading``
does not list, its stack -- and written to ``<dir>/why-alive.<pid>`` when the
worker exits. Kept in memory: a file written while a statement runs is a
file read cash sees, which changed what the tests measured.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Any

from cash.notebook import call_effects
from cash.notebook.statement import processor
from cash.tracking import trace_recorder as tr

_SEEN: list[str] = []


def _unlisted(listed: set[int | None]) -> list[str]:
    stacks = []
    frames = sys._current_frames()
    try:
        for ident, frame in frames.items():
            if ident in listed or tr._hosts_unlisted(frame):
                continue
            chain = []
            while frame is not None:
                chain.append(f"{frame.f_globals.get('__name__')}:{frame.f_code.co_name}")
                frame = frame.f_back
            stacks.append(" < ".join(chain))
    finally:
        frames.clear()
    return stacks


def _why(notebook_ns: Any) -> str:
    """The first reason `other_thread_alive` finds, in its own order."""
    me = threading.current_thread()
    threads = threading.enumerate()
    listed = {t.ident for t in threads}
    if tr._unlisted_python_thread(listed):
        return f"unlisted {_unlisted(listed)}"
    for t in threads:
        if t is me or t in tr.HOST_THREADS or not t.is_alive():
            continue
        what = f"{t.name!r} {tr._named(type(t))} daemon={t.daemon} code={tr._code_of(t)!r}"
        if tr.runs_user_code(t, notebook_ns):
            return f"user {what}"
        if not tr._kernel_code(t):
            name = getattr(tr._code_of(t), "__module__", None)
            if not (isinstance(name, str) and (name == "cash" or name.startswith("cash."))):
                return f"other {what}"
    return "no reason found again (the thread ended meanwhile)"


def install(folder: str) -> None:
    alive = tr.other_thread_alive

    def logged(notebook_ns: Any) -> bool:
        yes = alive(notebook_ns)
        if yes:
            _SEEN.append(f"{os.environ.get('PYTEST_CURRENT_TEST', '?')}\t{_why(notebook_ns)}\n")
        return yes

    for module in (tr, call_effects, processor):
        module.other_thread_alive = logged

    def dump() -> None:
        if _SEEN:
            with open(os.path.join(folder, f"why-alive.{os.getpid()}"), "a", encoding="utf-8") as fh:
                fh.writelines(_SEEN)

    atexit.register(dump)
