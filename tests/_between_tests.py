"""What a test leaves in the process that no later test needs, dropped after it.

A share check (`cash.notebook.shared_objects`) and the other heap reads cost
time in proportion to every object the process holds (``gc.get_referrers``
scans them all). An xdist worker runs thousands of tests in one process, and
each test left ~100 objects behind for good: one process running
``tests/test_notebook`` held 0.7M objects after 250 tests, 1.2M after 4,000
and 1.9M after 10,000, and a test that takes 2 s in a fresh process took 15
to 60 s late in a CI worker.

What kept them:

* cash's process-wide memos (`cash._memo.LruMemo` and ``functools``
  caches of 256 entries and more, and the name tables of
  `cash.tracking.trace_items`): bounded, but
  at thousands of entries each, and an entry keyed by a statement's source or
  code object keeps its trees, analyses and the weak-keyed memos beside them
  alive. A user session reuses them; a test never reuses another test's
  statements. Dropping an entry is always allowed (a full memo drops the
  least recently used one), so a test cannot depend on one kept from before.
* IPython's output history, which is one dict on the ``HistoryManager``
  class, shared by every shell and emptied only by ``reset()``: each display
  a real shell made (a badge, a result) stayed for the process.
* every `cash.Cash` a test made: its exit work (`Cash.shutdown`, the
  backend's last writes) is registered with ``atexit``, which holds it, its
  backend and its stored-key record until the process ends, long after the
  instance itself is gone (`track_cash_exit_work`).
* the instance of a test class: pytest keeps it on its item for the whole
  session, and with it whatever ``setup_method`` put there (a Cash, its
  magics, a mocked backend and every call the mock recorded).
* pytest's own terminal reporter in an xdist worker, which keeps three
  reports per test for a summary only the controller prints.
"""

from __future__ import annotations

import functools
import sys
import weakref
from collections.abc import Callable
from typing import Any

#: A memo this big holds one entry per statement, code object or file and
#: keeps filling; a smaller one, or an unbounded ``functools.cache``, holds
#: a fact about the environment (the stdlib's folders, a type table), which
#: every test asks again and which does not grow.
_GROWS = 256
_memos: list[Any] = []
_modules_scanned = 0


def _scan_memos() -> None:
    """Collect every memo a ``cash.*`` module holds at its top level, or on
    an object it holds there (`purity_analyzer`'s analyzer). Rescanned when
    a module has been imported since."""
    global _modules_scanned
    if len(sys.modules) == _modules_scanned or "cash._memo" not in sys.modules:
        return
    _modules_scanned = len(sys.modules)
    lru_memo = sys.modules["cash._memo"].LruMemo
    seen = {id(m) for m in _memos}

    def take(value: Any, module: str) -> None:
        if id(value) in seen:
            return
        if isinstance(value, lru_memo):
            size = value.maxsize
        elif isinstance(value, functools._lru_cache_wrapper) and getattr(value, "__module__", None) == module:
            size = value.cache_parameters()["maxsize"]
        else:
            return
        if size is not None and size >= _GROWS:
            seen.add(id(value))
            _memos.append(value)

    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("cash."):
            continue
        for value in list(vars(module).values()):
            take(value, name)
            if type(value).__module__.startswith("cash.") and hasattr(value, "__dict__"):
                for held in list(vars(value).values()):
                    take(held, name)


def forget_cash_memos() -> None:
    _scan_memos()
    for memo in _memos:
        if isinstance(memo, functools._lru_cache_wrapper):
            memo.cache_clear()
        else:
            memo.clear()
    trace_items = sys.modules.get("cash.tracking.trace_items")
    if trace_items is not None:
        trace_items._TABLES.clear()
        trace_items._DIGESTS.clear()


def forget_ipython_outputs() -> None:
    history = sys.modules.get("IPython.core.history")
    if history is not None:
        history.HistoryManager.outputs.clear()


def forget_test_instance(item: Any) -> None:
    """Empty the test class instance *item* ran on, once its teardown ran."""
    instance = getattr(item, "instance", None)
    attributes = getattr(instance, "__dict__", None)
    if attributes is not None:
        attributes.clear()


def forget_passed_reports(config: Any) -> None:
    """In an xdist worker only: its terminal reporter keeps every report for
    a summary nobody sees (the controller counts the reports it is sent).
    Run without workers, the same lists give the final ``N passed``."""
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        for category in ("", "passed"):
            reporter.stats.pop(category, None)


#: (the instance, its exit work) for every Cash made in this process.
_exit_work: list[tuple[weakref.ref, Callable[[], None]]] = []


def track_cash_exit_work() -> None:
    """Note each Cash's exit work as `Cash.__init__` registers it, so that it
    can be taken off ``atexit`` once the instance is collected
    (`forget_dead_cash_exit_work`). Not run then: what it finishes (the
    stored-key record, writes still queued) is the test's and over, and the
    writer threads are daemons. Caught at the registration, not by wrapping
    ``__init__``: cash reads the frames around a Cash being made."""
    from cash import core

    register = core.register_atexit
    if getattr(register, "_tracks_exit_work", False):
        return

    @functools.wraps(register)
    def register_atexit(fn: Callable[..., object], *args: Any, **kwargs: Any) -> None:
        register(fn, *args, **kwargs)
        if type(getattr(fn, "__self__", None)) is core._ExitWork:
            instance = sys._getframe(1).f_locals.get("self")
            if isinstance(instance, core.Cash):
                _exit_work.append((weakref.ref(instance), fn))

    register_atexit._tracks_exit_work = True  # type: ignore[attr-defined]
    core.register_atexit = register_atexit


def forget_dead_cash_exit_work() -> None:
    if not _exit_work:
        return
    from cash.tracking.process_registries import unregister_atexit

    alive = []
    for instance, run in _exit_work:
        if instance() is None:
            unregister_atexit(run)
        else:
            alive.append((instance, run))
    _exit_work[:] = alive
