"""IPython's ``%autoreload``, run where a plain kernel runs it: before the cell.

The extension reloads the modules whose files changed from a ``pre_run_cell``
event. cash runs a cell's statements itself and hands IPython only a ``pass``
at the end, so that event fired after the statements: the cell ran the old
code and was keyed as if nothing had changed. A module cash tracks is
reloaded by cash's own check; one it does not (``%aimport mylib``) reached no
key at all.

`run_autoreload_now` runs the extension's check at the point the cell's
module check runs, and says which modules it reloaded, so cash treats them
as an edit it saw itself: tracked from then on, and their readers re-keyed.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from collections.abc import Iterator
from typing import Any

from ...tracking.function_tracker import is_local_module

__all__ = ["run_autoreload_now"]

logger = logging.getLogger(__name__)


def run_autoreload_now(shell: Any) -> set[str]:
    """Run ``%autoreload``'s check now, when the extension is loaded and on,
    and return the names of the modules it reloaded. Its later
    ``pre_run_cell`` check then finds nothing left to do."""
    registry = getattr(getattr(shell, "magics_manager", None), "registry", None)
    magics = registry.get("AutoreloadMagics") if isinstance(registry, dict) else None
    reloader = getattr(magics, "_reloader", None)
    mtimes = getattr(reloader, "modules_mtimes", None)
    if not getattr(reloader, "enabled", False) or not isinstance(mtimes, dict):
        return set()
    before = dict(mtimes)
    try:
        with _only_local_modules(reloader):
            reloader.check()
    except Exception:  # noqa: BLE001 - the extension reports its own failures
        logger.debug("autoreload's check failed", exc_info=True)
    # A module seen for the first time is only recorded, not reloaded.
    return {name for name, mtime in mtimes.items() if name in before and before[name] != mtime}


@contextlib.contextmanager
def _only_local_modules(reloader: Any) -> Iterator[None]:
    """Narrow the check to the user's own modules: with ``%autoreload 2`` it
    stats every loaded module, 5-15 ms with a data stack loaded, and its own
    ``pre_run_cell`` check still runs after the cell for the rest. A
    reloader without the attributes this narrows checks everything."""
    check_all = getattr(reloader, "check_all", None)
    marked = getattr(reloader, "modules", None)
    if not isinstance(check_all, bool) or not isinstance(marked, dict):
        yield
        return
    candidates = list(sys.modules) if check_all else list(marked)
    local = {name: True for name in candidates if _is_local(sys.modules.get(name))}
    reloader.check_all, reloader.modules = False, local
    try:
        yield
    finally:
        reloader.check_all, reloader.modules = check_all, marked


#: Module name -> (the module, whether it is the user's), asked once per module object.
_LOCAL: dict[str, tuple[types.ModuleType, bool]] = {}


def _is_local(module: Any) -> bool:
    if not isinstance(module, types.ModuleType):
        return False
    name = getattr(module, "__name__", None)
    held = _LOCAL.get(name) if isinstance(name, str) else None
    if held is not None and held[0] is module:
        return held[1]
    try:
        verdict = is_local_module(module)
    except (TypeError, AttributeError):
        verdict = False
    if isinstance(name, str):
        _LOCAL[name] = (module, verdict)
    return verdict
