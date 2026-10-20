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

A module cash reloads itself while the extension is on is reloaded the
extension's way (`in_place_reload`): its old classes and functions are
updated in place, and objects made before the edit run the new methods, as
in a plain kernel. A plain ``importlib.reload`` made new classes, so
``type(orders[0]) is mylib.Order`` was False and ``orders[0].total()`` ran
the old method.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import sys
import tokenize
import types
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from cash._user_attrs import attribute_or, has_attribute

from ...tracking.function_tracker import is_local_module

__all__ = ["in_place_reload", "run_autoreload_now"]

logger = logging.getLogger(__name__)


def _reloader_of(shell: Any) -> Any:
    """The ``%autoreload`` extension's reloader, when it is loaded."""
    registry = attribute_or(attribute_or(shell, "magics_manager", None), "registry", None)
    magics = registry.get("AutoreloadMagics") if isinstance(registry, dict) else None
    return attribute_or(magics, "_reloader", None)


def in_place_reload(shell: Any) -> Callable[[types.ModuleType, Callable[[types.ModuleType], Any]], bool] | None:
    """A reload for cash's own module check that does what ``%autoreload``
    does to a module it reloads, or None when the extension is not on.

    The returned function reloads a module the extension covers (every
    module with ``%autoreload 2`` or ``3``, the ``%aimport``-ed ones with
    ``1``) through IPython's ``superreload`` with the given plain reload, and
    returns True; any other module it leaves alone and returns False.
    ``superreload`` remembers the module's classes and functions, reloads it,
    and updates the old ones in place: a class's methods and attributes, a
    function's code, and the class of each object of an old class.
    """
    reloader = _reloader_of(shell)
    old_objects = attribute_or(reloader, "old_objects", None)
    if not attribute_or(reloader, "enabled", False) or not isinstance(old_objects, dict):
        return None
    try:
        from IPython.extensions.autoreload import superreload
    except ImportError:
        return None
    try:
        accepted = set(inspect.signature(superreload).parameters)
    except (TypeError, ValueError):
        return None

    def reload(module: types.ModuleType, plain_reload: Callable[[types.ModuleType], Any]) -> bool:
        name = attribute_or(module, "__name__", None)
        if not isinstance(name, str) or not _covers(reloader, name):
            return False
        kwargs: dict[str, Any] = {}
        if attribute_or(reloader, "autoload_obj", False):
            # ``%autoreload 3`` also binds what the reload added.
            if "shell" in accepted:
                kwargs["shell"] = shell
            tracker = attribute_or(reloader, "import_from_tracker", None)
            if "import_from_tracker" in accepted and attribute_or(tracker, "imports_froms", None):
                kwargs["import_from_tracker"] = tracker
        superreload(module, plain_reload, old_objects, **kwargs)
        return True

    return reload


def _covers(reloader: Any, name: str) -> bool:
    """Whether the extension reloads module *name* when its file changes."""
    skipped = attribute_or(reloader, "skip_modules", None)
    if isinstance(skipped, dict) and name in skipped:
        return False
    if attribute_or(reloader, "check_all", False):
        return True
    marked = attribute_or(reloader, "modules", None)
    return isinstance(marked, dict) and name in marked


def run_autoreload_now(shell: Any, reloaded: Iterable[str] = ()) -> set[str]:
    """Run ``%autoreload``'s check now, when the extension is loaded and on,
    and return the names of the modules it reloaded. Its later
    ``pre_run_cell`` check then finds nothing left to do.

    *reloaded* are the modules cash has just reloaded itself: the extension
    takes them as current and leaves them alone. IPython's deduplicating
    reload (``deduperreload``, ``%autoreload 2``'s default) is off for the
    check, so a module it reloads is reloaded whole, as a fresh import loads
    it. That reload compiles each changed function from ``ast.unparse`` of
    its definition alone: the code says it starts on line 1 of the module's
    file, so ``inspect.getsource`` read the module's docstring for it, and a
    function the edit added got a copy of the module's namespace for its
    globals. Keys built from that code differed from a fresh process's, so
    nothing computed after an edit was reused after a restart, and a failed
    call took its arguments for changed (the helper's source read as none).
    """
    reloader = _reloader_of(shell)
    mtimes = attribute_or(reloader, "modules_mtimes", None)
    if not attribute_or(reloader, "enabled", False) or not isinstance(mtimes, dict):
        return set()
    current = set(reloaded)
    for name in current:
        _take_as_current(reloader, name)
    before = dict(mtimes)
    try:
        with _only_local_modules(reloader), _whole_module_reloads(reloader):
            reloader.check()
    except Exception:
        logger.debug("autoreload's check failed", exc_info=True)
    # A module seen for the first time is only recorded, not reloaded.
    found = {name for name, mtime in mtimes.items() if name in before and before[name] != mtime}
    _note_sources(reloader, found | current)
    return found


def _take_as_current(reloader: Any, name: str) -> None:
    """Record module *name*'s file as loaded, as the extension does after it
    reloads one itself: cash already reloaded it from that file."""
    module = sys.modules.get(name)
    try:
        filename, mtime = reloader.filename_and_mtime(module)
    except Exception:  # noqa: BLE001 - an extension without it: its check decides
        return
    if filename is None:
        return
    reloader.modules_mtimes[name] = mtime
    failed = attribute_or(reloader, "failed", None)
    if isinstance(failed, dict):
        failed.pop(filename, None)


@contextlib.contextmanager
def _whole_module_reloads(reloader: Any) -> Iterator[None]:
    """Turn the deduplicating reload off for the check (see
    `run_autoreload_now`), so the extension falls back to its whole-module
    reload. An extension without one is left as it is."""
    deduper = attribute_or(reloader, "deduper_reloader", None)
    if deduper is None or not has_attribute(type(deduper), "enabled"):
        yield
        return
    was = deduper.enabled
    try:
        deduper.enabled = False
    except Exception:  # noqa: BLE001 - not switchable: its check runs as it is
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            deduper.enabled = was


def _note_sources(reloader: Any, names: set[str]) -> None:
    """Give the deduplicating reload the source each of *names* was reloaded
    from, so a later edit is compared with that one, not the one before."""
    sources = attribute_or(attribute_or(reloader, "deduper_reloader", None), "source_by_modname", None)
    if not isinstance(sources, dict):
        return
    for name in names:
        filename = attribute_or(sys.modules.get(name), "__file__", None)
        if name not in sources or not filename or not filename.endswith(".py"):
            continue
        try:
            with tokenize.open(filename) as handle:
                sources[name] = handle.read()
        except (OSError, SyntaxError, UnicodeDecodeError):
            sources[name] = ""


@contextlib.contextmanager
def _only_local_modules(reloader: Any) -> Iterator[None]:
    """Narrow the check to the user's own modules: with ``%autoreload 2`` it
    stats every loaded module, 5-15 ms with a data stack loaded, and its own
    ``pre_run_cell`` check still runs after the cell for the rest. A
    reloader without the attributes this narrows checks everything."""
    check_all = attribute_or(reloader, "check_all", None)
    marked = attribute_or(reloader, "modules", None)
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
    name = attribute_or(module, "__name__", None)
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
