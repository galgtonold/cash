"""Letting a cached function from the running script be pickled by name, so
process pools can send it to their workers."""

from __future__ import annotations

import ast
import importlib.util
import io
import logging
import os
import sys
from collections.abc import Callable

from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..tracking.tracker_context import untracked

logger = logging.getLogger(__name__)


#: script path -> "does its top level have an `if __name__ == '__main__':`?"
_MAIN_GUARD: dict[str, bool] = {}


def _has_main_guard(path: str) -> bool:
    """Is the script's own work behind ``if __name__ == "__main__":``?

    Only then is importing it in a worker harmless: its top level defines
    things, and the work it does when run stays in the process that ran it.
    """
    known = _MAIN_GUARD.get(path)
    if known is not None:
        return known
    found = False
    try:
        # Untracked: cash reading the script is nobody's input, and a cached
        # call this runs inside would have recorded it as one.
        with untracked(), io.FileIO(path, "rb") as fh:
            tree = ast.parse(fh.read())
        for node in tree.body:
            test = getattr(node, "test", None) if isinstance(node, ast.If) else None
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                sides = [test.left, *test.comparators]
                names = {s.id for s in sides if isinstance(s, ast.Name)}
                values = {s.value for s in sides if isinstance(s, ast.Constant)}
                if names == {"__name__"} and values == {"__main__"}:
                    found = True
                    break
    except (OSError, SyntaxError, ValueError):
        found = False
    _MAIN_GUARD[path] = found
    return found


def expose_script_function(func: Callable, wrapper: Callable) -> None:
    """Let a cached function from the running script be pickled BY NAME.

    A function defined in the script you run belongs to ``__main__``, and
    joblib's process workers (cloudpickle) send such a function BY VALUE:
    its code and closure. A cached function's closure holds the ``Cash``
    instance, locks and all, so ``Parallel(n_jobs=2)(delayed(work)(i) ...)``
    failed with "Could not pickle the task to send it to the workers" -- and
    even a copy that pickled would arrive without what the decorator
    registered, keyed differently from the parent's entries.

    So the script's module is ALSO registered under the name an import would
    give it (``model`` for model.py -- the name the cache key already uses),
    and the wrapper names that module. Pickling then records ``model.work``;
    a worker imports ``model`` and decorates ``work`` itself, exactly as a
    ``multiprocessing`` spawn worker re-imports the script. Keys agree, so the
    workers and the parent share entries.

    Only for a script whose work sits behind ``if __name__ == "__main__":``,
    because a worker that imports the script runs its top level. Without the
    guard nothing changes here, and pickling fails with a message that says
    so (``Cash.__reduce__``). Also skipped when the name is a DIFFERENT module
    already, or would import a different file: registering it would shadow
    that module.
    """
    g = getattr(func, "__globals__", None)
    if not isinstance(g, dict) or g.get("__name__") not in MAIN_MODULE_NAMES:
        return
    path = g.get("__file__")
    if not isinstance(path, str) or not path:
        return
    name = resolve_main_module(func)
    if name in MAIN_MODULE_NAMES or not name.isidentifier():
        return
    module = sys.modules.get(g["__name__"])
    if module is None or getattr(module, "__dict__", None) is not g:
        return
    try:
        present = sys.modules.get(name)
        if present is None:
            if not _has_main_guard(path):
                return
            spec = importlib.util.find_spec(name)
            origin = getattr(spec, "origin", None)
            if not origin or not os.path.exists(origin) or not os.path.samefile(origin, path):
                return
            sys.modules[name] = module
        elif present is not module:
            return
        wrapper.__module__ = name
    except (ImportError, ValueError, OSError):
        logger.debug("could not expose %s for pickling by name", name, exc_info=True)
