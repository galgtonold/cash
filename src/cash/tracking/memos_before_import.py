"""Memos of the user's that already held entries when ``import cash`` ran
(``KEY-MEMO-BEFORE-IMPORT``).

cash watches file reads from ``import cash`` on. A loader memoised with
``functools.lru_cache`` / ``functools.cache`` that was called before that read
its file unseen, and every later call is served from the memo, so a cached
function that gets its config from it records no file and an edit to the file
is not noticed.

What is looked at is bounded and cheap: the modules already imported whose
file is the user's (`is_user_module`; the standard library and installed
packages are judged by their path and never opened), the ``functools`` memo
wrappers among their module-level names, and each one's ``cache_info()``,
which only reads a counter.
"""

from __future__ import annotations

import functools
import sys

from cash.diagnostics import warn_diagnostic
from cash.exceptions import CashCacheIneffectiveWarning
from cash.install_paths import is_user_module

__all__ = ["CODE", "warn_memos_filled_before_import"]

CODE = "KEY-MEMO-BEFORE-IMPORT"

#: The type ``functools.lru_cache`` and ``functools.cache`` return. Under the
#: pure-Python fallback (no C ``_functools``) the wrapper is a plain function,
#: which is not looked for.
_LRU_WRAPPER = getattr(functools, "_lru_cache_wrapper", None)

#: At most this many user modules and names in each are looked at: a project
#: with more is not one whose import should pay for a hint.
_MAX_MODULES = 500
_MAX_NAMES = 2000

#: How many memos the message names.
_NAMED = 3


def _filled_memos(mod: object) -> list[object]:
    """The memo wrappers defined at *mod*'s top level that hold entries."""
    try:
        namespace = vars(mod)
    except TypeError:
        return []
    name = namespace.get("__name__")
    found = []
    for i, value in enumerate(list(namespace.values())):
        if i >= _MAX_NAMES:
            break
        if type(value) is not _LRU_WRAPPER or getattr(value, "__module__", None) != name:
            continue
        try:
            if value.cache_info().currsize > 0:
                found.append(value)
        except Exception:  # noqa: BLE001 - a hint is never worth an import error
            continue
    return found


def find_memos_filled_before_import() -> list[object]:
    """Every user memo that holds entries now, in import order."""
    if _LRU_WRAPPER is None:
        return []
    found: list[object] = []
    looked = 0
    for mod in list(sys.modules.values()):
        try:
            if not is_user_module(mod):
                continue
        except Exception:  # noqa: BLE001 - an odd entry in sys.modules
            continue
        looked += 1
        if looked > _MAX_MODULES:
            break
        found.extend(_filled_memos(mod))
    return found


def _label(memo: object) -> str:
    module = getattr(memo, "__module__", "?")
    return f"{module}.{getattr(memo, '__qualname__', getattr(memo, '__name__', '?'))}"


def warn_memos_filled_before_import() -> None:
    """Warn once, at ``import cash``, naming the user's memos that already
    hold entries. Silent when there are none."""
    memos = find_memos_filled_before_import()
    if not memos:
        return
    names = ", ".join(_label(m) for m in memos[:_NAMED])
    if len(memos) > _NAMED:
        names += f" and {len(memos) - _NAMED} more"
    what = (
        f"{names} (functools memo) already held results when cash was imported, "
        "so a file it read then is not tracked. cash watches file reads from "
        "import cash on; a cached function that gets data from this memo "
        "records no file, and an edit to that file is not noticed."
    )
    fix = (
        "import cash before the modules that call it, or pass the file's path "
        'as an argument (load_cfg("config.yaml")) from inside the cached function.'
    )
    location = None
    code = getattr(getattr(memos[0], "__wrapped__", None), "__code__", None)
    if code is not None:
        location = (code.co_filename, code.co_firstlineno)
    warn_diagnostic(CashCacheIneffectiveWarning, CODE, what, fix, location=location)
