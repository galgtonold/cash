"""What a ``warnings.warn`` does, as far as a call's result can see it.

``warnings`` is library code, so a key holds none of it. Yet what a warning
does is decided at the time of the call, outside the function's code: the
filters (``simplefilter("error")`` makes it an exception) and the two
display hooks a program may rebind (``warnings.showwarning = mine``,
``warnings.formatwarning = mine``) run whatever the program put there. A
cached ``f(1)`` that warned and returned under one setting was served under
another, where plain Python raised or ran the user's hook.

Any code a call reaches may warn -- a library's included -- so every key
folds `fingerprint` (`KeyBuilder._code_state_post`), which is None, and
folds nothing, while nothing a call could see differs from plain Python's
defaults: no filter that raises and the hooks ``warnings`` was loaded with.
Otherwise it is the filters up to the last one that raises (or all of
them, with ``defaultaction`` ``"error"``) and the code of each hook rebound.
What only decides whether a line is printed (``ignore``, ``always``, a
filter added behind the last ``"error"``) does not reach a key, and the
hooks ``catch_warnings`` swaps and puts back are not read at all.

The fast hit path checks the same state on every hit in C (`watch`).
"""

from __future__ import annotations

import sys
import threading
import warnings
from typing import Any

from ..code_digest import callable_identity

__all__ = ["fingerprint", "filters_now", "watch"]

#: Context-aware warnings (3.14, the default on free-threaded builds) keep
#: the filters per context; ``warnings.filters`` is then not what is used.
_GETTER = getattr(warnings, "_get_filters", None) if getattr(sys.flags, "context_aware_warnings", False) else None
_ORIGINAL = {
    "showwarning": getattr(warnings, "_showwarning_orig", warnings.showwarning),
    "formatwarning": getattr(warnings, "_formatwarning_orig", warnings.formatwarning),
}
_HOOKS = tuple(_ORIGINAL)

_lock = threading.Lock()
#: ``(state read, fingerprint)`` of the last `fingerprint`.
_last: list[Any] = [None, None]
#: ``id(hook) -> (hook, its code's digest)``; the hook is held, so its id
#: is not reused while the entry stands.
_hook_digests: dict[int, tuple[Any, str]] = {}


def filters_now() -> Any:
    """The filters a ``warnings.warn`` made now consults."""
    return _GETTER() if _GETTER is not None else warnings.filters


def _read() -> tuple:
    return (warnings.showwarning, warnings.formatwarning, warnings.defaultaction, *filters_now())


def _pattern(p: Any) -> Any:
    if p is None or isinstance(p, str):
        return p
    return (getattr(p, "pattern", repr(p)), getattr(p, "flags", 0))


def _filter_text(item: Any) -> str:
    try:
        action, message, category, module, lineno = item
    except (TypeError, ValueError):
        return repr(item)
    name = f"{getattr(category, '__module__', '?')}.{getattr(category, '__qualname__', repr(category))}"
    return repr((action, _pattern(message), name, _pattern(module), lineno))


def _hook_digest(hook: Any) -> str:
    held = _hook_digests.get(id(hook))
    if held is not None and held[0] is hook:
        return held[1]
    digest = callable_identity(hook)
    if len(_hook_digests) > 64:
        _hook_digests.clear()
    _hook_digests[id(hook)] = (hook, digest)
    return digest


def _compute(state: tuple) -> str | None:
    hooks, default, filters = state[:2], state[2], state[3:]
    parts = []
    if default == "error":
        reach = len(filters)
        parts.append(f"default={default!r}")
    else:
        reach = 0
        for i, item in enumerate(filters):
            if type(item) is tuple and item and item[0] == "error":
                reach = i + 1
    parts.extend(_filter_text(item) for item in filters[:reach])
    for name, hook in zip(_HOOKS, hooks, strict=True):
        if hook is not _ORIGINAL[name]:
            parts.append(f"{name}={_hook_digest(hook)}")
    return "|".join(parts) if parts else None


def fingerprint() -> str | None:
    """What a key folds of the warnings state now (see the module doc), or
    None when it is plain Python's default."""
    state = _read()
    with _lock:
        if _last[0] == state:
            return _last[1]
    fp = _compute(state)
    with _lock:
        _last[0], _last[1] = state, fp
    return fp


#: The fingerprint the fast hit path's entries were offered under.
_SERVED: list[Any] = [None]


def _moved(bump: Any) -> Any:
    def moved() -> None:
        # The C side saw the state change since its last look: the entries
        # remembered are keyed under the old fingerprint unless it is the same.
        fp = fingerprint()
        if fp != _SERVED[0]:
            _SERVED[0] = fp
            bump()

    return moved


def watch(cw: Any) -> None:
    """Have the fast hit path (`_cwatch.fast_call`) compare the warnings
    state on every hit, and drop every remembered hit (the cash epoch) when
    what a key folds of it moved."""
    if not hasattr(cw, "warn_watch"):
        return
    _SERVED[0] = fingerprint()
    cw.warn_watch(warnings.__dict__, _GETTER, _moved(cw.bump_cash))
    # A hit remembered before is keyed under a state not looked at here.
    cw.bump_cash()
