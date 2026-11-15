"""What a ``warnings.warn`` does, as far as a call's result can see it.

``warnings`` is library code, so a key holds none of it. Yet what a warning
does is decided at the time of the call, outside the function's code: the
filters (``simplefilter("error")`` makes it an exception) and the two
display hooks a program may rebind (``warnings.showwarning = mine``,
``warnings.formatwarning = mine``) run whatever the program put there. A
cached ``f(1)`` that warned and returned under one setting was served under
another, where plain Python raised or ran the user's hook.

Only a call that warns depends on any of it, and any code it reaches may
warn, a library's included. So a run records whether a warning reached the
process's filters while its body ran (`recording`): a probe at the front of
the filters (`_Probe`) is asked about every warning that gets that far --
whatever filter then decides it, ignored ones included -- and none that the
body's own filters (``catch_warnings()`` + ``simplefilter``) decide first.
An entry of a run that warned keeps `fingerprint` as it was (`stamp`); a
lookup serves it only while `fingerprint` is the same (`holds`).

`fingerprint` is what of the state a result can see: the filters up to the
last one that raises (all of them, with ``defaultaction`` ``"error"``) and
the code of each display hook rebound to the user's own code. What only
decides whether or how a line is printed -- ``ignore``, ``always``, a filter
behind the last ``"error"``, a hook of a library's or the interpreter's
(IPython's, ``logging.captureWarnings``) -- is not in it, so a notebook and
a script share entries. A user's hook is judged by its code alone: what it
reads besides is not in it (the code trace of the run that called it has
it).

The fast hit path compares the state on every hit in C and drops every
remembered hit when `fingerprint` moved (`watch`).

A warning a ``"default"``/``"once"``/``"module"`` filter showed before is
skipped by ``warnings`` before any filter is asked (its
``__warningregistry__``) while the filters stay as they are: the outermost
recording marks them changed (as ``catch_warnings`` does on entering and
leaving), so such a warning reaches the probe again -- and is shown again
once, as after any ``catch_warnings`` block.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys
import threading
import warnings
from collections.abc import Iterator
from typing import Any

from ..code_digest import callable_identity
from ..install_paths import is_cash_path, is_user_code_file

__all__ = ["fingerprint", "filters_now", "holds", "note_warned", "recording", "stamp", "watch"]

#: Context-aware warnings (3.14, the default on free-threaded builds) keep
#: the filters per context; ``warnings.filters`` is then not what is used.
_GETTER = getattr(warnings, "_get_filters", None) if getattr(sys.flags, "context_aware_warnings", False) else None
_ORIGINAL = {
    "showwarning": getattr(warnings, "_showwarning_orig", warnings.showwarning),
    "formatwarning": getattr(warnings, "_formatwarning_orig", warnings.formatwarning),
}
_HOOKS = tuple(_ORIGINAL)

_lock = threading.RLock()
#: ``(state read, fingerprint)`` of the last `fingerprint`.
_last: list[Any] = [None, None]
#: ``id(hook) -> (hook, its digest or None for a library's)``; the hook is
#: held, so its id is not reused while the entry stands.
_hook_digests: dict[int, tuple[Any, str | None]] = {}


def filters_now() -> Any:
    """The filters a ``warnings.warn`` made now consults."""
    return _GETTER() if _GETTER is not None else warnings.filters


# -- the probe ---------------------------------------------------------------


class _Probe:
    """The message pattern of the filter `recording` puts first: asked
    about every warning that reaches the process's filters, it notes it for
    every body being recorded and matches none.

    It reads as a pattern that matches nothing (``pattern``, ``flags``) for
    code that copies the filters elsewhere (scikit-learn hands them to its
    workers as ``filterwarnings(message=f.pattern)``). A warning cash itself
    issues, or a filter lookup cash makes (`diagnostics`), is not the
    body's: not noted."""

    __slots__ = ()

    #: What the filter reads as when copied: a pattern no text matches.
    pattern = "(?!)"
    #: Told apart by `process_registries`, which compares the filters
    #: around a call: cash's, not a change the call made.
    cash_probe = True
    flags = 0

    def match(self, text: Any, *args: Any) -> None:
        if _ACTIVE and not _asked_by_cash():
            with _lock:
                for flag in _ACTIVE:
                    flag[0] = True
        return None

    def search(self, text: Any, *args: Any) -> None:
        return None

    def fullmatch(self, text: Any, *args: Any) -> None:
        return None

    def __repr__(self) -> str:
        return "<cash: notes the warnings a cached call issues>"

    def __reduce__(self) -> Any:
        return (_the_probe, ())

    def __copy__(self) -> _Probe:
        return self

    def __deepcopy__(self, memo: Any) -> _Probe:
        return self


#: Modules whose frames lie between a warning's issuer and the probe.
_WARNINGS_MODULES = frozenset({"warnings", "_py_warnings"})
#: code file -> is it cash's own.
_CASH_FILES: dict[str, bool] = {}


def _asked_by_cash() -> bool:
    """Is the warning the probe is asked about cash's own (issued by, or
    looked up from, a frame of cash's)? ``warnings.warn`` is C: the frame
    below the probe is the issuer's, past any of a pure-Python ``warnings``."""
    frame = sys._getframe(2)
    while frame is not None and frame.f_globals.get("__name__") in _WARNINGS_MODULES:
        frame = frame.f_back
    if frame is None:
        return False
    name = frame.f_code.co_filename
    own = _CASH_FILES.get(name)
    if own is None:
        own = _CASH_FILES[name] = is_cash_path(name)
    return own


def _the_probe() -> _Probe:
    return _ITEM[1]


#: The filter `recording` puts first; matches no warning.
_ITEM = ("default", _Probe(), Warning, None, 0)
#: The flags (``[warned]``) of the bodies being recorded, on any thread: a
#: warning a thread a body started issues counts for it.
_ACTIVE: list[list[bool]] = []
#: The filters lists the probe was put in front of, and how many bodies
#: wanted it there.
_PLACED: list[Any] = []
_DEPTH = [0]


def _place() -> None:
    try:
        filters = filters_now()
    except Exception:  # noqa: BLE001 - no filters to put it in
        return
    if type(filters) is list and not (filters and filters[0] is _ITEM):
        filters.insert(0, _ITEM)
        _PLACED.append(filters)


def _registries_stale() -> None:
    """Mark the filters changed, so every ``__warningregistry__`` is
    cleared at its next warning and none is skipped unasked."""
    try:
        mutated = getattr(warnings, "_filters_mutated", None)
        if mutated is not None:
            mutated()
            return
        mutated = getattr(warnings, "_filters_mutated_lock_held", None)
        lock = getattr(warnings, "_lock", None)
        if mutated is not None and lock is not None:
            with lock:
                mutated()
    except Exception:  # noqa: BLE001 - a warnings module without it
        pass


def _take_back() -> None:
    lists = list(_PLACED)
    with contextlib.suppress(Exception):
        lists.append(filters_now())
    _PLACED.clear()
    for filters in lists:
        if type(filters) is not list:
            continue
        for i in range(len(filters) - 1, -1, -1):
            if filters[i] is _ITEM:
                del filters[i]


@contextlib.contextmanager
def recording() -> Iterator[list[bool]]:
    """Note whether a warning reached the filters while the block ran
    (``flag[0]``); a block nested in it counts for the outer one too."""
    flag = [False]
    with _lock:
        _ACTIVE.append(flag)
        _DEPTH[0] += 1
        _place()
        if _DEPTH[0] == 1:
            _registries_stale()
    try:
        yield flag
    finally:
        with _lock:
            # By identity: two flags that say the same are equal lists.
            for i in range(len(_ACTIVE) - 1, -1, -1):
                if _ACTIVE[i] is flag:
                    del _ACTIVE[i]
                    break
            _DEPTH[0] -= 1
            if _DEPTH[0] == 0:
                _take_back()
            else:
                _place()  # a filters list the body left in place has it first again


def note_warned() -> None:
    """A cached call served inside the bodies being recorded warned when it
    ran: they depend on the warnings state as much as it does."""
    with _lock:
        for flag in _ACTIVE:
            flag[0] = True


def stamp(flag: list[bool]) -> str | None:
    """What an entry of a run that recorded *flag* keeps: `fingerprint`
    (``""`` for plain Python's defaults), or None when it did not warn."""
    if not flag[0]:
        return None
    return fingerprint() or ""


def holds(stamped: str) -> bool:
    """Does the warnings state a result can see stand as *stamped* (`stamp`)?"""
    return (fingerprint() or "") == stamped


# -- the fingerprint ---------------------------------------------------------


def _read() -> tuple:
    filters = filters_now()
    return (
        warnings.showwarning,
        warnings.formatwarning,
        warnings.defaultaction,
        *(item for item in filters if item is not _ITEM),
    )


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


def _code_of(hook: Any) -> Any:
    """The code object *hook* runs, or None (a builtin, a C method)."""
    seen = 0
    while seen < 16:
        seen += 1
        if isinstance(hook, functools.partial):
            hook = hook.func
            continue
        wrapped = getattr(hook, "__wrapped__", None)
        code = getattr(inspect.unwrap(hook) if wrapped is not None else hook, "__code__", None)
        if code is not None:
            return code
        func = getattr(hook, "__func__", None)
        if func is not None:
            hook = func
            continue
        call = getattr(type(hook), "__call__", None)
        if call is not None and not isinstance(hook, type) and getattr(call, "__code__", None) is not None:
            return call.__code__
        return None
    return None


def _hook_digest(hook: Any) -> str | None:
    """The digest of *hook*'s code when it is the user's, else None: a
    library's or the interpreter's hook shows a warning, and a key does not
    read how a line is shown."""
    held = _hook_digests.get(id(hook))
    if held is not None and held[0] is hook:
        return held[1]
    code = _code_of(hook)
    digest = callable_identity(hook) if code is not None and is_user_code_file(code.co_filename) else None
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
            digest = _hook_digest(hook)
            if digest is not None:
                parts.append(f"{name}={digest}")
    return "|".join(parts) if parts else None


def fingerprint() -> str | None:
    """What of the warnings state a result can see now (see the module
    doc), or None when it is plain Python's default."""
    state = _read()
    with _lock:
        if _last[0] == state:
            return _last[1]
    fp = _compute(state)
    with _lock:
        _last[0], _last[1] = state, fp
    return fp


# -- the fast hit path -------------------------------------------------------

#: The fingerprint the fast hit path's entries were offered under.
_SERVED: list[Any] = [None]


def _moved(bump: Any) -> Any:
    def moved() -> None:
        # The C side saw the state change since its last look: an entry
        # stamped under the old fingerprint may not hold, unless it is the same.
        fp = fingerprint()
        if fp != _SERVED[0]:
            _SERVED[0] = fp
            bump()

    return moved


def watch(cw: Any) -> None:
    """Have the fast hit path (`_cwatch.fast_call`) compare the warnings
    state on every hit, and drop every remembered hit (the cash epoch) when
    `fingerprint` moved."""
    if not hasattr(cw, "warn_watch"):
        return
    _SERVED[0] = fingerprint()
    cw.warn_watch(warnings.__dict__, _GETTER, _moved(cw.bump_cash))
    # A hit remembered before was checked under a state not looked at here.
    cw.bump_cash()
