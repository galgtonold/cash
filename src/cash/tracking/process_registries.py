"""What a statement registers with the process, outside any variable.

``atexit.register(flush)``, ``signal.signal(SIGUSR1, on_usr1)``,
``sys.excepthook = hook``, ``copyreg.pickle(K, reduce_k)``,
``logging.getLogger("lib").addHandler(h)``: each changes a registry the
process keeps, which no variable holds and no entry stores. A statement
restored from the cache does not change it again: after a restart the
handler was never installed, and a re-run that plain Python makes
register a second callback registered none.

`snapshot` reads each registry before a statement, where it can be read
cheaply and exactly, and `changed` names those that are not as they were
after it: the statement is not stored (`StatementProcessor`), and a kept
call is not kept (`call_unit`). What cash itself registers is left out:
its loggers (``cash.*``), its import hooks, the ``atexit`` callbacks it
registers through `register_atexit`, and the audit events of its own work.

Not read: ``sys.modules``, whose copy and comparison cost 50 us a statement
with two thousand modules loaded, and whose writes cash already sees in
the statement's text and the functions it calls, as it does for the
seeds of ``random`` and NumPy, which it puts back with the entry.

Two registries have no way to be read, so their calls are counted
instead: ``codecs.register`` (wrapped here, from the first snapshot on)
and ``sys.addaudithook`` (the ``sys.addaudithook`` audit event, which
Python raises to the hooks already installed: cash's own is,
`io_watch`).
"""

from __future__ import annotations

import _signal  # noqa: PLC2701 - signal.getsignal wraps it in Python, fifteen times as slow
import atexit
import codecs
import os
import signal
import sys
import threading
import warnings
from collections.abc import Callable
from typing import Any

from .._user_attrs import attribute_or, has_attribute

__all__ = [
    "IMPORT_FINALIZERS",
    "changed",
    "finalizers_made",
    "register_atexit",
    "registering_imports",
    "snapshot",
    "unregister_atexit",
]

#: ``atexit`` callbacks cash registered (`register_atexit`): not the user's.
_OWN_ATEXIT = [0]
#: ``atexit``'s own, before `_count_atexit` wraps them: cash's callbacks are
#: counted here alone.
_atexit_register = atexit.register
_atexit_unregister = atexit.unregister


def register_atexit(fn: Callable[..., object], *args: Any, **kwargs: Any) -> None:
    """``atexit.register``, counted as cash's own."""
    before = atexit._ncallbacks()
    _atexit_register(fn, *args, **kwargs)
    _OWN_ATEXIT[0] += atexit._ncallbacks() - before


def unregister_atexit(fn: Callable[..., object]) -> None:
    """``atexit.unregister`` of a callback `register_atexit` registered."""
    before = atexit._ncallbacks()
    _atexit_unregister(fn)
    _OWN_ATEXIT[0] -= before - atexit._ncallbacks()


#: Calls of ``codecs.register`` and audit hooks added, by anyone but cash.
_CODECS = [0]
_AUDIT_HOOKS = [0]
_counting = False
#: ``callback -> times registered`` through ``atexit.register`` since
#: counting began, and how many of those ``atexit.unregister`` took back:
#: ``atexit._ncallbacks()`` never goes down (``joblib.Parallel`` registers
#: a clean-up for each run and unregisters it when done).
_ATEXIT_REGISTERED: dict[Any, int] = {}
_ATEXIT_TAKEN_BACK = [0]
#: And those a library's code registered itself: its clean-up of what it
#: made for itself (``joblib.Parallel``'s temporary folder, tqdm's monitor
#: thread), which a hit, making none of it, does not need. Judged by who
#: calls ``register``, not by the callback: ``atexit.register(df.to_csv,
#: path)`` in a cell is the user's, and writes the file at exit.
_ATEXIT_LIBRARY = [0]


def _atexit_callbacks() -> int:
    return atexit._ncallbacks() - _OWN_ATEXIT[0] - _ATEXIT_TAKEN_BACK[0] - _ATEXIT_LIBRARY[0]


def _library_called(frame: Any) -> bool:
    """Whether *frame*, the caller of ``atexit.register``, runs a library's
    code, not the user's (a cell, a module of theirs)."""
    from ..install_paths import is_user_code_module

    name = attribute_or(frame, "f_globals", {}).get("__name__")
    module = sys.modules.get(name) if isinstance(name, str) else None
    return module is not None and not is_user_code_module(module)


def _count_atexit() -> None:
    register, unregister = _atexit_register, _atexit_unregister

    def counted_register(func: Callable[..., object], *args: Any, **kwargs: Any) -> Callable[..., object]:
        before = atexit._ncallbacks()
        result = register(func, *args, **kwargs)
        made = atexit._ncallbacks() - before
        if _library_called(sys._getframe(1)):
            _ATEXIT_LIBRARY[0] += made
            return result
        try:
            _ATEXIT_REGISTERED[func] = _ATEXIT_REGISTERED.get(func, 0) + made
        except TypeError:  # unhashable: never taken back as far as the count goes
            pass
        return result

    def counted_unregister(func: Callable[..., object]) -> None:
        unregister(func)
        try:
            _ATEXIT_TAKEN_BACK[0] += _ATEXIT_REGISTERED.pop(func, 0)
        except TypeError:
            pass

    for wrapper, original in ((counted_register, register), (counted_unregister, unregister)):
        wrapper.__module__ = "atexit"
        wrapper.__name__ = wrapper.__qualname__ = original.__name__
        wrapper.__doc__ = original.__doc__
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    atexit.register = counted_register  # type: ignore[assignment]
    atexit.unregister = counted_unregister  # type: ignore[assignment]


def _count_calls() -> None:
    """Start counting what cannot be read (see the module docstring)."""
    global _counting
    if _counting:
        return
    _counting = True
    register = codecs.register

    def counted_register(search_function: Callable[[str], object]) -> None:
        _CODECS[0] += 1
        return register(search_function)

    counted_register.__module__ = "codecs"
    counted_register.__name__ = counted_register.__qualname__ = "register"
    counted_register.__doc__ = register.__doc__
    counted_register.__wrapped__ = register  # type: ignore[attr-defined]
    codecs.register = counted_register
    _count_atexit()

    def added(_args: tuple) -> None:
        _AUDIT_HOOKS[0] += 1

    from . import io_watch

    io_watch.subscribe_always("sys.addaudithook", added)


_SIGNALS: tuple[int, ...] | None = None


def _signals() -> tuple[Any, ...]:
    # ``_signal.getsignal``: ``signal.getsignal`` makes an enum of each
    # handler in Python, fifteen times as slow for all of them.
    global _SIGNALS
    if _SIGNALS is None:
        found = []
        for number in sorted(signal.valid_signals()):
            try:
                _signal.getsignal(number)
            except (ValueError, OSError):
                continue
            found.append(number)
        _SIGNALS = tuple(found)
    return tuple(map(_signal.getsignal, _SIGNALS))


def _hooks() -> tuple[Any, ...]:
    return (
        sys.excepthook,
        sys.displayhook,
        sys.breakpointhook,
        sys.unraisablehook,
        threading.excepthook,
        sys.gettrace(),
        sys.getprofile(),
        threading.gettrace(),
        threading.getprofile(),
        sys.getrecursionlimit(),
        sys.getswitchinterval(),
        sys.get_int_max_str_digits(),
    )


def _cash_s(obj: Any) -> bool:
    module = attribute_or(obj, "__module__", None) or attribute_or(type(obj), "__module__", None)
    return isinstance(module, str) and (module == "cash" or module.startswith("cash."))


def _imports() -> tuple[Any, ...]:
    return (
        tuple(sys.path),
        tuple(finder for finder in sys.meta_path if not _cash_s(finder)),
        tuple(sys.path_hooks),
    )


#: A logger's state as `logging.getLogger` makes it.
_NEW_LOGGER = (0, True, False, (), ())


def _loggers() -> tuple[int, dict[str, tuple[Any, ...]]] | None:
    logging = sys.modules.get("logging")
    if logging is None:
        return None
    manager = logging.Logger.manager
    states = {"": _logger_state(logging.root)}
    for name, logger in list(manager.loggerDict.items()):
        if type(logger) is logging.PlaceHolder or name == "cash" or name.startswith("cash."):
            continue
        states[name] = _logger_state(logger)
    return manager.disable, states


def _logger_state(logger: Any) -> tuple[Any, ...]:
    # From its own ``__dict__``: no property of a subclass runs.
    state = vars(logger)
    return (
        state.get("level"),
        state.get("propagate"),
        state.get("disabled"),
        tuple(state.get("handlers") or ()),
        tuple(state.get("filters") or ()),
    )


def _same_loggers(before: Any, after: Any) -> bool:
    if before is None or after is None:
        return before is after
    if before[0] != after[0]:
        return False
    old, new = before[1], after[1]
    if old == new:
        return True
    # One `getLogger` made is as plain Python makes it again: not a change.
    return all(old.get(name, _NEW_LOGGER) == state for name, state in new.items()) and all(name in new for name in old)


_MISSING = object()


def _environment() -> dict[Any, Any]:
    data = attribute_or(os.environ, "_data", None)
    return dict(data) if isinstance(data, dict) else dict(os.environ)


def _matplotlib() -> dict[str, Any] | None:
    matplotlib = sys.modules.get("matplotlib")
    params = attribute_or(matplotlib, "rcParams", None) if matplotlib is not None else None
    return dict(dict.items(params)) if isinstance(params, dict) else None


def _numpy() -> tuple[Any, ...] | None:
    numpy = sys.modules.get("numpy")
    if numpy is None or not has_attribute(numpy, "geterr"):
        return None
    return (numpy.geterr(), numpy.get_printoptions())


def _decimal() -> tuple[Any, ...] | None:
    decimal = sys.modules.get("decimal")
    if decimal is None:
        return None
    context = decimal.getcontext()
    return (
        context.prec,
        context.rounding,
        context.Emin,
        context.Emax,
        context.capitals,
        context.clamp,
        dict(context.traps),
    )


#: Each registry: what the warning calls it, how it is read, and how two
#: reads are told the same (``==`` when None).
def _warning_filters() -> tuple[Any, tuple[Any, ...]]:
    return warnings.filters, tuple(warnings.filters)


def _same_filters(old: tuple[Any, tuple[Any, ...]], new: tuple[Any, tuple[Any, ...]]) -> bool:
    # Another list: inside a ``warnings.catch_warnings()`` (cash's own
    # around a call it captures), which puts the first back when it ends.
    return old[0] is not new[0] or old[1] == new[1]


_REGISTRIES: tuple[tuple[str, Callable[[], Any], Callable[[Any, Any], bool] | None], ...] = (
    ("atexit's callbacks (atexit.register)", _atexit_callbacks, None),
    ("a signal's handler (signal.signal)", _signals, None),
    (
        "a hook or limit of the interpreter's (sys.excepthook, sys.displayhook, threading.excepthook, "
        "sys.settrace, sys.setprofile, sys.setrecursionlimit and the like)",
        _hooks,
        None,
    ),
    ("the warnings filters (warnings.filterwarnings, warnings.simplefilter)", _warning_filters, _same_filters),
    ("a logger's level, handlers or filters (logging)", _loggers, _same_loggers),
    ("pickle's reducers (copyreg.pickle)", lambda: dict(_copyreg().dispatch_table), None),
    ("the codec search functions (codecs.register)", lambda: _CODECS[0], None),
    ("the audit hooks (sys.addaudithook)", lambda: _AUDIT_HOOKS[0], None),
    ("where imports are found (sys.path, sys.meta_path, sys.path_hooks)", _imports, None),
    ("the environment (os.environ)", _environment, None),
    ("the working directory (os.chdir)", os.getcwd, None),
    ("matplotlib's rcParams", _matplotlib, None),
    ("NumPy's error handling or print options (np.seterr, np.set_printoptions)", _numpy, None),
    ("decimal's context", _decimal, None),
)


def _copyreg() -> Any:
    import copyreg

    return copyreg


def _read() -> list[Any]:
    out = []
    for _name, read, _same in _REGISTRIES:
        try:
            out.append(read())
        except Exception:  # noqa: BLE001 - a registry not read is not compared
            out.append(_MISSING)
    return out


def _same_read(index: int, old: Any, new: Any) -> bool:
    if old is _MISSING or new is _MISSING:
        return old is new
    same = _REGISTRIES[index][2]
    try:
        return same(old, new) if same is not None else old == new
    except Exception:  # noqa: BLE001 - not told the same
        return False


def _compare(before: list[Any], into: set[str]) -> None:
    for (name, read, same), old in zip(_REGISTRIES, before, strict=True):
        if old is _MISSING or name in into:
            continue
        try:
            new = read()
            equal = same(old, new) if same is not None else old == new
        except Exception:  # noqa: BLE001 - read before, not now: not as it was
            equal = False
        if not equal:
            into.add(name)


class Snapshot:
    """The registries as a statement or a call found them (`snapshot`),
    and those it changed outside an import so far."""

    __slots__ = ("__weakref__", "before", "found", "imported")

    def __init__(self) -> None:
        self.before = _read()
        self.found: set[str] = set()
        #: The modules first imported meanwhile that registered something
        #: (`registering_imports`).
        self.imported: list[str] = []


#: The snapshots still waiting for their `changed`; one dropped unasked
#: (its statement interrupted) goes with it.
_open: Any = None
#: How deep the imports under way on this thread are (``depth``).
_importing = threading.local()


def _load_unlocked(spec: Any) -> Any:
    """``importlib._bootstrap._load_unlocked``: what a module does to the
    registries while it is first imported is the module's, made once a
    process whoever imports it, which plain Python does not make again on
    the next run either (torch's loggers, warnings filters and ``atexit``
    callbacks, made on an optimizer's first step). On a registry the import
    changed, what the statement changed before is noted and what it finds
    after is where it starts from."""
    if not _open or attribute_or(_importing, "depth", 0):
        return _original_load(spec)
    watching = list(_open)
    pre = _read()
    _importing.depth = 1
    first = finalizers_made()
    try:
        return _original_load(spec)
    finally:
        _importing.depth = 0
        made = None
        if first is not None:
            made = finalizers_made() or first
            IMPORT_FINALIZERS.append((first, made))
        registered = made is not None and made != first
        # Only what the import itself changed, and only that registry
        # compared before it: in the middle of a statement, cash's own
        # scopes have swapped what they put back when they end (its
        # ``warnings.catch_warnings()`` around a call, IPython's display
        # hook while a statement's output is captured).
        now = _read()
        for index, (old, new) in enumerate(zip(pre, now, strict=True)):
            if _same_read(index, old, new):
                continue
            registered = True
            for each in watching:
                start = each.before[index]
                # Already where the import left it: another thread's import
                # running meanwhile ended first and moved it on.
                if not _same_read(index, start, old) and not _same_read(index, start, new):
                    each.found.add(_REGISTRIES[index][0])
                each.before[index] = new
        name = attribute_or(spec, "name", None)
        if registered and isinstance(name, str) and name in sys.modules:
            for each in watching:
                each.imported.append(name)


#: The ``weakref.finalize`` numbers (``info.index``) the imports made while
#: a snapshot was open, as ``(first, past the last)``: the module's, as its
#: registrations are (`cash.notebook.statement.unseen_writes.finalizer_left`).
IMPORT_FINALIZERS: list[tuple[int, int]] = []


def finalizers_made() -> int | None:
    """The number the next ``weakref.finalize`` takes, read without taking
    it; None when it cannot be read."""
    import weakref

    text = repr(weakref.finalize._index_iter)  # type: ignore[attr-defined]
    if text.startswith("count(") and text.endswith(")"):
        try:
            return int(text[6:-1])
        except ValueError:
            return None
    return None


_original_load: Callable[[Any], Any] | None = None


def _watch_imports() -> None:
    global _open, _original_load
    if _open is not None:
        return
    from .._weak import WeakSet

    _open = WeakSet()
    bootstrap = sys.modules["importlib._bootstrap"]
    _original_load = bootstrap._load_unlocked
    bootstrap._load_unlocked = _load_unlocked


def snapshot() -> Snapshot:
    """Each registry as it is now, for `changed`."""
    _count_calls()
    _watch_imports()
    if not _open:
        IMPORT_FINALIZERS.clear()
    taken = Snapshot()
    _open.add(taken)
    return taken


def registering_imports(before: Snapshot) -> list[str]:
    """The modules first imported since *before* (`snapshot`) whose import
    registered something (a logger's level, an ``atexit`` callback, a
    finalizer): a served entry imports them as the run did
    (`cash.notebook.cache_key.import_body_modules`), or after a restart
    the registration was never made (``zzlazy.sub`` reached through a
    package's ``__getattr__``)."""
    return list(dict.fromkeys(before.imported))


def changed(before: Snapshot) -> list[str]:
    """The registries not as *before* (`snapshot`) found them, by name,
    but for what an import made in between did to them."""
    _open.discard(before)
    _compare(before.before, before.found)
    return [name for name, _read_it, _same in _REGISTRIES if name in before.found]
