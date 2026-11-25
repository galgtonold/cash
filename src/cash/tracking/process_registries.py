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

Three registries have no way to be read, so their calls are counted
instead: ``codecs.register`` (wrapped here, from the first snapshot on),
``sys.addaudithook`` (the ``sys.addaudithook`` audit event, which
Python raises to the hooks already installed: cash's own is,
`io_watch`) and the environment ``os.putenv`` and ``os.unsetenv`` change
outside ``os.environ`` (their audit events).

An environment write the user waived where it was made (``with
cash.assume_safe():`` around it, ``# @cash:assume-safe`` on its line or on
the ``def`` line of a function on the way, as a file write is waived:
`register_env_waiver`) is not counted, and the key it wrote is left out when
``os.environ`` is compared. Every write through ``os.environ`` calls
``os.putenv`` or ``os.unsetenv``, so a later write of that key the user did
not waive is still counted.

The variable cash hands its pool workers (`cash._location.LAUNCH_ENV`),
which it writes again each time it resolves its settings, is cash's own: a
write of it, from cash's work in the middle of a statement, is neither
counted nor compared.

What a waived stretch of a cached call changes in the other registries is
left out the way an import's is: `read_registries` where the stretch
begins, `waive_since` where it ends (`cash.effect_observer`).
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

from .._location import LAUNCH_ENV
from .._user_attrs import attribute_or, has_attribute
from ..loaded_modules import loaded_module

__all__ = [
    "IMPORT_FINALIZERS",
    "IMPORT_STEP_CODE",
    "changed",
    "drop_trace_stand_in",
    "finalizers_made",
    "importing",
    "read_registries",
    "register_atexit",
    "register_env_waiver",
    "registering_imports",
    "snapshot",
    "stand_in_trace",
    "unregister_atexit",
    "waive_since",
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
#: Calls of ``os.putenv`` and ``os.unsetenv``: they change the environment
#: a child process and C code read, and not ``os.environ``.
_NATIVE_ENVIRONMENT = [0]
#: The keys of the environment (as ``os.environ`` holds them) a waived
#: ``os.putenv`` or ``os.unsetenv`` wrote: not compared (`_same_environment`).
_WAIVED_ENVIRONMENT: set[Any] = set()
#: The keys of the environment cash writes for itself (`LAUNCH_ENV`), in
#: each spelling ``os.environ`` and the audit events hold them: never the
#: statement's write.
_OWN_ENVIRONMENT = frozenset({LAUNCH_ENV, LAUNCH_ENV.upper(), os.fsencode(LAUNCH_ENV)})
#: Asked, in order, whether the user waived the environment write being
#: audited: True or False, or None to leave it to the next. None from all:
#: not waived.
_ENV_WAIVERS: list[Callable[[], bool | None]] = []
_counting = False
#: ``callback -> times registered`` through ``atexit.register`` since
#: counting began, and how many of those ``atexit.unregister`` took back
#: while ``atexit._ncallbacks()`` stayed as it was: up to Python 3.13 it
#: never goes down, from 3.14 an unregister takes the entries out of it
#: (``joblib.Parallel`` registers a clean-up for each run and unregisters
#: it when done).
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
        before = atexit._ncallbacks()
        unregister(func)
        removed = before - atexit._ncallbacks()
        try:
            registered = _ATEXIT_REGISTERED.pop(func, 0)
        except TypeError:
            registered = 0
        # What the count dropped by (3.14 on) is taken back already; the
        # rest of the user's registrations of *func* is taken back here.
        users = min(registered, removed)
        _ATEXIT_TAKEN_BACK[0] += registered - users
        # The rest of the drop: callbacks a library registered itself, which
        # it takes back as its run ends (joblib's temporary-folder clean-up,
        # unregistered once the folder is deleted). Only when the library
        # takes them back: the user's unregister of a library's callback
        # changes what runs at exit.
        if removed > users and _library_called(sys._getframe(1)):
            _ATEXIT_LIBRARY[0] -= removed - users

    for wrapper, original in ((counted_register, register), (counted_unregister, unregister)):
        wrapper.__module__ = "atexit"
        wrapper.__name__ = wrapper.__qualname__ = original.__name__
        wrapper.__doc__ = original.__doc__
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
    atexit.register = counted_register  # type: ignore[assignment]
    atexit.unregister = counted_unregister  # type: ignore[assignment]


def register_env_waiver(waiver: Callable[[], bool | None], first: bool = False) -> None:
    """Ask *waiver*, from now on, whether the user waived an ``os.putenv``
    or ``os.unsetenv`` being audited (``with cash.assume_safe():``, ``#
    @cash:assume-safe``): True or False where it can tell (a cached call it
    observes), None to leave it to the others. *first*: asked before those
    registered already."""
    if first:
        _ENV_WAIVERS.insert(0, waiver)
    else:
        _ENV_WAIVERS.append(waiver)


def _env_write_waived() -> bool:
    for waiver in tuple(_ENV_WAIVERS):
        try:
            verdict = waiver()
        except Exception:  # noqa: BLE001 - a waiver that cannot tell does not waive
            return False
        if verdict is not None:
            return verdict
    return False


def _environment_key(key: Any) -> Any:
    """*key* of an ``os.putenv`` event as ``os.environ`` holds it: upper
    case on Windows, bytes as given elsewhere."""
    if os.name == "nt" and isinstance(key, str):
        return key.upper()
    return key


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

    def set_natively(args: tuple) -> None:
        if args and _environment_key(args[0]) in _OWN_ENVIRONMENT:
            return
        if args and _env_write_waived():
            _WAIVED_ENVIRONMENT.add(_environment_key(args[0]))
            return
        _NATIVE_ENVIRONMENT[0] += 1

    io_watch.subscribe_always(("os.putenv", "os.unsetenv"), set_natively)


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


#: ``id(trace function) -> [it, the trace function it stands in for, how
#: many installs of it are live]``: one cash installs for a while
#: (`stand_in_trace`), read as the one it replaced. Counted, because the
#: same function stands in on several threads at once (a thread pool of
#: waived calls), and one thread's call ending must not end another's.
_TRACE_STAND_INS: dict[int, list[Any]] = {}
_STAND_INS_LOCK = threading.Lock()


def stand_in_trace(installed: Any, replaced: Any) -> None:
    """Read ``sys.gettrace()`` as *replaced* while it is *installed*, a
    trace function of cash's own (`drop_trace_stand_in` ends it)."""
    with _STAND_INS_LOCK:
        entry = _TRACE_STAND_INS.get(id(installed))
        if entry is not None and entry[0] is installed:
            entry[2] += 1
        else:
            _TRACE_STAND_INS[id(installed)] = [installed, replaced, 1]


def drop_trace_stand_in(installed: Any) -> None:
    """Read ``sys.gettrace()`` as it is again once *installed* is gone
    (on every thread, once each `stand_in_trace` of it has ended)."""
    with _STAND_INS_LOCK:
        entry = _TRACE_STAND_INS.get(id(installed))
        if entry is None or entry[0] is not installed:
            return
        entry[2] -= 1
        if entry[2] <= 0:
            del _TRACE_STAND_INS[id(installed)]


def _trace() -> Any:
    current = sys.gettrace()
    stand_in = _TRACE_STAND_INS.get(id(current)) if _TRACE_STAND_INS else None
    return stand_in[1] if stand_in is not None and stand_in[0] is current else current


def _hooks() -> tuple[Any, ...]:
    return (
        sys.excepthook,
        sys.displayhook,
        sys.breakpointhook,
        sys.unraisablehook,
        threading.excepthook,
        _trace(),
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
    logging = loaded_module("logging")
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


def _environment() -> tuple[dict[Any, Any], int]:
    """``os.environ`` but for cash's own keys (`_OWN_ENVIRONMENT`), and how
    many times ``os.putenv`` or ``os.unsetenv`` was called (the audit
    events: also when imported by name), which change the process's
    environment and leave ``os.environ`` alone."""
    data = attribute_or(os.environ, "_data", None)
    found = dict(data) if isinstance(data, dict) else dict(os.environ)
    for key in _OWN_ENVIRONMENT:
        found.pop(key, None)
    return found, _NATIVE_ENVIRONMENT[0]


def _same_environment(old: tuple[dict[Any, Any], int], new: tuple[dict[Any, Any], int]) -> bool:
    """The same but for the keys a waived write changed (`_WAIVED_ENVIRONMENT`):
    each write of the user's not waived is counted, and the count compared."""
    if old[1] != new[1]:
        return False
    if old[0] == new[0]:
        return True
    waived = _WAIVED_ENVIRONMENT
    keys = (old[0].keys() | new[0].keys()) - waived
    return all(old[0].get(key, _MISSING) == new[0].get(key, _MISSING) for key in keys)


def _matplotlib() -> dict[str, Any] | None:
    matplotlib = loaded_module("matplotlib")
    params = attribute_or(matplotlib, "rcParams", None) if matplotlib is not None else None
    return dict(dict.items(params)) if isinstance(params, dict) else None


def _numpy() -> tuple[Any, ...] | None:
    numpy = loaded_module("numpy")
    if numpy is None or not has_attribute(numpy, "geterr"):
        return None
    return (numpy.geterr(), numpy.get_printoptions())


def _decimal() -> tuple[Any, ...] | None:
    decimal = loaded_module("decimal")
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
def _not_cash(item: Any) -> bool:
    # The filter cash puts first while a cached body runs, on any thread
    # (`warning_state.recording`): not the call's.
    return not (type(item) is tuple and len(item) > 1 and getattr(item[1], "cash_probe", False) is True)


def _warning_filters() -> tuple[Any, ...]:
    # ``defaultaction`` too: the action of a warning no filter matches.
    return warnings.filters, tuple(filter(_not_cash, warnings.filters)), warnings.defaultaction


def _same_filters(old: tuple[Any, ...], new: tuple[Any, ...]) -> bool:
    # Another list: inside a ``warnings.catch_warnings()`` (cash's own
    # around a call it captures), which puts the first back when it ends.
    return old[0] is not new[0] or old[1:] == new[1:]


def _warning_display() -> tuple[Any, ...]:
    # Read off the module, where ``warnings.warn`` looks them up: a call
    # that rebinds one (``warnings.formatwarning = f``) changes how every
    # later warning of the process reads, and a hit would not rebind it.
    return warnings.filters, warnings.formatwarning, warnings.showwarning


def _same_display(old: tuple[Any, ...], new: tuple[Any, ...]) -> bool:
    # Another filters list: inside a ``catch_warnings()`` as above, which
    # puts both hooks back too when it ends.
    return old[0] is not new[0] or all(a is b for a, b in zip(old[1:], new[1:], strict=True))


_REGISTRIES: tuple[tuple[str, Callable[[], Any], Callable[[Any, Any], bool] | None], ...] = (
    ("atexit's callbacks (atexit.register)", _atexit_callbacks, None),
    ("a signal's handler (signal.signal)", _signals, None),
    (
        "a hook or limit of the interpreter's (sys.excepthook, sys.displayhook, threading.excepthook, "
        "sys.settrace, sys.setprofile, sys.setrecursionlimit and the like)",
        _hooks,
        None,
    ),
    (
        "the warnings filters (warnings.filterwarnings, warnings.simplefilter, warnings.defaultaction)",
        _warning_filters,
        _same_filters,
    ),
    ("how warnings are shown (warnings.formatwarning, warnings.showwarning)", _warning_display, _same_display),
    ("a logger's level, handlers or filters (logging)", _loggers, _same_loggers),
    ("pickle's reducers (copyreg.pickle)", lambda: dict(_copyreg().dispatch_table), None),
    ("the codec search functions (codecs.register)", lambda: _CODECS[0], None),
    ("the audit hooks (sys.addaudithook)", lambda: _AUDIT_HOOKS[0], None),
    ("where imports are found (sys.path, sys.meta_path, sys.path_hooks)", _imports, None),
    ("the environment (os.environ, os.putenv, os.unsetenv)", _environment, _same_environment),
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
    with _open_lock:
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
                _start_from(each, index, old, new)
        name = attribute_or(spec, "name", None)
        if registered and isinstance(name, str) and name in sys.modules:
            for each in watching:
                each.imported.append(name)


def _start_from(each: Snapshot, index: int, old: Any, new: Any) -> None:
    """*each* starts registry *index* from *new*, which a stretch it does not
    count (an import, a waiver) moved it to from *old*: a change of the
    caller's own made before the stretch is noted first."""
    start = each.before[index]
    # Already where the stretch left it: another thread's import running
    # meanwhile ended first and moved it on.
    if not _same_read(index, start, old) and not _same_read(index, start, new):
        each.found.add(_REGISTRIES[index][0])
    each.before[index] = new


def read_registries() -> list[Any]:
    """Each registry as it is now, where a waived stretch of a call begins
    (`waive_since`)."""
    return _read()


def waive_since(before: Snapshot, start: list[Any]) -> None:
    """Leave out of *before* (`snapshot`) what the registries changed since
    *start* (`read_registries`): a stretch of the call the user waived
    (``with cash.assume_safe():``, ``# @cash:assume-safe``). What the call
    changed before it, or changes after it, still counts."""
    now = _read()
    for index, (old, new) in enumerate(zip(start, now, strict=True)):
        if not _same_read(index, old, new):
            _start_from(before, index, old, new)


def importing() -> bool:
    """Is a module being first imported on this thread while a snapshot is
    open? What it does then is the module's, as what it does to the
    registries is (`_load_unlocked`)."""
    return bool(attribute_or(_importing, "depth", 0))


#: The code of `_load_unlocked`, which `snapshot` puts in ``importlib``'s
#: place for the rest of the process: a frame of it is a step of the
#: import, as ``importlib``'s own are (`cash.backends.no_file_writes`).
IMPORT_STEP_CODE = _load_unlocked.__code__


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


#: Held while the open snapshots are added to, taken from or listed, and
#: while the import step is put in place: snapshots are taken on every
#: thread that runs a cached call, and a set changed by one thread while
#: another lists it raises. Re-entrant: listing can start a collection,
#: whose finalizers may import.
_open_lock = threading.RLock()


def _watch_imports() -> None:
    global _open, _original_load
    if _open is not None:
        return
    from .._weak import WeakSet

    with _open_lock:
        if _open is not None:
            return
        bootstrap = sys.modules["importlib._bootstrap"]
        current = bootstrap._load_unlocked
        # Never this module's own step as the one it calls on: two threads
        # taking their first snapshot at once each put it in place, the
        # second over the first, and every import after that called itself
        # until the stack ran out.
        if current is not _load_unlocked:
            _original_load = current
            bootstrap._load_unlocked = _load_unlocked
        # Last: the step passes straight through until there is a set.
        _open = WeakSet()


def snapshot() -> Snapshot:
    """Each registry as it is now, for `changed`."""
    _count_calls()
    _watch_imports()
    taken = Snapshot()
    with _open_lock:
        if not _open:
            IMPORT_FINALIZERS.clear()
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
    with _open_lock:
        _open.discard(before)
    _compare(before.before, before.found)
    return [name for name, _read_it, _same in _REGISTRIES if name in before.found]
