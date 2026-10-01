"""How a function is named and identified by its code: its namespace
(`func_key`), the digest of its source (`hash_callable_source`), the
fingerprint of its compiled code, and the source pinned when it was
decorated (`OwnSourcePins`)."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import pickle
import types
import weakref
from collections.abc import Callable
from typing import Any

from .._memo import CODE_OBJECTS, LruMemo
from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning
from ..object_hashing import stable_key_repr
from ..source_norm import (
    bytecode_identity,
    callable_identity,
    code_consts_without_docstring,
    compiled_identity,
    loaded_class_identity,
    loaded_code_matches_disk,
    own_source_digest,
    source_digest,
    unwrap_partials,
)

logger = logging.getLogger(__name__)


# id(code object) -> (the object itself, its source digest). Keyed by IDENTITY,
# and the object is retained so the id cannot be recycled under us -- the same
# guard `function_tracker._source_cache` uses.
#
# NOT keyed on the code object directly: CodeType implements __eq__/__hash__
# BY VALUE, and co_filename is not part of that equality, so two helpers with
# the same body in different modules would share one slot.
#
# A redefinition (reloaded module, re-run cell) compiles a NEW code object, so
# identity keying still cannot serve a digest for code that is no longer
# running. Editing a .py file WITHOUT reloading leaves the old code object
# live, and the old digest is then the correct answer.
#
# Load-bearing, not a micro-optimisation. `hash_callable_source` is the live
# per-call identity of every transitive helper, and it calls
# `inspect.getsource`, which re-reads and RE-TOKENISES the source block on
# every call: without the memo, most of a hit's key computation.
#
# Module-level rather than per-instance: the digest depends only on the code
# object, so two Cash instances cannot legitimately disagree about it.
SOURCE_HASH_MEMO: LruMemo[int, tuple[Any, str]] = LruMemo(CODE_OBJECTS)
#: ``id(code) -> (code, path, size, mtime_ns, text digest)``: the stat of the
#: file whose text a function's key was read from, taken just before reading
#: it, and that text's digest. The store compares both with the file now
#: (`FileDeps.code_moved_since_keyed`).
CODE_KEYED_STATS: LruMemo[int, tuple[Any, str, int, int, str]] = LruMemo(CODE_OBJECTS)


#: Source files already reported as edited-since-load, one notice per file.
_SOURCE_CHANGED_WARNED: set[str] = set()


def stat_code_file(fn: Any) -> tuple[Any, str, int, int] | None:
    """``(code, path, size, mtime_ns)`` for *fn*'s source file, or None."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if not path or path.startswith("<"):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (code, path, st.st_size, st.st_mtime_ns)


def warn_source_changed_since_load(fn: Callable) -> None:
    """Say, once per file, that a helper is keyed by its loaded code."""
    code = getattr(fn, "__code__", None)
    path = getattr(code, "co_filename", "") or ""
    if path in _SOURCE_CHANGED_WARNED:
        return
    _SOURCE_CHANGED_WARNED.add(path)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "a function")
    try:
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "KEY-SOURCE-CHANGED",
            f"{path} was edited after this process loaded it, so the code running "
            f"{name}() is the old version while the file holds a new one. cash "
            f"keys it by the code actually running, so results stay correct for "
            f"this process -- but they are not the new code's results, and they "
            f"will not be reused once the process restarts.",
            "restart the process to run the new code. If a deploy puts new files "
            "on disk before the restart, this is the window it opens.",
        )
    except Exception:  # a notice must never break a call
        logger.debug("Could not emit the source-changed notice", exc_info=True)


def func_key(func: Callable) -> str:
    """Return a module-qualified key for a function.

    Uses ``func.__module__ + '.' + func.__qualname__`` to avoid collisions
    when different modules define functions with the same ``__qualname__``
    (e.g. a notebook's ``dep()`` vs a library module's ``dep()``).

    ``__main__`` is resolved to the name the module would have when
    imported — see `resolve_main_module`.

    A ``functools.partial`` is named after the function it wraps plus a
    digest of what it binds, a callable instance after its class. Anything
    else without ``__qualname__`` or ``__name__`` falls back to ``repr`` so
    keying it never crashes.
    """
    if isinstance(func, functools.partial):
        # `repr(partial)` holds the wrapped function's ADDRESS, so every
        # process took a fresh namespace and none of them ever hit. Name it after what it
        # wraps, plus what it binds -- two partials of one function stay
        # two namespaces, and each is the same in every process.
        # The bound values' content, not their ``repr``: that holds an
        # ordinary object's address, and a new process never found the
        # namespace again. Only a name: the values themselves are keyed per
        # call (`ClosureFold.fold_bound_partial`).
        inner = func_key(func.func)
        try:
            bound = hashlib.sha256(
                pickle.dumps(stable_key_repr((func.args, sorted(func.keywords.items()))), protocol=4),
            ).hexdigest()[:12]
        except Exception:  # noqa: BLE001 - an unpicklable argument keys on its type
            shape = [type(v).__qualname__ for v in (*func.args, *func.keywords.values())]
            bound = hashlib.sha256(repr((shape, sorted(func.keywords))).encode("utf-8")).hexdigest()[:12]
        return f"{inner}[partial:{bound}]"
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if not qualname and callable(func) and not isinstance(func, type):
        # A callable INSTANCE (`cash.cache(Scaler(2))`) reprs with its address,
        # so every process took a fresh namespace and none ever hit. Named
        # after its class; what it holds is keyed per call
        # (`ClosureFold.fold_bound_self`).
        cls = type(func)
        module = getattr(cls, "__module__", None) or "__unknown__"
        if module in MAIN_MODULE_NAMES:
            call = getattr(cls, "__call__", None)
            module = resolve_main_module(call) if hasattr(call, "__globals__") else module
        return f"{module}.{cls.__qualname__}[instance]"
    module = getattr(func, "__module__", None) or "__unknown__"
    if module in MAIN_MODULE_NAMES:
        module = resolve_main_module(func)
    return f"{module}.{qualname or repr(func)}"


def hash_callable_source(fn: Callable) -> str:
    """Return a stable hex digest representing *fn*'s body.

    Used by `register_hasher` to embed the hasher's source
    identity in the cache key, so that changing a hasher's body
    invalidates dependent cache entries even when the hasher's
    output coincidentally matches the old one. Also the
    ``hash_callable`` injected into ``SysModulesHelperResolver``,
    which makes it the live per-call identity of every transitive
    HELPER -- so what this returns decides whether editing a helper
    recomputes its callers.

    This is `cash.source_norm.callable_identity`, plus a memo per code
    object and a check that the file still holds the code that runs.

    Resolution order:

    1. ``own_source(fn)`` - primary (for a ``functools.wraps`` wrapper its
       own code, with the identity of what it wraps folded in), reduced via
       ``source_identity_digest`` so that a comment, a reformat, or a
       change to cash's own ``@....cache`` decorator arguments in a
       helper does not invalidate the functions that call it. Works
       for module-level functions and lambdas defined in a
       discoverable source file.
    2. ``bytecode_identity(fn)`` - fallback. Works for functions defined
       in a REPL or via ``exec()``, and for callable instances (it reads
       ``__call__``), so two instances of the same callable class share
       one identity. Folds consts/names/varnames, NOT ``co_code`` alone:
       a const load's operand is an index, so bare ``co_code`` cannot
       see ``return "alpha"`` become ``return "omega"``. Bytecode is
       stable within a Python version; an upgrade conservatively
       invalidates the cache.
    3. ``opaque_identity(fn)`` (``module.qualname``) - last resort, for a
       builtin, a ufunc or a partial. Doesn't differentiate instances of
       the same class; stable across processes, but coarse.
    """
    memo_owner: Any = getattr(fn, "__code__", None)
    if (memo_owner is None and isinstance(fn, type)) or (
        isinstance(fn, types.FunctionType) and hasattr(fn, "__wrapped__")
    ):
        # A class has no code object; a `functools.wraps` wrapper shares
        # its code with every function its decorator wraps, and its
        # identity includes the one it wraps -- so it is memoized as itself.
        memo_owner = fn
    memo_key = id(memo_owner) if memo_owner is not None else None
    if memo_key is not None:
        entry = SOURCE_HASH_MEMO.get(memo_key)
        # ``is``, not ``==``: confirms this is the SAME object and not a
        # recycled id, and sidesteps CodeType's by-value equality.
        if entry is not None and entry[0] is memo_owner:
            return entry[1]

    # The source on disk may no longer be the code that is running: a file
    # edited after this process imported it (new files land, the restart
    # comes later) gives the NEW text for the OLD code object; an entry keyed
    # by the new text but computed by the old code would be served to the
    # restarted process. Key such a helper by what actually runs.
    # One os.stat in the normal case; see `loaded_code_matches_disk`.
    if not loaded_code_matches_disk(fn):
        digest = loaded_class_identity(fn) if isinstance(fn, type) else compiled_identity(fn)
        if digest is not None:
            warn_source_changed_since_load(fn)
            if memo_key is not None:
                SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
            return digest

    # `callable_identity`, in its two halves: only a digest read from the
    # file is recorded against the file's stat.
    keyed_stat = stat_code_file(fn)
    digest = source_digest(fn)
    if digest is None:
        digest = compiled_identity(fn)
        # A function with no readable source (a dataclass's generated
        # `__init__`, an exec'd helper) is its bytecode, which its code
        # object fixes: memoized like a source digest, or every hit re-ran a
        # failing `inspect.getsource` per generated method.
        if memo_key is not None and isinstance(memo_owner, types.CodeType):
            SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
        return digest
    if memo_key is not None:
        SOURCE_HASH_MEMO[memo_key] = (memo_owner, digest)
    if keyed_stat is not None:
        own = digest if memo_owner is not fn else own_source_digest(fn)
        if own is not None:
            CODE_KEYED_STATS[id(keyed_stat[0])] = (*keyed_stat, own)
    return digest


def code_fingerprint(code: types.CodeType) -> str:
    """A digest of what a code object DOES, independent of where it sits.

    Source text is not enough on its own for a lambda: two different
    lambdas written on the SAME physical line share their
    ``inspect.getsource`` result, so ``a(lambda: "AAA"), a(lambda: "BBB")``
    fingerprint identically and collide. Their code objects differ, which
    is the signal this reads.

    Deliberately built from ``co_code``/``co_names``/``co_varnames`` and the
    constants, never from ``repr`` of a nested code object -- that carries a
    memory address, which would make the key unstable across processes and
    turn every restart into a miss. Nested code (a lambda inside a lambda)
    recurses instead, however deep: code objects form a finite tree, and a
    level left out would let two lambdas differing only there collide.
    """
    parts: list[str] = [
        code.co_code.hex(),
        repr(code.co_names),
        repr(code.co_varnames),
        repr(code.co_freevars),
    ]
    for const in code_consts_without_docstring(code):
        if isinstance(const, types.CodeType):
            parts.append(code_fingerprint(const))
        elif isinstance(const, frozenset):
            # `x in {"a", "b"}` compiles to a frozenset, whose repr follows
            # the per-process string hash: its sorted members instead.
            parts.append(f"frozenset({sorted(map(repr, const))})")
        else:
            parts.append(repr(const))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class OwnSourcePins:
    """The identity of each cached function's own source, pinned per
    function object when it is decorated (`OwnSourcePins.pin_own_source`)."""

    #: Pins held at once. Not a memo: a pin is the text on disk when the
    #: decorator ran and cannot be taken again later, so a full table keeps
    #: the pins it has (see `pin_own_source`).
    OWN_PINS_MAX = 4096

    def __init__(self) -> None:
        # id(func) -> (reference to func, decoration-pinned own-source
        # identity). The reference is checked on every read: a redefined
        # function's id can go to a later definition once the old one dies.
        self._own_pins: dict[int, tuple[Callable[[], Any], str]] = {}
        # Pins taken at decoration whose file has not yet been compared with
        # the loaded code; the first call does it once (see `pin_own_source`).
        self._own_pins_unverified: set[int] = set()

    def pin_own_source(self, func: Callable, source_hash: str | None = None) -> str:
        """Identity of *func* itself, pinned per function object.

        The state hash's root component must describe the function the
        wrapper EXECUTES, not the live ``source_hashes[qualname]`` registry
        slot — a redefinition (notebook cell re-run) or a second lambda
        sharing the ``<lambda>`` qualname overwrites that slot, letting a
        stale wrapper store its results under the new function's identity.

        For named functions the pin is the registration-time source hash.
        Lambdas additionally fold `code_fingerprint`: two lambdas defined on
        the SAME source line share their source text, and only their code,
        nested code included, tells them apart.

        Taken when the decorator runs (*source_hash* is the hash registration
        just computed), because that is when the text on disk is the text the
        import compiled. Taken at the first call instead, a deploy that lands
        new files between import and that call would key the old body's
        result by the new body's text, and every restarted process would
        serve it as an ordinary hit.
        The first call still compares the loaded code with the file once, to
        say so. A pin not taken at decoration (the table is full) falls back to
        the same loaded-vs-disk check helpers get.
        """
        key = id(func)
        owner = func
        # A partial has no code of its own, and the fallbacks below then keyed
        # on a repr carrying the wrapped function's ADDRESS -- a different pin
        # in every process, so a cached partial never hit across processes.
        # What it wraps is the code that runs; what it binds is already in the
        # namespace name (`get_func_key`).
        func = unwrap_partials(func)
        # An id outlives nothing: once a redefined function dies, a later
        # definition can get its address. So an entry counts only while it
        # still refers to this very object, and the decorator (which passes
        # the hash it just computed) always takes a fresh pin.
        entry = self._own_pins.get(key)
        pin = entry[1] if entry is not None and source_hash is None and entry[0]() is owner else None
        if pin is not None:
            if self._own_pins_unverified and key in self._own_pins_unverified:
                self._own_pins_unverified.discard(key)
                if not loaded_code_matches_disk(func):
                    warn_source_changed_since_load(func)
                    # The file changed before the decorator ran -- after the
                    # module was compiled, while it was still importing -- so
                    # the text the pin was read from is not the code that runs.
                    # Keyed by what runs instead: the result belongs to the old
                    # body, and a process running the new one keys by the new
                    # text and recomputes.
                    live = bytecode_identity(func)
                    if live is not None:
                        pin = live
                        self._own_pins[key] = (entry[0], live)
            return pin
        at_decoration = source_hash is not None
        keyed_stat = stat_code_file(func)
        if source_hash is None:
            if loaded_code_matches_disk(func):
                source_hash = callable_identity(func)
            else:
                source_hash = bytecode_identity(func) or callable_identity(func)
                warn_source_changed_since_load(func)
                keyed_stat = None  # keyed by what runs, not by the file
        if keyed_stat is not None and id(keyed_stat[0]) not in CODE_KEYED_STATS:
            disk_digest = own_source_digest(func)
            if disk_digest is not None:
                CODE_KEYED_STATS[id(keyed_stat[0])] = (*keyed_stat, disk_digest)
        pin = source_hash
        if getattr(func, "__name__", "") == "<lambda>":
            code = getattr(func, "__code__", None)
            if code is not None:
                pin = hashlib.sha256(f"{pin}:{code_fingerprint(code)}".encode("utf-8")).hexdigest()
        self._own_pins_unverified.discard(key)
        if key in self._own_pins or len(self._own_pins) < self.OWN_PINS_MAX:
            self._own_pins[key] = (self._pin_owner_ref(owner, key), pin)
            if at_decoration:
                self._own_pins_unverified.add(key)
        return pin

    def _pin_owner_ref(self, owner: Any, key: int) -> Callable[[], Any]:
        """A zero-argument callable returning *owner* while it lives.

        A weak reference drops the pin when the function dies, so the table
        does not keep every redefinition alive; the few callables that take
        no weak reference are held strongly, which also keeps their id theirs.
        """
        pins, unverified = self._own_pins, self._own_pins_unverified

        def _drop(ref: weakref.ref) -> None:
            entry = pins.get(key)
            if entry is not None and entry[0] is ref:
                pins.pop(key, None)
                unverified.discard(key)

        try:
            return weakref.ref(owner, _drop)
        except TypeError:
            return lambda: owner
